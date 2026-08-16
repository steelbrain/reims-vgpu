//! Vulkan-backend half of the [`super`] draw encode path: sampled-source
//! resolution, zero-copy load paths, metal2vulkan draw submission, and the
//! host-authoritative GVA/surface Store ownership.
//!
//! The whole module is gated on `backend-vulkan` at its declaration in
//! [`super`], which also re-exports these items flat so callers keep addressing
//! them as `crate::runtime::draw::<name>`. `use super::*` pulls in the
//! parent's imports, which this half shares.

use super::*;
use crate::contract::pixel_format::solid_bgra8;

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
    multisampled: bool,
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
    // Plain 2D, which every other shape is a single flag away from. `cube` is
    // false in all of them: the two cube kinds decline below rather than
    // reaching here, so nothing this function returns ever sets it.
    let d2 = SampledImageShape {
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        layers: 1,
    };
    Some(match kind {
        SampledImageKind::D1 => SampledImageShape {
            one_dim: true,
            ..d2
        },
        SampledImageKind::D1Array => SampledImageShape {
            one_dim: true,
            arrayed: true,
            ..d2
        },
        SampledImageKind::D2 => d2,
        SampledImageKind::D2Multisample => SampledImageShape {
            multisampled: true,
            ..d2
        },
        SampledImageKind::D2Array => SampledImageShape {
            arrayed: true,
            ..d2
        },
        SampledImageKind::D3 => SampledImageShape { volume: true, ..d2 },
        SampledImageKind::Cube | SampledImageKind::CubeArray => return None,
    })
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
    // The seeded solid colour of attachment 0, as its recipe rather than as its
    // pixels.
    //
    // This used to be the `Vec<u8>` itself, built by `solid_rgba8` inside the
    // seed loop. Only one of this function's five exits returns it — the
    // clear-only one at the bottom, where the record encoded no draw — and the
    // three that matter under compositing (`chain_resident_established`, an armed
    // Store, a real `draw_rgba`) each return something else and dropped it
    // unread. On a macos-11 Safari-torture leg the seed loop landed 2.87 GB of
    // solid colour through `write_gva_solid8`, which converts one row and repeats
    // it; the full-surface image built beside it was the same 2.87 GB of
    // allocate-and-fill, and 188 ms of the leg's 892 ms `prep_seed_us` sat
    // outside both landing spans, which is where it was.
    //
    // Held as `(width, height, colour)` so the recipe costs a few words and the
    // exit that wants pixels calls the same constructor with the same arguments.
    let mut color0_solid: Option<(u32, u32, [f64; 4])> = None;
    // The engine draw's own refusal slug, kept so the skipped-draw tail can name
    // why its draws were skipped instead of guessing. `None` means the engine
    // draw was never attempted — this record carried no pipeline or no vertices.
    let mut engine_refusal: Option<&'static str> = None;
    // Solid CLEAR seed Stores only when this record owns guest writeback
    // (last of a serialized chain, or unified always-writeback).
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::PrepSeed);
    if writeback_guest && clear_seed_enabled() {
        for (i, c) in colors.iter().enumerate() {
            // Reports an out-of-contract value; the gate below is unchanged, and
            // it is the site that disagrees with this module's writeback loop
            // about what such a value means. See `super::store_action_in_contract`.
            let _ = super::store_action_in_contract(req.pipeline_ref, c.store_action);
            if !crate::contract::pass_action::store_action_publishes_single_sample(c.store_action) {
                continue;
            }
            if c.load_action != MTL_LOAD_ACTION_CLEAR && c.load_action != MTL_LOAD_ACTION_DONT_CARE
            {
                // Load/composite needs real encode (metal2vulkan) — skip Store.
                continue;
            }
            if c.width == 0 || c.height == 0 {
                continue;
            }
            // Neither writer below takes a full-surface RGBA copy — the GVA
            // landing repeats a single row and the type-11 landing builds its own
            // image in the mapping's order — so the only reader of one is the
            // clear-only exit, and it is the one place that builds it.
            let solid = (i == 0).then_some((c.width, c.height, c.clear_color));
            // Which branch this loop actually takes, how many bytes it lands and
            // what the landing costs. `prep_seed_us` is 8.6 µs of a 41 µs chain
            // on the `blur=40` dial and rebuilding the images without their two
            // redundant passes did not move it by a hundredth, so the cost is in
            // one of these two writes and there was no reading that said which.
            let seed_kb = u64::from(c.width)
                .saturating_mul(u64::from(c.height))
                .saturating_mul(u64::from(RGBA8_BPP))
                / 1024;
            let ok = if c.target_gva != 0 {
                let _span = StoreCostSpan::new("clear_seed_gva_us");
                crate::runtime::drain::note_store_route("clear_seed_gva");
                crate::runtime::drain::note_store_route_n("clear_seed_gva_kb", seed_kb);
                // A solid landing, so the writer converts one row rather than
                // this surface's thousand identical ones. The full RGBA image
                // above is built for `color0_rgba` and is not this write's
                // source.
                write_gva_solid8(
                    state,
                    host,
                    req.task_id,
                    c.target_gva,
                    c.width,
                    c.height,
                    c.row_stride,
                    c.format,
                    &c.clear_color,
                )
                .is_ok()
            } else if c.mapping_id != 0 {
                // Type-11 CLEAR. `write_bgra8` takes guest scanout order and
                // converts to the mapping's native format per row; it handles a
                // fragmented mapping too, staging native rows and landing them
                // through `mapper::write_mapping_bytes`. (A comment here used to
                // call it contig-only, which it has not been.)
                //
                // Built from the swapped *pixel* rather than by exchanging the
                // channels of the RGBA image: a solid image is one repeated
                // word, so the exchange belongs to the word and doing it per
                // texel cost an allocation and two passes over the whole
                // surface. See `contract::pixel_format::solid_bgra8`.
                let _span = StoreCostSpan::new("clear_seed_t11_us");
                crate::runtime::drain::note_store_route("clear_seed_t11");
                crate::runtime::drain::note_store_route_n("clear_seed_t11_kb", seed_kb);
                let bgra = solid_bgra8(c.width, c.height, &c.clear_color);
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
                    color0_solid = solid;
                }
                crate::observe::line(format!(
                    "linux_clear_store mid={} gva={:#x} {}x{} pipe={} load={}",
                    c.mapping_id, c.target_gva, c.width, c.height, req.pipeline_ref, c.load_action
                ));
            }
        }
    }

    // Pages the eager GVA fallback below is allowed to reach, resolved
    // **before** any GPU work.
    //
    // The write was documented as needing no bound because "the command
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
    // Resolved here rather than at the fallback write so the set predates the
    // submit. The host-authoritative path keeps no pages from this walk; its
    // eventual transfer performs one fresh walk and verifies the allocation
    // generation before writing. `None` here means there is no GVA target, no
    // writeback, or the walk cannot name the span — an unresolvable span is not
    // an authorisation for the eager fallback to write anywhere.
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::PrepPages);
    let sync_store_pages =
        sync_store_allowed_pages(state, host, req.task_id, colors.first(), writeback_guest);
    // Back to `Prep`, which is now the residue: everything in this function
    // before the metal2vulkan call that is neither of the two spans above.
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Prep);
    // metal2vulkan path: load MTLB → AIR → SPIR-V → internal Vulkan engine offscreen.
    let mut draw_rgba: Option<Vec<u8>> = None;
    // Physical order of `draw_rgba`. A type-11 composite Store renders into a
    // BGRA `Surface` resident, so its readback is already in guest scanout
    // order; the pooled and GVA targets stay RGBA. Carried instead of assumed —
    // which of those a record hit depends on whether an identity resolved, and
    // that is not a condition the Store block can re-derive.
    let mut draw_bgra = false;
    // Type-11 composite Store: the frame was written into the mapping's guest
    // pages by `store_surface_resident`, so this encode owes the caller nothing
    // further.
    let mut surface_store_armed = false;
    // GVA render Store: the frame remains authoritative in the resident and a
    // resource-scoped debt records the future transfer. The twin of
    // `surface_store_armed`, and it returns through the same door.
    let mut gva_store_armed = false;
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
            Ok(M2vDrawSpan::ResidentGvaStore { identity }) => {
                let _store_span = StoreCostSpan::new("gva_store_us");
                note_type11_store_route("gva_flush");
                // Metal Store preserves the attachment in host GPU memory. It
                // does not synchronize that texture into guest backing; the
                // resource-validity protocol asks for that separately. The
                // live resource retains its transfer backing until explicit
                // discard or delete; the debt records only content ownership.
                let landed = req.colors.first().is_some_and(|c0| {
                    crate::runtime::writeback_debt::arm_gva(state, host, req.task_id, c0, &identity)
                });
                if landed {
                    note_type11_store_route("gva_resident_authoritative");
                    gva_store_armed = true;
                } else {
                    // The copying rail: read the resident the draw just
                    // rendered into and let the synchronous Store block below
                    // run exactly as it does for a Store that never skipped its
                    // readback. `read_resident_chain` fail-logs a lost resident.
                    note_type11_store_route("gva_store_sync");
                    draw_rgba = read_resident_chain(req, &identity);
                    crate::observe::line(format!(
                        "linux_m2v_draw ok resident_gva_store pipe={} {}x{} gva={:#x} rgba={}",
                        req.pipeline_ref,
                        pass_w,
                        pass_h,
                        req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                        draw_rgba.is_some() as u8
                    ));
                }
            }
            Ok(M2vDrawSpan::ResidentSurfaceStore {
                identity,
                guest_store,
            }) => {
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
                let stored = c0_store
                    .map(|(mid, cw, ch, _)| {
                        store_surface_resident(state, host, &identity, mid, cw, ch, guest_store)
                    })
                    .unwrap_or(false);
                match (stored, c0_store) {
                    (true, Some((mid, cw, ch, fmt))) => {
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
                            "linux_m2v_draw ok resident_surface_store pipe={} {}x{} mid={mid}",
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
                        draw_rgba = read_resident_chain(req, &identity);
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
                //
                // Kept for the tail below, because that latch is exactly what
                // makes the skipped-draw line unreadable on its own: this fires
                // once per (reason, pipeline) and the tail fires once per
                // packet, so a hundred skipped draws sit behind one decline.
                engine_refusal = Some(crate::observe::Decline::slug(&e));
                linux_m2v_draw_failure(&e, req).fail_once(req.pipeline_ref as u64);
            }
        }
    }

    // Everything below is Store routing, for both kinds of record, so charge it
    // as Store for both.
    //
    // `Phase::Store` used to be entered only *inside* `try_metal2vulkan_draw`,
    // which a record with no pipeline or no vertices never calls — so such a
    // record kept `Prep` open through its whole guest writeback, and `prep_us`
    // was the chain head for one class of record and the entire body for
    // another. That is not a phase. It is also not a small error: `prep_us` read
    // 0.89 µs a chain on one driven Maps boot and 4.69 on the next of the same
    // binary, with `store_us` moving 0.53 → 0.96 beside it, which is the ratio
    // of the two classes changing and nothing about the code.
    //
    // For a record that did draw, the open phase is already `Store` and this
    // charges the same accumulator it reopens, so the drawing path is unchanged
    // by construction rather than by a condition.
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Store);
    // A resident render-pass chain intermediate: the exec loop reads
    // `chain_resident_established` and arms the next record's LoadFromTarget.
    if req.chain_resident_established {
        return (EncodeStatus::Ok, None);
    }

    // Deferred type-11 composite Store: the window names the pinned resident and
    // the guest write lands on first access. `None`, not the frame, for the same
    // reason the `Owned` route returns `None` — `writeback_guest` is granted only
    // to the last record of a packet, so there is no record N+1 to seed.
    if surface_store_armed || gva_store_armed {
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
                // What this Store would cost if it were served the way the
                // type-11 surface Store is served.
                //
                // That rail lands its frame with `copy_target_to_guest_pages` —
                // the GPU writes the guest's pages and no byte crosses host
                // memory. This one reads the whole resident back to the host
                // first (`read_resident_chain`, a blocking fence) and then
                // writes it out again row by row. On a driven Safari drag the
                // split is 14 330 Stores here against 9 870 there, so the
                // copying rail carries 59 % of them.
                //
                // A buffer→image copy converts nothing, so the GPU rail can
                // only ever serve a destination whose bytes are already the
                // resident's. That is not a channel order this rail may name for
                // itself: `gva_resident_format` builds the resident from the
                // order the guest declared, so a GVA target is RGBA or BGRA
                // according to `c0.format` and nothing else. `format` is the
                // gate, and it is recorded rather than assumed: whether this
                // class is worth a GPU rail at all is exactly the question of
                // how many of the 14 330 clear it, and `convert_rgba8_to_row`
                // below is a per-row conversion for every one that does not.
                crate::runtime::drain::note_store_route(
                    if c0.format == pixel_format::MTL_FORMAT_RGBA8_UNORM
                        || c0.format == pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB
                    {
                        "gva_store_fmt_byte_identical"
                    } else {
                        "gva_store_fmt_needs_conversion"
                    },
                );
                if crate::observe::first_sight("gva_store_fmt", u64::from(c0.format)) {
                    crate::observe::off(format!(
                        "gva_store_fmt fmt={:#x} {}x{} stride={} \
                         (destination format of a GVA render Store)",
                        c0.format, c0.width, c0.height, c0.row_stride
                    ));
                }
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
                    sync_store_pages.as_ref().map(|p| p.membership()),
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
                        // Inside `if gva_ok`: this arm only runs when
                        // `write_gva_rgba8_within` landed the same bytes in the
                        // guest's pages, so they are re-derivable from there.
                        true,
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
                // reader of it in `exec` sits inside the record loop that just
                // ended. The intermediate handoff that *is* live returned above,
                // before the Store arms. Returning the frame here handed a whole
                // framebuffer to a binding that is dropped unread.
                return (EncodeStatus::Ok, None);
            }
        }
    }

    if any_store {
        if req.vertex_count > 0 || req.indexed.is_some() {
            // This used to end in the literal `(m2v pending)`, which was a
            // hardcoded guess and was wrong. The tail is reached whenever the
            // engine draw did not land, for any reason, and a translation still
            // being in flight is only one of them — on a driven macos-26 boot
            // the guess accounted for **none** of the 114 lines it printed.
            // Pipeline 160 alone produced 105 of them, from t=36351 to
            // t=112002, while its one translation was queued at t=36340 and
            // reported `done ... ok` at t=36351. The real refusals were a
            // sampled-texture validation check and the driver quarantine.
            //
            // `refused_by` is deliberately not spelled `reason=`: the census
            // ranks fail-channel lines on that key, and the underlying slug is
            // already counted once at its own emitter. Naming it twice would
            // make one refusal read as two.
            crate::observe::fail(format!(
                "linux_clear_store draws_skipped reason=draws_skipped_after_engine_refusal \
                 pipe={} vtx={} refused_by={}",
                req.pipeline_ref,
                req.vertex_count,
                engine_refusal.unwrap_or("engine_draw_not_attempted")
            ));
            // The line above dedupes on `(pipeline, slug)` and the count does
            // not, so the two answer different questions and only this one can
            // be added up. Reading the line count as the draw count understates
            // it by however many times one pipeline was refused — which on a
            // rail that composites the same layer every frame is the entire
            // magnitude. Nothing else in the census counts a draw the engine
            // refused, so "what did this refusal cost the guest" had no
            // instrument at all; a bare zero here now means no draw was
            // skipped, rather than meaning nobody was counting.
            //
            // The vertices are banded beside the draws because a skipped
            // six-vertex full-screen quad and a skipped fifty-four-vertex pass
            // are the same 1 in a draw count and are not the same loss.
            //
            // Deliberately **not** split by `engine_refusal` here. That slug is
            // already this crate's vocabulary and is counted at its own
            // emitter, so keying a census entry on it would merge two
            // populations under one name and make one refusal read as two —
            // the same reason `refused_by=` above is not spelled `reason=`.
            // The split lives on the fail line; the magnitude lives here.
            crate::runtime::drain::note_store_route("draws_skipped_after_engine_refusal");
            crate::runtime::drain::note_store_route_n(
                "draws_skipped_after_engine_refusal_vertices",
                u64::from(req.vertex_count),
            );
        }
        // The one exit that hands the seed on: this record encoded no draw, so
        // the solid colour it landed *is* this chain's colour-0 content, and the
        // next record loads it as its seed. Built here rather than in the loop
        // above because the four exits before this one return without it.
        //
        // The route is what says the deferral is worth having: read it against
        // `clear_seed_gva` + `clear_seed_t11`, which count the seeds, and the
        // difference is the full-surface images that used to be built and
        // dropped. A boot where the two are equal has nothing to save here.
        (
            EncodeStatus::Ok,
            color0_solid.map(|(w, h, clear)| {
                crate::runtime::drain::note_store_route("clear_seed_color0_image");
                crate::runtime::drain::note_store_route_n(
                    "clear_seed_color0_image_kb",
                    u64::from(w)
                        .saturating_mul(u64::from(h))
                        .saturating_mul(u64::from(RGBA8_BPP))
                        / 1024,
                );
                solid_rgba8(w, h, &clear)
            }),
        )
    } else {
        (EncodeStatus::NoMetal("draw_vk_nothing_stored"), None)
    }
}

/// Sampled texture source + geometry for an engine draw.
pub(super) enum SampledSourceRequest {
    /// Shared texel bytes + optional producer identity (see
    /// [`LinearSampleIdentity`]) + what those texels are; the Arc lets memoized
    /// repeat binds skip the per-draw copy and the engine skip re-hashing.
    ///
    /// The third field is a [`SampledByteFormat`] and not a bare `TexelLayout`
    /// because a layout is linear by construction. While it was one, every CPU
    /// upload of an sRGB guest texture reached the sampler through a `_UNORM`
    /// view and was never decoded, while the zero-copy rails beside it — which
    /// carry a resolved host format — bound the `_SRGB` spelling and were. One
    /// guest texture, two colours, and which one it got decided by a cost
    /// threshold. Each producer answers from the format it *loaded from*, so a
    /// convert that reorders channels keeps the transfer function it never
    /// touched.
    Bytes(
        std::sync::Arc<Vec<u8>>,
        Option<LinearSampleIdentity>,
        SampledByteFormat,
        crate::backend::vulkan::engine::SampledByteOrigin,
    ),
    /// Engine-resident allocation plus the exact view format this sampled
    /// texture declared. Allocation identity and view interpretation are
    /// separate parts of the texture contract.
    Target(
        crate::backend::vulkan::engine::TargetIdentity,
        ash::vk::Format,
    ),
    /// Zero-copy guest gather: the engine copies the texel bytes from
    /// imported guest RAM inside the draw CB — no CPU read, no memo, no
    /// hash. Carries the native texel layout the image is created with.
    /// Guest-RAM runs the engine gathers from, the byte layout of those texels,
    /// an optional copied-content identity, and what the guest-write witness
    /// says that identity is worth (see [`crate::runtime::gather_witness`]).
    ///
    /// A resource-owned direct image has no copied content to witness and
    /// carries no identity. A copy-backed source carries the identity that lets
    /// the engine bind a retained gathered image without gathering again. If a
    /// backend declines a supplied direct image, the absent identity makes the
    /// copy fallback gather conservatively on that bind.
    /// The last field is the **format's own** channel plan, not the guest's
    /// view swizzle: this rail binds guest bytes untouched, so a format whose
    /// Metal channels do not sit identically on the Vulkan format carrying them
    /// needs that difference expressed on the image view. It is composed with
    /// the type-8 view swizzle at the push site. Identity for every format but
    /// `A8Unorm`.
    GuestRuns(
        crate::backend::vulkan::engine::GuestRunSource,
        TexelLayout,
        /// Exact Vulkan image/view format. This is distinct from the texel
        /// layout because linear and sRGB formats carry the same bytes while
        /// applying different fixed-function sampling conversions.
        ash::vk::Format,
        /// Consecutive depth planes carried by the source window.
        u32,
        Option<LinearSampleIdentity>,
        crate::runtime::gather_witness::GatherVouch,
        pixel_format::SwizzlePlan,
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
    SampledByteFormat,
);
type LoadedLinearSample = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    SampledByteFormat,
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
    /// The attachment's `MTLClearColor`, which the guest states in the colour
    /// space the attachment decodes *to*. Metal encodes it on the way into an
    /// sRGB attachment and decodes it back on sample, so the value a shader
    /// reads is the one written here — this device stores that value directly
    /// and binds it linear. No attachment format is carried because none is
    /// needed: the round trip cancels.
    Clear([f64; 4]),
    /// The attachment's prior contents, read back out of its guest pages, plus
    /// the format they were **stored** in.
    ///
    /// The format is here and not on [`Self::Clear`] because a seed is stored
    /// bytes rather than a decoded value: an sRGB attachment holds encoded ones,
    /// and a bind that samples them linear hands the shader an undecoded value
    /// the next attachment write then encodes a second time.
    Seed(&'a [u8], u16),
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
        MTL_LOAD_ACTION_CLEAR => Some((
            color.width,
            color.height,
            AttachmentAliasSample::Clear(color.clear_color),
        )),
        MTL_LOAD_ACTION_LOAD => {
            if let Some(seed) = color
                .target_seed_rgba
                .as_deref()
                .filter(|seed| seed.len() == need)
            {
                return Some((
                    color.width,
                    color.height,
                    AttachmentAliasSample::Seed(seed, color.format),
                ));
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

/// The object-list entry and descriptor bytes behind a sampled `texture_ref`,
/// reporting when the ref names something that is not a texture.
///
/// Four rails resolve a texture ref through the second rung of the object-list
/// ladder — `mipmap`, `draw::texture_view` and `compute_exec` name
/// `[OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT]` to
/// [`objects::resolve_descriptor`], and `draw::render_target` writes the pair
/// out — and the two sampled-path sites in this file did not. They went
/// `lookup_list_entry` -> `read_descriptor` -> `decode_texture_descriptor`, so
/// **any** object whose descriptor is at least `TEXTURE_DESC_GEOMETRY_LEN`
/// bytes decodes as a texture and yields a plausible extent rather than a
/// refusal. That is the hazard `contract::iosurface_pages` documents for the
/// `TEXTURE_DESC_WIDTH` name collision, reached a different way.
///
/// # It reports and does not refuse, on purpose
///
/// Adding the rung as a *decline* would turn a resolve that currently produces
/// geometry into `DrawPreparationDecline::TextureResolveMissing`, which loses
/// the draw — a behaviour change on the pathway this device is verified on,
/// justified by nothing yet measured. Whether a guest ever binds a non-texture
/// ref here is exactly what is not known, and the four rails that do check
/// cannot answer it because they refuse before anything is counted. So the
/// answer stays what it was and the log now says when it was reached; a driven
/// boot reading this slug is what decides whether the rung becomes a refusal.
pub(super) fn sampled_texture_descriptor<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(
    crate::runtime::decode::resource::ListObjectEntry,
    std::sync::Arc<[u8]>,
)> {
    let resource = objects::resolve_resource(state, host, task_id, texture_ref).ok()?;
    let entry = resource.entry;
    use crate::runtime::decode::resource::{OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT};
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        let slug = crate::observe::ladder_slug!("draw_sampled_texture", wrong_type);
        if degrade_log_first(texture_ref, slug) {
            crate::observe::fail(format!(
                "sampled_source_degraded reason={slug} task={task_id} ref={texture_ref} \
                 object_type={} (decoded as a texture descriptor anyway; geometry \
                 comes from a record of another kind)",
                entry.object_type
            ));
        }
    }
    Some((entry, std::sync::Arc::clone(&resource.descriptor)))
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
///
/// On a unified host, an admitted guest-backed target changes the first fact:
/// the resident is not a copy of the pages, it is the guest allocation itself.
/// Sampling it binds the same resource the render pass attached, so guest CPU
/// writes cannot make a second copy stale and the guest-write currency ladder
/// has no question to answer. `t11sample_ready_guest_allocation` and
/// `t11sample_ready_device_allocation` keep that population measurable. The
/// copied resident retains every rung and witness below.
pub(super) fn resolve_sampled_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    if texture_ref == 0 {
        return None;
    }

    // Opcode-9 buffer-backed texture (type-8): the sampled bytes are an MTLBuffer's
    // guest storage, not a view over another texture. Resolve it directly before
    // the view/surface paths (which would mis-decode the opcode-9 descriptor).
    // `resource` (when supplied by the caller) serves every classification and
    // descriptor consumer below from the one retained object.
    if let Some(bt) = buffer_texture_descriptor(
        state,
        host,
        task_id,
        texture_ref,
        resource.as_deref(),
    ) {
        // The opcode-9 descriptor's own pixel format. The loader converts to
        // RGBA8 order and decodes nothing, so the transfer function the guest
        // declared is still the one these bytes carry.
        let source = bt.desc.pixel_format;
        let (w, h, rgba) = load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt)?;
        return Some((
            w,
            h,
            0,
            SampledSourceRequest::Bytes(
                std::sync::Arc::new(rgba),
                None,
                SampledByteFormat::from_source(TexelLayout::Rgba8, source),
                crate::backend::vulkan::engine::SampledByteOrigin::BufferBackedTexture,
            ),
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
    // The retained resource already carries the total typed decode. Type 11 can
    // therefore fill the slot directly from that object rather than looking up
    // and decoding the same reference a second time. It returns `None` for every
    // other type, so it can only fill a slot the classification above left empty.
    // Hence an `Option`, not a list of candidates to choose between.
    let mut is_linear_tex = false;
    let mut is_type5 = false;
    let mut type5_view: Option<objects::Type5TextureView> = None;
    let mut surface: Option<u32> = None;
    let resolved_resource = resource.or_else(|| {
        objects::resolve_resource(state, host, task_id, texture_ref).ok()
    });
    if let Some(resource) = resolved_resource.as_ref() {
        let entry = resource.entry;
        if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
            is_type5 = true;
            let desc = &resource.descriptor;
            if let Ok(t5) = reims_vgpu_wire::device_desc::type5_header(desc) {
                let sid = t5.surface_id.get();
                if sid != 0 {
                    type5_view = objects::decode_type5_texture_view(desc);
                    surface = Some(sid);
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
        surface = surface.or_else(|| {
            resolved_resource.as_ref().and_then(|resource| {
                objects::resolve_type11_resource(state, task_id, texture_ref, resource)
            })
        });
    }

    if let Some(mid) = surface {
        // Ensure type-4 pages exist for this surface id.
        let _ = objects::ensure_surface_for_texture_bind(state, host, mid);
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
                if let Some(src) = resolved_resource.as_ref().and_then(|resource| {
                    try_type5_sample_zero_copy(state, host, mid, view, resource.lifetime_ref())
                }) {
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
                    SampledSourceRequest::Bytes(
                        rgba,
                        Some(identity),
                        byte_format,
                        crate::backend::vulkan::engine::SampledByteOrigin::SerializedSurfaceView,
                    ),
                ));
            }
        }
        if let Some(m) = state.mappings.get(&mid) {
            if m.has_geom && m.width > 0 && m.height > 0 {
                let (w, h) = (m.width, m.height);
                // Compute the resident-surface identity once and reuse it for
                // both the readiness check and the direct bind.
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
                let resident_backing = resolved_resource
                    .as_ref()
                    .filter(|resource| {
                        resource_type_owns_surface_resident(resource.entry.object_type)
                    })
                    .map(|resource| resource.resident_target_backing(&resident_id))
                    // An unclassified ref has no resource object to own a
                    // lease. Keep its existing query path so compatibility
                    // traffic still reaches the copying rails.
                    .unwrap_or_else(|| {
                        crate::backend::vulkan::engine::resident_content_backing(&resident_id)
                    });
                let resident_ready = resident_backing
                    != crate::backend::vulkan::engine::ResidentContentBacking::NotReady;
                if resident_ready {
                    crate::runtime::drain::note_store_route(match resident_backing {
                        crate::backend::vulkan::engine::ResidentContentBacking::GuestAllocation => {
                            "t11sample_ready_guest_allocation"
                        }
                        crate::backend::vulkan::engine::ResidentContentBacking::DeviceAllocation => {
                            "t11sample_ready_device_allocation"
                        }
                        crate::backend::vulkan::engine::ResidentContentBacking::NotReady => {
                            unreachable!("resident_ready excludes this arm")
                        }
                    });
                }
                // A render attachment and a sampled binding name the same
                // texture resource. When that resource's storage is the guest
                // allocation itself, a guest CPU write changes the resource;
                // it cannot make a second resident copy stale because there is
                // no second copy. Bind it directly before entering the currency
                // ladder, which exists for device-allocation mirrors.
                //
                // A non-identity view still falls through. The target bind
                // cannot carry that view's channel remap yet, so treating it as
                // direct would preserve the bytes and sample the wrong logical
                // channels.
                if guest_allocation_sample_is_direct(resident_backing, may_bind_resident) {
                    crate::runtime::drain::note_store_route("t11rung_resident");
                    let format = resident_id.resident_format();
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Target(resident_id, format),
                    ));
                }

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
                // A bind whose view remaps channels cannot take a resident
                // directly — the engine hands the swizzle to the image view and
                // the direct bind has none — so it falls straight to a byte rung
                // that can apply it. Counted on its own route and *outside* the
                // refusal below: nothing was replaced and there is nothing to
                // merge, so borrowing `_refused` would report a repaint that did
                // not happen and run a merge for it.
                if resident_ready && !guest_replaced && !may_bind_resident {
                    note_type11_sample_rung("t11rung_resident_swizzled", guest_write);
                } else if resident_ready {
                    if !guest_replaced {
                        note_type11_sample_rung("t11rung_resident", guest_write);
                        let format = resident_id.resident_format();
                        return Some((
                            w,
                            h,
                            mid,
                            SampledSourceRequest::Target(resident_id, format),
                        ));
                    }
                    note_type11_sample_rung("t11rung_resident_refused", guest_write);
                    match guest_owned {
                        Some(ranges) => {
                            // The merge is what makes falling through sound: it
                            // puts the resident's half into every page the guest
                            // did not write, so the rungs below read a surface
                            // holding both. When it does not land, that premise
                            // is false — the halves are still split, and the
                            // pages below hold only the guest's, which for a
                            // composite the Store deliberately left GPU-side is
                            // nothing at all. Falling through anyway is how a
                            // sampled backdrop comes back blank.
                            //
                            // So refuse the bind instead of choosing a half.
                            // `merge_guest_writes_into_pages` has already named
                            // the stage that failed on the fail channel; this
                            // turns its `false` into a decline the draw reports,
                            // which is what its doc asks the caller for.
                            if !merge_guest_writes_into_pages(
                                state,
                                host,
                                mid,
                                w,
                                h,
                                &resident_id,
                                ranges,
                            ) {
                                crate::runtime::drain::note_store_route(
                                    "t11sample_resident_merge_unlanded",
                                );
                                return None;
                            }
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

                // Falling through because the resident is *gone* is not the same
                // as falling through because it is stale, and until this line
                // the two were indistinguishable here —
                // `resident_content_ready` is `is_some_and(content_ready)`, so
                // absent and not-ready-yet are both `false`.
                //
                // The stale case above is sound because it merges the
                // resident's half into the pages first, and refuses when that
                // merge does not land, for the reason its own comment gives: the
                // pages below then hold only the guest's half, "which for a
                // composite the Store deliberately left GPU-side is nothing at
                // all". A reclaimed resident has exactly that property and there
                // is nothing left to merge from — the image is destroyed — yet
                // it takes this fall-through with no merge and no refusal.
                //
                // For most surfaces that is correct: a type-11 surface's pages
                // are its content, the flush rails write them, and reading them
                // back is what `resolve_type11_load_seed` already calls "a cache
                // miss is a reason to read them".
                //
                // # The unsound case this line was added to count is closed
                //
                // It used to be reachable. A resident whose pixels were never
                // written to those pages at all — an MRT secondary attachment,
                // never pinned, never written back, and still carrying a real
                // `Gva`/`Surface` identity — could be aged out, and serving it
                // from its pages substituted an unrelated earlier frame.
                //
                // `ResidentTargetSlot::gpu_only_content` closed it at the
                // reclaim end, which is the only end that can be closed: both
                // allocation-pressure recovery skips such a slot at any
                // population. So **a resident that reaches this arm was, by
                // construction, not the sole copy of its pixels when it was
                // reclaimed** — something had copied them out, which is what
                // cleared the flag.
                //
                // The guarantee therefore does not live here and cannot be
                // asserted here; it lives beside those two selectors, held by
                // `elapsed_time_never_reclaims_a_live_resident`,
                // `the_capacity_walk_finds_no_victim_rather_than_destroy_the_only_copy`
                // and `no_reclaim_cause_may_take_the_only_copy_of_a_frame` —
                // the last of which is exhaustive over `ResidentReclaim`, so a
                // fourth way to lose a resident has to answer this question
                // before it compiles.
                //
                // What the line below still reports is a **cost**, not a
                // soundness risk: this device paid for the reclaim by re-reading
                // guest pages. It stays on the fail channel because the reclaim
                // cutoff is a measured trade (see `IDLE_MAINTENANCE_START_MS`) and the
                // reliance on it should stay visible, not because a firing means
                // something was lost.
                if !resident_ready {
                    if let Some((cause, since_ms)) =
                        crate::backend::vulkan::engine::resident_absent_after_reclaim(&resident_id)
                    {
                        crate::runtime::drain::note_store_route("t11sample_reclaimed_from_pages");
                        // How long after we destroyed it the guest came back.
                        // This is the half `resident_resample_peak_ms` cannot
                        // see: that peak only observes residents that survived
                        // to be read, so every gap longer than the cutoff is
                        // censored out of it and a reclaim policy tuned from it
                        // is tuned from data it destroyed the tail of. A
                        // resident read here had gone at least
                        // `IDLE_MAINTENANCE_START_MS + since_ms` between uses.
                        crate::runtime::drain::note_store_route(reclaimed_resample_band(since_ms));
                        if crate::observe::first_sight("sampled_resident_reclaimed", u64::from(mid))
                        {
                            crate::observe::fail(format!(
                                "sampled_resident_reclaimed reason=sampled_resident_reclaimed \
                                 mid={mid} {w}x{h} prior={} since_reclaim_ms={since_ms} \
                                 (reclaimed after its pixels were copied out; re-reading \
                                 its guest pages costs an upload, not a frame)",
                                cause.slug()
                            ));
                        }
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
                    // BGRA8 by construction — it is a type-4 scanout cache — but
                    // the *values* in it are the surface's, and this cache is
                    // filled from a writeback that reorders channels and decodes
                    // nothing. So the transfer function is the mapping's declared
                    // one, exactly as it is on the guest-page rungs below.
                    let source = crate::runtime::draw::mapping_declared_format(state, mid, None);
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(
                            bgra,
                            identity,
                            SampledByteFormat::from_source(TexelLayout::Bgra8, source),
                            crate::backend::vulkan::engine::SampledByteOrigin::SurfaceHostCache,
                        ),
                    ));
                }

                // 2) Guest pages, which are what the surface *is*. Reached only
                // when no host-side copy served the bind — no resident, or one
                // the guest has written over — so the gather always runs and the
                // guest bytes are taken unconditionally. Declining the gather is
                // expected control flow — the CPU byte loader below serves the
                // same pixels — so it stays quiet, like the type-2/3 rail's.
                if let Some(src) = resolved_resource.as_ref().and_then(|resource| {
                    try_type11_sample_zero_copy(state, host, mid, w, h, resource.lifetime_ref())
                }) {
                    note_type11_sample_rung("t11rung_zero_copy", guest_write);
                    return Some((w, h, mid, src));
                }
                // The memo skips the convert/alloc on unchanged content and
                // returns a content identity so the engine skips re-hash+upload;
                // its census (T11Memo hit / T11Guest fill) is emitted internally.
                let memo_source = crate::runtime::draw::mapping_declared_format(state, mid, None);
                if let Some((rgba, identity)) = load_type11_rgba_memoized(state, host, mid) {
                    note_type11_sample_rung("t11rung_guest_memo", guest_write);
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(
                            rgba,
                            Some(identity),
                            SampledByteFormat::from_source(TexelLayout::Rgba8, memo_source),
                            crate::backend::vulkan::engine::SampledByteOrigin::SurfaceGuestFallback,
                        ),
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
        // Zero-copy gather for large Vulkan-native linear textures: replaces
        // the CPU host-cache/memo byte paths below for eligible formats (the
        // lin_memo full-window re-read + memcmp per bind was the dominant
        // per-draw cost under compositor load).
        // The object map retains the typed construction descriptor for the
        // resource lifetime. Both linear loaders consume that same object here;
        // neither needs to revisit guest construction bytes.
        if let Some(tex) =
            resolved_resource
                .as_ref()
                .and_then(|resource| match resource.decoded() {
                    Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => Some(tex),
                    _ => None,
                })
        {
            // Above the gather: a span whose pages a render Store published and
            // nothing has written since is already an engine image, so there is
            // nothing to gather and — the point of the rung — no writeback to
            // wait for. See [`try_gva_resident_sample`].
            if may_bind_resident {
                if let Some((w, h, src)) =
                    try_gva_resident_sample(state, host, task_id, texture_ref, tex)
                {
                    return Some((w, h, 0, src));
                }
            }
            if let Some((w, h, src)) = try_linear_sample_zero_copy(
                state,
                host,
                task_id,
                texture_ref,
                tex,
                resolved_resource.as_ref()?.lifetime_ref(),
            ) {
                return Some((w, h, 0, src));
            }
            if let Some((w, h, rgba, identity, byte_format)) =
                load_linear_from_host_caches(state, host, task_id, texture_ref, tex)
            {
                return Some((
                    w,
                    h,
                    0,
                    SampledSourceRequest::Bytes(
                        rgba,
                        identity,
                        byte_format,
                        crate::backend::vulkan::engine::SampledByteOrigin::LinearTexture,
                    ),
                ));
            }
        }
    }

    // The last-resort sampled rung. The geometry comes from the decoded texture
    // descriptor and from nowhere else. Neither a payload shorter than the
    // descriptor's own extent nor a descriptor naming no extent at all is a
    // geometry this call may invent one for: the caller turns `None` into a
    // typed `DrawPreparationDecline::TextureResolveMissing`, which names the ref
    // and the stage. `TextureDescriptor::extent` owns the second check and says
    // what clamping the two fields up would have bound instead.
    //
    // The layout is carried rather than assumed. This rung answered
    // `TexelLayout::Rgba8` unconditionally and sized its own length check at
    // four bytes a texel, so it was the one place a half-float texture could
    // still be quantised after every rail above it learned not to — and it is
    // the rung a `RGBA16Float` display-profile LUT actually lands on, because
    // the three rungs above are reached only for a resource the draw already
    // knows is a linear texture.
    let (mut bytes, layout) = load_sampled_rgba_static(
        state,
        host,
        task_id,
        texture_ref,
        native_uploads_asking_host(),
        crate::runtime::render_writeback::SettleSite::LinearTextureSampled,
    )?;
    let (_entry, desc) = sampled_texture_descriptor(state, host, task_id, texture_ref)?;
    let tex = decode_texture_descriptor(&desc).ok()?;
    let (w, h) = tex.extent()?;
    let planes = tex.levels.first()?.planes();
    let need = (w as usize)
        .saturating_mul(h as usize)
        .saturating_mul(planes as usize)
        .saturating_mul(layout.layout().bytes_per_texel() as usize);
    if bytes.len() < need {
        return None;
    }
    bytes.truncate(need);
    Some((
        w,
        h,
        0,
        SampledSourceRequest::Bytes(
            std::sync::Arc::new(bytes),
            None,
            layout,
            crate::backend::vulkan::engine::SampledByteOrigin::LinearTexture,
        ),
    ))
}

/// Whether a decoded resource object owns the resident reached by the surface
/// branch above.
///
/// Base surfaces, texture views, and IOSurface textures are three construction
/// forms of a texture object. Each carries one stable resource reference for
/// its lifetime; a view additionally names its parent, but that does not make
/// its own reference transient. Other object kinds can reach this resolver as
/// probes, but cannot produce the `surface` value whose resident is retained.
fn resource_type_owns_surface_resident(object_type: u8) -> bool {
    matches!(
        object_type,
        objects::OBJECT_TYPE_SURFACE
            | objects::OBJECT_TYPE_REF_TEXTURE
            | crate::runtime::decode::resource::OBJECT_TYPE_IOSURFACE
    )
}

/// Whether a decoded resource object owns a linear GVA texture resident.
///
/// Normal textures and their serialized variants carry one stable resource
/// reference from construction until deletion. Their level-zero GVA identity
/// may therefore retain the engine allocation for that same lifetime. Surface
/// texture forms use [`resource_type_owns_surface_resident`] instead, while an
/// anonymous attachment keeps the registry-query fallback.
fn resource_type_owns_gva_resident(object_type: u8) -> bool {
    matches!(
        object_type,
        crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE
            | crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VARIANT
    )
}

#[cfg(test)]
mod resource_resident_ownership_tests {
    use super::*;

    #[test]
    fn every_surface_texture_construction_form_owns_its_resident() {
        for object_type in [
            objects::OBJECT_TYPE_SURFACE,
            objects::OBJECT_TYPE_REF_TEXTURE,
            crate::runtime::decode::resource::OBJECT_TYPE_IOSURFACE,
        ] {
            assert!(resource_type_owns_surface_resident(object_type));
        }
        for object_type in [
            crate::runtime::decode::resource::OBJECT_TYPE_BUFFER,
            crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE,
            crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VARIANT,
        ] {
            assert!(!resource_type_owns_surface_resident(object_type));
        }
    }

    #[test]
    fn linear_texture_construction_forms_own_their_gva_resident() {
        for object_type in [
            crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE,
            crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VARIANT,
        ] {
            assert!(resource_type_owns_gva_resident(object_type));
        }
        for object_type in [
            objects::OBJECT_TYPE_SURFACE,
            objects::OBJECT_TYPE_REF_TEXTURE,
            crate::runtime::decode::resource::OBJECT_TYPE_IOSURFACE,
            crate::runtime::decode::resource::OBJECT_TYPE_BUFFER,
        ] {
            assert!(!resource_type_owns_gva_resident(object_type));
        }
    }
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
/// This is not a degradation the caller absorbs. `exec` resolves the pass load
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

/// A type-11 attachment's prior contents, kept in the representation of the
/// freshest rung that supplied them.
enum Type11LoadSeed {
    /// Host-owned bytes from the render cache, or the universal converted
    /// fallback when the native guest-page view cannot be described.
    Host(
        std::sync::Arc<Vec<u8>>,
        crate::backend::vulkan::engine::SeedOrder,
    ),
    /// The mapping's native texels as bounded guest-RAM runs. The engine imports
    /// them when possible and uses the runs themselves for its CPU fallback.
    Guest(crate::backend::vulkan::engine::GuestTargetSeed),
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
/// under `MTL_LOAD_ACTION_LOAD` with no caller-supplied seed. With the served
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

/// The prior contents of a type-11 attachment under `MTL_LOAD_ACTION_LOAD`,
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
/// The guest-pages rung preserves the mapping's native texels as bounded runs.
/// The engine imports or gathers those runs and copies them straight into the
/// same-format attachment; when the host cannot expose stable aliases, the
/// existing RGBA reader remains the universal fallback. Both arms use the
/// mapping's latched geometry, and the engine validates the exact strided span
/// before recording the copy. Any writeback debt is paid before the page view is
/// built, so it observes this device's latest Store rather than pre-Store bytes.
///
/// The sibling Metal path already had rung 2: type-11 `seed_color_load` falls
/// through to the same reader via `load_sampled_rgba_static`. Only the Vulkan arm
/// stopped at the cache.
///
/// `None` means the guest's LOAD could not be honoured at all, and
/// [`note_type11_load_seed`] has already said which check refused.
/// Band how long after pressure recovery the guest wanted the resident again.
/// The fixed reference interval keeps existing census buckets comparable; it
/// has no role in deciding whether the resident remains alive.
fn reclaimed_resample_band(since_ms: u64) -> &'static str {
    let cutoff = crate::backend::vulkan::engine::IDLE_MAINTENANCE_START_MS;
    if since_ms < cutoff {
        "t11sample_reclaimed_within_1x_cutoff"
    } else if since_ms < cutoff * 2 {
        "t11sample_reclaimed_within_2x_cutoff"
    } else if since_ms < cutoff * 4 {
        "t11sample_reclaimed_within_4x_cutoff"
    } else {
        "t11sample_reclaimed_past_4x_cutoff"
    }
}

/// Whether one shader stage occupies any binding number in the **sampled band**,
/// which holds textures and samplers together.
///
/// The band's relocation — `separate_sampled` — is what stops one stage's
/// binding number landing on the other's, so the question it has to be triggered
/// from is about the whole band and not about half of it. Asking only about
/// textures leaves a stage that binds a sampler and no texture invisible to the
/// trigger, and Metal argument tables are sticky across draws in an encoder, so
/// a vertex sampler routinely survives a re-bind that zeroed the vertex
/// textures. With the two stages unseparated their sampler at index 0 resolves
/// to one binding, `push_smp` is first-writer-wins, and the loser's filter,
/// address mode and LOD clamp are dropped while the layout's
/// `VERTEX | FRAGMENT` stage flags let it go on sampling through the winner's.
///
/// A ref of zero is not a bind: it is the guest leaving a slot empty, which is
/// the same reading `push_smp`'s own callers take.
fn stage_uses_sampled_band(textures: &[TextureBind], samplers: &[SamplerBind]) -> bool {
    textures.iter().any(|t| t.texture_ref != 0) || samplers.iter().any(|s| s.sampler_ref != 0)
}

fn try_type11_target_guest_seed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
    target_format: ash::vk::Format,
) -> Option<crate::backend::vulkan::engine::GuestTargetSeed> {
    use crate::backend::vulkan::engine::{GuestRunSource, GuestTargetSeed};
    use crate::runtime::mapping_write::type11_sample_window;

    if w == 0 || h == 0 || !mapper::ensure_resolved_for_scanout(state, host, mapping_id) {
        return None;
    }
    let (base_off, bpr, layout) = {
        let mapping = state.mappings.get(&mapping_id)?;
        if !mapping.mapped
            || mapping.page_entries.is_empty()
            || !mapping.has_geom
            || mapping.width != w
            || mapping.height != h
        {
            return None;
        }
        let format = if mapping.format == 0 {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        } else {
            mapping.format
        };
        let layout = pixel_format::store_texel_order(format)?;
        let (base_off, bpr, _) = type11_sample_window(mapping, w, h, format)?;
        (base_off, u64::from(bpr), layout)
    };
    let source_format = translate::pixel::vk_texel_layout(layout);
    if source_format != target_format {
        return None;
    }
    let (span, row_length_texels) =
        strided_window_extent(w, h, u64::from(layout.bytes_per_texel()), bpr)?;

    // A debt is not submitted work, so queue order cannot put it before the
    // seed read until this call turns it into work. A submitted payment and the
    // draw use the same queue; no CPU settle is needed between them.
    crate::runtime::writeback_debt::pay_for_mapping(state, host, mapping_id);
    let (gpas, runs) = mapping_window_guest_runs(state, host, mapping_id, base_off, span)?;
    let page = state.page_size();
    Some(GuestTargetSeed {
        source: GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: span,
            row_length_texels,
            pages: guest_page_window(host, gpas, page, base_off % page, span),
            direct_image: None,
        },
        format: source_format,
    })
}

fn resolve_type11_load_seed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
    target_format: ash::vk::Format,
) -> Option<Type11LoadSeed> {
    use crate::backend::vulkan::engine::SeedOrder;
    // The cache is the other host-side copy of these pages, and it sits above
    // the only rung that reads the pages themselves — so a stale hit here is
    // never corrected by anything below it. It takes the same witness the
    // sampled path's read of this same map takes (`resolve_sampled_source`'s
    // `guest_replaced`), spelled the same way and for the same reason: the
    // coarse token says whether the guest wrote the *allocation*, and only when
    // it did is the page list walked to say whether it wrote the *pixels*.
    //
    // Without it this rung served exactly the frame the elision gate two frames
    // up had just refused. The caller computes `type11_guest_wrote_since_store`
    // to decide whether the resident may carry the chain, declines when it says
    // written, and then arrived here to be handed the same stale bytes out of
    // the host cache. The pass composites onto them and its Store publishes
    // them back over the guest's pages, which is the fixpoint this file's own
    // note above `type11_load_currency_query` calls "renders correctly for a few
    // frames then stays corrupted".
    let guest_write = mapping_guest_write_verdict(state, host, mapping_id);
    let site = guest_wrote_allocation(guest_write)
        .then(|| guest_write_site(state, host, mapping_id, w, h));
    let guest_replaced = !matches!(site, None | Some(GuestWriteSite::Elsewhere));
    if guest_replaced {
        crate::runtime::drain::note_store_route("t11seed_cache_refused_guest_wrote");
    }
    let cached = (!guest_replaced)
        .then(|| crate::runtime::surface_cache::get_shared(state, mapping_id, w, h))
        .flatten();
    let served = if let Some(bgra) = cached {
        Some((
            Type11LoadSeed::Host(bgra, SeedOrder::Bgra8),
            Type11SeedRung::Cache,
        ))
    } else {
        try_type11_target_guest_seed(state, host, mapping_id, w, h, target_format)
            .map(|seed| (Type11LoadSeed::Guest(seed), Type11SeedRung::GuestPages))
            .or_else(|| {
                load_type11_mapping_rgba(state, host, mapping_id, None)
                    .map(|(_, _, r)| r)
                    .filter(|rgba| rgba.len() == (w as usize) * (h as usize) * 4)
                    .map(|rgba| {
                        (
                            Type11LoadSeed::Host(std::sync::Arc::new(rgba), SeedOrder::Rgba8),
                            Type11SeedRung::GuestPages,
                        )
                    })
            })
    };
    note_type11_load_seed(state, mapping_id, w, h, served.as_ref().map(|s| s.1));
    served.map(|(seed, _)| seed)
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
    let (base_off, surface_bpr, span_end, pages_n, base_w, base_h, base_fmt, map_gen) = {
        let Some(m) = state.mappings.get(&mapping_id) else {
            return fail(Type5ViewDecline::NoMapping);
        };
        let Some((base_off, surface_bpr, span_end)) = mapping_write::type5_sample_window(
            m,
            view.plane_index,
            view.width,
            view.height,
            view.pixel_format,
        ) else {
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
        )
    };
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
        mapping_write::SurfaceWindow {
            base_off,
            bpr: surface_bpr,
            span_end,
            bpp,
        },
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: view.width,
            height: view.height,
        },
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
    // The ten-bit pair (`'x420'`, `R16Unorm` / `RG16Unorm`) takes the same
    // native rail for the same reason and one more: `texel_to_rgba8` has no arm
    // for them, because an arm would have to narrow ten bits of graded luma to
    // eight. `TexelLayout::has_cpu_loader_arm` is where that is stated.
    //
    // The half-float colour pair is deliberately **not** here yet. It belongs
    // by the same argument the linear rails took — `texel_to_rgba8`'s arm for
    // it clamps to `[0, 1]` and quantizes to 256 levels — but nothing has ever
    // measured a type-5 view arriving in one, and this rail is the video-plane
    // rail. `type5_view_narrowed` below is the measurement; add the arm when it
    // fires, not before.
    //
    // The packed 32-bit colour formats take the native rail for the same
    // reason and a sharper one: their channel boundaries are not byte
    // boundaries, so `TexelLayout::Rgba8` would not merely quantize them, it
    // would read the word as four unrelated bytes. Four bytes wide is exactly
    // what the default arm below tests for and exactly what makes that wrong,
    // which is why they are named here rather than left to it.
    // The layout is the view's own; the transfer function travels with it,
    // because the default arm below converts to RGBA8 order and decodes nothing.
    let byte_format = SampledByteFormat::from_source(
        match view.pixel_format {
            pixel_format::MTL_FORMAT_RG8_UNORM => TexelLayout::Rg8,
            pixel_format::MTL_FORMAT_R16_UNORM => TexelLayout::R16Unorm,
            pixel_format::MTL_FORMAT_RG16_UNORM => TexelLayout::Rg16Unorm,
            pixel_format::MTL_FORMAT_RGB10A2_UNORM => TexelLayout::Rgb10a2Unorm,
            pixel_format::MTL_FORMAT_BGR10A2_UNORM => TexelLayout::Bgr10a2Unorm,
            pixel_format::MTL_FORMAT_RG11B10_FLOAT => TexelLayout::Rg11b10Float,
            _ => TexelLayout::Rgba8,
        },
        view.pixel_format,
    );
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
            "type5_draw_view ok task={task_id} ref={texture_ref} sid={mapping_id} map_gen={map_gen} view={}x{} fmt={:#x} bpp={bpp} base={base_w}x{base_h} base_fmt={base_fmt:#x} off={base_off} bpr={surface_bpr} span_end={span_end} src={generation_source} rgb_nz={nz} max_rgb={max}",
            view.width,
            view.height,
            view.pixel_format,
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
    let rgba: std::sync::Arc<Vec<u8>> = if byte_format.layout() == TexelLayout::Rgba8 {
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
        // The third CPU convert in this crate, and the third that said nothing
        // when it lost precision. `byte_format`'s table above names the video
        // plane formats natively and folds everything else here, so a half-float
        // type-5 view is quantized on the same terms the linear rails were.
        crate::runtime::draw::note_sampled_narrowing(
            "type5_view_narrowed",
            texture_ref,
            view.pixel_format,
            view.width,
            view.height,
        );
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
            // The type-5 view path re-derives this from the view's own pixel
            // format on every call, hit or miss, so storing it is a statement of
            // what the bytes are rather than the source anything reads.
            layout: byte_format.layout(),
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

/// Copied sampled-gather floor: below this the CPU byte path (one small read +
/// memo) is cheaper than a recorded buffer-to-image transfer. Performance
/// threshold only — never a correctness gate and never a gate on a direct
/// buffer-backed image.
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
/// Resource construction is independent of byte length: when a retained guest
/// allocation can back the sampled image directly, that is the resource and no
/// crossover is consulted. This floor applies only after direct construction
/// is unavailable and the remaining GPU path would copy into another image.
pub(super) const SAMPLED_GATHER_MIN_BYTES: u64 = 64 * 1024;

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
pub(super) fn task_gva_guest_run_window<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Result<(Vec<u64>, Vec<crate::backend::vulkan::engine::GuestRun>), WindowRefusal> {
    let page = state.page_size();
    let gpas =
        gva_mem::task_gva_page_gpas(host, &state.tasks, task_id, gva, span, state.page_shift);
    let wanted = reims_vgpu_paging::span::pages_spanned(gva, span, page);
    if gpas.len() as u64 != wanted {
        return Err(WindowRefusal::SpanUnmapped);
    }
    let runs = coalesce_pages_to_runs(host, &gpas, page, gva % page, span)?;
    Ok((gpas, runs))
}

/// Why a guest-page window could not be built.
///
/// Typed rather than a bare `None` because these are **degradations that
/// repeat**. A bind that lands here is not cached — only resolutions are held
/// (see [`crate::runtime::bound_buffers`]) — so the same reference re-walks the
/// task page table and re-pays the CPU staging read on every draw for as long
/// as the guest keeps binding it. That is the part of the per-draw cost the
/// held-resolution registry does not reach, and until these were counted there
/// was no way to say how large it is: the two silent `None`s this replaces were
/// the only unnamed exits in a rail whose every other outcome has a route.
///
/// Each caller maps these onto its own route prefix rather than counting them
/// here. The buffer rail and the linear sampled rail differ by two orders of
/// magnitude in volume, and one shared counter would report their sum as though
/// it were either — the same conflation [`band_runs`] already carries and says
/// so about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowRefusal {
    /// A stretch resolved, but the view the host built for it is one the host
    /// owes a release for, so the engine cannot hold it.
    ///
    /// This is a per-stretch answer and not a host verdict: the engine gathers
    /// from these pointers when the submission reaches the GPU, which is after
    /// this call returns, so only a view that outlives the release may go. Both
    /// shims hand back guest RAM itself for a host-VA-packed request and build a
    /// view otherwise, and a coalesced stretch is GPA-contiguous by
    /// construction, so this is the rare answer rather than the standing one.
    /// It was the standing one for as long as the question was asked of the
    /// device instead of the call.
    NoAlias,
    /// Some page of the span does not resolve under the task's page table.
    ///
    /// The one a mapped-range record could answer without walking.
    SpanUnmapped,
    /// Every page resolved, but a GPA-contiguous stretch would not import.
    ///
    /// A walk that finished and still could not bind, so no range record would
    /// have saved it.
    Untileable,
}

/// The bounded guest-memory references behind a set of runs, one per maximal
/// GPA-contiguous stretch, when this host can import guest RAM at all.
///
/// `None` is the routing answer for every host that cannot import — no
/// `VK_EXT_external_memory_host`, an operator who turned the rail off, a shim
/// that cannot say where guest RAM lives, or a GPA the imports do not cover —
/// and the caller gathers on the CPU exactly as it did before. Every refusal is
/// named on the always-on sink by [`crate::runtime::guest_ram_map`], so a fall
/// back to the copy is never silent.
///
/// # Why this asks for runs rather than one bind range
///
/// It asked for one, and a driven boot priced what that cost. `zc_buffer_gathered`
/// read 371 422 against `zc_buffer_imported` at **zero** on a host whose
/// `vk_caps` said `host_pointer_import=supported`, and the banded census said
/// why with no ambiguity left in it: not one window in the boot was refused for
/// a missing import, a declined pointer, an unbacked GPA or a range outside
/// one. Every single one was refused for being scattered, 98.5 % of them into
/// 9-32 stretches, and **nothing at all** at one or two. The guest backs a
/// surface in 16 KiB physically-contiguous granules, so a rail that takes only
/// one stretch is not a rail that rarely fires — it is one that cannot fire.
///
/// The bytes have to be gathered somewhere regardless: a vertex or storage bind
/// must name one contiguous range and these windows are not one in GPA space.
/// So the only question was whether the CPU or the GPU does it, and the CPU was
/// answering it at 3.6 GB/s of `memcpy` — 105 ms per second of wall clock, two
/// thirds of every draw's staging phase. Handing the runs to the caller lets it
/// submit one `VkBufferCopy` per stretch into device-local memory instead, which
/// crosses the bus once where the CPU path crossed it once and paid a full
/// core's memcpy on top.
///
/// # Four call sites, and the counters are shared
///
/// This serves the draw-time buffer rail and three sampled ones. From the boot
/// above, through `engine_delta`:
///
/// | rail | gathers | bytes the CPU moved |
/// |---|---:|---:|
/// | buffer (`stage_phase`'s `runs`) | 15 758 per second | 3.6 GB **per second** |
/// | sampled (`sampled_gathers`) | 211 for the boot | 254 MB for the boot |
///
/// So a reading of `zc_buf_runs_*` is both populations, and the sampled one is
/// around two orders of magnitude smaller. Only the buffer rail consumes the
/// runs; the sampled rail binds a one-run window directly and otherwise still
/// gathers on the CPU, which its own volume does not justify changing.
fn guest_page_window<M: HostOps>(
    host: &mut M,
    gpas: Vec<u64>,
    page: u64,
    head_offset: u64,
    span: u64,
) -> Option<std::sync::Arc<Vec<crate::runtime::guest_ram_map::GuestWindowRun>>> {
    use crate::runtime::guest_ram_map::MapRefusal;
    match crate::runtime::guest_ram_map::references_for_runs(host, &gpas, page, head_offset, span) {
        Ok(runs) => {
            // Banded on the way through as well as on the refusals below,
            // because the count is what decides whether a window binds straight
            // into the draw (one run) or costs a copy region per stretch — and
            // a rail whose regions grew without anyone noticing would read here
            // first.
            crate::runtime::drain::note_store_route(band_runs(runs.len()));
            Some(std::sync::Arc::new(runs))
        }
        Err(refusal) => {
            crate::runtime::drain::note_store_route(match refusal {
                MapRefusal::NoBackendImport => "zc_buf_no_import",
                MapRefusal::HostRefused(_) => "zc_buf_host_refused",
                MapRefusal::NoUsableRegion { .. } => "zc_buf_no_region",
                // Its own band and not folded into `zc_buf_no_import`: this
                // host has the extension and would import, and what refused is
                // the size of the guest against the size of its heaps. A boot
                // reading this is one where raising the heap or lowering `-m`
                // would restore the rail, which is not true of any other band.
                MapRefusal::ImportExceedsHeap { .. } => "zc_buf_over_heap",
                MapRefusal::GpaNotInAnyImport { .. } => "zc_buf_gpa_unbacked",
                MapRefusal::OutsideImport(_) => "zc_buf_outside_import",
                // `references_for_runs` reaches this only for a window it could
                // not tile at all — an empty page list, a zero length, an
                // overflowing range. A merely scattered window is now a success
                // with several runs, counted above.
                MapRefusal::Scattered { .. } => "zc_buf_untileable",
            });
            None
        }
    }
}

/// Band a window's stretch count for the census.
///
/// Banded, not exact: what these decide is how many copy regions a window costs
/// the GPU gather, which is a question about the order of magnitude. An exact
/// count would also need an unbounded set of static strings, which
/// `note_store_route` does not take.
///
/// The low bands are the ones a driven boot was first measured in, kept so a
/// later reading is comparable with that one: 42 windows at 3-4 stretches,
/// 4 322 at 5-8, **370 716 at 9-32** and 1 261 above — and nothing at all at one
/// or two, which is what made the single-reference rail unreachable.
///
/// # Why they reach past 64
///
/// These bands stopped at `>32` while the Vulkan engine capped its GPU gather at
/// 64 regions, so every window that cap turned away landed in one bucket that
/// *starts below the cap* — a reading of it could not say whether a refused
/// window overshot by one region or by five hundred, and the cap's own
/// justification was written from exactly that bucket.
///
/// Widening them answered it and retired the cap: on a driven boot the
/// distribution is bimodal, 99.66 % of windows at 1-32 stretches and a second
/// population of full-screen surfaces at 257-512, with **nothing between 33 and
/// 256**. 64 was not a threshold between two regimes; it sat in the empty space
/// between them, and any value from 33 to 256 would have refused the same 1 162
/// windows. The bands stay wide because that shape is what a future cap
/// proposal has to be argued against.
fn band_runs(runs: usize) -> &'static str {
    match runs {
        0..=1 => "zc_buf_runs_1",
        2 => "zc_buf_runs_2",
        3..=4 => "zc_buf_runs_3_4",
        5..=8 => "zc_buf_runs_5_8",
        9..=32 => "zc_buf_runs_9_32",
        33..=64 => "zc_buf_runs_33_64",
        65..=128 => "zc_buf_runs_65_128",
        129..=256 => "zc_buf_runs_129_256",
        257..=512 => "zc_buf_runs_257_512",
        513..=1024 => "zc_buf_runs_513_1024",
        _ => "zc_buf_runs_gt1024",
    }
}

/// Coalesce GPA-contiguous stretches of `window` into packed host-VA runs
/// covering `span` bytes from `head_off` into the first page.
///
/// The stretch arithmetic is `reims_vgpu_paging::runs::coalesce_window`; what
/// this adds is the host side — one aliasing request per stretch. A stretch is
/// GPA-contiguous, which on both shims is the shape that resolves to QEMU's own
/// RAMBlock mapping, so the answer is guest RAM itself and the pointer outlives
/// every submission that gathers from it.
///
/// [`HostOps::stable_page_alias`] rather than `map_pages` because that is the
/// whole requirement: the engine reads these pointers after this call returns,
/// so a view the host owes a release for must be released here and refused
/// rather than handed on.
///
/// Refuses if any stretch fails to import or comes back transient, or if the
/// window runs out before `span` — a partial gather would hand the GPU a short
/// buffer, which is a wrong frame rather than a slow one.
fn coalesce_pages_to_runs<M: HostOps>(
    host: &mut M,
    window: &[u64],
    page: u64,
    head_off: u64,
    span: u64,
) -> Result<Vec<crate::backend::vulkan::engine::GuestRun>, WindowRefusal> {
    use crate::backend::vulkan::engine;
    let stretches = reims_vgpu_paging::runs::coalesce_window(window, page, head_off, span)
        .ok_or(WindowRefusal::Untileable)?;
    let mut runs: Vec<engine::GuestRun> = Vec::with_capacity(stretches.len());
    for s in stretches {
        let pages = &window[s.pages];
        let Some(mapped) = host.map_pages(pages, page as usize) else {
            return Err(WindowRefusal::Untileable);
        };
        if mapped.alias != crate::runtime::host::PageAlias::Stable {
            host.unmap_pages(mapped.ptr, pages.len().saturating_mul(page as usize));
            note_transient_guest_run_alias();
            return Err(WindowRefusal::NoAlias);
        }
        runs.push(engine::GuestRun {
            host_ptr: (mapped.ptr as u64 + s.start_offset) as usize,
            len: s.len,
        });
    }
    Ok(runs)
}

/// Name the first GPA-contiguous stretch of the process that came back as a
/// view rather than as guest RAM.
///
/// Once, because the interesting thing is that it happened at all: a stretch is
/// contiguous by construction, so a host that reconstructs one is a host whose
/// packing rule this device has not understood, and every draw bind on it takes
/// the CPU byte loader. The rate lives on the callers' route counters.
fn note_transient_guest_run_alias() {
    static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::observe::fail(String::from(
            "guest_run_rail off reason=stretch_alias_transient \
             (a GPA-contiguous stretch was not guest RAM; draw binds take the CPU byte loader)",
        ));
    }
}

/// Ensure one linear resource has the packed host allocation shared by all of
/// its buffer offsets or texture planes, returning whether that allocation is
/// available. The caller then borrows it from
/// [`crate::runtime::bound_buffers::BoundBuffers`]. A negative result is held
/// under the same retirement rules as a positive one: mappings do not become
/// complete without a map/object notification, and those notifications remove
/// the entry.
#[derive(Clone, Copy)]
pub(super) enum PackedResourceRail {
    Buffer,
    LinearSample,
}

/// Band a scattered packed window by how many maximal GPA runs it is made of.
///
/// A window that refuses [`crate::runtime::guest_ram_map::reference_for_pages`]
/// costs a `vkAllocateMemory` and a host virtual alias, and the alternative —
/// one RAMBlock reference per run, which `references_for_runs` already builds
/// for the gather rail — is only cheaper while the run count is small. A two-run
/// window is two binds against one allocation; a five-hundred-run window is not.
/// Nothing measured the distribution, so the choice cannot be made from the run
/// count without this, and a `Scattered` refusal's own `runs` field is deduped
/// by `report_once` and so reports the first window of a boot rather than the
/// population.
pub(super) fn packed_scatter_band(gpas: &[u64], page: u64) -> &'static str {
    match reims_vgpu_paging::runs::contig_run_count(gpas, page) {
        0 | 1 => "zc_packed_scatter_runs_1",
        2 => "zc_packed_scatter_runs_2",
        3..=4 => "zc_packed_scatter_runs_3_4",
        5..=8 => "zc_packed_scatter_runs_5_8",
        9..=16 => "zc_packed_scatter_runs_9_16",
        17..=64 => "zc_packed_scatter_runs_17_64",
        _ => "zc_packed_scatter_runs_65_up",
    }
}

pub(super) fn ensure_packed_resource<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    resource_ref: u32,
    backing: &BufferBacking,
    rail: PackedResourceRail,
) -> bool {
    use crate::runtime::bound_buffers::{PackedBuffer, PackedBufferResolution};

    if let Some(held) = state.bound_buffers.packed(task_id, resource_ref) {
        let matches = match held {
            PackedBufferResolution::Available(buffer) => {
                buffer.gva == backing.gva && buffer.size == backing.size
            }
            PackedBufferResolution::Unavailable { gva, size } => {
                *gva == backing.gva && *size == backing.size
            }
        };
        if matches {
            return matches!(held, PackedBufferResolution::Available(_));
        }
    }

    let unavailable = || PackedBufferResolution::Unavailable {
        gva: backing.gva,
        size: backing.size,
    };
    let made = (|| {
        let page = state.page_size();
        let page_base = backing.gva & !(page - 1);
        let head = backing.gva - page_base;
        let map_len =
            crate::contract::checked::align_up_u64(head.checked_add(backing.size)?, page)?;
        // The one admission rule, which asks the map's standing refusal before
        // the latches. Assembling it here from the latches alone is what let
        // this rail import on a host that had already refused the whole map.
        let align = crate::runtime::guest_ram_map::packed_alias_import_align(host, map_len)?;
        let gpas = gva_mem::task_gva_page_gpas(
            host,
            &state.tasks,
            task_id,
            page_base,
            map_len,
            state.page_shift,
        );
        if gpas.len() as u64 != map_len / page {
            return None;
        }
        // Stable specifically: both the `runs` entry below and, on the
        // `zc_packed_alias_import` arm, the import itself outlive this call, and
        // a Vulkan guest import is released after the last fence rather than at
        // the drop. A view the host owes a release for is released here instead
        // and the whole resolution refuses.
        let host_base = host.stable_page_alias(&gpas, page as usize)?;
        // A packed view answers a **scatter**: Vulkan host-pointer memory takes
        // one contiguous host range, and a linear guest resource may name guest
        // pages that are not contiguous. When this window's pages *are* one run
        // there is no scatter to answer, and the bytes are already inside the
        // RAMBlock import this device holds for the VM's lifetime — so importing
        // them again allocates a second device memory over memory already
        // imported, and pays the driver's page pinning for it.
        //
        // That was not a small cost. One driven Maps boot made **13 681**
        // host-pointer imports totalling 5.05 s of `vkAllocateMemory`, and
        // **12 380 of them were 4096 bytes** — a single page, which is one run by
        // construction, so every one of those was answering a scatter that could
        // not exist. `host_ram_import`'s own census doc states the invariant this
        // broke: "one or two for a whole boot. A count that tracks the workload
        // is a per-resource import".
        //
        // `reference_for_pages` is the existing resolver for exactly this
        // question — it checks the run itself and refuses `Scattered` with a run
        // count — so the contiguous case is routed to it rather than re-deciding
        // contiguity here. The alias above stays: on a contiguous window it hands
        // back a direct RAMBlock alias, which is a lookup, and `runs` needs that
        // host pointer either way.
        let ramblock = crate::runtime::guest_ram_map::reference_for_pages(
            host,
            &gpas,
            page,
            head,
            backing.size,
        )
        .ok()
        .and_then(|guest| {
            // `head` is an offset *into the import*, because every consumer
            // spends it as `import.slice(head + offset, span)`. Against the
            // RAMBlock import that is where this window's first byte sits in the
            // block, not where it sits in a freshly mapped alias.
            let base = guest.import().gpa_base()?;
            let head_in_import = gpas.first()?.checked_add(head)?.checked_sub(base)?;
            Some((std::sync::Arc::clone(guest.import()), head_in_import, guest))
        });
        let (import, head, guest) = match ramblock {
            Some(resolved) => {
                crate::runtime::drain::note_store_route("zc_packed_ramblock");
                resolved
            }
            None => {
                crate::runtime::drain::note_store_route("zc_packed_alias_import");
                crate::runtime::drain::note_store_route(packed_scatter_band(&gpas, page));
                let import = std::sync::Arc::new(
                    crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                        host_base, map_len, align,
                    )
                    .ok()?,
                );
                let whole = import.slice(head, backing.size).ok()?;
                let guest = crate::runtime::guest_ram::GuestRef::new(
                    std::sync::Arc::clone(&import),
                    whole,
                )
                .ok()?;
                (import, head, guest)
            }
        };
        Some(PackedBufferResolution::Available(PackedBuffer {
            gva: backing.gva,
            size: backing.size,
            head,
            import,
            gpas: std::sync::Arc::new(gpas),
            runs: std::sync::Arc::new(vec![crate::backend::vulkan::engine::GuestRun {
                host_ptr: host_base.checked_add(head as usize)?,
                len: backing.size,
            }]),
            pages: std::sync::Arc::new(vec![
                crate::runtime::guest_ram_map::GuestWindowRun {
                    window_offset: 0,
                    guest,
                },
            ]),
        }))
    })()
    .unwrap_or_else(unavailable);

    crate::runtime::drain::note_store_route(match (rail, &made) {
        (PackedResourceRail::Buffer, PackedBufferResolution::Available(_)) => {
            "zc_buffer_packed_alias"
        }
        (PackedResourceRail::Buffer, PackedBufferResolution::Unavailable { .. }) => {
            "zc_buffer_packed_unavailable"
        }
        (PackedResourceRail::LinearSample, PackedBufferResolution::Available(_)) => {
            "zc_lin_packed_alias"
        }
        (PackedResourceRail::LinearSample, PackedBufferResolution::Unavailable { .. }) => {
            "zc_lin_packed_unavailable"
        }
    });
    let available = matches!(made, PackedBufferResolution::Available(_));
    state
        .bound_buffers
        .insert_packed(task_id, resource_ref, made);
    available
}

pub(super) fn slice_packed_buffer(
    packed: &crate::runtime::bound_buffers::PackedBuffer,
    offset: u64,
    span: u64,
) -> Option<crate::runtime::bound_buffers::BoundBuffer> {
    offset.checked_add(span).filter(|&end| end <= packed.size)?;
    Some(crate::runtime::bound_buffers::BoundBuffer {
        gva: packed.gva.checked_add(offset)?,
        span,
        source_offset: offset,
        runs: std::sync::Arc::clone(&packed.runs),
        pages: Some(std::sync::Arc::clone(&packed.pages)),
    })
}

pub(super) fn sampled_backing_from_packed(
    packed: &crate::runtime::bound_buffers::PackedBuffer,
    level_offset: u64,
    row_pitch: u64,
    span: u64,
    owner: crate::model::TaskResourceLifetimeRef,
) -> Option<crate::backend::vulkan::engine::GuestSampledBacking> {
    let plane_offset = packed.head.checked_add(level_offset)?;
    plane_offset
        .checked_add(span)
        .filter(|end| *end <= packed.import.len())?;
    Some(crate::backend::vulkan::engine::GuestSampledBacking {
        backing: crate::backend::vulkan::engine::GuestTargetBacking {
            allocation_host_ptr: packed.import.host_base(),
            allocation_len: packed.import.len(),
            plane_offset,
            row_pitch,
        },
        import: std::sync::Arc::clone(&packed.import),
        owner,
        origin: crate::backend::vulkan::engine::SampledByteOrigin::LinearTexture,
    })
}

/// Build the single-plane direct sampled request from a borrowed retained
/// allocation. Only the execution payloads take new strong references; the
/// allocation geometry and physical construction list stay with the resource.
// Each argument is an independently decoded piece of the guest's sampled-source
// contract; grouping them into a struct would only move the same fields.
#[allow(clippy::too_many_arguments)]
pub(super) fn direct_linear_sample_from_packed(
    packed: &crate::runtime::bound_buffers::PackedBuffer,
    level_offset: u64,
    row_pitch: u64,
    span: u64,
    row_length_texels: u32,
    native: TexelLayout,
    format: ash::vk::Format,
    native_components: pixel_format::SwizzlePlan,
    owner: crate::model::TaskResourceLifetimeRef,
) -> Option<SampledSourceRequest> {
    let direct_image = sampled_backing_from_packed(
        packed,
        level_offset,
        row_pitch,
        span,
        owner,
    )?;
    Some(SampledSourceRequest::GuestRuns(
        crate::backend::vulkan::engine::GuestRunSource {
            runs: std::sync::Arc::clone(&packed.runs),
            source_offset: level_offset,
            total_len: span,
            row_length_texels,
            pages: Some(std::sync::Arc::clone(&packed.pages)),
            direct_image: Some(direct_image),
        },
        native,
        format,
        1,
        None,
        crate::runtime::gather_witness::GatherVouch::Fresh,
        native_components,
    ))
}

/// Build one directly sampled plane from the mapping's retained allocation.
/// The complete allocation is the resource whose plane offset and row pitch
/// the descriptor names; no copied-image freshness witness participates in
/// this resource-owned disposition.
struct MappedSamplePlane {
    mapping_id: u32,
    base_off: u64,
    row_pitch: u64,
    span: u64,
    row_length_texels: u32,
    origin: crate::backend::vulkan::engine::SampledByteOrigin,
    owner: crate::model::TaskResourceLifetimeRef,
}

fn mapped_sampled_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plane: MappedSamplePlane,
) -> Option<crate::backend::vulkan::engine::GuestRunSource> {
    use crate::backend::vulkan::engine::{GuestRun, GuestRunSource};
    use crate::runtime::guest_ram::GuestRef;

    let MappedSamplePlane {
        mapping_id,
        base_off,
        row_pitch,
        span,
        row_length_texels,
        origin,
        owner,
    } = plane;

    let (import, _footprint) =
        crate::runtime::mapper::ensure_contig_import_with_footprint(state, host, mapping_id)?;
    let end = base_off.checked_add(span)?;
    if end > import.len() {
        return None;
    }
    let whole = import.slice(0, import.len()).ok()?;
    let guest = GuestRef::new(std::sync::Arc::clone(&import), whole).ok()?;
    let direct_image = Some(crate::backend::vulkan::engine::GuestSampledBacking {
        backing: crate::backend::vulkan::engine::GuestTargetBacking {
            allocation_host_ptr: import.host_base(),
            allocation_len: import.len(),
            plane_offset: base_off,
            row_pitch,
        },
        import: std::sync::Arc::clone(&import),
        owner,
        origin,
    });
    Some(GuestRunSource {
        runs: std::sync::Arc::new(vec![GuestRun {
            host_ptr: import.host_base(),
            len: import.len(),
        }]),
        source_offset: base_off,
        total_len: span,
        row_length_texels,
        pages: Some(std::sync::Arc::new(vec![
            crate::runtime::guest_ram_map::GuestWindowRun {
                window_offset: 0,
                guest,
            },
        ])),
        direct_image,
    })
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

/// Byte window and Vulkan row pitch for one complete linear texture level.
/// Depth planes are consecutive at `row_stride * height`; only the final
/// plane's final-row padding is outside the sampled window.
pub(super) fn strided_level_extent(
    layout: &crate::runtime::decode::resource::TextureLevelLayout,
    bpp: u64,
) -> Option<(u64, u32)> {
    let (last_plane, row_length_texels) = strided_window_extent(
        layout.width,
        layout.height,
        bpp,
        layout.row_stride,
    )?;
    let preceding_planes = u64::from(layout.planes() - 1)
        .checked_mul(layout.row_stride)?
        .checked_mul(u64::from(layout.height))?;
    Some((preceding_planes.checked_add(last_plane)?, row_length_texels))
}

/// Gather `span` bytes from `base_off` into mapping `mid`'s guest pages as host
/// runs, landing any deferred writeback that aliases them first. Returns the
/// window's own page list beside the runs, for the same reason
/// [`task_gva_guest_run_window`] does.
///
/// Shared by the type-11 attachment seed and the type-11/type-5 sampled rails,
/// which reach the same pages through different window math.
///
/// # No settle, for the reason its linear twin already states
///
/// This produced a settle until it was measured at 945 waits and 0.63 s on a
/// driven boot, justified as "the coherence rule the CPU loaders obey: a
/// resident-authoritative window covering this mapping must land before the GPU
/// reads, or the gather sees the pre-Store bytes". That rule is the CPU
/// loaders', and this is not one of them. Nothing here reads a pixel byte: it
/// resolves a page list and coalesces it into runs, and the *GPU* reads those
/// runs when the draw's command buffer executes.
///
/// A guest-page writeback is a GPU command on the same single queue, submitted
/// before this call can return, so queue order already puts it ahead of the
/// gather. [`try_linear_sample_zero_copy`] states the same argument for the
/// linear gather and has never taken a settle; this is the arm that diverged.
fn mapping_window_guest_runs<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    base_off: u64,
    span: u64,
) -> Option<(Vec<u64>, Vec<crate::backend::vulkan::engine::GuestRun>)> {
    let gpas = mapper::mapping_page_gpas(state, host, mid)?;
    let page = state.page_size();
    if (gpas.len() as u64).saturating_mul(page) < base_off.checked_add(span)? {
        return None;
    }
    let first_page = (base_off / page) as usize;
    let head_off = base_off % page;
    let need_pages = (head_off + span).div_ceil(page) as usize;
    let window = gpas.get(first_page..first_page + need_pages)?;
    let runs = coalesce_pages_to_runs(host, window, page, head_off, span).ok()?;
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
///
/// The guest bind itself has no length, so `size - offset` remains the admission
/// window and the fallback whenever reflection cannot prove a tighter answer.
/// A reflected bounded object or invocation-bounded footprint narrows only the
/// bytes walked and moved. Unbounded pointers, unknown access, and indexed
/// vertex access keep the full window.
fn try_buffer_zero_copy_resolved<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    backing: &BufferBacking,
    offset: u64,
    extent_cap: Option<u64>,
) -> Option<crate::runtime::bound_buffers::BoundBuffer> {
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        // The guest bound past the end of the allocation it named. Counted
        // rather than dropped: it is the guest disagreeing with its own
        // descriptor, and it is the one route here that is not about paging.
        crate::runtime::drain::note_store_route("zc_buffer_offset_past_end");
        return None;
    }
    let Some(span) = host_alloc_len(size - offset)
        .filter(|&n| n > 0)
        .map(|n| n as u64)
    else {
        // A declared length this process cannot address. `offset < size` above
        // makes the `n > 0` arm unreachable, so this is the width check alone.
        crate::runtime::drain::note_store_route("zc_buffer_span_unusable");
        return None;
    };
    // The shader's proven reach, when it has one. `min` and not the cap alone:
    // a declared object larger than what is left of the allocation is the guest
    // and the shader disagreeing, and the allocation is the side that bounds
    // what this device may read.
    let full = span;
    // The floor gates the **gather**, not the rail.
    //
    // `ZERO_COPY_BUFFER_MIN_BYTES` earns its keep against one outcome and its
    // own doc says which: removing it moved `submit_us` by 13 % because "every
    // gathered bind is a recorded GPU gather", while `stage_us` — the CPU read
    // it was once believed to be saving — barely moved. But it used to refuse
    // here, at the top of the rail, which also refused the two outcomes below
    // that record no gather at all: `zc_buffer_imported`, an offset into the
    // RAMBlock import this device already holds for the VM's lifetime, and the
    // retained packed alias. Neither costs a submit, and neither was ever what
    // the measurement was about.
    //
    // What that cost is the sentence the route family's own doc opens with: **a
    // resolution is cached; a refusal is not.** An admitted bind lands in the
    // held-resolution registry and answers `zc_buffer_held` on every later draw;
    // a refused one re-resolves its backing and re-reads its bytes on the CPU
    // every draw for as long as the guest keeps binding it. A driven macos-13
    // Maps leg scored `zc_buffer_below_floor` 2 909 326 against
    // `zc_buffer_held` 1 820 542 — the refusal was the *majority* route, and
    // `binds_us` was 31 % of per-chain cost with two thirds of it in the vertex
    // loads. The floor's own table was measured where it governed one bind in
    // eight, on a different workload; `AGENTS.md`'s rule against generalising
    // across workloads applies to it as much as to a pathway.
    //
    // So the span is computed unconditionally and the floor is asked again at
    // the one place a gather is what would actually be recorded.
    let span = extent_cap.map_or(full, |cap| full.min(cap));
    let gather_eligible = gather_span_if_eligible(full, extent_cap).is_some();
    // Counted only once the rail has actually taken the bind. Counting at the
    // narrowing instead credited this rail with bytes the bind then went and
    // read on the CPU path anyway, which is a saving that did not happen.
    if span < full {
        crate::runtime::drain::note_store_route("zc_buffer_extent_narrowed");
        crate::runtime::drain::note_store_route_n("zc_buffer_extent_saved_bytes", full - span);
    }
    // Packed construction is not behind the floor, and the registry census is
    // why. A packed alias is retained once per `(task, reference)` and every
    // offset slices it, which is the representation this rail is built around —
    // `load_buffer_content_resolved` explicitly declines to add a per-offset
    // entry when one exists, "the per-offset registry the packed representation
    // exists to remove".
    //
    // Admitting sub-floor binds while leaving this behind the floor sent them to
    // the direct window instead, and that path *does* key per offset: one driven
    // Maps leg took `bound_buffers` from `entries=0 peak=0` — the registry was
    // not in use at all — to `entries=9770 peak=16723`, with 1926 pairs holding
    // more than one offset and one holding 56. Against that, a retained alias
    // per resource is ~2138 pairs total and each answers every offset in O(1).
    if ensure_packed_resource(
        state,
        host,
        task_id,
        buffer_ref,
        backing,
        PackedResourceRail::Buffer,
    ) {
        let packed = state.bound_buffers.packed_available(
            task_id,
            buffer_ref,
            backing.gva,
            backing.size,
        )?;
        if let Some(bound) = slice_packed_buffer(packed, offset, span) {
            crate::runtime::drain::note_store_route("zc_buffer_imported");
            return Some(bound);
        }
    }
    // No settle here, for the reason `try_linear_sample_zero_copy` states at
    // length: this rail hands the engine guest-RAM runs and the *GPU* reads them
    // when the draw's command buffer executes, so a guest-page writeback — a GPU
    // command already on the same single queue — is ordered ahead of it by
    // submission order. Only the CPU readers, which touch the pages with this
    // thread, owe the block.
    // Walk exactly the bound range. Resolving the whole backing and slicing out
    // the bind would translate every page of the allocation to serve one bind,
    // and would refuse a bind whose allocation has an unmapped tail page even
    // though the bind itself resolves.
    let (gpas, runs) = match task_gva_guest_run_window(state, host, task_id, gva + offset, span) {
        Ok(window) => window,
        Err(refusal) => {
            crate::runtime::drain::note_store_route(match refusal {
                WindowRefusal::NoAlias => "zc_buffer_no_alias",
                WindowRefusal::SpanUnmapped => "zc_buffer_span_unmapped",
                WindowRefusal::Untileable => "zc_buffer_untileable",
            });
            return None;
        }
    };
    let page = state.page_size();
    let pages = guest_page_window(host, gpas, page, (gva + offset) % page, span);
    // Here, and only here, is it known whether this bind would record a gather,
    // which is the one thing the floor governs. See [`window_binds_zero_copy`].
    //
    // The walk above is spent either way and is not cached on this exit, so the
    // refusing arm is a real new cost for a *scattered* sub-floor buffer. It is
    // counted apart from the binds that never reached the walk, so a boot can
    // say how much of the family that is.
    if !window_binds_zero_copy(pages.is_none(), gather_eligible) {
        crate::runtime::drain::note_store_route("zc_buffer_below_floor");
        crate::runtime::drain::note_store_route("zc_buffer_below_floor_walked");
        return None;
    }
    crate::runtime::drain::note_store_route(if pages.is_some() {
        "zc_buffer_imported"
    } else {
        "zc_buffer_gathered"
    });
    Some(crate::runtime::bound_buffers::BoundBuffer {
        gva: gva + offset,
        span,
        source_offset: 0,
        runs: std::sync::Arc::new(runs),
        pages,
    })
}

/// The engine's view of a held resolution.
///
/// One spelling for the fresh walk and the lookup, so a resolution cannot mean
/// one thing on the draw that built it and another on every draw after.
fn bound_buffer_content(
    bound: &crate::runtime::bound_buffers::BoundBuffer,
) -> crate::backend::vulkan::engine::BufferContent {
    use crate::backend::vulkan::engine;
    engine::BufferContent::GuestRuns(engine::GuestRunSource {
        runs: std::sync::Arc::clone(&bound.runs),
        source_offset: bound.source_offset,
        total_len: bound.span,
        row_length_texels: 0,
        pages: bound.pages.clone(),
        direct_image: None,
    })
}

/// Load one draw-time buffer bind: the zero-copy rail when allowed and
/// eligible, else the CPU staging read. `allow_zero_copy` is false for
/// buffers feeding Constant-step attributes (the engine prepends a CPU
/// base-instance prefix to those).
///
/// # The `zc_buffer_*` route family, and what it is for
///
/// Every bind that reaches here with `allow_zero_copy` and a resolvable
/// backing takes **exactly one** route, so the family sums to the attempts:
///
/// ```text
/// held                                    the registry answered
/// offset_past_end + span_unusable         the descriptor disagrees with itself
/// below_floor                             too small to be worth the rail
/// no_alias + span_unmapped + untileable   the rail was tried and refused
/// imported + gathered                     the rail ran
/// ```
///
/// The split exists to answer one question the held-resolution registry cannot:
/// **how much of the per-draw cost is being paid over and over.** A resolution
/// is cached; a refusal is not. So a reference in the last-but-one group
/// re-walks the task page table *and* re-pays the CPU staging read on every
/// draw the guest binds it, for as long as it keeps binding it — and before
/// these routes existed, that path was the only outcome in this function with
/// no name at all. A steady rate there is repeats, because the guest's live
/// reference set is bounded and the bind rate is not.
///
/// Only `span_unmapped` is a refusal a mapped-range record could answer without
/// walking. `untileable` walked successfully and still could not bind, so
/// nothing upstream of the walk would have saved it — which is why the two are
/// counted apart rather than as one "the window failed".
///
/// `extent_cap` is the byte extent the shader on this draw proved it cannot read
/// past, from [`crate::runtime::spirv_bind::reflected_buffer_extent`]. `None`
/// keeps the whole-allocation window this function has always bound. It is part
/// of the registry key rather than of the resolution, because it describes the
/// shader and not the bind — see [`crate::runtime::bound_buffers`].
///
/// The CPU staging and GPU gather rails apply the same cap. Rail admission is
/// deliberately different: it uses the full guest window, not the narrowed
/// extent, so a small reflected object keeps the registry and execution route
/// the guest bind would otherwise have used. [`gather_span_if_eligible`] owns
/// that relation.
/// The bytes a gather-eligible guest buffer window actually has to move.
///
/// Admission is a property of the full window the guest bound; reflection only
/// bounds the shader's reach within it. Keeping both decisions in this function
/// makes it impossible to save bytes by accidentally moving a small reflected
/// object off the gather rail and its held-resolution registry.
pub(super) fn gather_span_if_eligible(full: u64, extent_cap: Option<u64>) -> Option<u64> {
    (full >= ZERO_COPY_BUFFER_MIN_BYTES).then(|| extent_cap.map_or(full, |cap| full.min(cap)))
}

/// Whether a walked guest window may bind zero-copy, once it is known whether
/// binding it would record a GPU gather.
///
/// The whole of what [`ZERO_COPY_BUFFER_MIN_BYTES`] governs, in one place. That
/// floor was measured against exactly one outcome — a recorded gather, which
/// costs a submit — and it used to be asked before the window was walked, where
/// "would this gather?" is not yet answerable. Asking it there refused two
/// outcomes that record no gather: a window resolving to pages is an offset into
/// an import this device already holds for the VM's lifetime, and costs nothing
/// a submit would notice.
///
/// The asymmetry is the point, so the truth table is the test: size may only
/// ever veto a *gather*.
pub(super) fn window_binds_zero_copy(needs_gather: bool, gather_eligible: bool) -> bool {
    !needs_gather || gather_eligible
}

/// Answer a retained zero-copy resolution without touching the object table or
/// walking the task page table.
fn held_buffer_content(
    state: &mut DeviceState,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    extent_cap: Option<u64>,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    // The guest contract is buffer-plus-offset. Once the whole resource has a
    // packed alias, derive the bind directly from that one retained object:
    // neither the object descriptor nor the task page table changes between
    // offsets, and both announce the events that retire this entry.
    if let Some(crate::runtime::bound_buffers::PackedBufferResolution::Available(packed)) =
        state.bound_buffers.packed(task_id, buffer_ref)
    {
        if offset < packed.size {
            let full = packed.size - offset;
            // No floor here. Slicing a retained packed alias records no gather —
            // the alias is already imported and this hands back a sub-range of
            // it — so `ZERO_COPY_BUFFER_MIN_BYTES`, which governs gathers alone,
            // has nothing to say. It used to be asked, and that sent every
            // sub-floor bind past the one branch that answers in O(1) and down
            // the full resolve-and-reconstruct path on *every* draw: one driven
            // Maps leg scored `zc_buffer_imported` 1 924 454 against
            // `zc_buffer_held` 1 389 939, for binds whose alias was sitting
            // right here the whole time.
            let span = extent_cap.map_or(full, |cap| full.min(cap));
            if let Some(bound) = slice_packed_buffer(packed, offset, span) {
                crate::runtime::drain::note_store_route("zc_buffer_held");
                return Some(bound_buffer_content(&bound));
            }
        }
    }
    // The registry is keyed on the same cap the walk uses, or a lookup could
    // answer with a shorter span than this shader needs.
    // A held resolution answers before anything is resolved at all: the walk
    // below produces the same runs until the guest moves the addresses, and it
    // announces every such move. This is the whole point of the registry — see
    // `crate::runtime::bound_buffers`.
    if let Some(bound) = state
        .bound_buffers
        .get(task_id, buffer_ref, offset, extent_cap)
    {
        let content = bound_buffer_content(bound);
        crate::runtime::drain::note_store_route("zc_buffer_held");
        return Some(content);
    }
    None
}

/// Resolve a previously validated backing through the zero-copy ladder, with
/// the CPU read as the capability fallback.
// Hot buffer-load path: the arguments are the decoded bind plus the host and
// device state it resolves against, and threading a struct through would only
// rename them.
#[allow(clippy::too_many_arguments)]
fn load_buffer_content_resolved<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    allow_zero_copy: bool,
    extent_cap: Option<u64>,
    backing: &BufferBacking,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    // Resolve the backing (object-list entry + descriptor) ONCE and share it
    // between the zero-copy attempt and the CPU fallback. Sub-floor binds used
    // to walk the task PT twice — once in the failed ZC attempt, once in the
    // CPU read.
    if allow_zero_copy {
        if let Some(bound) = try_buffer_zero_copy_resolved(
            state, host, task_id, buffer_ref, backing, offset, extent_cap,
        ) {
            let content = bound_buffer_content(&bound);
            // A packed resource is already retained by `(task, reference)` and
            // the content above shares its whole-buffer source. Holding this
            // offset as a second entry would recreate the per-offset registry
            // the packed representation exists to remove.
            let packed = matches!(
                state.bound_buffers.packed(task_id, buffer_ref),
                Some(crate::runtime::bound_buffers::PackedBufferResolution::Available(_))
            );
            if !packed {
                state
                    .bound_buffers
                    .insert(task_id, buffer_ref, offset, extent_cap, bound);
            }
            return Some(content);
        }
    }
    let bytes =
        read_buffer_bytes_resolved(state, host, task_id, buffer_ref, backing, offset, extent_cap)?;
    Some(crate::backend::vulkan::engine::BufferContent::from(bytes))
}

// Same shape as `load_buffer_content_resolved`, which this forwards to.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_buffer_content<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&crate::model::TaskResource>,
    offset: u64,
    allow_zero_copy: bool,
    extent_cap: Option<u64>,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    if allow_zero_copy {
        if let Some(content) =
            held_buffer_content(state, task_id, buffer_ref, offset, extent_cap)
        {
            return Some(content);
        }
    }
    // Resolve the backing (object-list entry + descriptor) ONCE and share it
    // between the zero-copy attempt and the CPU fallback. Sub-floor binds used
    // to walk the task PT twice — once in the failed ZC attempt, once in the
    // CPU read.
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref, resource)?;
    load_buffer_content_resolved(
        state,
        host,
        task_id,
        buffer_ref,
        offset,
        allow_zero_copy,
        extent_cap,
        &backing,
    )
}

/// Retain an indexed draw's exact guest-buffer window for the Vulkan vertex
/// input stage. Unlike the Metal fallback, this does not materialize the index
/// array on the CPU: Vulkan consumes the bounded resource directly when the
/// command buffer executes.
fn load_index_content_reason<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<crate::backend::vulkan::engine::BufferContent, IndexLoadReason> {
    let (backing, need) = resolve_index_window_reason(state, host, task_id, info)?;
    let extent = Some(need as u64);
    if let Some(content) = held_buffer_content(
        state,
        task_id,
        info.index_buffer_ref,
        info.index_buffer_offset,
        extent,
    ) {
        return Ok(content);
    }
    load_buffer_content_resolved(
        state,
        host,
        task_id,
        info.index_buffer_ref,
        info.index_buffer_offset,
        true,
        extent,
        &backing,
    )
    .ok_or(IndexLoadReason::ReadFail)
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
///
/// **Every gate names itself on the `zc_lin_*` route set, including the ones
/// that decline before the walk.** Only the three walk refusals used to, and
/// the whole set read zero on a driven Safari-drag boot — which says the walk
/// was never reached and says nothing at all about why. The rail declined
/// 91 687 times in that boot's steady state and the caller fell to
/// `load_linear_guest_memoized`, whose own doc concedes it re-reads the full
/// `bpr * h` span out of guest memory and memcmps it against the memo on every
/// bind, hit or miss. That was 62 µs of a 191 µs draw. A rail that costs the
/// device its largest per-draw item when it declines may not decline silently;
/// the routes below are what turn "the gather is not running" into which gate.
/// Serve a sampled GVA span off the engine resident a render Store published
/// into it — the GVA twin of `t11rung_resident`.
///
/// # What it is worth
///
/// The rung below this one reads the span out of guest pages, and before it can
/// read it has to wait for any render writeback landing in those same pages.
/// That wait was **94 % of `sampled_phase`'s `resolve_us`** on a driven Safari
/// drag — 275 ms of a 610 ms drain second, the largest single item in the
/// device — and narrowing cannot touch it, because the pages the reader wants
/// are exactly the pages the Store is writing:
/// `settle_linear_memo_read_overlap` was 4 700 of 4 717 waits.
///
/// A probe on that rung counted the join before this was built. Of the sampled
/// binds it saw, **20 688** named a GVA target a Store had stamped and
/// **16 202 of those read
/// [`crate::runtime::gva_store_witness::GvaWriteReach::Quiet`]** — the pages
/// still hold the Store's frame, so the resident does too and nobody has to
/// read anything.
/// Genuine refusals (the guest repainted, or the Store never stamped) were 79.
///
/// # Why the whole key and not the address
///
/// The generation is recomputed here from the span's *current* page set rather
/// than taken from any entry found at this address. The witness map can hold an
/// orphan — a target whose pages have since moved, keyed under the hash they
/// had — and an orphan's token still answers about pages that now belong to
/// something else. Recomputing means a moved page list simply misses, which is
/// [`crate::runtime::gva_store_witness`]'s own rule: the wrong answer is
/// unreachable rather than guarded.
///
/// # The two conditions, and what each rules out
///
/// * [`crate::runtime::gva_store_witness::GvaWriteReach::Quiet`] — neither the
///   guest CPU nor another rail of this device has written the span since the
///   Store. Everything undecidable answers as written, so a host with no dirty
///   bitmap never reaches this rung.
/// * `resident_content_ready` — the engine still holds an image under this
///   identity. A reclaimed or never-filled slot refuses.
///
/// What is deliberately *not* a condition is "no draw has landed in the resident
/// since the Store". A draw landing there is the guest rendering into this very
/// texture, and Metal's model is that a render pass changes the texture it
/// targets — so the resident being ahead of the guest's pages is the content the
/// guest asked to sample, not a stale read.
///
/// # What it measured
///
/// Driven Safari drag, against the boot that supplied the numbers above:
///
/// ```text
///                                    before    after
/// gvarung_resident                        -   20 539
/// gvarung_resident_absent                 -        0
/// settle_linear_memo_read (waits)     4 717    1 446
/// settle_linear_memo_read_overlap     4 700        0
/// settle_linear_memo_read_us          6.70 s   3.74 s
/// fence (waits)                       9 475    6 071
/// fence_us                           11.92 s   5.77 s
/// sampled_phase resolve_us/s          290 ms   195 ms
/// ```
///
/// The row that says the rung did what it was built to do is the third one:
/// **the overlap class is gone**, not reduced. Every remaining memo wait is
/// `_unnamed` — a span whose page walk came up short, which this rung declines
/// before the witness. `gvarung_resident_absent` reading zero says every quiet
/// witness had a live resident behind it, so nothing is being armed and then
/// reclaimed out from under the rung.
///
/// Correctness was taken separately and not from a screenshot, because a rung
/// that serves a stale resident renders a *plausible* frame: the multi-round
/// recomposite run over a live Wikipedia article scored **PATCHED none,
/// UNSCOREABLE none** across six anchors with its reload and movement gates
/// both satisfied. A second workload (page loads and scrolling) served 11 103
/// binds off the rung with no loss on the fail channel.
/// The guest bytes one GVA render target occupies, as the rails that ask about
/// it name them.
///
/// One value rather than five parameters because the five only mean anything
/// together — a stride belongs to a height, and a format decides the channel
/// order the registry keys a resident on — and because two callers assembling
/// the same five by hand is how they come to disagree about one of them.
#[derive(Clone, Copy, Debug)]
pub(super) struct GvaSpan {
    pub texture_ref: u32,
    pub gva: u64,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    /// The guest's declared pixel format, not a host one:
    /// [`gva_resident_format`] turns it into the `format` half of the key.
    pub format: u16,
}

/// Why a GVA span's resident may not stand in for its guest pages.
///
/// Kept apart from a bare `None` because the three have nothing in common: one
/// is a span with no identity at all, one is a target something has written
/// since the Store, and one is a target the engine no longer holds. Each caller
/// names them on its own census routes — the rule is shared, the vocabulary is
/// not, so a reading says which rung refused as well as why.
pub(super) enum GvaResidentRefusal {
    /// The resource has no complete initial transfer backing, so it has no
    /// usable resident identity yet.
    NoGeneration,
    /// The witness will not call the pages quiet.
    Wrote(crate::runtime::gva_store_witness::GvaWriteReach),
    /// Quiet, but the engine is not holding an image under this identity.
    NoResident,
}

fn retained_resident_is_ready(
    backing: Option<crate::backend::vulkan::engine::ResidentContentBacking>,
    registry_query: impl FnOnce() -> bool,
) -> bool {
    use crate::backend::vulkan::engine::ResidentContentBacking;

    match backing {
        Some(
            ResidentContentBacking::GuestAllocation | ResidentContentBacking::DeviceAllocation,
        ) => true,
        Some(ResidentContentBacking::NotReady) => false,
        None => registry_query(),
    }
}

/// Whether the resident named by a GVA texture is still usable.
///
/// A named texture owns a retained allocation lease, so warm binds answer from
/// the texture object without re-entering the global engine. Anonymous and
/// unclassified spans have no protocol lifetime to hold such a lease and keep
/// the fail-closed registry query.
fn gva_resident_ready(
    state: &DeviceState,
    task_id: u32,
    texture_ref: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let backing = (texture_ref != 0)
        .then(|| state.task_resources.get(task_id, texture_ref))
        .flatten()
        .filter(|resource| resource_type_owns_gva_resident(resource.entry.object_type))
        .map(|resource| resource.resident_target_backing(identity));
    let retained = backing.is_some();
    let ready = retained_resident_is_ready(backing, || {
        crate::backend::vulkan::engine::resident_content_ready(identity)
    });
    crate::runtime::drain::note_store_route(match (retained, ready) {
        (true, true) => "gva_ready_resource",
        (true, false) => "gva_not_ready_resource",
        (false, true) => "gva_ready_registry",
        (false, false) => "gva_not_ready_registry",
    });
    ready
}

/// The one currency test behind every GVA resident shortcut: does the engine
/// still hold, under this span's own identity, what the render Store published
/// into these guest pages?
///
/// Two rails ask it — the sampled bind below and the colour LOAD seed — and it
/// is written once because a copied version of this rule is the next
/// divergence. The callers differ only in what they do with the answer.
///
/// A named resource uses its stable host-texture generation and retained
/// transfer backing. The page-set fallback remains only for an attachment with
/// no resource reference and therefore no protocol lifetime to carry.
pub(super) fn gva_resident_if_current<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    span: GvaSpan,
) -> Result<crate::backend::vulkan::engine::TargetIdentity, GvaResidentRefusal> {
    use crate::runtime::gva_store_witness::{reach, GvaTargetKey};

    let GvaSpan {
        texture_ref,
        gva,
        row_stride,
        width: w,
        height: h,
        format,
    } = span;
    if gva == 0 || w == 0 || h == 0 {
        return Err(GvaResidentRefusal::NoGeneration);
    }
    let span_bytes = u64::from(row_stride).saturating_mul(u64::from(h));
    let generation = if texture_ref != 0 {
        crate::runtime::writeback_debt::gva_resource_generation(
            state,
            host,
            crate::runtime::writeback_debt::GvaResourceKey {
                task_id,
                texture_ref,
            },
            gva,
            span_bytes,
        )
    } else {
        gva_span_alloc_generation(state, host, task_id, gva, row_stride, h)
    };
    if generation == 0 {
        return Err(GvaResidentRefusal::NoGeneration);
    }
    let resident_format = gva_resident_format(format);
    let identity = crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva,
        width: w,
        height: h,
        generation,
        format: resident_format,
    };
    // An unpaid Store says the guest pages are deliberately stale and this
    // image is authoritative. The older witness below answers the opposite
    // state: a Store was copied out and both locations still agree. Keeping
    // those states distinct prevents a skipped copy from masquerading as a
    // statement about bytes that were never written.
    if crate::runtime::writeback_debt::gva_resident_authoritative(state, &identity) {
        return gva_resident_ready(state, task_id, texture_ref, &identity)
            .then_some(identity)
            .ok_or(GvaResidentRefusal::NoResident);
    }
    let verdict = reach(
        state,
        host,
        GvaTargetKey {
            gva,
            generation,
            width: w,
            height: h,
            // From the identity built just above, not from `resident_format`
            // again. `GvaTargetKey::of` builds this same key from the same
            // identity on the other side of the witness, and the two must
            // agree — a channel order written by hand at two sites is how they
            // stop agreeing.
            bgra: identity.is_bgra(),
        },
    );
    if !verdict.is_quiet() {
        return Err(GvaResidentRefusal::Wrote(verdict));
    }
    gva_resident_ready(state, task_id, texture_ref, &identity)
        .then_some(identity)
        .ok_or(GvaResidentRefusal::NoResident)
}

#[cfg(test)]
mod gva_resident_ownership_tests {
    use super::*;
    use crate::backend::vulkan::engine::ResidentContentBacking;
    use std::cell::Cell;

    #[test]
    fn a_retained_texture_answers_readiness_without_a_registry_query() {
        for backing in [
            ResidentContentBacking::GuestAllocation,
            ResidentContentBacking::DeviceAllocation,
        ] {
            assert!(retained_resident_is_ready(Some(backing), || {
                panic!("a retained allocation must not query the registry")
            }));
        }
        assert!(!retained_resident_is_ready(
            Some(ResidentContentBacking::NotReady),
            || panic!("a named resource's failed retain is already authoritative")
        ));
    }

    #[test]
    fn an_anonymous_gva_span_keeps_the_registry_fallback() {
        let queries = Cell::new(0_u32);
        assert!(retained_resident_is_ready(None, || {
            queries.set(queries.get() + 1);
            true
        }));
        assert_eq!(queries.get(), 1);
    }
}

/// May the colour LOAD seed at this GVA attachment be skipped, because the
/// engine still holds what the render Store published into these pages?
///
/// Answering `true` **obliges the encode side** to chain or to re-seed:
/// `colors[0].target_seed_rgba` goes out `None` while the attachment still says
/// LOAD, so a pass that does neither loads an undefined attachment.
/// `try_metal2vulkan_draw` owns that obligation.
///
/// # Why the rung this replaces was thought not to exist
///
/// `settle_linear_texture_seed` is the device's largest remaining wait — 4 701
/// per driven drag, **4 692 of them genuine overlaps**, because a
/// `MTLLoadActionLoad` over a GVA target reads the attachment's own guest pages
/// on the CPU while the render Store that published them is still writing. The
/// sampled twin of that wait was removed by `try_gva_resident_sample`, and the
/// same currency test applies here.
///
/// A cross-pass version of this rung existed and was **deleted for reading
/// zero**. That zero was an artifact of where it was sampled: it sat downstream
/// of `mrt_draw_request`, which produced the seed *eagerly* for every GVA LOAD,
/// so by the time it ran no draw had a seedless GVA LOAD target and its
/// denominator was empty by construction. It could not have fired whatever the
/// guest did.
///
/// Asked here instead — at the production site, before `seed_color_load` reads
/// anything — the same question answers, on one driven Safari drag against
/// `load_seed_ok_color` 4 862, which was every colour LOAD seed that boot
/// produced:
///
/// ```text
/// gvaseed_elided       4 849   99.7 %
/// gvaseed_not_quiet       11
/// gvaseed_no_resident      2
/// gvaseed_no_generation    0
/// ```
///
/// # What eliding them did
///
/// ```text
///                                    before    after
/// load_seed_ok_color                  4 862       11
/// settle_linear_texture_seed (waits)  4 792        3
/// settle_linear_texture_seed_us        1.69 s   11 ms
/// fence (waits)                       6 403    3 136
/// fence_us                             6.17 s   4.88 s
/// ```
///
/// `gvaseed_chained` equalled `gvaseed_elided` exactly — 4 475 of each — so
/// every elision was honoured at encode time and `gvaseed_reseeded` never
/// fired. The race is real and must stay handled; it is simply not hot.
///
/// Correctness was not taken from a screenshot, because this class renders a
/// *plausible* frame when it is wrong and this file records a reverted attempt
/// at it that gave a black screen with orange fragments. The multi-round
/// recomposite run over a live Wikipedia article scored **PATCHED none,
/// UNSCOREABLE none** with both its gates satisfied, on five CLEAN offsets and
/// one CHURN.
///
/// # The copying arm never reaches this, and that is the design
///
/// `gva_store_witness` is armed only by the GPU-direct Store rail, so a host
/// without the guest-RAM import stamps nothing and this can never answer yes.
/// Confirmed rather than argued: a `REIMS_VGPU_GUEST_IMPORT=off` boot reads
/// `gvaseed_not_quiet` **3 246 against `load_seed_ok_color` 3 246** — every
/// seed built, none elided — with `gvaseed_elided` and `gvarung_resident` both
/// absent and zero bound imports. That arm keeps the behaviour it had before
/// either rung existed.
pub(super) fn gva_load_seed_elidable<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    span: GvaSpan,
) -> bool {
    use crate::runtime::drain::note_store_route;
    let answer = gva_resident_if_current(state, host, task_id, span);
    note_store_route(match answer {
        Ok(_) => "gvaseed_elided",
        Err(GvaResidentRefusal::NoGeneration) => "gvaseed_no_generation",
        Err(GvaResidentRefusal::Wrote(_)) => "gvaseed_not_quiet",
        Err(GvaResidentRefusal::NoResident) => "gvaseed_no_resident",
    });
    answer.is_ok()
}

pub(super) fn try_gva_resident_sample<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
) -> Option<(u32, u32, SampledSourceRequest)> {
    use crate::runtime::drain::note_store_route;

    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let (w, h) = (layout.width, layout.height);
    let row_stride = u32::try_from(layout.row_stride).ok()?;
    let identity = match gva_resident_if_current(
        state,
        host,
        task_id,
        GvaSpan {
            texture_ref,
            gva,
            row_stride,
            width: w,
            height: h,
            format: tex.pixel_format,
        },
    ) {
        Ok(identity) => identity,
        Err(GvaResidentRefusal::NoGeneration) => return None,
        Err(GvaResidentRefusal::Wrote(verdict)) => {
            note_store_route(verdict.route());
            return None;
        }
        Err(GvaResidentRefusal::NoResident) => {
            note_store_route("gvarung_resident_absent");
            return None;
        }
    };
    let declared_format = tex.declared_pixel_format()?;
    let format = translate::pixel::translate(declared_format).ok()?.vk;
    note_store_route("gvarung_resident");
    Some((w, h, SampledSourceRequest::Target(identity, format)))
}

/// Which repair would let the linear zero-copy rung carry this format, as a
/// route name.
///
/// The three named arms are the three ways [`translate::pixel::sampled_pixels`]
/// declines, and each points at different work: teach the decode contract an
/// ordinal, name a byte layout for a format the contract already defines, or
/// give the image view a component mapping. `_other` is a healthy zero — a
/// firing means `sampled_pixels` grew a fourth decline that this split does not
/// name, not that a format was lost.
fn zc_lin_no_layout_route(reason: translate::TranslateReason) -> &'static str {
    use translate::TranslateReason as R;
    match reason {
        R::UnknownPixelFormat(_) => "zc_lin_no_layout_undefined_format",
        R::NoSampledLayout(_) => "zc_lin_no_layout_no_texel_layout",
        _ => "zc_lin_no_layout_other",
    }
}

pub(super) fn try_linear_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
    owner: crate::model::TaskResourceLifetimeRef,
) -> Option<(u32, u32, SampledSourceRequest)> {
    use crate::backend::vulkan::engine;
    // The object-list entry + descriptor are resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in as `tex`; the
    // cache fallback shares the same decode.
    let Some(declared_format) = tex.declared_pixel_format() else {
        crate::runtime::drain::note_store_route("zc_lin_no_declared_format");
        return None;
    };
    // sRGB variants ride the same rail as their linear siblings: the layout is
    // identical and the CPU loaders never decoded either. The qualifier is
    // still lost, so the census records it rather than letting the fold be
    // silent.
    // **Every layout `sampled_pixels` returns is admitted**, which is the same
    // rule `try_type5_sample_zero_copy` states and applies: that function is
    // the answer to "which guest bytes sample byte-identically through the
    // matching Vulkan image", and a layout it hands back has already passed
    // the identity-components test inside it. The engine creates the image
    // with `vk_texel_layout(native)`, so the texel size and channel order come
    // from the same answer and cannot disagree with it.
    //
    // This rail used to narrow that set again, to four-byte colour plus the
    // single-channel floats, on the stated grounds that "R8/Rg8 video planes
    // keep their existing CPU/type-5 rails". The premise was that R8 and Rg8
    // only ever arrive as video; they do not. A Safari window drag with no
    // video playing produced 37 704 `Rg8` binds, 49 % of every linear sampled
    // bind in the boot, and each one fell to `load_linear_guest_memoized`'s
    // full-span guest re-read plus memcmp. The narrowing was never a
    // correctness rule — `texel_to_rgba8` expands `Rg8` to `(r, g, 0, 255)`
    // and an `R8G8_UNORM` image samples `(r, g, 0, 1)`, which is the same
    // texel — so what it bought was the CPU path on half the binds.
    //
    // `R32_SFLOAT` keeps its extra condition, and it is a host capability
    // rather than a layout one: LUTs are sampled with interpolation and that
    // format's linear-filter feature is optional (absent on Apple/MoltenVK).
    //
    // The two ways a format declines are separated, because they want opposite
    // fixes and a single count reads the same for both. `sampled_pixels`
    // answering `Err` means the *contract* carries no [`TexelLayout`] for the
    // format at all — either it is undefined, or its Metal channels do not sit
    // identically on their Vulkan ones, which is a component mapping this rail
    // does not yet carry. Answering `Ok` with a layout this rail does not admit
    // now means only the host filter gate below.
    //
    // The plan beside the layout is the format's own channel mapping — identity
    // for all but `A8Unorm`, whose byte rides in `R8_UNORM`. It is composed with
    // the guest's type-8 view swizzle where the image view is built, so this
    // rail no longer has to refuse a format for having one.
    let (native, sampled_vk_format, native_components) =
        match translate::pixel::sampled_pixels(declared_format) {
        // Deduped per declared format, which is a handful of values a boot
        // enumerates in a handful of lines. The number is the guest's own
        // `MTLPixelFormat` ordinal, so it names the format without this device
        // having to hold a second spelling of Apple's table.
        // The reason is kept, not discarded. `Err` here has three causes that
        // want three different repairs — the format is outside the decode
        // contract, the contract defines it but no rail names a byte layout for
        // it, or the layout exists but its channels need a swizzle — and a
        // single count reads the same for all three. The sub-route is the
        // reason's own slug, so it cannot drift from the taxonomy in
        // `translate::reason`, and the total is still recorded beside it so the
        // split adds up.
        Err(reason) => {
            crate::runtime::drain::note_store_route("zc_lin_format_no_layout");
            crate::runtime::drain::note_store_route(zc_lin_no_layout_route(reason));
            if crate::observe::first_sight("zc_lin_format_no_layout", u64::from(declared_format)) {
                crate::observe::off(format!(
                    "zc_lin_format_no_layout fmt={declared_format:#x} {reason} \
                     (no sampled TexelLayout; the bind falls to the CPU \
                     re-read + memcmp rung)"
                ));
            }
            return None;
        }
        Ok((layout, _decline, components)) => {
            // Every layout is asked about, not just the one that was known to
            // be optional. This rail hands the guest's bytes to a sampler that
            // interpolates them, so "can this host filter this format" is a
            // question about the layout, and a table indexed by the layout
            // cannot be missing an entry for one that was added later.
            if !engine::supports_sampled_layout_linear_filter(layout) {
                crate::runtime::drain::note_store_route("zc_lin_layout_unfilterable");
                return None;
            }
            let format = translate::pixel::translate(declared_format).ok()?.vk;
            (layout, format, components)
        }
    };
    let bpp = native.bytes_per_texel();
    let Some((gva, layout)) = tex.level_gva(0, state.page_shift) else {
        crate::runtime::drain::note_store_route("zc_lin_no_level_gva");
        return None;
    };
    let (w, h) = (layout.width, layout.height);
    if w == 0 || h == 0 {
        crate::runtime::drain::note_store_route("zc_lin_no_extent");
        return None;
    }
    let Some((span, row_length_texels)) = strided_level_extent(layout, bpp as u64)
    else {
        crate::runtime::drain::note_store_route("zc_lin_unstrideable");
        return None;
    };
    let planes = layout.planes();
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        crate::runtime::drain::note_store_route("zc_lin_past_allocation");
        return None;
    }
    // No settle here, and that is the difference between this rail and the CPU
    // loaders it replaces.
    //
    // A CPU loader reads the guest's pages with this thread, which nothing
    // orders against a submitted-but-unexecuted writeback, so it has to block
    // until the writeback has landed. This rail does not read anything: it hands
    // the engine guest-RAM runs and the *GPU* reads them when the draw's command
    // buffer executes. A guest-page writeback is a GPU command on the same
    // single queue, and `copy_image_level0_to_buffer` submits it before
    // returning — it is already on the queue by the time the debt flag that a
    // settle consults is even set. Queue order therefore already puts the
    // writeback ahead of this gather, and a CPU fence wait buys an ordering that
    // holds without it.
    //
    // `try_type11_sample_zero_copy` and `try_type5_sample_zero_copy` are the two
    // rails that were already written this way, and this one is now consistent
    // with them.
    //
    // That argument is about a **submitted** writeback, and it does not extend
    // to an owed one: a writeback debt is a frame this device rendered and
    // deliberately did not write down, so there is no command on any queue for
    // queue order to order and the pages hold the frame before it. The payment
    // is what puts it on the queue; then the paragraph above applies again.
    //
    // Paid through the texture ref, the same call this rail's CPU twin makes,
    // because a linear texture's bytes may alias a surface this device owes a
    // frame and only `pay_for_texture` resolves one id namespace to the other.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    // Retain the texture's complete allocation once. A sampled image needs the
    // allocation base, level offset and row pitch together; reducing it to the
    // level's page runs would throw away the resource shape and force a copy.
    let allocation = tex
        .allocation_base_gva(state.page_shift)
        .filter(|_| tex.allocation_size != 0)
        .map(|allocation_gva| BufferBacking {
            gva: allocation_gva,
            size: tex.allocation_size,
        });

    let available = allocation.as_ref().is_some_and(|backing| {
        ensure_packed_resource(
            state,
            host,
            task_id,
            texture_ref,
            backing,
            PackedResourceRail::LinearSample,
        )
    });

    // A warm resource bind borrows the retained allocation directly. The
    // single-plane request retains only the import/run payloads execution
    // needs; allocation geometry and the physical construction list remain on
    // the task resource.
    if available && planes == 1 {
        let backing = allocation.as_ref()?;
        if let Some(packed) = state.bound_buffers.packed_available(
            task_id,
            texture_ref,
            backing.gva,
            backing.size,
        ) {
            if let Some(request) = direct_linear_sample_from_packed(
                packed,
                layout.offset,
                layout.row_stride,
                span,
                row_length_texels,
                native,
                sampled_vk_format,
                native_components,
                owner.clone(),
            ) {
                if span < SAMPLED_GATHER_MIN_BYTES {
                    crate::runtime::drain::note_store_route("zc_lin_direct_below_gather_floor");
                }
                return Some((w, h, request));
            }
        }
    }

    // The witness recorder mutably borrows `state`, so the multi-plane arm
    // owns its resource state across that call. Single-plane binds return above
    // and never pay this construction-state clone.
    let packed = if available && planes != 1 {
        allocation.as_ref().and_then(|backing| {
            state
                .bound_buffers
                .packed_available(task_id, texture_ref, backing.gva, backing.size)
                .cloned()
        })
    } else {
        None
    };

    if let Some(packed) = packed {
        if !native.a_cost_floor_may_decline() || span >= SAMPLED_GATHER_MIN_BYTES {
            let page = state.page_size();
            let packed_offset = packed.head.checked_add(layout.offset)?;
            let first_page = usize::try_from(packed_offset / page).ok()?;
            let head_off = packed_offset % page;
            let page_count = usize::try_from(head_off.checked_add(span)?.div_ceil(page)).ok()?;
            let witness_gpas = packed
                .gpas
                .get(first_page..first_page.checked_add(page_count)?)?;
            let witness_runs = [engine::GuestRun {
                host_ptr: packed
                    .import
                    .host_base()
                    .checked_add(usize::try_from(packed_offset).ok()?)?,
                len: span,
            }];
            let seen = crate::runtime::gather_witness::note_gather(
                state,
                host,
                crate::runtime::gather_witness::GatherRail::Linear,
                crate::runtime::gather_witness::GatherKey::TaskGva { task_id, gva },
                crate::runtime::gather_witness::GatherWindow {
                    gpas: witness_gpas,
                    runs: &witness_runs,
                    span,
                    page_size: page as usize,
                },
            );
            return Some((
                w,
                h,
                SampledSourceRequest::GuestRuns(
                    engine::GuestRunSource {
                        runs: std::sync::Arc::clone(&packed.runs),
                        source_offset: layout.offset,
                        total_len: span,
                        row_length_texels,
                        pages: Some(std::sync::Arc::clone(&packed.pages)),
                        direct_image: None,
                    },
                    native,
                    sampled_vk_format,
                    planes,
                    Some(LinearSampleIdentity::from(seen.identity)),
                    seen.vouch,
                    native_components,
                ),
            ));
        }
    }

    // Only the copied fallback has a size crossover. A direct image above was
    // constructed from the guest allocation exactly as the resource contract
    // describes, so its size never selects a different ownership model.
    // Single-channel float LUTs have no equivalent CPU loader arm and therefore
    // cannot be declined on cost grounds even on this fallback.
    if native.a_cost_floor_may_decline() && span < SAMPLED_GATHER_MIN_BYTES {
        crate::runtime::drain::note_store_route(if span < 4 * 1024 {
            "zc_lin_below_floor_lt4k"
        } else if span < 16 * 1024 {
            "zc_lin_below_floor_lt16k"
        } else {
            "zc_lin_below_floor_lt64k"
        });
        return None;
    }

    // The copy-backed fallback still covers exactly the bound level window.
    let (gpas, runs) = match task_gva_guest_run_window(state, host, task_id, gva, span) {
        Ok(window) => window,
        Err(refusal) => {
            crate::runtime::drain::note_store_route(match refusal {
                WindowRefusal::NoAlias => "zc_lin_no_alias",
                WindowRefusal::SpanUnmapped => "zc_lin_span_unmapped",
                WindowRefusal::Untileable => "zc_lin_untileable",
            });
            return None;
        }
    };
    let page = state.page_size() as usize;
    let seen = crate::runtime::gather_witness::note_gather(
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
                source_offset: 0,
                total_len: span,
                row_length_texels,
                pages: guest_page_window(host, gpas, page as u64, gva % page as u64, span),
                direct_image: None,
            },
            native,
            sampled_vk_format,
            planes,
            Some(LinearSampleIdentity::from(seen.identity)),
            seen.vouch,
            native_components,
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
    owner: crate::model::TaskResourceLifetimeRef,
) -> Option<SampledSourceRequest> {
    use crate::backend::vulkan::engine;
    use crate::runtime::mapping_write::type11_sample_window;
    if w == 0 || h == 0 {
        return None;
    }
    let (native, sampled_vk_format, base_off, bpr) = {
        let m = state.mappings.get(&mid)?;
        if !m.mapped || m.page_entries.is_empty() {
            return None;
        }
        let format = if m.format != 0 {
            m.format
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        // This rail binds the mapping's raw bytes with the view mapping it was
        // given, so it carries no format plan. Identity is required rather than
        // assumed: `is_four_byte_color` happens to exclude the one non-identity
        // format (`A8Unorm` is a single byte), but that is a coincidence of
        // widths and not the rule this line depends on.
        let native = match translate::pixel::sampled_pixels(format) {
            Ok((layout, _decline, components)) if layout.is_four_byte_color() => {
                if !pixel_format::swizzle_is_identity(&components) {
                    crate::runtime::drain::note_store_route("zc_t11_needs_swizzle");
                    return None;
                }
                layout
            }
            _ => return None,
        };
        let sampled_vk_format = translate::pixel::translate(format).ok()?.vk;
        let (base_off, bpr_u32, _span_end) = type11_sample_window(m, w, h, format)?;
        (native, sampled_vk_format, base_off, bpr_u32 as u64)
    };
    // From the layout the translation chose, as the type-5 rail does, so the
    // texel size cannot disagree with the image the engine creates. The
    // `is_four_byte_color` gate above already fixes it at four.
    let (span, row_length_texels) =
        strided_window_extent(w, h, native.bytes_per_texel() as u64, bpr)?;
    // The owed frame, before anything looks at the pages that owe it.
    //
    // The comment on the linear rail explains why this rail needs no *settle* —
    // a submitted writeback is already ahead of this gather in queue order, so a
    // CPU fence wait buys nothing. A writeback **debt** is a different object: it
    // has not been submitted, there is no command for queue order to order, and
    // the surface's pages hold the frame before the one the resident is holding.
    // Gathering them without paying binds the guest a stale frame.
    //
    // Paid before `note_gather` and not after, because the payment writes those
    // pages: the witness has to see the write, or it vouches for bytes that
    // changed underneath the vouch.
    //
    // Free when nothing is owed — `pay_for_mapping` is one emptiness check on
    // the ledger, which is the answer on nearly every call.
    crate::runtime::writeback_debt::pay_for_mapping(state, host, mid);
    if let Some(source) = mapped_sampled_source(
        state,
        host,
        MappedSamplePlane {
            mapping_id: mid,
            base_off,
            row_pitch: bpr,
            span,
            row_length_texels,
            origin: crate::backend::vulkan::engine::SampledByteOrigin::SurfaceGuestFallback,
            owner,
        },
    ) {
        if span < SAMPLED_GATHER_MIN_BYTES {
            crate::runtime::drain::note_store_route("zc_t11_direct_below_gather_floor");
        }
        return Some(SampledSourceRequest::GuestRuns(
            source,
            native,
            sampled_vk_format,
            1,
            None,
            crate::runtime::gather_witness::GatherVouch::Fresh,
            pixel_format::swizzle_identity(),
        ));
    }
    if span < SAMPLED_GATHER_MIN_BYTES {
        return None;
    }
    let (gpas, runs) = mapping_window_guest_runs(state, host, mid, base_off, span)?;
    let page = state.page_size() as usize;
    let seen = crate::runtime::gather_witness::note_gather(
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
            source_offset: 0,
            total_len: span,
            row_length_texels,
            pages: guest_page_window(host, gpas, page as u64, base_off % page as u64, span),
            direct_image: None,
        },
        native,
        sampled_vk_format,
        1,
        Some(LinearSampleIdentity::from(seen.identity)),
        seen.vouch,
        // Identity: this rail admitted the format only after checking its plan
        // was identity, so there is nothing to fold in.
        pixel_format::swizzle_identity(),
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
pub(super) fn try_type5_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    view: objects::Type5TextureView,
    owner: crate::model::TaskResourceLifetimeRef,
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
    let (native, sampled_vk_format, bpp, base_off, bpr) = {
        let m = state.mappings.get(&mid)?;
        if !m.mapped || m.page_entries.is_empty() {
            return None;
        }
        // Native formats whose guest bytes sample byte-identically through the
        // matching Vulkan image (the CPU loader's `texel_to_rgba8` is a
        // pass-through/swizzle for exactly these); everything else stays CPU.
        // The texel size comes from the layout the translation chose, so it can
        // never disagree with the image the engine creates.
        // A multiplanar view's planes are the video luma/chroma formats, all of
        // which sit identically on their Vulkan spellings. Required rather than
        // assumed, for the reason the type-11 rail states.
        let (native, bpp) = match translate::pixel::sampled_pixels(view.pixel_format) {
            Ok((layout, _decline, components)) => {
                if !pixel_format::swizzle_is_identity(&components) {
                    crate::runtime::drain::note_store_route("zc_t5_needs_swizzle");
                    return None;
                }
                (layout, layout.bytes_per_texel())
            }
            Err(_) => return None,
        };
        let (base_off, bpr_u32, _span_end) =
            type5_sample_window(m, view.plane_index, w, h, view.pixel_format)?;
        let sampled_vk_format = translate::pixel::translate(view.pixel_format).ok()?.vk;
        (native, sampled_vk_format, bpp, base_off, bpr_u32 as u64)
    };
    let (span, row_length_texels) = strided_window_extent(w, h, bpp as u64, bpr)?;
    // Same rule as the linear rail's floor: a cost threshold may only turn a
    // plane away onto a CPU arm that produces the same pixels. The type-11 rail
    // needs no such qualifier because `is_four_byte_color` already fixes what
    // reaches its floor; this one admits every layout the translation names.
    // The owed frame, for the reason `try_type11_sample_zero_copy` states: a
    // debt is not a submitted writeback and queue order cannot order it.
    crate::runtime::writeback_debt::pay_for_mapping(state, host, mid);
    if let Some(source) = mapped_sampled_source(
        state,
        host,
        MappedSamplePlane {
            mapping_id: mid,
            base_off,
            row_pitch: bpr,
            span,
            row_length_texels,
            origin: crate::backend::vulkan::engine::SampledByteOrigin::SerializedSurfaceView,
            owner,
        },
    ) {
        if span < SAMPLED_GATHER_MIN_BYTES {
            crate::runtime::drain::note_store_route("zc_t5_direct_below_gather_floor");
        }
        return Some(SampledSourceRequest::GuestRuns(
            source,
            native,
            sampled_vk_format,
            1,
            None,
            crate::runtime::gather_witness::GatherVouch::Fresh,
            pixel_format::swizzle_identity(),
        ));
    }
    if native.a_cost_floor_may_decline() && span < SAMPLED_GATHER_MIN_BYTES {
        return None;
    }
    let (gpas, runs) = mapping_window_guest_runs(state, host, mid, base_off, span)?;
    let page = state.page_size() as usize;
    let seen = crate::runtime::gather_witness::note_gather(
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
            source_offset: 0,
            total_len: span,
            row_length_texels,
            pages: guest_page_window(host, gpas, page as u64, base_off % page as u64, span),
            direct_image: None,
        },
        native,
        sampled_vk_format,
        1,
        Some(LinearSampleIdentity::from(seen.identity)),
        seen.vouch,
        // Identity: this rail admitted the format only after checking its plan
        // was identity, so there is nothing to fold in.
        pixel_format::swizzle_identity(),
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
/// A straight upload — RGBA8, BGRA8 kept native, or half-float colour kept at
/// its own width — gathers each row with a plain copy (padding skipped, no
/// swizzle) and reports its native format; every other format converts to
/// RGBA8 per row. Shared by the guest-linear memo's miss-fill so its padded and
/// tight branches agree byte-for-byte with the direct loader.
///
/// The straight-upload output is sized from the chosen layout's own texel
/// width. It was sized from `RGBA8_BPP` while only four-byte layouts could be
/// chosen, and the half-float arms are eight and four bytes a texel — so a
/// hard-coded four here would under-allocate an `RGBA16Float` image by half and
/// the row copy would refuse rather than write past it, which is a lost bind
/// dressed as a decline.
fn native_scratch_to_upload(
    scratch: &[u8],
    w: u32,
    h: u32,
    planes: u32,
    bpr: u64,
    sample_fmt: u16,
    tight: u64,
) -> Option<(Vec<u8>, TexelLayout)> {
    let native = native_uploads_for(sample_fmt);
    let bpr = bpr as usize;
    if let Some(fmt) = linear_native_upload_format(sample_fmt, native)
        .filter(|fmt| tight == (w as u64).saturating_mul(fmt.bytes_per_texel() as u64))
    {
        let row_bytes = tight as usize;
        let rows = (h as usize).checked_mul(planes as usize)?;
        let mut out = vec![0u8; row_bytes.checked_mul(rows)?];
        for row_index in 0..rows {
            let src = row_index.checked_mul(bpr)?;
            let dst = row_index * row_bytes;
            out.get_mut(dst..dst + row_bytes)?
                .copy_from_slice(scratch.get(src..src + row_bytes)?);
        }
        return Some((out, fmt));
    }
    let out_row = (w as usize).checked_mul(RGBA8_BPP as usize)?;
    let rows = (h as usize).checked_mul(planes as usize)?;
    let out_len = out_row.checked_mul(rows)?;
    let trow = tight as usize;
    let mut out = vec![0u8; out_len];
    // This rung carries nearly all of the pathway's sampled traffic, so a
    // narrowing taken here is the one most likely to be the narrowing that
    // matters — and until this line existed the rung reported none at all while
    // the general loader reported its own. Same key as the others, so a format
    // narrowed on both rails is two lines and not one.
    crate::runtime::draw::note_sampled_narrowing("linear_memo_narrowed", 0, sample_fmt, w, h);
    for row_index in 0..rows {
        let src = row_index.checked_mul(bpr)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_fmt,
            scratch.get(src..src + trow)?,
            w,
            &mut out[row_index * out_row..],
        ) {
            return None;
        }
    }
    Some((out, TexelLayout::Rgba8))
}

/// Which native sampled layouts the CPU byte rails may hand the engine for this
/// guest format on this host.
///
/// [`NativeUploads`] is a parameter of the loaders and not a constant because
/// the answer has a capability half that `runtime/draw/texture_view.rs` cannot
/// ask: an image is created at the layout's own `VkFormat`, and a host that
/// cannot linearly filter that format would sample it through a sampler that
/// asks for filtering anyway. This is the one place that asks, so the two
/// halves of the answer are decided together.
///
/// `Bgra8` is unconditional: `B8G8R8A8_UNORM` carries
/// `SAMPLED_IMAGE_FILTER_LINEAR` on every Vulkan implementation by mandate, and
/// the rail that first took it argues the same.
///
/// **Keyed on the format so the common one never asks an irrelevant question.**
/// The host capability is a lock-free device-lifetime snapshot, but this sits
/// on the rung carrying essentially all of the pathway's sampled traffic and
/// only two guest formats can make use of the answer.
/// [`pixel_format::narrows_to_unorm8`] is exactly the set of formats whose CPU
/// arm is lossy, which is exactly the set the half-float flag can change the
/// answer for — so keying on it is the same rule stated once, not a fast path
/// that could disagree with the slow one.
fn native_uploads_for(sample_format: u16) -> NativeUploads {
    if !pixel_format::narrows_to_unorm8(sample_format) {
        return NativeUploads::BGRA8;
    }
    native_uploads_asking_host()
}

/// The same answer for a caller that does not yet know the guest format.
///
/// The last-resort sampled rung resolves the format inside the loader, so it
/// cannot key the question the way the hot rung does — and it does not need to:
/// `settle_linear_texture_sampled` read **0** across a four-rail sweep, because
/// every rung above it serves. A lock on a path that does not run is not a cost.
fn native_uploads_asking_host() -> NativeUploads {
    use crate::backend::vulkan::engine;
    NativeUploads {
        // One flag for both half-float layouts, so the answer is the
        // conjunction: a host that filters one and not the other keeps neither
        // on the native rail. Nothing on record separates them — both carry
        // `SAMPLED_IMAGE_FILTER_LINEAR` by mandate — and a per-layout flag
        // would be two fields nobody could point at a host that needed them.
        float16: engine::supports_sampled_layout_linear_filter(TexelLayout::Rgba16Float)
            && engine::supports_sampled_layout_linear_filter(TexelLayout::Rg16Float),
        ..NativeUploads::BGRA8
    }
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
/// - The staleness check is exact, not sampled. The full
///   `bpr * h * depth_planes` native span
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
    texture_ref: u32,
    tex: &TextureDescriptor,
    gva: u64,
    w: u32,
    h: u32,
) -> Option<(
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    SampledByteFormat,
)> {
    let declared_format = tex.declared_pixel_format()?;
    let sample_fmt = effective_view_sample_format(declared_format, None)?;
    let (_, layout) = tex.level_gva(0, state.page_shift)?;
    let bpr = layout.row_stride;
    let planes = layout.planes();
    let tight = pixel_format::tight_row_bytes(w, declared_format)? as u64;
    // Padded strides ride the same memo now — the native read below covers the
    // full `bpr*h*planes` span (padding included, so a write anywhere is
    // observed) and
    // `native_scratch_to_upload` gathers the tight rows. A sub-tight stride
    // (impossible geometry) or a zero dimension declines to the fallback.
    if bpr < tight || w == 0 || h == 0 {
        return None;
    }
    // `bpr*h*planes` and not `TextureLevelLayout::slice_read_span`, which every
    // reader that walks only the tight rows uses instead. This one really does
    // read the last row's padding — that is what makes the memo's byte-for-byte
    // compare able to notice a guest write into it — so it is charged for what
    // it touches.
    //
    // The consequence is a third way to decline, and the one most worth knowing:
    // an image the guest sized to `offset + read_span` exactly is refused here
    // and served by the general loader below, which uses the tighter rule. That
    // is a slower path, not lost work.
    let span = bpr.checked_mul(h as u64)?.checked_mul(u64::from(planes))?;
    let native_len = host_alloc_len(span)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Same coherence rule as the general loader: land any resident-authoritative
    // writeback *aliasing the sampled span* before reading it — and only then.
    //
    // This is the device's largest single wait, 11.5 s across a driven
    // Safari-drag boot, and almost none of it was owed: a writeback lands in one
    // surface's pages while this reader is usually somewhere else entirely. The
    // walk below runs only when something is outstanding, so the binds that
    // dominate this rail — the ones with a clear debt flag — still pay one
    // atomic load.
    //
    // A short walk is `None` and settles. `pages_spanned` is the count the
    // resolver would have produced with nothing dropped, and a dropped page is
    // one this reader cannot rule out.
    // Census, pay, settle — the whole obligation of a CPU read of one named
    // resource's guest bytes. See `writeback_debt::settle_for_texture`.
    crate::runtime::writeback_debt::settle_for_texture(
        state,
        host,
        task_id,
        texture_ref,
        gva,
        span,
        crate::runtime::render_writeback::SettleSite::LinearMemoRead,
    );
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
    let key = (task_id, gva, w, h, planes, sample_fmt);
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
            Some((m.rgba.clone(), m.generation, m.layout))
        }
        Some(_) => {
            crate::runtime::drain::note_store_route("lin_memo_changed");
            None
        }
    };
    if let Some((rgba, generation, fmt)) = hit {
        state.guest_linear_scratch = scratch;
        return Some((
            rgba,
            Some(LinearSampleIdentity {
                key: gva,
                generation,
            }),
            // The memo stores the layout it converted to; the transfer function
            // is `sample_fmt`'s and is re-derived on hit and miss alike, so a
            // retained entry cannot carry a stale one.
            SampledByteFormat::from_source(fmt, sample_fmt),
        ));
    }
    // First sight or native bytes changed: convert fresh, new generation.
    let Some((rgba, fmt)) =
        native_scratch_to_upload(&scratch, w, h, planes, bpr, sample_fmt, tight)
    else {
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
            layout: fmt,
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
        SampledByteFormat::from_source(fmt, sample_fmt),
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
/// A second driven boot on the same workload read 28 / 28 / 0 and a third
/// 32 / 32 / 0 — the same partition three times, so the zero is not one boot's
/// luck. The identity to check is `with_host_entry == host_agrees +
/// host_content`; it is what catches a miscount before the zero is believed.
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
/// [`crate::runtime::draw::seed_color_load`]'s stated rule, "exact target
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
    byte_format: SampledByteFormat,
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
    // `draw::seed_color_load`), neither of which consults the write
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
        load_linear_guest_memoized(state, host, task_id, texture_ref, tex, gva, w, h)
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
        // The other half of a channel-order reading. A GVA span is *written* by
        // the render Store at the attachment's declared format and *read* here
        // at the sampled descriptor's, and on the copying rail those are two
        // interpretations of one buffer rather than one typed image — so the
        // pair has to be joinable. `gva_flush_gpu_declined` names the write's
        // format against the same `gva=`; this names the read's. Latched per
        // (gva, format) so a steady bind stays at one line and a *change* of
        // interpretation still surfaces.
        if crate::observe::first_sight(
            "lin_serve_fmt",
            gva ^ (u64::from(tex.pixel_format) << 48),
        ) {
            crate::observe::off(format!(
                "lin_serve_fmt task={task_id} ref={texture_ref} gva={gva:#x} {w}x{h} \
                 fmt={:#x} bytes={byte_format:?}",
                tex.pixel_format
            ));
        }
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

/// Guest pages the Vulkan draw's eager GVA fallback may write.
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
) -> Option<super::StoreTargetPages> {
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
/// type11_seed_provided      242      t11elide_le_256x256        4
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
/// all of the composite seed traffic, and `type11_seed_provided` at 242 against
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
/// measured `type11_seed_elided` 283 against `type11_seed_provided` 23, so the
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
fn note_draw_coverage(
    scissor: ScissorRect,
    target_w: u32,
    target_h: u32,
    load_action: Option<u16>,
    seeded: bool,
    from_target: bool,
) {
    let covers = scissor.covers(target_w, target_h);
    crate::runtime::drain::note_store_route(if covers {
        "draw_scissor_full"
    } else {
        "draw_scissor_partial"
    });
    // Into the union before the early return, and clamped to the target: a
    // full-coverage draw is exactly the case that makes a pass's union total,
    // so leaving it out would measure only the passes that were already cheap.
    note_pass_scissor_rect(ScissorRect {
        x: scissor.x.min(target_w),
        y: scissor.y.min(target_h),
        width: scissor.width.min(target_w),
        height: scissor.height.min(target_h),
    });
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
        Some(MTL_LOAD_ACTION_LOAD) if from_target => "draw_partial_load_from_target",
        Some(MTL_LOAD_ACTION_LOAD) if seeded => "draw_partial_load_seeded",
        Some(MTL_LOAD_ACTION_LOAD) => "draw_partial_load_unseeded",
        Some(MTL_LOAD_ACTION_CLEAR) => "draw_partial_clear",
        Some(MTL_LOAD_ACTION_DONT_CARE) => "draw_partial_dontcare",
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
    let area = (scissor.width as u64).saturating_mul(scissor.height as u64);
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
fn note_pass_scissor_rect(rect: ScissorRect) {
    use std::sync::atomic::Ordering;
    let (x0, y0) = (rect.x.min(u16::MAX as u32), rect.y.min(u16::MAX as u32));
    let x1 = rect.x.saturating_add(rect.width).min(u16::MAX as u32);
    let y1 = rect.y.saturating_add(rect.height).min(u16::MAX as u32);
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
    c0: &crate::runtime::draw::ColorRtRequest,
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
    guest_holds_bytes: bool,
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
            task_id,
            texture_ref,
            width,
            height,
            bgra.clone(),
            gva,
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
            guest_holds_bytes,
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
    /// resident with `skip_readback`: the caller lands the frame from that
    /// resident, which is where the pixels are — this record produced none on
    /// the host.
    ///
    /// `identity` is the key the draw registered, carried rather than re-derived
    /// — see [`Self::ResidentSurfaceStore`] for what a second derivation costs.
    ResidentGvaStore {
        identity: crate::backend::vulkan::engine::TargetIdentity,
    },
    /// Type-11 composite Store executed into its registry resident with
    /// `skip_readback`: the caller copies that image into the mapping's guest
    /// pages through [`crate::runtime::render_writeback`], which never brings
    /// the frame across host memory.
    ///
    /// Distinct from [`Self::ResidentGvaStore`] because the destination is a
    /// mapping rather than a raw task GVA, and the two reach guest memory by
    /// different routes. Distinct from [`Self::Pixels`] because there are no
    /// pixels — a caller that treated an empty frame as one would write a blank
    /// framebuffer into guest memory.
    ///
    /// # Why the identity travels with the span
    ///
    /// `identity` is the exact key this record handed `registry_ensure`, so the
    /// image the Store reads is the image the draw rendered into by
    /// construction. The Store used to call `type11_store_identity` a second
    /// time instead, on the grounds that it is the same function that produced
    /// `DrawRequest::target_identity` — but it is not the same *value*. That
    /// identity carries `MappingEntry::map_generation`, the draw mutates
    /// `DeviceState` between the two calls, and any writer that revalidates the
    /// mapping advances the generation. The Store then asked the registry for a
    /// key one generation ahead of the one the draw registered, which is
    /// `read_target_unknown_identity diverges=generation asked_gen=N
    /// held_gen=N-1` — the whole Maps frame lost, on the only render-target
    /// rail a host without `VK_EXT_external_memory_host` has.
    ResidentSurfaceStore {
        identity: crate::backend::vulkan::engine::TargetIdentity,
        guest_store: GuestStoreStatus,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct GuestStoreStatus {
    guest_backed: bool,
    recorded: bool,
    footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
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
/// The first-appearance line answers "is this route reachable" and cannot answer
/// "how often". Both questions are live: reachability is what says a route the
/// census reads zero for was offered at all, and the rate is what prices the
/// route — `engine_delta`
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
/// list. `Ok(empty)` ⇒ the guest declared a single render target and this is the
/// classic single-RT path. A fragment shader that writes `location` 1.. has
/// those outputs rendered rather than discarded; each secondary persists as a
/// registry resident keyed by its protocol identity, exactly as slot 0 does.
///
/// Strict by construction — any ambiguity is an `Err` and the caller refuses the
/// draw, rather than a guessed attachment: requires a resident primary,
/// contiguous slots (0,1,2,… matching the shader's `location`s), matching
/// framebuffer geometry, a known color-renderable format, and a resolvable
/// identity.
///
/// # Why `Err` and not an empty vector
///
/// The empty vector used to mean both "the guest asked for one target" and
/// "the guest asked for several and this device could not build them", and the
/// caller could not tell those apart — so it took the single-RT path for both
/// and **executed the draw**. A guest that asks for N render targets then gets
/// 1, with no error anywhere it can see: the shader's `location` 1.. outputs
/// are discarded and a later pass sampling that attachment reads whatever was
/// in those pages before. Refusing is what a GPU does with a render pass it
/// cannot build.
///
/// The Metal arm is what settled the question rather than an argument about
/// what Vulkan ought to do: `backend::metal::render` attaches every entry of
/// this same colour list at its own slot number and has never degraded, so the
/// two arms disagreed about one wire form and only one of them was silent.
#[allow(
    clippy::too_many_arguments,
    reason = "every argument is a distinct wire-derived input to the attachment set"
)]
pub(super) fn build_secondary_targets<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    colors: &[ColorRtRequest],
    pipeline: &crate::runtime::decode::resource::RenderPipelineDescriptor,
    primary: &crate::backend::vulkan::engine::TargetIdentity,
    fb_w: u32,
    fb_h: u32,
    blend_constants: [f32; 4],
) -> Result<
    Vec<crate::backend::vulkan::engine::SecondaryColorTarget>,
    crate::runtime::census::present_proxy::SecondaryMrtRefusal,
> {
    use crate::backend::vulkan::engine::{SecondaryColorTarget, TargetIdentity};
    use crate::runtime::census::present_proxy::SecondaryMrtRefusal;
    if colors.len() <= 1 {
        return Ok(Vec::new());
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
            return Err(SecondaryMrtRefusal {
                slot: c.slot,
                reason: crate::runtime::census::present_proxy::MrtDrop::NonContiguousSlot,
            });
        }
        // MRT requires every attachment to share the framebuffer geometry.
        if c.width != fb_w || c.height != fb_h {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::GeometryMismatch,
                c.width,
                c.height,
            );
            return Err(SecondaryMrtRefusal {
                slot: c.slot,
                reason: crate::runtime::census::present_proxy::MrtDrop::GeometryMismatch,
            });
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
                return Err(SecondaryMrtRefusal {
                    slot: c.slot,
                    reason: crate::runtime::census::present_proxy::MrtDrop::UnknownFormat,
                });
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
                generation: if c.texture_ref != 0 {
                    crate::runtime::writeback_debt::gva_resource_generation(
                        state,
                        host,
                        crate::runtime::writeback_debt::GvaResourceKey {
                            task_id,
                            texture_ref: c.texture_ref,
                        },
                        c.target_gva,
                        u64::from(c.row_stride).saturating_mul(u64::from(c.height)),
                    )
                } else {
                    gva_span_alloc_generation(
                        state,
                        host,
                        task_id,
                        c.target_gva,
                        c.row_stride,
                        c.height,
                    )
                },
                // The format this attachment's image is actually created with,
                // not a re-derivation of it. `registry_ensure_attachment` takes
                // `format` — resolved just above by `color_attachment` — so
                // answering the key from anything else lets the identity claim
                // one format while the image holds another. It did: a
                // `R16G16_SFLOAT` secondary is admitted by `color_attachment`
                // and got an identity saying eight-bit RGBA.
                format,
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
            return Err(SecondaryMrtRefusal {
                slot: c.slot,
                reason: crate::runtime::census::present_proxy::MrtDrop::NoIdentity,
            });
        };
        // A secondary aliasing the primary target is a degenerate feedback loop
        // the engine rejects, and a pass that reads and writes one image through
        // two attachments has no correct rendering — so the draw is refused
        // rather than run with the alias quietly removed.
        //
        // `aliases` and not `==`: the destination is the conflict, not the
        // registry slot. Two attachments over one guest span at two formats are
        // two images, so `==` says no and the span is still written twice.
        //
        // The rule is pairwise over the whole attachment set rather than a test
        // against slot 0. Two *secondaries* over one destination write it twice
        // in one pass for exactly the reason the primary case does, and this
        // named `primary` alone — so slots 1 and 2 over one span were admitted,
        // built into a framebuffer and drawn.
        let aliases_a_sibling = out
            .iter()
            .any(|s: &SecondaryColorTarget| identity.aliases(&s.identity));
        if identity.aliases(primary) || aliases_a_sibling {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::AliasesPrimary,
                c.width,
                c.height,
            );
            return Err(SecondaryMrtRefusal {
                slot: c.slot,
                reason: crate::runtime::census::present_proxy::MrtDrop::AliasesPrimary,
            });
        }
        // This slot's declared load action, carried rather than narrowed. The
        // census is the primary's, banded, and it is the instrument this arm
        // never had: slot 0 has reported its three declarations for as long as
        // `LoadAction` has existed, and the secondaries reported nothing, so a
        // boot could not say whether this arm ever saw a DontCare at all.
        let load = crate::contract::pass_action::LoadAction::from_declared(c.load_action);
        let (declared_n, declared_area) =
            load.census_routes(crate::contract::pass_action::AttachmentBand::Color1Plus);
        crate::runtime::drain::note_store_route(declared_n);
        crate::runtime::drain::note_store_route_n(
            declared_area,
            u64::from(c.width).saturating_mul(u64::from(c.height)),
        );
        // Read whatever the action is; the engine consults it only under
        // `Clear`, which is Metal's own rule for `clearColor`. Reading it here
        // unconditionally is what a plain struct field means -- the narrowing
        // that mattered was making the *action* two-valued, which sent DontCare
        // into the arm that applies this colour.
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
                match translate::blend::state(a, blend_constants) {
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
    Ok(out)
}

/// Translate one decoded vertex attribute's Metal format to the engine's.
///
/// The decline names the attribute that could not be translated — location and
/// buffer index, plus the raw ordinal — because a draw refused for an
/// untranslatable format is otherwise indistinguishable from one refused for the
/// attribute next to it.
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
    translate::vertex::step_function(attribute.declared_step_function).map_err(|reason| {
        DrawPreparationDecline::VertexStepFunctionUnsupported {
            location: attribute.location,
            buffer_index: attribute.buffer_index,
            reason,
        }
    })
}

/// Discharge [`DrawEncodeRequest::gva_load_from_resident`]: either chain colour0's
/// LOAD off the engine resident, or put the CPU seed back.
///
/// `mrt_draw_request` skipped the seed because the engine still held what the
/// render Store published into these guest pages, so `colors[0].target_seed_rgba`
/// arrives `None` while the attachment still says `MTL_LOAD_ACTION_LOAD`. **That
/// makes producing content here an obligation, not an optimisation**: an encode
/// that neither chains nor re-seeds hands the pass an undefined attachment, and
/// this class renders a *plausible* frame when it is wrong.
///
/// Returns the identity to chain from, and sets `chain_load_from_target`, when the
/// resident is ready; otherwise re-reads the seed and returns `None`. Exactly one
/// of `gvaseed_chained` / `gvaseed_reseeded` is counted per discharged request.
///
/// A cross-pass rung of this shape used to sit inline at the call site and was
/// deleted for reading zero — `xpass_c0_gva_load_window_open` 0 against
/// `xpass_c0_not_gva_load` 4859/5616 over two driven boots — on the reading that
/// no draw arriving there has a colour0 that is a seedless LOAD'd GVA target. That
/// was true and it was circular: the seed was built eagerly upstream, so nothing
/// could arrive seedless however the guest behaved. Asked at the production site
/// instead, the same question answers `gvaseed_elided` 4 849 against 13 refusals.
///
/// **The re-seed is not a formality.** `req.gva_alloc_gen` is recomputed after the
/// request is built, so a page set that moved in between names a different target
/// and the resident under it is not ready. `gvaseed_reseeded` says how often the
/// race is real — zero on the boots measured so far, which is why it is a function
/// with a test rather than a branch trusted to a workload that never took it.
///
/// This is a function rather than a block because it is the only seam a test can
/// reach: `try_metal2vulkan_draw` loads a guest pipeline and translates two shader
/// blobs before it gets here, so nothing can drive the arm end to end without a
/// GPU. See `a_gva_load_from_resident_draw_with_no_resident_puts_the_seed_back`.
pub(super) fn honour_gva_load_elision<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    chain_load_from_target: &mut bool,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    if !req.gva_load_from_resident || *chain_load_from_target {
        return None;
    }
    let ready = gva_chain_identity(req).filter(|identity| {
        let texture_ref = req.colors.first().map(|c0| c0.texture_ref).unwrap_or(0);
        gva_resident_ready(state, req.task_id, texture_ref, identity)
    });
    match ready {
        Some(identity) => {
            crate::runtime::drain::note_store_route("gvaseed_chained");
            *chain_load_from_target = true;
            Some(identity)
        }
        None => {
            crate::runtime::drain::note_store_route("gvaseed_reseeded");
            // `first()`, not `[0]`. This arm is reached whenever
            // `gva_chain_identity` declined, and one of the things it declines
            // for is an empty colour list — so indexing here would turn a
            // decline into a panic on the drain worker.
            let c0 = req.colors.first()?;
            let (tex_ref, gva, cw, ch) = (c0.texture_ref, c0.target_gva, c0.width, c0.height);
            let seed = crate::runtime::draw::seed_color_load(
                state,
                host,
                req.task_id,
                tex_ref,
                gva,
                cw,
                ch,
            );
            if seed.is_none() {
                crate::observe::fail(format!(
                    "gvaseed_reseed_miss ref={tex_ref} {cw}x{ch} gva={gva:#x} \
                     (the elision was decided on a page set that has since \
                     moved, and the re-read found nothing: this pass loads \
                     an undefined attachment)"
                ));
            }
            if let Some(c0) = req.colors.first_mut() {
                c0.target_seed_rgba = seed;
            }
            None
        }
    }
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
    // Before anything is resolved or uploaded: a bind naming a slot past its
    // argument table refuses the draw, once, for all three classes and both
    // stages. Every consumer below therefore takes the slot as in-range and
    // spells no bound of its own.
    if let Some(bind) = crate::runtime::draw::first_bind_past_table(req) {
        return Err(DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::BindSlotPastTable {
                pipeline_ref: req.pipeline_ref,
                bind,
            },
        ));
    }
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::PipelineGen);
    // Name the color0 GVA target's allocation before anything can render into
    // it, and once: the pinned Store identity, the cross-pass Load identity and
    // the deferred window's stored copy are all keyed on this value, and two
    // walks of one address across a submit are two answers.
    req.gva_alloc_gen = gva_alloc_generation(state, host, req);
    // One call for the pipeline descriptor and both translated shaders. It is
    // memoized on the three objects' list entries — see
    // `crate::runtime::pipeline_resolve` for what that identity is and what it
    // does not cover — so the object-list walks, descriptor reads, MTLB reads,
    // AIR carves and content hashes behind it happen once per pipeline object
    // rather than once per draw. The sub-phases below still bracket the parts,
    // so a boot's `chain_phase` line says how much of the span survived.
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::PipelineDesc);
    let resolved =
        crate::runtime::pipeline_resolve::resolve(state, host, req.task_id, req.pipeline_ref)
            .map_err(DrawError::DrawPreparation)?;
    let pd = &resolved.desc;
    let v_shader = resolved.vertex.clone();
    let f_shader = resolved.fragment.clone();

    // Request construction obtains the attachment count from this immutable
    // pipeline before LOAD/CLEAR and Store policy can inspect it. Keep the
    // equality explicit at the backend boundary: reaching here with a different
    // count is an internal contract violation, not a shape Vulkan may repair.
    let pipeline_sample_count = pd.raster_sample_count.max(1);
    if pipeline_sample_count > 1 {
        let color = req.colors.first();
        let key = (u64::from(req.pipeline_ref) << 32)
            | u64::from(color.map_or(0, |color| color.texture_ref));
        if crate::observe::first_sight("render_multisample_contract", key) {
            crate::observe::off(format!(
                "render_multisample_contract task={} pipe={} raster_samples={} \
                 colors={} color_ref={} source_ref={} mid={} gva={:#x} {}x{} \
                 fmt={:#x} load={} store={} depth_ref={}",
                req.task_id,
                req.pipeline_ref,
                pipeline_sample_count,
                req.colors.len(),
                color.map_or(0, |color| color.texture_ref),
                color.map_or(0, |color| color.multisample_source_ref),
                color.map_or(0, |color| color.mapping_id),
                color.map_or(0, |color| color.target_gva),
                color.map_or(0, |color| color.width),
                color.map_or(0, |color| color.height),
                color.map_or(0, |color| color.format),
                color.map_or(0, |color| color.load_action),
                color.map_or(0, |color| color.store_action),
                req.depth_attach.as_ref().map_or(0, |depth| depth.texture_ref),
            ));
        }
    }
    if let Some(color) = req
        .colors
        .iter()
        .find(|color| color.sample_count != pipeline_sample_count)
    {
        return Err(DrawError::Unsupported(
            crate::backend::vulkan::engine::reason::DrawReason::MultisampleAttachmentSampleCountMismatch {
                attachment: color.sample_count,
                raster: pipeline_sample_count,
            },
        ));
    }

    for (stage, shader, textures) in [
        ("vertex", &v_shader, &req.vertex_textures),
        ("fragment", &f_shader, &req.fragment_textures),
    ] {
        if let Some((index, descriptor)) =
            crate::runtime::spirv_bind::first_non_sampled_texture_descriptor(&shader.reflection)
        {
            let access = match descriptor.access {
                crate::runtime::spirv_bind::ReflectedTextureAccess::Storage => "storage",
                crate::runtime::spirv_bind::ReflectedTextureAccess::Unknown => "unknown",
                crate::runtime::spirv_bind::ReflectedTextureAccess::Sampled => continue,
            };
            return Err(DrawError::DrawPreparation(
                DrawPreparationDecline::TextureAccessUnsupported {
                    stage,
                    index,
                    texture_ref: textures
                        .iter()
                        .find(|texture| texture.index == index)
                        .map(|texture| texture.texture_ref)
                        .unwrap_or(0),
                    binding: descriptor.binding,
                    access,
                },
            ));
        }
    }
    for (stage, expected_stage, shader) in [
        (
            "vertex",
            metal2vulkan::reflect::ShaderStage::Vertex,
            &v_shader,
        ),
        (
            "fragment",
            metal2vulkan::reflect::ShaderStage::Fragment,
            &f_shader,
        ),
    ] {
        if let Some(unsupported) = crate::runtime::spirv_bind::first_unsupported_vulkan_interface(
            &shader.reflection,
            expected_stage,
        ) {
            return Err(DrawError::DrawPreparation(
                DrawPreparationDecline::ReflectedInterfaceUnsupported {
                    stage,
                    feature: unsupported.feature,
                    count: unsupported.count,
                },
            ));
        }
        if let Some(resource) =
            crate::runtime::spirv_bind::first_unsupported_vulkan_resource(&shader.reflection)
        {
            let kind =
                crate::runtime::spirv_bind::unsupported_vulkan_resource_kind_name(resource.kind)
                    .expect("helper returned an unsupported Vulkan resource");
            return Err(DrawError::DrawPreparation(
                DrawPreparationDecline::ReflectedResourceUnsupported {
                    stage,
                    index: resource.metal_index,
                    binding: resource.descriptor.map(|descriptor| descriptor.binding),
                    kind,
                },
            ));
        }
    }

    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Pipeline);
    let Some((w, h)) = req.colors.first().map(|c0| (c0.width, c0.height)) else {
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

    // The two modules in the numbering this draw will use, from the translation
    // cache. Each carries the walks of its own numbering beside it — see
    // `m2v_cache::ShaderVariant` — so nothing here re-walks a module per draw.
    // A vertex module never relocates, so it is always the base variant.
    let v_variant = v_shader.variant(false, false);
    let v_words = v_variant.words.clone();
    #[allow(unused_mut)]
    let mut f_variant = f_shader.variant(false, false);

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
        //
        // Which indices those are, and which the attribute list names at all,
        // are both functions of the pipeline's attribute list and nothing else,
        // so they are resolved with the pipeline rather than rebuilt per draw —
        // see [`crate::runtime::pipeline_resolve::VertexBindPlan`], which also
        // carries why the second set is deliberately unfiltered. Not to be
        // confused with `stage_in_bufs` further down: that one is filled during
        // the attribute walk, holds only the indices that actually carried
        // bytes, and decides storage binding.
        let bind_plan = &resolved.bind_plan;
        let render_buffer_bounds = crate::runtime::spirv_bind::RenderBufferIndexBounds::new(
            req.first_vertex,
            req.vertex_count,
            req.base_instance,
            req.instance_count,
            req.indexed.is_some(),
        );
        let mut vtx_storage: Vec<(u32, crate::backend::vulkan::engine::BufferContent)> = Vec::new();
        // The three `bind_phase` spans below divide `chain_phase`'s `binds_us`,
        // which is this draw path's largest column and covered three costs with
        // one number. Each is a lexical scope so an early `return Err` charges
        // the span it left from rather than losing the time.
        let vertex_span =
            crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::VertexLoad);
        for b in req.vertex_buffers.iter() {
            if b.buffer_ref == 0 {
                continue;
            }
            let allow_zc = !bind_plan.is_constant_step(b.index);
            // A vertex buffer is read twice on this path — as the declared
            // argument reflection describes, and as the byte source for every
            // stage-in attribute naming this index, which it does not. Only the
            // first is what `Unused` is about, so an index the pipeline's
            // attribute list names keeps its guest bytes whatever reflection
            // says about the argument.
            let feeds_stage_in = bind_plan.feeds_stage_in(b.index);
            // The vertex shader's own reflection bounds its own `[[buffer(n)]]`
            // binds, and a stage-in index is excluded — see that function's doc
            // for why the exclusion is not implied by the translator's output.
            let cap = crate::runtime::spirv_bind::vertex_buffer_extent(
                &v_shader.reflection,
                b.index,
                feeds_stage_in,
                render_buffer_bounds,
            );
            let access =
                crate::runtime::spirv_bind::reflected_buffer_access(&v_shader.reflection, b.index);
            crate::runtime::bind_phase::note_access(access);
            let content = if crate::runtime::spirv_bind::may_serve_neutral(access, feeds_stage_in) {
                crate::runtime::bind_phase::note_neutral_served();
                crate::backend::vulkan::engine::BufferContent::Bytes(
                    crate::runtime::spirv_bind::neutral_bind_bytes(),
                )
            } else {
                crate::runtime::bind_phase::note_unused_staged(access);
                let Some(content) = load_buffer_content(
                    state,
                    host,
                    req.task_id,
                    b.buffer_ref,
                    b.resource.as_deref(),
                    b.offset,
                    allow_zc,
                    cap,
                ) else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::VertexBufferMissing {
                            index: b.index,
                            buffer_ref: b.buffer_ref,
                            offset: b.offset,
                        },
                    ));
                };
                content
            };
            vtx_storage.push((b.index, content));
        }
        drop(vertex_span);
        let mut frag_storage: Vec<(u32, crate::backend::vulkan::engine::BufferContent)> =
            Vec::new();
        let fragment_span =
            crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::FragmentLoad);
        for b in req.fragment_buffers.iter() {
            if b.buffer_ref == 0 {
                continue;
            }
            // The fragment shader's reflection, for the same reason. The two
            // stages are looked up separately because one Metal buffer index
            // names a different argument in each, and a cap taken from the
            // wrong stage would bound a bind the other stage never declared.
            let cap = crate::runtime::spirv_bind::reflected_render_buffer_extent(
                &f_shader.reflection,
                b.index,
                render_buffer_bounds,
            );
            let access =
                crate::runtime::spirv_bind::reflected_buffer_access(&f_shader.reflection, b.index);
            crate::runtime::bind_phase::note_access(access);
            // No stage-in exclusion here: `[[stage_in]]` is a vertex-stage
            // concept and `pd.vertex_attributes` names vertex buffer indices,
            // which are a different index space from the fragment stage's.
            let content = if crate::runtime::spirv_bind::may_serve_neutral(access, false) {
                crate::runtime::bind_phase::note_neutral_served();
                crate::backend::vulkan::engine::BufferContent::Bytes(
                    crate::runtime::spirv_bind::neutral_bind_bytes(),
                )
            } else {
                crate::runtime::bind_phase::note_unused_staged(access);
                let Some(content) = load_buffer_content(
                    state,
                    host,
                    req.task_id,
                    b.buffer_ref,
                    b.resource.as_deref(),
                    b.offset,
                    true,
                    cap,
                ) else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::FragmentBufferMissing {
                            index: b.index,
                            buffer_ref: b.buffer_ref,
                            offset: b.offset,
                        },
                    ));
                };
                content
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
            // `setVertexBuffer:offset:attributeStride:atIndex:` overrides what
            // the pipeline's `MTLVertexBufferLayoutDescriptor` declared for this
            // buffer index, so it is resolved before the stride is read — a
            // pipeline built for a dynamic stride declares one this device
            // cannot use, and the guard below would drop the attribute for it.
            //
            // On this backend the stride reaches the pipeline through
            // `AttrKey::stride`, which is already part of the key: Vulkan's
            // per-binding stride is `VkVertexInputBindingDescription::stride`
            // and is not dynamic below `vkCmdBindVertexBuffers2`, core in 1.3
            // against this device's 1.2 floor. So two draws sharing shaders and
            // differing only in a guest-supplied stride already get their own
            // pipelines, with no change to the key.
            let stride =
                super::bind_attribute_stride(&req.vertex_buffers, a.buffer_index, a.stride);
            if a.format == 0 || stride == 0 {
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
                        stride,
                    },
                ));
            }
            let step = prepare_vertex_step_function(a).map_err(DrawError::DrawPreparation)?;
            let step_rate = a.step_rate();
            attrs.push(crate::backend::vulkan::engine::VertexAttributeResource {
                location: a.location,
                // One Vulkan binding per location (archive render_draw_core).
                binding: a.location,
                format,
                offset: a.offset,
                stride,
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
        // The sampled band holds textures **and** samplers, and the relocation is
        // what stops one stage's binding number landing on the other's. So the
        // trigger asks about the whole band: a stage with a sampler and no
        // texture is still occupying binding numbers in it.
        //
        // Textures alone is what stood here, and Metal argument tables are sticky
        // across draws in an encoder — a vertex sampler survives a re-bind that
        // zeroed the vertex textures. With `has_vtx_tex` false the two stages'
        // sampler at index 0 both resolve to `SAMPLER_BINDING_BASE + 0`, and
        // `push_smp`'s `sampler_binds.insert` is first-writer-wins: the vertex
        // sampler takes the binding and the fragment one is *dropped*. The layout
        // gives every sampler `VERTEX | FRAGMENT` stage flags, so the fragment
        // module then samples its textures through the vertex stage's filter,
        // address mode and LOD clamp, with nothing refused and nothing counted.
        let has_vtx_sampled = stage_uses_sampled_band(&req.vertex_textures, &req.vertex_samplers);
        let has_frag_sampled =
            stage_uses_sampled_band(&req.fragment_textures, &req.fragment_samplers);
        let reflected_sampled_collision =
            reflected_sampled_binding_collision(&v_shader.reflection, &f_shader.reflection);
        let separate_sampled =
            (has_vtx_sampled && has_frag_sampled) || buf_collide || reflected_sampled_collision;
        // Sampled relocation first (archive order), then buffer band. The
        // buffer band lands at [104,136), clear of the [96,104) ColorInput /
        // framebuffer-fetch band, which neither relocation touches. The
        // sampled-with-buffer coupling is kept so the engine's image/sampler
        // binding base mirrors one flag pair, not a third variant.
        if separate_sampled || buf_collide {
            f_variant = f_shader.variant(separate_sampled, buf_collide);
        }
        let f_words = f_variant.words.clone();

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
        // too whenever the adopted vertex reflection structurally declares a
        // buffer at that binding (never name-keyed).
        let mut storage: Vec<crate::backend::vulkan::engine::StorageBufferResource> = Vec::new();
        for (idx, content) in &vtx_storage {
            if !vertex_buffer_needs_storage_binding(
                &v_shader.reflection,
                *idx,
                stage_in_bufs.contains(idx),
            ) {
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
        // kinds only; ColorInput / ThreadgroupBuffer reach the shader by other
        // paths, while storage textures were declined before bind preparation.
        // Verified non-flooding: 0 fires across a full x86 boot (desktop convergence
        // + Safari + CSS gradients + a 23-binding compositor shader), so any fire is
        // a genuine bind gap, not expected control flow.
        // The guard below reports; its value is the population the repair after
        // the texture loop acts on. Empty on every draw that binds what it
        // samples, which is the hot path.
        let frag_unbound_used_textures: Vec<u32> = {
            // Membership predicates over the (tiny) provided-resource slices — the
            // scan allocates nothing on the all-bound hot path. Unsupported
            // reflected resource families have already been refused above.
            let unbound = frag_unbound_scan(
                &f_shader.reflection.bindings,
                |i| frag_storage.iter().any(|(x, _)| *x == i),
                |i| {
                    req.fragment_textures
                        .iter()
                        .any(|t| t.index == i && t.texture_ref != 0)
                },
                |i| {
                    req.fragment_samplers
                        .iter()
                        .any(|s| s.index == i && s.sampler_ref != 0)
                },
                // Same relocation the bind path applies, so the question is
                // asked of the binding the module would actually carry.
                |i| {
                    let base_off = if separate_sampled {
                        FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                    } else {
                        0
                    };
                    crate::runtime::spirv_bind::declares_descriptor(
                        &f_words,
                        TEXTURE_BINDING_BASE + i + base_off,
                    )
                },
            );
            // Declaration is not the bar the specification sets — a layout must
            // contain a descriptor for every resource the shader *statically
            // uses*, and a declared-and-never-referenced variable is legal to
            // omit. The scan above asks the weaker question because it is the
            // cheap one and it runs per draw; this asks the real one, only for
            // the handful the scan already flagged, and counts each answer so a
            // boot says which population these firings belong to.
            let uses: Vec<_> = unbound
                .iter()
                .map(|gap| {
                    (
                        *gap,
                        frag_unbound_static_use(gap, &f_words, separate_sampled),
                    )
                })
                .collect();
            for (_, use_) in &uses {
                crate::runtime::drain::note_store_route(use_.slug());
            }
            if !unbound.is_empty() {
                // Cold path only: build the provided-index sets for the log detail.
                let bufs: std::collections::BTreeSet<u32> =
                    frag_storage.iter().map(|(i, _)| *i).collect();
                let texs: std::collections::BTreeSet<u32> = req
                    .fragment_textures
                    .iter()
                    .filter(|t| t.texture_ref != 0)
                    .map(|t| t.index)
                    .collect();
                let smps: std::collections::BTreeSet<u32> = req
                    .fragment_samplers
                    .iter()
                    .filter(|s| s.sampler_ref != 0)
                    .map(|s| s.index)
                    .collect();
                // Each gap carries its own verdict, because a line that says
                // only "unbound=[tex0]" cannot be ranked: the same text is
                // written for a specification violation and for a variable the
                // module declares and never references.
                let detail = uses
                    .iter()
                    .map(|(gap, use_)| format!("{gap}:{}", use_.slug()))
                    .collect::<Vec<_>>()
                    .join(",");
                let violations = uses.iter().filter(|(_, u)| u.is_violation()).count();
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound reason=frag_declared_descriptor_unbound \
                     pipe={} unbound=[{detail}] violations={violations}/{} \
                     provided_buf={bufs:?} provided_tex={texs:?} \
                     provided_smp={smps:?} {}x{}",
                    req.pipeline_ref,
                    uses.len(),
                    w,
                    h
                ));
            }
            // Textures only, and violations only. A buffer or sampler gap is
            // reported above and not repaired here: the sampler class already
            // provisions its own default further down, and a storage buffer gap
            // has no neutral this device can invent. `DeclaredUnused` is legal
            // to omit and must stay omitted, or the census this block exists to
            // keep cannot tell the two populations apart.
            frag_unbound_textures_to_neutralize(&uses)
        };

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
        // The four `sampled_phase` spans below divide this phase's `sampled_us`,
        // the same way `bind_phase` divides `binds_us`. Counted here rather than
        // where a span opens, so a draw that samples nothing is still in the
        // denominator.
        crate::runtime::sampled_phase::note_sampled();
        // Sampled textures + samplers (metal2vulkan bands: textures 32+N, samplers 64+M).
        // Texture and sampler **indices are independent** (live logo SPIR-V: image
        // binding 35 = texture(3), sampler binding 64 = sampler(0)). Pairing
        // sampler to texture index left sampler 67 empty → black samples.
        // Fragment sampled resources use +FRAG_SAMPLED when either both stages
        // sample or fragment buffers moved into the sampled/static-sampler band.
        let mut images: Vec<crate::backend::vulkan::engine::SampledImageResource> = Vec::new();
        let mut samplers: Vec<crate::backend::vulkan::engine::SamplerResource> = Vec::new();
        let mut sampler_binds: std::collections::BTreeSet<u32> = Default::default();
        // Where each provisioned sampler's state came from, keyed by binding, for
        // the hang trail. A `SamplerResource` cannot be asked this after the
        // fact: a translated guest sampler that happens to be `Linear`/`Linear`
        // and one this device invented are the same value, and only one of them
        // is something the guest asked for. See
        // [`crate::runtime::gpu_hang_trail::SamplerNote`].
        let mut sampler_origin: std::collections::BTreeMap<u32, u8> = Default::default();
        {
            let mut push_tex = |index: u32,
                                texture_ref: u32,
                                retained: Option<&std::sync::Arc<crate::model::TaskResource>>,
                                frag_stage: bool|
             -> Result<(), DrawError> {
                if texture_ref == 0 {
                    return Ok(());
                }
                let base_off = if frag_stage && separate_sampled {
                    FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                } else {
                    0
                };
                let reflection = if frag_stage {
                    &f_shader.reflection
                } else {
                    &v_shader.reflection
                };
                let reflected_descriptor =
                    crate::runtime::spirv_bind::reflected_texture_descriptor(reflection, index);
                let img_bind = reflected_descriptor
                    .map(|descriptor| descriptor.binding)
                    .unwrap_or(TEXTURE_BINDING_BASE + index)
                    + base_off;
                if let Some(descriptor) = reflected_descriptor {
                    use crate::runtime::spirv_bind::ReflectedTextureAccess;
                    let unsupported = match descriptor.access {
                        ReflectedTextureAccess::Sampled => None,
                        ReflectedTextureAccess::Storage => Some("storage"),
                        ReflectedTextureAccess::Unknown => Some("unknown"),
                    };
                    if let Some(access) = unsupported {
                        return Err(DrawError::DrawPreparation(
                            DrawPreparationDecline::TextureAccessUnsupported {
                                stage: if frag_stage { "fragment" } else { "vertex" },
                                index,
                                texture_ref,
                                binding: img_bind,
                                access,
                            },
                        ));
                    }
                }
                // The two guest reads this bind needs before anything can decide
                // where its texels come from. `sampled_phase::Part::Lookup` is
                // this pair and nothing else, so the object-list walk is priced
                // against the resolve below rather than summed into it.
                let (texture_resource, view_swizzle) = {
                    let _s = crate::runtime::sampled_phase::Span::open(
                        crate::runtime::sampled_phase::Part::Lookup,
                    );
                    let texture_resource = retained.cloned().or_else(|| {
                        objects::resolve_resource(state, host, req.task_id, texture_ref).ok()
                    });
                    // A type-8 view's channel remap. Resolved here rather than in
                    // the loaders because it describes how the bind READS the
                    // texture, not what the texture contains: the engine hands it
                    // to the image view as a component mapping and the hardware
                    // applies it at sample time, so the texels stay untouched and
                    // the bind keeps whatever content rail it was already on.
                    let view_swizzle = texture_resource
                        .as_ref()
                        .filter(|resource| {
                            resource.entry.object_type
                                == crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VIEW
                        })
                        .and_then(|_| resolve_texture_view(state, host, req.task_id, texture_ref))
                        .and_then(|view| view.swizzle)
                        .filter(|plan| !pixel_format::swizzle_is_identity(plan));
                    (texture_resource, view_swizzle)
                };
                // Where the texels come from, which is the part with the cache
                // behind it. Scoped to a block so it closes at the resolve and
                // not at the end of the bind: everything after it — the
                // reflection read, the shape fold, the pushes — is deliberately
                // unbracketed, and a span held to the closure's end would
                // swallow it and make the four parts look like they summed.
                // A bind that declines from inside here charges its remainder
                // to `Resolve`, because the span commits on `Drop`.
                let (tw, th, loaded) = {
                    // The probe is charged to the alias part it belongs to, and
                    // the span is handed off to `ResolveSource` on the branch
                    // where the probe found nothing — so the two parts partition
                    // this scope rather than overlapping it.
                    let alias_span = crate::runtime::sampled_phase::Span::open(
                        crate::runtime::sampled_phase::Part::ResolveAlias,
                    );
                    let attachment_alias = frag_stage
                        .then(|| fragment_attachment_alias_sample(req, index, texture_ref))
                        .flatten();
                    if let Some((aw, ah, alias)) = attachment_alias {
                        match alias {
                            AttachmentAliasSample::Clear(clear) => (
                                aw,
                                ah,
                                SampledSourceRequest::Bytes(
                                    std::sync::Arc::new(solid_rgba8(aw, ah, &clear)),
                                    None,
                                    SampledByteFormat::synthesised(TexelLayout::Rgba8),
                                    crate::backend::vulkan::engine::SampledByteOrigin::AttachmentAlias,
                                ),
                            ),
                            AttachmentAliasSample::Seed(seed, stored) => (
                                aw,
                                ah,
                                SampledSourceRequest::Bytes(
                                    std::sync::Arc::new(seed.to_vec()),
                                    None,
                                    SampledByteFormat::from_source(TexelLayout::Rgba8, stored),
                                    crate::backend::vulkan::engine::SampledByteOrigin::AttachmentAlias,
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
                                (
                                    identity.width(),
                                    identity.height(),
                                    SampledSourceRequest::Target(
                                        identity.clone(),
                                        req.colors
                                            .iter()
                                            .find(|color| {
                                                color.slot == index
                                                    && color.texture_ref == texture_ref
                                            })
                                            .and_then(|color| {
                                                translate::pixel::color_attachment(color.format)
                                                    .ok()
                                                    .map(|resolved| resolved.0)
                                            })
                                            .unwrap_or_else(|| identity.resident_format()),
                                    ),
                                )
                            }
                        }
                    } else {
                        drop(alias_span);
                        let _s = crate::runtime::sampled_phase::Span::open(
                            crate::runtime::sampled_phase::Part::ResolveSource,
                        );
                        let Some(loaded) = resolve_sampled_source(
                            state,
                            host,
                            req.task_id,
                            texture_ref,
                            texture_resource.clone(),
                            view_swizzle.is_none(),
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
                    }
                };
                let mut bytes_identity = None;
                let mut byte_origin = crate::backend::vulkan::engine::SampledByteOrigin::Synthetic;
                // Byte layout of a CPU-origin bind. Default RGBA8; a source that
                // already holds its bytes in an uploadable order keeps them —
                // BGRA8 from the type-4 scanout cache, a native single/dual-channel
                // video plane — and the host spelling is applied once, where the
                // engine resource is built (`vk_texel_layout` below).
                let sampled_vk_format;
                // How the bound texels' channels sit on the host format, from
                // the rail that produced them. Identity for every CPU-origin
                // bind, because those loaders have already put the channels
                // where Metal presents them; non-identity only where a rail
                // handed the guest's own bytes over untouched.
                let mut sampled_components = pixel_format::swizzle_identity();
                let mut source_planes = 1;
                let source_is_target = matches!(&loaded, SampledSourceRequest::Target(_, _));
                let source = match loaded {
                    SampledSourceRequest::Bytes(rgba, identity, byte_format, origin) => {
                        bytes_identity = identity;
                        sampled_vk_format = translate::pixel::vk_sampled_bytes(byte_format);
                        byte_origin = origin;
                        crate::backend::vulkan::engine::SampledSource::Bytes(rgba)
                    }
                    SampledSourceRequest::Target(identity, format) => {
                        sampled_vk_format = format;
                        // The source resolver carries the sampled texture's
                        // exact view format beside the allocation identity. A
                        // resident attachment view is not necessarily the view
                        // this bind names; collapsing the two loses both sRGB
                        // interpretation and physical channel order.
                        //
                        // The resource's `swizzle` below remains independent and
                        // is composed once with the format's component plan.
                        //
                        // This arm used to `return Ok(())` — no resource pushed
                        // at all. That was not a decline: the unbound scan had
                        // already counted `texture_ref != 0` as provided, so no
                        // neutral image was substituted either, and the binding
                        // went missing from a layout the fragment module
                        // statically uses. The engine's
                        // `used_binding_absent_from_layout` then refused the
                        // whole draw, which cost the guest every pixel of it.
                        crate::backend::vulkan::engine::SampledSource::Target(identity)
                    }
                    SampledSourceRequest::GuestRuns(
                        src,
                        _native,
                        format,
                        planes,
                        identity,
                        vouch,
                        components,
                    ) => {
                        sampled_vk_format = format;
                        sampled_components = components;
                        source_planes = planes;
                        bytes_identity = identity;
                        crate::backend::vulkan::engine::SampledSource::GuestRuns(src, vouch)
                    }
                };
                let array_element = reflected_descriptor
                    .map(|descriptor| descriptor.array_element)
                    .unwrap_or(0);
                let descriptor_count = reflected_descriptor
                    .map(|descriptor| descriptor.descriptor_count)
                    .unwrap_or(1);
                // Texture dimensionality comes solely from the translator's reflection,
                // keyed on the UN-relocated descriptor binding. The always-on
                // `census_reflection_wellformed` guard (m2v_cache) proves the reflection
                // is internally consistent per translate. `Absent` is an unused/unbound
                // sampler slot (Metal permits it) — default 2D silently (expected control
                // flow). `Unsupported` is a texture shape reflection carries but the
                // sampled path can't express — log fail-visibly, then keep the 2D default
                // so the draw still paints rather than dropping content.
                use crate::runtime::spirv_bind::{ReflectedSampledKind, SampledImageKind};
                let image_kind = match crate::runtime::spirv_bind::reflected_sampled_kind(
                    reflection,
                    reflected_descriptor
                        .map(|descriptor| descriptor.binding)
                        .unwrap_or(TEXTURE_BINDING_BASE + index),
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
                    multisampled,
                    mut layers,
                } = shape;
                if multisampled && !source_is_target {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::TextureDimensionUnsupported {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                            index,
                            texture_ref,
                            binding: img_bind,
                            kind: format!("{image_kind:?}"),
                        },
                    ));
                }
                if volume {
                    layers = texture_resource
                        .as_ref()
                        .and_then(|resource| match resource.decoded() {
                            Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => {
                                tex.levels.first().map(|level| level.planes())
                            }
                            _ => None,
                        })
                        .unwrap_or(source_planes);
                }
                // A Vulkan 1D image is defined to have height 1; the descriptor
                // may report the LUT's texel count in either axis, so collapse
                // to a single row and fold the other axis into the width the
                // sampled bytes are validated against.
                let (tw, th) = if one_dim {
                    (tw.saturating_mul(th).max(1), 1)
                } else {
                    (tw, th)
                };
                images.push(crate::backend::vulkan::engine::SampledImageResource {
                    binding: img_bind,
                    array_element,
                    descriptor_count,
                    width: tw,
                    height: th,
                    layers,
                    arrayed,
                    volume,
                    cube,
                    one_dim,
                    multisampled,
                    source,
                    byte_origin,
                    format: sampled_vk_format,
                    identity: bytes_identity.map(|i| {
                        crate::backend::vulkan::engine::SampledContentIdentity {
                            key: i.key,
                            generation: i.generation,
                        }
                    }),
                    // The guest's view swizzle applied *after* the format's own
                    // channel plan, folded into the one mapping the image view
                    // can carry. Composed unconditionally rather than behind a
                    // "does this need it" branch: identity is the unit on both
                    // sides, so the fold is a no-op for every bind that does not
                    // need it, and there is no case left to forget.
                    swizzle: view_swizzle.unwrap_or_default().after(&sampled_components),
                });
                Ok(())
            };
            for t in req.vertex_textures.iter() {
                push_tex(t.index, t.texture_ref, t.resource.as_ref(), false)?;
            }
            for t in req.fragment_textures.iter() {
                push_tex(t.index, t.texture_ref, t.resource.as_ref(), true)?;
            }
        }
        // Repair the gaps the guard found. A fragment texture the module
        // statically uses and this draw did not bind is absent from the
        // descriptor set layout *entirely* — `engine/exec.rs` builds the layout
        // from provided resources alone, so it is not an unwritten slot in a
        // layout that has the binding. Vulkan requires a descriptor for every
        // statically-used resource, and on Mesa's Intel driver the omission is
        // fatal rather than undefined: it sizes its binding array to
        // `max_binding + 1`, zero-fills every number nothing declared, and
        // scores each *used* binding as `(use_count << 7) / array_size`, so the
        // hole divides by zero and the host process dies of `SIGFPE` inside
        // pipeline creation with nothing returned for this device to decline on.
        //
        // Cold path: the vector is empty on every draw that binds what it
        // samples, so this costs one `is_empty` on the hot path.
        for &index in &frag_unbound_used_textures {
            use crate::runtime::spirv_bind::{ReflectedSampledKind, SampledImageKind};
            let base_off = if separate_sampled {
                FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
            } else {
                0
            };
            let img_bind = TEXTURE_BINDING_BASE + index + base_off;
            if images.iter().any(|img| img.binding == img_bind) {
                continue;
            }
            // The shape has to be the one the module declared: a plain 2D view
            // bound where the shader samples an array is a different violation,
            // not a repair. The reflection is asked in the translator's
            // numbering, which is what `reflected_sampled_kind` takes — the
            // relocation above applies to the SPIR-V, not to the signature.
            let kind = match crate::runtime::spirv_bind::reflected_sampled_kind(
                &f_shader.reflection,
                TEXTURE_BINDING_BASE + index,
            ) {
                ReflectedSampledKind::Kind(k) => k,
                ReflectedSampledKind::Absent | ReflectedSampledKind::Unsupported => {
                    SampledImageKind::D2
                }
            };
            let Some(shape) = sampled_image_shape(kind) else {
                // Cube and cube-array need six faces, and this engine declines
                // them where they are bound too. The hole stays and is named,
                // rather than papered over with a shape the shader did not
                // declare — which would be a second violation wearing the
                // repair's clothes.
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound \
                     reason=frag_neutral_texture_shape_unsupported \
                     pipe={} idx={index} binding={img_bind} kind={kind:?}",
                    req.pipeline_ref
                ));
                continue;
            };
            if shape.multisampled {
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound \
                     reason=frag_neutral_texture_multisample_unrepresentable \
                     pipe={} idx={index} binding={img_bind} kind={kind:?}",
                    req.pipeline_ref
                ));
                continue;
            }
            // A repair that succeeded, not a success: the shader samples a
            // texture whose contents this device invented, so it stays on the
            // fail channel and the reliance stays measurable.
            crate::observe::fail(format!(
                "shader_resource_declared_unbound \
                 reason=frag_neutral_texture_substituted \
                 pipe={} idx={index} binding={img_bind} kind={kind:?} 1x1",
                req.pipeline_ref
            ));
            images.push(crate::backend::vulkan::engine::SampledImageResource {
                binding: img_bind,
                array_element: 0,
                descriptor_count: 1,
                width: 1,
                height: 1,
                layers: shape.layers,
                arrayed: shape.arrayed,
                volume: shape.volume,
                cube: shape.cube,
                one_dim: shape.one_dim,
                multisampled: false,
                source: crate::backend::vulkan::engine::SampledSource::Bytes(std::sync::Arc::new(
                    crate::contract::pixel_format::solid_rgba8(1, 1, &[0.0; 4]),
                )),
                byte_origin: crate::backend::vulkan::engine::SampledByteOrigin::Synthetic,
                format: ash::vk::Format::R8G8B8A8_UNORM,
                identity: None,
                swizzle: Default::default(),
            });
        }
        {
            let mut push_smp = |index: u32,
                                sampler_ref: u32,
                                lod_clamp: Option<(u32, u32)>,
                                frag_stage: bool|
             -> Result<(), DrawError> {
                let base_off = if frag_stage && separate_sampled {
                    FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                } else {
                    0
                };
                let smp_bind = SAMPLER_BINDING_BASE + index + base_off;
                if sampler_binds.insert(smp_bind) {
                    let mut sampler = if sampler_ref != 0 {
                        sampler_origin.insert(smp_bind, b'g');
                        load_vulkan_sampler(state, host, req.task_id, sampler_ref, smp_bind)
                            .map_err(DrawError::DrawPreparation)?
                    } else {
                        sampler_origin.insert(smp_bind, b'd');
                        crate::backend::vulkan::engine::SamplerResource::normalized_default(
                            smp_bind,
                        )
                    };
                    // A bind record's own clamps override the sampler object's.
                    // That is what `setVertexSamplerStates:lodMinClamps:
                    // lodMaxClamps:withRange:` means: one sampler state bound
                    // at several slots, each clamped differently, without
                    // creating a sampler object per clamp. The compute rail
                    // applies the override in exactly this position.
                    if let Some((min_bits, max_bits)) = lod_clamp {
                        sampler.lod_min = min_bits;
                        sampler.lod_max = max_bits;
                    }
                    samplers.push(sampler);
                } else {
                    // A second sampler resolving to a binding this draw already
                    // provisioned. First-writer-wins, so this one's filter,
                    // address mode and LOD clamp are dropped — and because the
                    // layout gives every sampler `VERTEX | FRAGMENT` stage flags,
                    // the stage that lost goes on sampling through the stage that
                    // won.
                    //
                    // A **healthy zero** now that `separate_sampled` covers the
                    // whole sampled band: with both stages relocated apart, two
                    // stages cannot collide, and one stage cannot bind its own
                    // index twice. A firing is the bug, not the report — most
                    // likely a band whose relocation offset stopped separating
                    // the two.
                    crate::runtime::drain::note_store_route("sampler_bind_collided");
                    if crate::observe::first_sight("sampler_bind_collided", u64::from(smp_bind)) {
                        crate::observe::fail(format!(
                            "sampler_bind_collided binding={smp_bind} index={index} \
                             stage={} separate_sampled={separate_sampled} \
                             (a second sampler resolved to a binding this draw had \
                             already provisioned; its state is dropped and the \
                             other stage's is what samples)",
                            if frag_stage { "fragment" } else { "vertex" }
                        ));
                    }
                }
                Ok(())
            };
            // Stream sampler slots (often index 0 while texture is 3 for logo).
            // Both loops share one span rather than one per bind: the fix here
            // is a sampler object cache, which is the same fix whichever stage
            // asked, so two bars would be two views of one lever.
            let _s = crate::runtime::sampled_phase::Span::open(
                crate::runtime::sampled_phase::Part::Samplers,
            );
            for s in req.vertex_samplers.iter() {
                if s.sampler_ref != 0 {
                    push_smp(s.index, s.sampler_ref, s.lod_clamp, false)?;
                }
            }
            for s in req.fragment_samplers.iter() {
                if s.sampler_ref != 0 {
                    push_smp(s.index, s.sampler_ref, s.lod_clamp, true)?;
                }
            }
        }
        // Each shader variant carries the reflected sampler interface in the
        // same numbering as its executable words. Constexpr state therefore
        // cannot drift away from its relocated binding, and residual bindings
        // need no SPIR-V walk.
        {
            let _s = crate::runtime::sampled_phase::Span::open(
                crate::runtime::sampled_phase::Part::Reflect,
            );
            for (variant, stage) in [(&v_variant, "vertex"), (&f_variant, "fragment")] {
                for reflected in variant.samplers.iter() {
                    if sampler_binds.insert(reflected.binding) {
                        let binding = reflected.binding;
                        if let Some(state) = reflected.static_state {
                            sampler_origin.insert(binding, b'c');
                            samplers.push(
                                reflected_static_sampler_resource(stage, binding, state)
                                    .map_err(DrawError::DrawPreparation)?,
                            );
                        } else {
                            sampler_origin.insert(binding, b'd');
                            samplers.push(
                                crate::backend::vulkan::engine::SamplerResource::normalized_default(
                                    binding,
                                ),
                            );
                        }
                    }
                }
            }
        }
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Seed);
        // Colour load seed: LOAD → guest/host seed when present. `seed_order`
        // names what is in those bytes; the engine folds any needed R/B exchange
        // into its copy into the mapped staging span rather than making this
        // side materialize a converted frame.
        //
        // CLEAR is not a seed. It travels as `target_clear` and the render pass
        // does it, which is what `MTLLoadActionClear` asks for.
        let mut target_rgba8: Option<std::sync::Arc<Vec<u8>>> = None;
        let mut target_guest_seed = None;
        let mut target_clear = [0.0f32; 4];
        // Slot 0's declared action, travelling beside the seed rather than being
        // reconstructed from whether one arrived. `Clear` until a colour
        // attachment says otherwise, which is the same reason the type defaults
        // that way: a draw with no colour record at all has declared nothing,
        // and nothing must not mean "discard".
        let mut color0_load = crate::contract::pass_action::LoadAction::Clear;
        let mut seed_order = crate::backend::vulkan::engine::SeedOrder::Rgba8;
        let gpu_only_content_allowed =
            crate::backend::vulkan::engine::deferred_gpu_only_content_allowed();
        // Records 2+ of a resident render-pass chain load the prior record's
        // content directly from the engine target (no CPU seed, no re-upload).
        let mut chain_load_from_target = false;
        // Resolved once and read by both the Load gate below and the
        // `target_identity` assignment further down, so the record that loads
        // from a resident is by construction the record that renders into it.
        // Resolve and revalidate the shared allocation before minting the
        // resident identity below. Revalidation may advance the mapping
        // generation when the guest recycled a page, and the identity must
        // name that new generation rather than the one paired with the retired
        // alias.
        let type11_guest_backing = type11_guest_target_backing(state, host, req);
        let type11_resident_target = type11_store_identity(state, req, writeback_guest);
        if req.chain_from_resident && render_chain_identity(state, req).is_some() {
            // The serialized chain names the resident it intends to load;
            // existence and readiness are engine state and are validated
            // atomically with target acquisition by `execute_draw_request`.
            // A preparation-time query could only be a stale hint and cost
            // a second engine transaction for the same command.
            chain_load_from_target = true;
        }
        // Colour0's LOAD seed was skipped by `mrt_draw_request` because the
        // engine still held what the render Store published into its guest
        // pages. Honour that here, or put the seed back.
        let mut gva_load_identity =
            honour_gva_load_elision(state, host, req, &mut chain_load_from_target);
        // Type-11 composite Load. A retained guest-allocation target is the
        // guest's texture resource itself, so its LOAD is authoritative without
        // comparing two copies. A device-allocation target is a mirror: only
        // when it was stamped with the mapping's current
        // `surface_content_epoch` does it hold exactly the bytes
        // `resolve_type11_load_seed` would upload.
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
        // `render_writeback` — and there is no entry for the owner of the pages.
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
        // 41 389 against `type11_seed_provided` 242, at a mean 1.43 M texels per
        // elision, so revalidating by re-reading would move ~237 GB of guest
        // memory a session. The bitmap answers the same question in a word.
        if !chain_load_from_target {
            if let Some((identity, mapping_epoch)) = type11_load_currency_query(state, req) {
                // Both arms counted, into the same one-second window as
                // `drain_duty`. An elision count alone cannot tell "the seed was
                // skipped" from "this record was never a candidate", and the
                // ratio of the two is a within-boot number — the only kind that
                // survives the 1.8x `us_per_draw` drift between boots on this rig.
                let backing = req.colors.first().and_then(|c0| {
                    state
                        .task_resources
                        .get(req.task_id, c0.texture_ref)
                        .filter(|resource| {
                            resource_type_owns_surface_resident(resource.entry.object_type)
                        })
                        .map(|resource| resource.resident_target_backing(&identity))
                });
                let (resident_current, guest_wrote) =
                    type11_load_resident_is_current(backing, || {
                        let resident_epoch =
                            crate::backend::vulkan::engine::resident_content_epoch(&identity);
                        let mapping_id = req.colors.first().map(|c| c.mapping_id).unwrap_or(0);
                        let guest_wrote = type11_guest_wrote_since_store(state, host, mapping_id);
                        (
                            type11_resident_is_current(mapping_epoch, resident_epoch)
                                && !guest_wrote,
                            guest_wrote,
                        )
                    });
                if resident_current {
                    chain_load_from_target = true;
                    crate::runtime::drain::note_store_route("type11_seed_elided");
                    if backing
                        == Some(
                            crate::backend::vulkan::engine::ResidentContentBacking::GuestAllocation,
                        )
                    {
                        crate::runtime::drain::note_store_route("type11_seed_guest_allocation");
                    }
                    note_type11_elision_extent(w, h);
                } else {
                    crate::runtime::drain::note_store_route("type11_seed_provided");
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
            // Out of contract is DontCare here, same as on the Metal arm, and
            // it says so through the same helper — this arm used to take an
            // unknown load action into the `_ => {}` below and blank the
            // attachment in silence.
            let load_action = if super::load_action_in_contract(req.pipeline_ref, c0.load_action) {
                c0.load_action
            } else {
                MTL_LOAD_ACTION_DONT_CARE
            };
            color0_load = crate::contract::pass_action::LoadAction::from_declared(load_action);
            // The declared load action of slot 0, as a population **and** as
            // pixels.
            //
            // This is now the *declaration* beside the engine's
            // `passbegin_load` / `passbegin_clear` / `passbegin_dontcare`, which
            // is what this device resolved it to; the two differ exactly when a
            // declared Load arrived with no content, and reading them side by
            // side is the only way that case is visible as a count. Until the
            // engine grew a third route, `passbegin_clear` carried the Clears
            // and the DontCares together and the sum matched, so the collapse
            // read as a working census. The two cost very different amounts —
            // a CLEAR writes every texel of the attachment, and on this
            // pathway a quarter of all attachments are `VK_IMAGE_TILING_LINEAR`
            // over guest RAM with no fast clear and no colour compression, so
            // the write is the full plane at memory bandwidth. A DontCare
            // spelled `AttachmentLoadOp::DONT_CARE` writes none of it.
            //
            // Pixels rather than records because the price is proportional to
            // area and the population is dominated by small attachments: a
            // census of records would rank a boot's thousands of 64x64 passes
            // above its dozens of full-screen ones.
            let declared_px = u64::from(w).saturating_mul(u64::from(h));
            // `load_action` was already folded to DontCare above for anything
            // out of contract, so this fold is exact and the two spellings
            // cannot disagree.
            let (declared_n, declared_area) =
                crate::contract::pass_action::LoadAction::from_declared(load_action)
                    .census_routes(crate::contract::pass_action::AttachmentBand::Color0);
            crate::runtime::drain::note_store_route(declared_n);
            crate::runtime::drain::note_store_route_n(declared_area, declared_px);
            match load_action {
                MTL_LOAD_ACTION_LOAD if chain_load_from_target => {
                    // Resident target carries the chain; no CPU seed bytes.
                }
                MTL_LOAD_ACTION_CLEAR => {
                    // The pass clears the attachment. No seed: a seed would
                    // resolve this pass key to LOAD, which is the opposite of
                    // what the guest asked for, and would spend an allocation, a
                    // channel exchange and a staged upload writing one constant
                    // into every texel.
                    target_clear = [
                        c0.clear_color[0] as f32,
                        c0.clear_color[1] as f32,
                        c0.clear_color[2] as f32,
                        c0.clear_color[3] as f32,
                    ];
                }
                MTL_LOAD_ACTION_LOAD => {
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
                        // The attachment format follows the render identity,
                        // including an intermediate chain record that does not
                        // own the packet's final guest writeback. The Store-only
                        // identity below is deliberately narrower and would
                        // misclassify that record as a pooled RGBA target here.
                        let target_format = type11_render_identity(state, req)
                            .as_ref()
                            .map(|identity| identity.resident_format())
                            .unwrap_or(translate::pixel::RESIDENT_RGBA_FORMAT);
                        match resolve_type11_load_seed(
                            state,
                            host,
                            c0.mapping_id,
                            w,
                            h,
                            target_format,
                        ) {
                            Some(Type11LoadSeed::Host(bytes, order)) => {
                                target_rgba8 = Some(bytes);
                                seed_order = order;
                            }
                            Some(Type11LoadSeed::Guest(seed)) => target_guest_seed = Some(seed),
                            None => {}
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
                    note_load_seed_outcome(
                        seed_door,
                        target_rgba8.is_some() || target_guest_seed.is_some(),
                        c0,
                        w,
                        h,
                    );
                }
                // DontCare: the guest declared the prior contents undefined, so
                // arriving with no seed is the contract rather than a loss, and
                // it now *reaches* the engine as such —
                // `DrawRequest::color0_load` carries the word and the pass is
                // begun with `AttachmentLoadOp::DONT_CARE`. Its own arm, so the
                // third value of a three-valued enum stops sharing the catch-all
                // with the out-of-contract one two lines above; there is nothing
                // left for it to do, because leaving `target_rgba8` and
                // `target_clear` alone is exactly what DontCare means.
                MTL_LOAD_ACTION_DONT_CARE => {}
                _ => {}
            }
        }
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
        let mut resources = crate::backend::vulkan::engine::DrawRequest {
            pipeline_object: resolved.pipeline_object.clone(),
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
            // MTLTriangleFillMode / MTLDepthClipMode, both defaulting to 0.
            // Unlike the two above, the non-default arm of each needs a device
            // feature, so the engine may still decline the pipeline by name
            // after this maps cleanly: the mapping says what the guest asked
            // for, the capability check says whether the host can spell it.
            fill_mode: raster_or_default(
                req.fill_mode,
                translate::raster::fill_mode,
                crate::backend::vulkan::engine::FillMode::Fill,
                req.pipeline_ref,
                "fill_mode_unmapped",
            ),
            depth_clip: raster_or_default(
                req.depth_clip_mode,
                translate::raster::depth_clip_mode,
                crate::backend::vulkan::engine::DepthClipMode::Clip,
                req.pipeline_ref,
                "depth_clip_mode_unmapped",
            ),
            first_vertex: req.first_vertex,
            // Passed through. `decode::render`'s `wire_instance_count` is where
            // a zero instance count is decided, and it is decided once — a
            // second `.max(1)` here would re-apply that rule on this arm alone,
            // so a change made at the decode site would appear to take effect
            // everywhere while this path quietly kept the old answer.
            instance_count: Some(req.instance_count),
            primitive_topology: raster_or_default(
                Some(req.primitive_type),
                translate::raster::primitive_topology,
                crate::backend::vulkan::engine::PrimitiveTopology::Triangle,
                req.pipeline_ref,
                "primitive_type_unmapped",
            ),
            raster_sample_count: pd.raster_sample_count.max(1),
            color_sample_count: req
                .colors
                .first()
                .map(|color| color.sample_count.max(1))
                .unwrap_or(1),
            multisample_resolve: req
                .colors
                .first()
                .is_some_and(|color| color.multisample_source_ref != 0),
            ..crate::backend::vulkan::engine::DrawRequest::default()
        };
        let resolving_colors = req
            .colors
            .iter()
            .filter(|color| color.multisample_source_ref != 0)
            .count();
        if resolving_colors != 0
            && (resolving_colors != 1
                || req
                    .colors
                    .first()
                    .is_none_or(|color| color.multisample_source_ref == 0))
        {
            return Err(DrawError::Unsupported(
                crate::backend::vulkan::engine::reason::DrawReason::MultisampleResolveShapeUnsupported {
                    color_targets: req.colors.len() as u32,
                    depth: req.depth_attach.is_some(),
                    color_input: false,
                },
            ));
        }
        if let Some(color) = req.colors.first().filter(|color| color.multisample_source_ref != 0) {
            use crate::contract::pass_action::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
            if color.store_action != MTL_STORE_ACTION_MULTISAMPLE_RESOLVE {
                return Err(DrawError::Unsupported(
                    crate::backend::vulkan::engine::reason::DrawReason::MultisampleStoreActionUnsupported {
                        store_action: color.store_action,
                    },
                ));
            }
            if color.load_action == MTL_LOAD_ACTION_LOAD {
                return Err(DrawError::Unsupported(
                    crate::backend::vulkan::engine::reason::DrawReason::MultisampleLoadActionUnsupported {
                        load_action: color.load_action,
                    },
                ));
            }
        }
        resources.viewports = req
            .viewports
            .iter()
            .map(|vp| crate::backend::vulkan::engine::ViewportResource {
                x: vp[0] as f32,
                y: vp[1] as f32,
                width: vp[2] as f32,
                height: vp[3] as f32,
                min_depth: vp[4] as f32,
                max_depth: vp[5] as f32,
            })
            .collect();
        // The census takes slot 0, which is the rect it has always taken: with
        // one scissor it is the whole answer, and with several it is the one a
        // single-rect damage bound would have to start from.
        if let Some(scissor) = req.scissors.first() {
            note_draw_coverage(
                *scissor,
                w,
                h,
                req.colors.first().map(|c| c.load_action),
                target_rgba8.is_some() || target_guest_seed.is_some(),
                chain_load_from_target,
            );
        }
        // The mode is the guest's raw `MTLVisibilityResultMode`; the engine
        // takes the translated arm, and an ordinal outside the enum refuses the
        // draw by the translation's own name rather than being coerced into
        // one. `Ok(None)` cannot reach here — `runtime::exec` turns
        // `Disabled` into `req.visibility == None` — but it is handled rather
        // than asserted, because "the guest disarmed it" and "no query" are the
        // same draw either way.
        resources.occlusion_query = match req.visibility {
            None => None,
            Some(v) => translate::raster::visibility_result_mode(v.mode).map_err(|e| {
                DrawError::Unsupported(
                    crate::backend::vulkan::engine::reason::DrawReason::VisibilityResultMode(e),
                )
            })?,
        };
        resources.scissors = req
            .scissors
            .iter()
            .map(|s| crate::backend::vulkan::engine::ScissorResource {
                x: s.x,
                y: s.y,
                width: s.width,
                height: s.height,
            })
            .collect();
        if let Some(idx) = req.indexed.as_ref() {
            let index_type = translate::raster::index_type(idx.index_type).ok_or({
                DrawError::DrawPreparation(DrawPreparationDecline::IndexLoad {
                    reason: IndexLoadReason::TypeUnsupported,
                })
            })?;
            let content =
                load_index_content_reason(state, host, req.task_id, idx).map_err(|reason| {
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
                        reason: crate::runtime::draw::IndexLoadReason::BaseVertexOutOfRange,
                    })
                })?,
                content,
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
        resources.continues_render_pass = req.continues_render_pass;
        resources.render_pass_continues = req.render_pass_continues;
        resources.samplers = samplers;
        // Load seed always goes to the GPU (workstream D3). Premult One/OMSA is
        // hardware blend over the Load-seeded target — identical math to the
        // retired software `src + seed*(1-src.a)` path. Sampled alpha is
        // protocol data and must not be rewritten from an RGB content census;
        // content-gated keep-seed / alpha0-holes composites are retired.
        let store_is_store = req
            .colors
            .first()
            .map(|c| crate::contract::pass_action::store_action_publishes_single_sample(c.store_action))
            .unwrap_or(true);
        resources.target_rgba8 = target_rgba8;
        resources.target_guest_seed = target_guest_seed;
        resources.target_clear = target_clear;
        resources.color0_load = color0_load;
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
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::AssembleTarget);
        // Ephemeral resident render-pass rail: intermediate Store records render
        // into a protocol-keyed RGBA target on every Vulkan backend. This does
        // not leave guest-visible content GPU-only: portability devices read the
        // final record back and perform the normal synchronous guest Store.
        // Cross-pass deferred ownership remains gated below.
        let mut resident_render_chain = false;
        // Host-authoritative GVA Store rail: the final/single record also stays
        // on the registry resident (skip_readback). The caller records the
        // resource declaration and transfers only when synchronization or an
        // actual guest-page reader makes that copy observable.
        //
        // The rail's own resident, not a bool: the span this returns has to name
        // the key the draw registered, and a flag beside `resources` would let a
        // caller derive a second one. See `M2vDrawSpan::ResidentSurfaceStore`.
        let mut gva_resident_store: Option<crate::backend::vulkan::engine::TargetIdentity> = None;
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
                    resources.target_identity = Some(identity.clone());
                    resources.skip_readback = true;
                    gva_resident_store = Some(identity);
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
        //
        // The rail's own resident, not a bool, for the reason `gva_resident_store`
        // states: the span carries this value out to the Store.
        let mut surface_resident_store: Option<crate::backend::vulkan::engine::TargetIdentity> =
            None;
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
        if renders_into_surface_identity && !resources.skip_readback {
            resources.skip_readback = true;
            // `resources.target_identity`, which the comparison just proved is
            // this same value, and which is what `registry_ensure` will be
            // handed. Taken from here rather than unwrapped from the `Option`
            // above so the span cannot name a slot the draw did not register.
            surface_resident_store = type11_resident_target.clone();
        }
        resources.record_guest_store = surface_resident_store.is_some();
        // A first-failure classifier for a composite Store that still reads
        // back used to sit here. Its outer gate was never once true — none of
        // its four counters, nor its `else` arm, appears anywhere in the
        // always-on log across every boot it holds. Every type-11 Store either
        // skips its readback or is not a writeback Store.
        if chain_load_from_target {
            // The GVA Load elision validated its own identity and is the only
            // rail here whose target is not also claimed by a Store rail: a pass
            // that loads from a GVA resident need not be storing to one, so the
            // deferred-Store block above may have left this `None` for a reason
            // that is not a wiring bug. Supplying it here keeps the refusal below
            // meaning what it says.
            if resources.target_identity.is_none() {
                resources.target_identity = gva_load_identity.take();
            }
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
        // The backing belongs only to the type-11 surface identity it was
        // resolved from. A GVA or render-chain namespace may legitimately own
        // this draw's primary attachment instead, in which case handing it the
        // surface allocation would bind two unrelated resources together.
        if resources.target_identity == type11_render_identity(state, req) {
            resources.guest_target_memory = type11_guest_backing;
            resources.load_guest_target_backing = resources.guest_target_memory.is_some()
                && req.colors.first().is_some_and(|color| {
                    color.load_action == MTL_LOAD_ACTION_LOAD && color.target_seed_rgba.is_none()
                });
        }
        // Type-11 Load used to have a GPU rail here — ~170 lines of front-frame
        // retention policy resolving which resident image held the frame the
        // guest computes its damage against. It was reachable only under
        // `try_import`. A shared resident now keeps the guest allocation as
        // its own backing, while a copied resident is always landed before its
        // guest pages become a later seed, so neither needs a separate
        // front-frame retention policy.
        // Metal path always passes color0 blend into the encoder. Linux/engine
        // previously left `resources.blend = None` → opaque replace for every
        // draw, so Load seeds (gray/wallpaper/logo bases) were wiped by sparse
        // dock/chrome layers that Metal would alpha-blend over the attachment.
        // Contract: type-7 color attachment blend tags (decode/resource).
        // Outside the `blending_enabled` guard below, and deliberately: an
        // unblended attachment with a mask still leaves its unwritten channels
        // alone, so gating the mask on blending would drop it exactly where the
        // guest is replacing rather than compositing.
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
        resources.color_write_mask = pd.color0.write_mask;
        if pd.color0.blending_enabled {
            let constants = req.blend_color.unwrap_or([0.0; 4]);
            match translate::blend::state(&pd.color0, constants) {
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
        //
        // Unlike the instance count above, nothing upstream floors this one:
        // `decode::render` narrows it to 32 bits and refuses an over-wide value
        // by name, but a zero it decodes is a zero. So on a *non*-indexed draw
        // this clamp is the guest's own `vertexCount:` overruled, and one vertex
        // is drawn where none was asked for. Counted rather than changed,
        // because which arm is right is not decidable from here: the engine
        // validates the field it ignores, so passing a zero through would refuse
        // indexed draws that are perfectly well formed.
        //
        // A firing is the signal, and it is the reading that would let this be
        // scoped to the indexed case where the field is inert.
        if req.vertex_count == 0 {
            crate::runtime::drain::note_store_route("draw_vertex_count_zero");
        }
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
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::AssembleDepth);
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
                                .unwrap_or((1.0, MTL_LOAD_ACTION_CLEAR));
                            // `MTLLoadActionLoad` is carried through as the
                            // guest wrote it. Whether it can be *honoured* is
                            // not decidable here — it needs the depth resident's
                            // own content state, which only the engine holds —
                            // so the engine makes that call and names the
                            // degradation when it cannot. See
                            // `pools::registry_mark_depth_ready`.
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
                                identity: depth_chain_identity(req, stencil.is_some()),
                                test_enable: true,
                                write_enable: ds.depth_write_enabled,
                                compare,
                                clear_value,
                                load: load_action == MTL_LOAD_ACTION_LOAD,
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
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
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
                (resources.target_rgba8.is_some() || resources.target_guest_seed.is_some()) as u8,
                resources.indexed.is_some() as u8,
                resources.indexed.as_ref().map(|i| i.index_count).unwrap_or(0),
                attr_meta,
                ssbo_meta,
                sampler_meta
            ));
        }

        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::AssembleTrail);
        // Asked of the module rather than of m2v's reflection, which is the
        // whole point: the render path's existing unbound guard walks
        // `f_shader.reflection.bindings`, so a binding the translated SPIR-V
        // carries and the reflection omits is checked by nothing.
        // `descriptor_static_use` cannot close it either — it answers
        // `NotDeclared` for anything that is not a `UniformConstant`, which by
        // construction excludes every storage buffer.
        // Memoized on the `Arc`, because this is a per-draw walk of the whole
        // module and the words behind an `Arc` cannot change.
        let frag_declared_bindings =
            crate::runtime::spirv_bind::declared_binding_numbers_memoized(&f_words);
        let frag_layout_bindings: Vec<u32> = resources
            .storage_buffers
            .iter()
            .map(|s| s.binding)
            .chain(resources.sampled_images.iter().map(|i| i.binding))
            .chain(resources.samplers.iter().map(|s| s.binding))
            .collect();
        let frag_gap =
            crate::runtime::gpu_hang_trail::gap(&frag_declared_bindings, &frag_layout_bindings);
        // The hang trail, recorded here because this is the last point at which
        // the guest's pipeline ref and both translated module sizes are in scope
        // together: past it the engine keys on digests and the ref is gone. See
        // [`crate::runtime::gpu_hang_trail`] for what reads it and why a counter
        // could not answer the question.
        // What the draw is about to sample, lowest binding first. The trail's
        // whole subject is a fragment module that walks a pointer chain through
        // a sampled image, and until this it recorded the module's *size* and
        // nothing about its inputs — so a wedged boot could not say which rail
        // supplied the walked texture, what format the shader would read it as,
        // or whether the extent was the one the guest meant.
        //
        // Sorted here rather than relied upon: `sampled_images` is in the order
        // the two texture loops pushed it, vertex stage first, so the fragment
        // bindings are neither first nor contiguous.
        let mut sampled_notes = [crate::runtime::gpu_hang_trail::SampledNote::default();
            crate::runtime::gpu_hang_trail::SAMPLED_KEPT];
        let mut by_binding: Vec<&crate::backend::vulkan::engine::SampledImageResource> =
            resources.sampled_images.iter().collect();
        by_binding.sort_unstable_by_key(|i| i.binding);
        for (slot, image) in sampled_notes.iter_mut().zip(by_binding.iter()) {
            *slot = crate::runtime::gpu_hang_trail::SampledNote {
                binding: image.binding,
                kind: match &image.source {
                    crate::backend::vulkan::engine::SampledSource::Bytes(_) => 1,
                    crate::backend::vulkan::engine::SampledSource::Target(_) => 2,
                    crate::backend::vulkan::engine::SampledSource::GuestRuns(..) => 3,
                },
                format: image.format.as_raw() as u32,
                width: image.width,
                height: image.height,
                // Only the CPU-bytes rail has bytes here to read. The gather
                // rail's texels are in guest RAM and the target rail's are on
                // the GPU; reading either one would be a device-memory access
                // taken to write a log line, which is not a trade this makes.
                texel0: match &image.source {
                    crate::backend::vulkan::engine::SampledSource::Bytes(b) => b
                        .get(..4)
                        .map(|t| u32::from_le_bytes([t[0], t[1], t[2], t[3]]))
                        .unwrap_or(0),
                    _ => 0,
                },
            };
        }
        // And what it will sample them *through*. All four of the uber shader's
        // unbounded loops share one sampler, and a `LINEAR` filter on a texture
        // whose texels are the next UV walks a blend of two cells rather than
        // either — so the third of the wedge's three hypotheses is a property of
        // this list and of nothing the trail recorded before.
        let mut sampler_notes = [crate::runtime::gpu_hang_trail::SamplerNote::default();
            crate::runtime::gpu_hang_trail::SAMPLER_KEPT];
        let mut smp_by_binding: Vec<&crate::backend::vulkan::engine::SamplerResource> =
            resources.samplers.iter().collect();
        smp_by_binding.sort_unstable_by_key(|s| s.binding);
        for (slot, smp) in sampler_notes.iter_mut().zip(smp_by_binding.iter()) {
            use crate::backend::vulkan::engine::{
                SamplerAddressMode as A, SamplerFilter as F, SamplerMipFilter as M,
            };
            let filter = |f: F| match f {
                F::Nearest => b'N',
                F::Linear => b'L',
            };
            let address = |a: A| match a {
                A::ClampToEdge => b'e',
                A::MirrorClampToEdge => b'E',
                A::Repeat => b'r',
                A::MirrorRepeat => b'R',
                A::ClampToZero => b'z',
                A::ClampToBorderColor => b'b',
            };
            *slot = crate::runtime::gpu_hang_trail::SamplerNote {
                binding: smp.binding,
                min_filter: filter(smp.min_filter),
                mag_filter: filter(smp.mag_filter),
                mip_filter: match smp.mip_filter {
                    M::NotMipmapped => b'n',
                    M::Nearest => b'N',
                    M::Linear => b'L',
                },
                address_u: address(smp.address_mode_u),
                address_v: address(smp.address_mode_v),
                // `?` is a sampler that reached the list by a route that did not
                // record where its state came from, which is the reading that
                // would send the next session looking for a fourth path rather
                // than concluding anything about the three.
                provenance: sampler_origin.get(&smp.binding).copied().unwrap_or(b'?'),
                unnormalized: smp.unnormalized_coordinates,
            };
        }
        crate::runtime::gpu_hang_trail::note_draw(crate::runtime::gpu_hang_trail::DrawNote {
            sampled: sampled_notes,
            sampled_count: resources.sampled_images.len() as u32,
            samplers: sampler_notes,
            sampler_count: resources.samplers.len() as u32,
            pipeline_ref: req.pipeline_ref,
            vert_words: v_words.len() as u32,
            frag_words: f_words.len() as u32,
            width: w,
            height: h,
            vertex_count,
            instance_count: req.instance_count,
            // Asked of the module rather than of m2v's reflection, which is the
            // whole point: the render path's existing unbound guard walks
            // `f_shader.reflection.bindings`, so a binding the translated
            // SPIR-V carries and the reflection omits is checked by nothing.
            // `descriptor_static_use` cannot close it either — it answers
            // `NotDeclared` for anything that is not a `UniformConstant`, which
            // by construction excludes every storage buffer.
            frag_declared: frag_declared_bindings.len() as u32,
            // What the engine will build the layout from: the storage binds this
            // draw resolved, at the numbers they will carry, plus the textures
            // and samplers it provided at theirs.
            frag_provided: frag_layout_bindings.len() as u32,
            frag_gap: frag_gap.0,
            frag_gap_lo: frag_gap.1,
        });
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
        resources.vert_spirv = v_words;
        resources.frag_spirv = f_words;
        resources.width = w;
        resources.height = h;
        resources.vertex_count = vertex_count;
        if let Some(c0) = req.colors.first() {
            resources.color_attachment_format = Some(
                translate::pixel::color_attachment(c0.format)
                    .map_err(|reason| {
                        DrawError::Unsupported(
                            crate::backend::vulkan::engine::reason::DrawReason::ColorAttachmentFormat(reason),
                        )
                    })?
                    .0,
            );
        }
        // Attachment-count census, taken before the MRT gate rather than inside
        // it. `build_secondary_targets` returns empty for a single-attachment
        // draw without emitting, and every MRT counter below it therefore reads
        // zero whether the guest issues no MRT draw at all or issues them and
        // the producer drops them. Those are different facts and the log could
        // not tell them apart. `mrt_draw_single` is the denominator that proves
        // this probe runs.
        //
        // **It is still not the earliest sampling point, and reading it as one
        // is wrong.** `req.colors` is what `mrt_draw_request` *built*, not what
        // the guest declared: that builder skips an attachment whose geometry
        // differs from the first, so a two-attachment pass can arrive here with
        // one colour and be counted `mrt_draw_single`. The counters that close
        // the gap are `mrt_slot_attached` (what the guest declared) against
        // `mrt_slot_empty` and `mrt_slot_geometry_dropped` beside it. Compare
        // `mrt_slot_attached` with `mrt_draw_single + mrt_draw_multi` before
        // concluding a workload issues no MRT.
        //
        // With that comparison available, a driven x86/PCI/Vulkan boot — Safari
        // window drag plus System Settings and Spotlight, the vibrancy-bearing
        // panes the `secondary_mrt_drop` census names as its driving case —
        // reads `mrt_slot_attached=23112`, `mrt_slot_empty=0`,
        // `mrt_slot_geometry_dropped=0`, `mrt_draw_single=70332` and no
        // `mrt_draw_multi` at all. So this workload declares exactly one colour
        // attachment per pass and the MRT rails below are unexercised rather
        // than failing. Every `mrt_drop_*` reason reading zero is a statement
        // about the workload, not about the producer.
        crate::runtime::drain::note_store_route(if req.colors.len() > 1 {
            "mrt_draw_multi"
        } else {
            "mrt_draw_single"
        });
        // True MRT: render every color attachment (slot 1.. as engine secondary
        // residents) instead of dropping the shader's secondary outputs. Gated
        // on a resident primary; an `Ok(empty)` is the guest's own single-RT
        // draw and is byte-identical to the classic path.
        if let Some(primary_id) = resources.target_identity.clone() {
            let secs = build_secondary_targets(
                state,
                host,
                req.task_id,
                &req.colors,
                pd,
                &primary_id,
                w,
                h,
                req.blend_color.unwrap_or([0.0; 4]),
            );
            // Second half of the census: `built` vs `refused` separates "the
            // guest issued an MRT draw and we render every attachment" from
            // "it issued one and an attachment could not be built", which the
            // `mrt_drop_*` reasons alone cannot say because the whole feature
            // is silent when no MRT draw arrives. Counted before the refusal
            // returns, so the denominator survives the early exit.
            if req.colors.len() > 1 {
                crate::runtime::drain::note_store_route(if secs.is_err() {
                    "mrt_secondary_refused"
                } else {
                    "mrt_secondary_built"
                });
            }
            // A secondary attachment this device cannot build refuses the draw.
            // Executing it against slot 0 alone would render a frame whose
            // `location` 1.. outputs went nowhere, and nothing downstream — not
            // the guest, not this log — could tell that from a draw the guest
            // had only ever asked one target for.
            resources.secondary_targets = secs.map_err(|refusal| {
                DrawError::DrawPreparation(
                    crate::backend::vulkan::engine::DrawPreparationDecline::SecondaryTargetUnbuildable {
                        pipeline_ref: req.pipeline_ref,
                        refusal,
                    },
                )
            })?;
        }
        // The engine's own typed `DrawError` (a `vk_*` VkCall slug, a
        // `DrawReason` refusal, an interim `_untyped`) propagates unchanged so
        // the boundary below names the engine's specific check as the primary
        // `reason=` rather than flattening it into a `vk_engine: {e}` blob.
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Engine);
        let out = crate::backend::vulkan::engine::execute_draw_request(&resources)?;
        // Carried back on the request so `runtime::exec` can sum the chain's
        // draws into the guest's buffer. The engine reports per draw because a
        // Metal pass whose counter spans several draws is several Vulkan
        // queries; the sum belongs to whoever knows the offset, which is not
        // this arm.
        req.visibility_samples = out.occlusion_samples;
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
        if let Some(identity) = gva_resident_store {
            return Ok(M2vDrawSpan::ResidentGvaStore { identity });
        }
        if let Some(identity) = surface_resident_store {
            return Ok(M2vDrawSpan::ResidentSurfaceStore {
                identity,
                guest_store: GuestStoreStatus {
                    guest_backed: out.target_guest_backed,
                    recorded: out.guest_store_recorded,
                    footprint: out.guest_store_footprint,
                },
            });
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
/// The registry resident this draw's **depth** attachment renders into, if the
/// guest's pass descriptor named a depth texture.
///
/// The depth buffer is a guest resource with a guest lifetime. A pass descriptor
/// binds `MTLRenderPassDepthAttachment.texture`, this device decodes its ref, and
/// that ref is the identity — so one resident exists per guest depth texture and
/// survives for as long as the guest keeps the texture, instead of being
/// allocated and destroyed inside one draw.
///
/// # Why the generation is zero
///
/// Every other identity carries a generation because its resident holds content
/// that must not survive the guest reusing the key — a surface's
/// `map_generation` is the worked example. This one carries none, and the
/// argument is no longer the one an earlier version of this doc gave.
///
/// That version said the contents did not matter because the pass always
/// CLEARed, and that enabling depth LOAD would need a real per-texture
/// generation first. The first half was true and is now false: LOAD is honoured
/// (`DepthState::load` carries the guest's `loadAction`), so the contents are
/// load-bearing. The second half does not follow, and the reason is Metal's own
/// contract rather than anything this device arranges.
///
/// A texture ref names one live texture. The only way a resident can outlive
/// what it was created for is the guest destroying that texture and creating
/// another at the same ref, the same geometry and the same aspect — and a
/// **newly created `MTLTexture`'s contents are undefined until something writes
/// them**. So a pass that loads from one is reading undefined data by the
/// guest's own choice, and handing it a previous texture's depth is a
/// conformant answer to it. There is no case where a generation would turn a
/// wrong frame into a right one; it would only replace one undefined value with
/// a different undefined value.
///
/// **What this does not license is extending the same reasoning to colour.** A
/// colour target's contents are read back to guest pages and presented, so a
/// stale one is a wrong frame the guest can see rather than a value it declared
/// it did not care about. That is why the surface rail has a generation and this
/// one does not, and the difference is the readback, not the depth.
///
/// Geometry and aspect changes still recreate the image, through
/// `ResidentTargetSlot::reusable_for` and the `stencil` field of the key.
pub(super) fn depth_chain_identity(
    req: &DrawEncodeRequest,
    with_stencil: bool,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    let depth = req.depth_attach.as_ref()?;
    if depth.texture_ref == 0 {
        return None;
    }
    let c0 = req.colors.first()?;
    let (width, height) = (c0.width, c0.height);
    if width == 0 || height == 0 {
        return None;
    }
    Some(crate::backend::vulkan::engine::TargetIdentity::Texture {
        ref_: depth.texture_ref,
        width,
        height,
        generation: 0,
        stencil: with_stencil,
    })
}

pub(crate) fn render_chain_identity(
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
    if c0.mapping_id == 0
        || !crate::contract::pass_action::store_action_publishes_single_sample(c0.store_action)
    {
        return None;
    }
    render_chain_identity(state, req)
}

/// Stable shared allocation behind the type-11 primary attachment, when the
/// mapping's own view is one the engine may retain.
///
/// That question belongs to `ensure_contig_import_with_footprint`, which asks it
/// of the view this mapping holds rather than of the device — so there is no
/// pre-gate here. There was, and because it was a device-wide flag it took this
/// whole rail off the arm64 pathway for views that were guest RAM all along.
///
/// The mapping revalidation inside `ensure_contig_view` is part of the answer:
/// it retires an alias when the guest has recycled any of its pages, and that
/// advances the generation carried by `type11_render_identity`. The engine is
/// therefore handed a pointer and an identity derived from the same current
/// page ownership, never a cached pointer paired with a newly rewired surface.
fn type11_guest_target_backing<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    req: &DrawEncodeRequest,
) -> Option<crate::backend::vulkan::engine::GuestTargetMemory> {
    let c0 = req.colors.first()?;
    type11_render_identity(state, req)?;
    let (plane_offset, row_pitch, span_end) = {
        let mapping = state.mappings.get(&c0.mapping_id)?;
        let format = crate::runtime::mapping_write::mapping_store_format(mapping);
        crate::runtime::mapping_write::type11_sample_window(mapping, c0.width, c0.height, format)?
    };
    let (import, footprint) =
        crate::runtime::mapper::ensure_contig_import_with_footprint(state, host, c0.mapping_id)?;
    if span_end > import.len() {
        return None;
    }
    Some(crate::backend::vulkan::engine::GuestTargetMemory {
        backing: crate::backend::vulkan::engine::GuestTargetBacking {
            allocation_host_ptr: import.host_base(),
            allocation_len: import.len(),
            plane_offset,
            row_pitch: u64::from(row_pitch),
        },
        import,
        footprint,
    })
}

/// Whether this record's color0 LOAD is one the resident could serve at all —
/// it must be a LOAD, and no explicit seed may already have been selected for
/// it by RT provenance. Separate from the currency question so the two counters
/// on the branch below divide candidates, not all draws.
fn type11_load_is_a_seed_candidate(c0: &ColorRtRequest) -> bool {
    c0.load_action == MTL_LOAD_ACTION_LOAD && c0.target_seed_rgba.is_none()
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

/// Has the guest written this surface's pages since the Store that produced a
/// copied resident stamped them?
///
/// The device-side half of the currency test — the `surface_content_epoch`
/// comparison one function up — can only witness writers inside this crate.
/// A type-11 surface's pages are plain guest RAM, and the guest CPU stores
/// into them with no device operation, so the epoch does not move and the
/// resident silently stops matching what the seed would upload. This is the
/// only witness for that copied-allocation question, and it is the hypervisor's.
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
        crate::runtime::mapping_write::type11_sample_window(m, width, height, format)
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
///
/// `#[must_use]` because that is the whole contract: dropping the result leaves
/// the caller falling through to rungs that read pages holding only the guest's
/// half, and for a composite the Store deliberately left GPU-side those pages
/// were never written at all.
#[must_use]
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
/// `t11rung_resident_refused` counts binds where a ready resident existed and a
/// guest write to its pages sent the bind to the guest's pages instead — the
/// `guest_replaced` gate in [`resolve_sampled_source`]. It is the direct measure
/// of how much wrong content that rung used to serve.
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

fn guest_allocation_sample_is_direct(
    backing: crate::backend::vulkan::engine::ResidentContentBacking,
    may_bind_resident: bool,
) -> bool {
    backing == crate::backend::vulkan::engine::ResidentContentBacking::GuestAllocation
        && may_bind_resident
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

/// Decide whether a type-11 LOAD can use its retained target.
///
/// A guest allocation is the serialized texture's own storage, not a cached
/// copy, so a host or guest write changes the allocation the next render pass
/// loads. The copied-allocation currency query is supplied as a closure so the
/// shared case cannot accidentally pay either engine lookup or guest-write
/// lookup on a warm LOAD.
fn type11_load_resident_is_current(
    backing: Option<crate::backend::vulkan::engine::ResidentContentBacking>,
    copied_currency: impl FnOnce() -> (bool, bool),
) -> (bool, bool) {
    if backing == Some(crate::backend::vulkan::engine::ResidentContentBacking::GuestAllocation) {
        (true, false)
    } else {
        copied_currency()
    }
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
        // Both callers reach here only on a route that has already put these
        // pixels outside the image — the synchronous one through `write_bgra8`
        // into the mapping's guest pages, the deferred-readback one through
        // `publish_surface_store` into `surface_cache` plus the window's own
        // `Arc`. So the resident is no longer the sole copy and the reclaim
        // paths may take it, which is the whole reason those routes pay a
        // readback.
        //
        // Deliberately not on `arm_surface_resident_store`: that route skips the
        // readback precisely so no copy is made, keeps the frame in the image
        // alone, and holds it with a pin instead.
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
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
pub(crate) fn gva_page_set_hash(pages: &std::collections::HashSet<u64>) -> u64 {
    let mut hash: u64 = 0;
    for p in pages {
        hash ^= p.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    hash
}

/// Host-texture identity of this draw's color0 GVA resource.
///
/// A task-local texture reference keeps one generation from creation through
/// ordinary task map changes and transfer-backing discard. Explicit resource
/// delete ends that lifetime. A target with no resource reference falls back to
/// the page-set identity because it has no protocol lifetime to name instead.
fn gva_alloc_generation<M: HostMemory + HostOps>(
    state: &mut DeviceState,
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
    if c0.texture_ref != 0 {
        crate::runtime::writeback_debt::gva_resource_generation(
            state,
            host,
            crate::runtime::writeback_debt::GvaResourceKey {
                task_id: req.task_id,
                texture_ref: c0.texture_ref,
            },
            c0.target_gva,
            u64::from(c0.row_stride).saturating_mul(u64::from(c0.height)),
        )
    } else {
        gva_span_alloc_generation(
            state,
            host,
            req.task_id,
            c0.target_gva,
            c0.row_stride,
            c0.height,
        )
    }
}

/// The page-set generation of one `row_stride * height` GVA span under one task.
///
/// Every GVA render target is named this way, not only color0: the secondary MRT
/// attachments go through here too ([`build_secondary_targets`]). One spelling,
/// so no two callers can disagree about what names an allocation.
///
/// A short walk yields 0. An incomplete walk names no allocation, and hashing the
/// pages that happened to resolve would be an identity the guest never had.
pub(crate) fn gva_span_alloc_generation<M: HostMemory + HostOps>(
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
    if (pages.len() as u64) < reims_vgpu_paging::span::pages_spanned(gva, span, state.page_size()) {
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
        format: gva_resident_format(c0.format),
    })
}

/// The format the resident behind a GVA render target must hold: the one the
/// guest declared for that attachment.
///
/// This is [`crate::backend::vulkan::engine::TargetIdentity::resident_format`]'s
/// rule applied to the one namespace that has a declaration to follow, and it is
/// a function rather than an expression at each producer because the producers
/// key *the same registry slot*. A primary and a secondary attachment that
/// spelled it differently would render into one identity claiming two formats,
/// which `registry_ensure` answers by recreating the image every frame.
///
/// The guest's declaration is followed whenever the host can follow it. A
/// layout this device has no `TexelLayout` for, or one the host cannot render
/// to and blend into, falls back to the engine's neutral resident colour
/// format — which is what *every* render target got while the key held a
/// `bgra: bool` and could express nothing else.
///
/// The fallback is a fidelity loss and not a refusal. The draw still runs, and
/// the Store still lands correctly-shaped bytes for the guest's declared texel,
/// because [`crate::contract::pixel_format::convert_rgba8_to_row`] expands them
/// from eight bits — the guest reads a well-formed half-float frame carrying
/// eight bits of information. What the fallback costs is the range and the
/// precision the guest asked for: anything above 1.0 in a half-float
/// compositing target is clamped away before the compositor sees it.
///
/// Capability, never an API-version assumption.
/// `VK_FORMAT_R16G16B16A16_SFLOAT` is in Vulkan's mandatory format table for
/// both `COLOR_ATTACHMENT` and `COLOR_ATTACHMENT_BLEND`, and it is still asked
/// for per host: a widening that reads the spec's table instead of the device
/// is the shape AGENTS.md names, and this one would fail at `vkCreateImage`
/// rather than decline.
pub(crate) fn gva_resident_format(format: u16) -> ash::vk::Format {
    use crate::backend::vulkan::translate::pixel;
    // The declaration the *attachment* will be built from, folded onto its
    // allocation family, which is what `registry_ensure` creates the image in.
    //
    // Asked of `color_attachment` and not of `store_texel_order`, which is what
    // this used to ask and which is a different question — that one says whether
    // a resident's texels can be byte-copied into guest pages, and it answers
    // for three formats where `render_target_bpp` admits six. A guest render
    // target declared `R8Unorm`, `R16Float` or `RG16Float` therefore built its
    // image at one to four bytes a texel through `color_attachment` while its
    // identity claimed `RESIDENT_RGBA_FORMAT` and four, so everything keyed off
    // the identity — `bytes_per_texel`, the readback size, `stored_bytes_agree`
    // — read the wrong width for the image it was describing. `R16Float` and
    // `RG16Float` are both in the guest's vocabulary on boots on record.
    let Ok((attachment, _)) = pixel::color_attachment(format) else {
        return pixel::RESIDENT_RGBA_FORMAT;
    };
    let allocation = pixel::ResidentFormat::of(attachment).allocation();
    match pixel::texel_layout_of(allocation) {
        // Capability, never an API-version assumption: the host is asked whether
        // it renders to and blends this layout.
        Some(layout)
            if crate::backend::vulkan::engine::render_target_layout_supported(layout) =>
        {
            allocation
        }
        _ => pixel::RESIDENT_RGBA_FORMAT,
    }
}

/// Read a resident render-pass chain back to host memory so the exec loop can
/// land its pixels. Every failure is fail-visible; the guest keeps its pre-pass
/// bytes on loss.
///
/// # Every caller is now a refusal path
///
/// This used to be the ordinary Store of a GVA-targeted render, and the doc here
/// used to say so — 13 653 reads on a driven boot, 59 % of every render Store,
/// each one submitting and then blocking on a fence. That is no longer the
/// shape, and reading it as the hot path sends the next reader at a premise that
/// has already been fixed.
///
/// Both Store rails normally leave the frame host-authoritative. Their payments
/// use the GPU-direct guest-page copies in `render_writeback`; this function is
/// what either Store falls back to when it cannot arm or materialize that copy,
/// plus `writeback_chain_rgba`. The ordinary path therefore has no framebuffer
/// readback or CPU conversion at Store time.
///
/// So this is genuinely the abandon path, and it is a cost rather than a lost
/// frame — but it is the expensive one, and the reason to keep it narrow: it
/// reads the whole framebuffer back across the bus and blocks on a fence to do
/// it. A change that pushes traffic back onto it will not show up as a refusal,
/// only as `slot_us` and the fail-visible decline that sent it here.
///
/// What made the GPU-direct arm reachable was format. A buffer→image copy
/// converts nothing, so the resident has to already hold the bytes its
/// destination stores; the order used to be derived from the identity's *kind*,
/// which made every GVA resident RGBA and every Store a per-row conversion.
/// `TargetIdentity::Gva` now carries the order the guest declared for that
/// render target — see `is_bgra` on
/// [`crate::backend::vulkan::engine::TargetIdentity`] for why it is keyed on the
/// identity and what a change there has to keep true.
///
/// The identity is the caller's, never re-derived here: both callers hold the
/// key their own draw registered, carried out of `M2vDrawSpan`, and
/// `render_chain_identity` asked again after the draw can answer at a newer
/// mapping generation than the one the registry holds. See
/// [`M2vDrawSpan::ResidentSurfaceStore`].
pub(crate) fn read_resident_chain(
    req: &DrawEncodeRequest,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> Option<Vec<u8>> {
    match crate::backend::vulkan::engine::read_target(identity) {
        // Every caller of this function — `writeback_chain_rgba` and the GVA
        // arm-refusal fallback — has an RGBA contract, so the exchange happens
        // here, once, rather than at three call sites that would each have to
        // remember which namespace they were reading. `into_rgba8` uses the order
        // the engine reports for the image it copied, so it is a no-op on a
        // pooled resident and a whole-frame pass on anything the guest declared
        // in BGRA order — a surface, and a GVA target whose declared format
        // `gva_resident_format` could honour, which is most of them.
        //
        // That last clause used to read "a no-op on the pooled and GVA
        // residents", and it was wrong rather than imprecise: a GVA render
        // target declared `BGRA8Unorm` or `BGRA8Unorm_sRGB` is resident in BGRA
        // and owes the exchange. It was a true description of what the code did
        // — `ResidentReadSnapshot::bgra` answered "not BGRA" for the sRGB
        // spelling — so the comment documented the defect instead of catching
        // it, and the Store below exchanged R and B on its way into guest pages.
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
/// Land a type-11 render Store's frame in the guest's pages, from the resident
/// the draw just rendered into.
///
/// The Store never reads the frame back off the GPU: the copy's destination is
/// the guest's own pages, recorded into the engine's command stream and ordered
/// against the guest by the completion stamp. See
/// [`crate::runtime::render_writeback`] for what this replaced, and why the
/// deferred window it used to arm bought nothing on any measured workload.
///
/// `false` means the frame is not in the guest's pages and the caller must
/// materialize it the slow way — read the resident back and run the synchronous
/// Store block. That is a cost, never a lost frame.
///
/// The identity is the caller's, carried out of the draw on
/// [`M2vDrawSpan::ResidentSurfaceStore`], so the image read here is the image
/// the draw rendered into by construction.
///
/// It used to be [`type11_store_identity`] called again here, on the argument
/// that this is the same function that produced the draw's `target_identity`.
/// The same function is not the same value: that identity carries
/// `MappingEntry::map_generation`, and the draw mutates `DeviceState` between
/// the two calls. Read that variant's doc before reintroducing a derivation
/// anywhere on this path — deriving it a second *time* is as wrong as deriving
/// it a second *way*.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SurfaceStorePlan {
    SynchronizeGuestBacking,
    DeferCopy,
    CopyNow,
}

fn surface_store_plan(lazy_enabled: bool, guest_backed: bool) -> SurfaceStorePlan {
    if guest_backed {
        SurfaceStorePlan::SynchronizeGuestBacking
    } else if lazy_enabled {
        SurfaceStorePlan::DeferCopy
    } else {
        SurfaceStorePlan::CopyNow
    }
}

fn store_surface_resident<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    guest_store: GuestStoreStatus,
) -> bool {
    // The union belongs to the draws that just ran whether or not the write
    // below succeeds, and leaving it un-reset on a refused write would fold this
    // pass into the next Store's reading.
    note_pass_scissor_union(width, height);
    // A copied resident may defer the transfer until something reads the
    // mapping. A guest-backed resident may not: its Store is already the draw
    // into that allocation, so deferring creates an invented second operation
    // and makes correctness depend on the texture identity outliving the
    // command that synchronized it. Publish the existing alias eagerly and let
    // the completion stamp carry its queue ordering.
    let lazy = crate::runtime::writeback_debt::lazy_writeback_enabled();
    let plan = surface_store_plan(lazy, guest_store.guest_backed);
    if plan == SurfaceStorePlan::DeferCopy
        && arm_surface_writeback_debt(state, host, mapping_id, identity, width, height)
    {
        crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
        return true;
    }
    if plan == SurfaceStorePlan::SynchronizeGuestBacking {
        if lazy {
            crate::runtime::drain::note_store_route("target_store_shared_eager");
        }
        match crate::runtime::render_writeback::store_guest_backed_frame(
            state,
            mapping_id,
            identity,
            width,
            height,
            guest_store.recorded,
            guest_store.footprint,
        ) {
            Ok(()) => return true,
            Err(decline) => {
                crate::observe::Emit::decline("target_store_shared_declined", &decline)
                    .field("mapping", mapping_id)
                    .field("geom", format!("{width}x{height}"))
                    .fail_once(u64::from(mapping_id));
                crate::runtime::drain::note_store_route("target_store_shared_declined");
            }
        }
    }
    if !crate::runtime::render_writeback::store_render_frame(
        state, host, mapping_id, identity, width, height,
    ) {
        return false;
    }
    // The guest half of the write witness, recorded here rather than by the
    // caller because this rail returns straight out of `encode_draw` — it never
    // reaches `stamp_type11_resident`, and it is where nearly all type-11 Stores
    // go.
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

/// Record that `mapping_id` is owed this frame, and hand the currency witness to
/// the resident holding it.
///
/// `true` when the debt is armed and the caller owes the guest nothing further
/// this Store. `false` sends the caller down the ordinary eager Store, which is
/// what a mapping this device holds no entry for gets.
///
/// # Why the resident is stamped here, where no copy has happened
///
/// The stamp says the resident holds the mapping's content, and it is what
/// licenses the type-11 attachment LOAD to seed from that image instead of
/// reading a whole frame back out of guest memory — 802 elided against 36
/// uploaded on a driven boot. `registry_mark_ready` clears it on every draw that
/// renders into the resident, so a Store that did not re-stamp would hand the
/// next LOAD a refusal, the LOAD would read the guest's pages, and reading them
/// is exactly what pays the debt. The rail would collapse to the eager one with
/// extra bookkeeping.
///
/// It is sound for that consumer and it is the *only* consumer: the resident
/// holds this surface's newest pixels, which is what a LOAD asks for. It would
/// not be sound for a writeback that read the stamp to decide it could skip the
/// copy — with a debt outstanding the pages hold something older, not something
/// equal. `render_writeback`'s doc measures that elision as never once firing
/// and says not to build it; whoever revisits that has to read this first.
fn arm_surface_writeback_debt<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let Some(map_generation) = state.mappings.get(&mapping_id).map(|m| m.map_generation) else {
        return false;
    };
    // The host surface cache is the *other* host-side copy of this mapping, and
    // this Store has just superseded it without writing a byte anywhere. The
    // eager arm ends with `surface_cache::forget` for exactly this reason; this
    // arm cedes instead, because here the resident genuinely does hold the frame
    // and a cession says so — see [`surface_cache::cede_surface_to_resident`],
    // which was written for this call and had no caller.
    //
    // Without it the type-11 sampled ladder's host-cache rung serves the
    // *previous* frame for as long as the debt is outstanding: that rung is
    // gated on the guest-write witness alone, which by construction cannot see a
    // publish this device made itself, and it sits above both rungs that read
    // the guest's own pages, so nothing below corrects it. It repairs when the
    // debt is paid, which is what makes it a flicker rather than a stuck layer.
    //
    // A geometry this cache would not have stored is refused rather than armed,
    // per that function's contract: leaving a live entry beside a
    // resident-authoritative window is the state this whole call exists to
    // prevent, and the eager Store is always available.
    if !crate::runtime::surface_cache::cede_surface_to_resident(state, mapping_id, width, height) {
        crate::runtime::drain::note_store_route("wbdebt_uncedable_geometry");
        return false;
    }
    // The *third* host-side claim on this window, and the one this arm was
    // missing. `compute_storage_residency` records that a storage image and the
    // guest's pages both hold a window a compute dispatch wrote back, so
    // `compute_exec::stage_texture_raw` may serve the storage image and skip the
    // guest read entirely. This Store has just superseded both halves of that
    // claim without writing a byte, so the next dispatch staging the same window
    // would be fed the earlier dispatch's image and never see the render frame.
    //
    // The eager arm has always done this — `write_bgra8_from_resident_gpu` calls
    // `invalidate_storage_residency_window` over the same extent — so without it
    // here the two arms of `env::LAZY_WRITEBACK` disagree about what the GPU
    // observes, which is the one thing that switch's doc promises they never do.
    // Nothing else drops an entry from that map and no guest-write witness feeds
    // it, so a stale claim is held until the window is written some other way.
    if let Some(m) = state.mappings.get(&mapping_id) {
        let format = if m.format != 0 {
            m.format
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        if let Some((base_off, _bpr, span_end)) =
            crate::runtime::mapping_write::type11_sample_window(m, width, height, format)
        {
            state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
        }
    }
    // The one call that says "these pixels changed and the guest's pages do not
    // hold them yet". It advances the surface's content epoch, which is what the
    // stamp below records, and `ResourceValidity::host_published_seq`, which is
    // what orders this frame against the guest's own later claim to have written
    // the same pages — `writeback_debt::pay` reads that ordering back through
    // `resource_validity::licence_of` and abandons a frame the guest superseded.
    let epoch = state.note_surface_content_published(mapping_id);
    crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch);
    // Armed before the eviction is paid, so the ledger never holds two debts for
    // one mapping and the payment below cannot be the one just armed.
    let evicted = state
        .pending_writebacks
        .arm(mapping_id, identity.clone(), width, height, map_generation);
    if let Some(evicted) = evicted {
        crate::runtime::drain::note_store_route("wbdebt_evicted");
        if !crate::runtime::writeback_debt::pay_key(state, host, evicted) {
            let armed = state.pending_writebacks.take(mapping_id);
            debug_assert!(armed.is_some());
            crate::runtime::drain::note_store_route("wbdebt_capacity_fallback");
            return false;
        }
    }
    true
}

/// Readback-skip gate for the final/single record of a GVA render Store: the
/// record may leave its pixels on the engine registry resident and let the
/// caller read them back once, instead of taking a readback plus a fence wait
/// inside the record. All gates are protocol-shape checks (never content): the
/// caller must be able to replay the sync `write_gva_rgba8` exactly — identity
/// geometry == c0 geometry, convertible format, sane BPR.
fn gva_store_defer_eligible(req: &DrawEncodeRequest) -> bool {
    let Some(c0) = req.colors.first() else {
        return false;
    };
    if c0.mapping_id != 0
        || c0.target_gva == 0
        || c0.row_stride == 0
        || c0.texture_ref == 0
        || req.gva_alloc_gen == 0
    {
        return false;
    }
    let Some(identity) = gva_chain_identity(req) else {
        return false;
    };
    if identity.width() != c0.width || identity.height() != c0.height {
        return false;
    }
    pixel_format::tight_row_bytes(c0.width, c0.format)
        .and_then(|bytes| bytes.checked_mul(c0.sample_count.max(1)))
        .is_some_and(|tight| c0.row_stride >= tight)
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod vulkan_split_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn a_sampled_guest_allocation_does_not_enter_copy_currency_checks() {
        use crate::backend::vulkan::engine::ResidentContentBacking;

        assert!(guest_allocation_sample_is_direct(
            ResidentContentBacking::GuestAllocation,
            true
        ));
        assert!(!guest_allocation_sample_is_direct(
            ResidentContentBacking::DeviceAllocation,
            true
        ));
        assert!(!guest_allocation_sample_is_direct(
            ResidentContentBacking::NotReady,
            true
        ));
        assert!(!guest_allocation_sample_is_direct(
            ResidentContentBacking::GuestAllocation,
            false
        ));
    }

    #[test]
    fn a_guest_backed_store_is_never_turned_into_a_future_copy() {
        assert_eq!(
            surface_store_plan(true, true),
            SurfaceStorePlan::SynchronizeGuestBacking
        );
        assert_eq!(
            surface_store_plan(false, true),
            SurfaceStorePlan::SynchronizeGuestBacking
        );
        assert_eq!(surface_store_plan(true, false), SurfaceStorePlan::DeferCopy);
        assert_eq!(surface_store_plan(false, false), SurfaceStorePlan::CopyNow);
    }

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
            true,
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
            SampledByteFormat::synthesised(TexelLayout::Rgba8),
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
        note_guest_rung_blank(&state, &host, 1, 9, (gva, w, h), &blank, SampledByteFormat::synthesised(TexelLayout::Rgba8));
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
            SampledByteFormat::synthesised(TexelLayout::Rgba8),
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
        note_draw_coverage(
            ScissorRect {
                x: 0,
                y: 0,
                width: 200,
                height: 1000,
            },
            1000,
            1000,
            None,
            false,
            false,
        );
        note_draw_coverage(
            ScissorRect {
                x: 400,
                y: 0,
                width: 200,
                height: 1000,
            },
            1000,
            1000,
            None,
            false,
            false,
        );
        note_pass_scissor_union(1000, 1000);
        assert_eq!(
            store_route_count("pass_scissor_union_le99"),
            n + 1,
            "the union of two 20 % strips 400px apart is 60 %, not 20 %"
        );

        // The reset landed: a fresh single 4 % draw reads as 4 %, not 64 %.
        let n = store_route_count("pass_scissor_union_le5");
        note_draw_coverage(
            ScissorRect {
                x: 0,
                y: 0,
                width: 200,
                height: 200,
            },
            1000,
            1000,
            None,
            false,
            false,
        );
        note_pass_scissor_union(1000, 1000);
        assert_eq!(
            store_route_count("pass_scissor_union_le5"),
            n + 1,
            "the arm must reset the union, or it saturates across passes"
        );

        // A full-coverage draw takes the per-draw early return but must still
        // reach the union - it is the case that makes a pass unbounded.
        let n = store_route_count("pass_scissor_union_full");
        note_draw_coverage(
            ScissorRect {
                x: 0,
                y: 0,
                width: 40,
                height: 40,
            },
            1000,
            1000,
            None,
            false,
            false,
        );
        note_draw_coverage(
            ScissorRect {
                x: 0,
                y: 0,
                width: 1000,
                height: 1000,
            },
            1000,
            1000,
            None,
            false,
            false,
        );
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
            note_draw_coverage(
                ScissorRect {
                    x: 0,
                    y: 0,
                    width: sw,
                    height: sh,
                },
                1000,
                1000,
                None,
                false,
                false,
            );
            assert_eq!(
                store_route_count(slug),
                before + 1,
                "{sw}x{sh} of 1000x1000 belongs in {slug}"
            );
        }

        // The same rect against a target it fully covers is not a partial draw
        // at all, so it takes the `covers` early return and reaches no bucket.
        let before = store_route_count("draw_scissor_area_gt50");
        note_draw_coverage(
            ScissorRect {
                x: 0,
                y: 0,
                width: 800,
                height: 800,
            },
            800,
            800,
            None,
            false,
            false,
        );
        assert_eq!(
            store_route_count("draw_scissor_area_gt50"),
            before,
            "a full-coverage draw is counted by draw_scissor_full, not bucketed"
        );

        // A degenerate target must not divide by zero.
        note_draw_coverage(
            ScissorRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            0,
            0,
            None,
            false,
            false,
        );
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
            true,
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
            true,
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
            SampledByteFormat::synthesised(TexelLayout::Rgba8),
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
            SampledByteFormat::synthesised(TexelLayout::Rgba8),
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

    /// A GVA render target is named by its live task-local resource, whose host
    /// texture survives task virtual-memory map changes. Delete ends that
    /// lifetime; reusing the integer afterwards must mint another identity.
    #[test]
    fn a_gva_targets_identity_follows_resource_lifetime() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        state.define_task(1, 0x1_0000, 2);

        let mut req = one_page_gva_request();
        let gen_a = super::gva_alloc_generation(&mut state, &mut host, &req);
        assert_ne!(gen_a, 0, "a fully walked GVA span must name its allocation");
        req.gva_alloc_gen = gen_a;
        let id_a = super::gva_chain_identity(&req).expect("a GVA color0 has a chain identity");

        // The same buffer rendered again: same pages, so the same resident.
        assert_eq!(
            super::gva_alloc_generation(&mut state, &mut host, &req),
            gen_a,
            "an unchanged mapping must not mint a second identity"
        );

        // Ordinary virtual-memory remapping does not retarget the live resource.
        map_one_gva_page(&mut host, 5);
        let gen_b = super::gva_alloc_generation(&mut state, &mut host, &req);
        assert_eq!(gen_b, gen_a);
        req.gva_alloc_gen = gen_b;
        let id_b = super::gva_chain_identity(&req).expect("a GVA color0 has a chain identity");
        assert_eq!(id_a, id_b);

        assert!(crate::runtime::writeback_debt::retire_gva_resource(
            &mut state, 1, 7
        ));
        let gen_c = super::gva_alloc_generation(&mut state, &mut host, &req);
        assert_ne!(gen_c, gen_a, "delete ends the host texture's lifetime");
        assert_eq!(
            (id_a.width(), id_a.height()),
            (id_b.width(), id_b.height()),
            "and the generation must be the only thing that separates them"
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
            format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
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

    #[test]
    fn a_guest_allocation_load_never_queries_copy_currency() {
        use crate::backend::vulkan::engine::ResidentContentBacking;

        assert_eq!(
            type11_load_resident_is_current(
                Some(ResidentContentBacking::GuestAllocation),
                || panic!("one allocation has no second copy whose currency can be queried"),
            ),
            (true, false)
        );
        for backing in [
            Some(ResidentContentBacking::DeviceAllocation),
            Some(ResidentContentBacking::NotReady),
            None,
        ] {
            for copied_answer in [(false, true), (true, false)] {
                assert_eq!(
                    type11_load_resident_is_current(backing, || copied_answer),
                    copied_answer,
                    "{backing:?} must retain the copied-allocation currency test"
                );
            }
        }
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
            load_action: MTL_LOAD_ACTION_LOAD,
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
            load_action: MTL_LOAD_ACTION_CLEAR,
            ..Default::default()
        };
        assert!(!type11_load_is_a_seed_candidate(&c0));
    }

    /// A type-11 `LOAD` whose host cache misses seeds from the surface's own
    /// guest pages, and only refuses when those cannot serve the extent.
    ///
    /// Without the guest-pages rung this returns `None`, `target_rgba8` stays
    /// unset, and `exec` resolves the pass load action to `Clear` against the
    /// hardcoded `[0,0,0,0]` — so the guest's request to preserve its surface
    /// became a transparent-black wipe that the matching Store published. One
    /// x86/Vulkan boot measured 121 distinct (mapping, geometry) instances of that
    /// in ~170 s, four at the full 1920x1080 composite extent, with the host
    /// window 62-90 % near-black during a desktop drag against 0.001 % at idle.
    ///
    /// Every one of those 121 lines had `want == mapgeom` and `hostgen=0`: the
    /// cache had never held the surface and its pages were readable. That pair is
    /// what makes reading them the fix rather than a guess.
    /// The lazy type-11 Store publishes a frame nothing has written down yet, so
    /// the host surface cache — the other host-side copy of the same mapping —
    /// must stop naming the frame before it.
    ///
    /// The eager Store's GPU-direct arm ends in `surface_cache::forget` and says
    /// why; the lazy arm published a strictly newer frame and left the entry
    /// alone. The consumer that would read it is the type-11 sampled ladder's
    /// host-cache rung, gated on the guest-write witness alone — which cannot
    /// see a publish this device made itself — and sitting above both rungs that
    /// read the guest's pages, so nothing below it corrects the answer.
    ///
    /// Asserted through `get_shared_with_gen`, which is the accessor that rung
    /// actually calls: a cession leaves the entry present and empty, so a test
    /// that only asked whether the map contained the key would pass either way.
    #[test]
    fn arming_a_writeback_debt_stops_the_host_cache_naming_the_previous_frame() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 912u32;
        let pfn = 0x31u32;
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

        // Frame N, mirrored into the host cache by the write itself.
        let frame_n = vec![0x11u8; (w * h * 4) as usize];
        assert!(write_bgra8(
            &mut state,
            &mut host,
            mid,
            &frame_n,
            w * 4,
            w,
            h
        ));
        assert!(
            crate::runtime::surface_cache::get_shared_with_gen(&state, mid, w, h).is_some(),
            "the cache has to be warm or this test proves nothing"
        );

        // Frame N+1, rendered and deliberately not written down.
        let identity = crate::backend::vulkan::engine::TargetIdentity::Gva {
            gva: gpa,
            width: w,
            height: h,
            generation: 1,
            format: gva_resident_format(MTL_FORMAT_BGRA8_UNORM),
        };
        // The third host-side claim on the same window: a compute dispatch's
        // storage image, recorded as holding what the guest's pages hold. This
        // Store supersedes both halves of that claim.
        let (base_off, bpr, span_end) = crate::runtime::mapping_write::type11_sample_window(
            state.mappings.get(&mid).expect("mapped above"),
            w,
            h,
            MTL_FORMAT_BGRA8_UNORM,
        )
        .expect("a latched geometry resolves its window");
        let residency = crate::model::ComputeStorageResidencyKey {
            mapping_id: mid,
            map_generation: state.mappings[&mid].map_generation,
            surface_offset: base_off,
            surface_bpr: bpr,
            span_end,
            width: w,
            height: h,
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            texture_ref: 0,
        };
        state
            .compute_storage_residency
            .insert(residency, Default::default());

        assert!(
            arm_surface_writeback_debt(&mut state, &mut host, mid, &identity, w, h),
            "a mapped surface at a cacheable geometry arms"
        );
        assert!(
            crate::runtime::surface_cache::get_shared_with_gen(&state, mid, w, h).is_none(),
            "frame N is still on offer while frame N+1 exists only on the GPU"
        );
        assert!(
            !state.compute_storage_residency.contains_key(&residency),
            "the eager arm invalidates this window and the lazy one published a \
             strictly newer frame into it; leaving the claim standing feeds the \
             next dispatch the earlier dispatch's storage image instead of the \
             render frame, and the two arms of LAZY_WRITEBACK then disagree \
             about what the GPU observes"
        );
    }

    /// The Store lands the frame in the slot the *draw* registered, even when
    /// the mapping's generation has moved since.
    ///
    /// `map_generation` is part of [`crate::backend::vulkan::engine::TargetIdentity::Surface`],
    /// so a Store that re-derives its identity from `DeviceState` asks the
    /// registry for a key one generation ahead of the one `registry_ensure`
    /// was handed, and the registry answers `read_target_unknown_identity
    /// diverges=generation asked_gen=N held_gen=N-1`. Every Maps frame went
    /// that way on `REIMS_VGPU_SHARED_TARGET=off`, which is the only
    /// render-target rail a host without `VK_EXT_external_memory_host` has.
    ///
    /// The repair is that the identity travels out of the draw on
    /// [`M2vDrawSpan::ResidentSurfaceStore`] rather than being asked for twice,
    /// so this test bumps the generation between the two points and asserts the
    /// debt still names the draw's key. Before it, the ledger recorded the
    /// generation this test bumps to.
    #[test]
    fn the_store_names_the_slot_the_draw_registered_after_the_mapping_generation_moves() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::mapping_write::write_bgra8;

        if !crate::runtime::writeback_debt::lazy_writeback_enabled() {
            // The eager arm stores through the engine instead of the ledger and
            // has no debt to inspect. Reported rather than silently passing.
            eprintln!("skipped: REIMS_VGPU_LAZY_WRITEBACK=off selects the eager Store");
            return;
        }

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 913u32;
        let pfn = 0x37u32;
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
        // Warms the host cache, so the cession inside the arm has something to
        // cede and the arm is testing its own gate rather than an empty one.
        assert!(write_bgra8(
            &mut state,
            &mut host,
            mid,
            &vec![0x22u8; (w * h * 4) as usize],
            w * 4,
            w,
            h
        ));

        // The key the draw would have handed `registry_ensure`.
        let drawn = crate::runtime::present_identity::surface_identity(&state, mid, w, h);
        crate::model::DeviceState::bump_map_generation(state.mappings.get_mut(&mid).unwrap());
        assert_ne!(
            crate::runtime::present_identity::surface_identity(&state, mid, w, h),
            drawn,
            "the bump has to change what a second derivation answers or this \
             test cannot see the defect"
        );

        assert!(
            store_surface_resident(
                &mut state,
                &mut host,
                &drawn,
                mid,
                w,
                h,
                GuestStoreStatus::default(),
            ),
            "a copied resident at a cacheable geometry defers"
        );
        assert_eq!(
            state
                .pending_writebacks
                .get(mid)
                .expect("the deferred plan arms a debt")
                .identity,
            drawn,
            "the ledger has to name the image the draw rendered into"
        );
    }

    /// The type-11 zero-copy sampled rail hands the engine the surface's guest
    /// pages, so an owed frame has to be written into them first.
    ///
    /// The rail is exempt from the *settle* and its comment says why: a
    /// submitted writeback is already ahead of this gather in queue order. That
    /// argument does not reach a writeback **debt**, which is not submitted at
    /// all — there is no command for queue order to order, and the pages hold
    /// the frame before the one the resident is holding. The exemption was
    /// written before the debt rail existed and silently took the payment with
    /// it.
    #[test]
    fn the_type11_zero_copy_gather_pays_the_frame_those_pages_owe() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        // Keep this large enough that either the direct resource or the copied
        // fallback can take; the test is about debt payment, not ownership.
        let (w, h) = (128u32, 128u32);
        let span = (w * h * 4) as u64;
        assert!(span >= SAMPLED_GATHER_MIN_BYTES);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        // The rail refuses a host whose page views are transient before it ever
        // builds one — see `type11_zero_copy_declines_transient_host_mappings`.
        let mid = 913u32;
        let first_pfn = 0x40u32;
        let pages = (span >> PAGE_SHIFT_X86) as u32;
        host.map_range((first_pfn as u64) << PAGE_SHIFT_X86, span as usize, 0);
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = (0..pages)
                .map(|i| ((first_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                .collect();
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
        let resource =
            crate::model::TaskResource::new(Default::default(), std::sync::Arc::from([]));

        let map_generation = state.mappings.get(&mid).unwrap().map_generation;
        assert!(
            state
                .pending_writebacks
                .arm(
                    mid,
                    crate::runtime::writeback_debt::test_resident_identity(
                        mid,
                        w,
                        h,
                        u64::from(map_generation),
                    ),
                    w,
                    h,
                    map_generation,
                )
                .is_none(),
            "an empty ledger evicts nobody"
        );

        assert!(
            try_type11_sample_zero_copy(&mut state, &mut host, mid, w, h, resource.lifetime_ref())
                .is_some(),
            "the rail has to take, or this test proves nothing about what it does"
        );
        assert!(
            state.pending_writebacks.get(mid).is_none(),
            "the gather bound pages still owed a frame this device never wrote down"
        );
    }

    /// A stage that binds a sampler and no texture is still in the sampled band,
    /// so it still triggers the band's relocation.
    ///
    /// Asking only about textures is what stood here, and Metal argument tables
    /// are sticky across draws in an encoder: a vertex sampler survives a re-bind
    /// that zeroed the vertex textures. Unseparated, both stages' sampler at
    /// index 0 resolves to `SAMPLER_BINDING_BASE`, `push_smp` takes the first
    /// writer, and the fragment module goes on sampling through the vertex
    /// stage's filter, address mode and LOD clamp with nothing refused.
    #[test]
    fn a_stage_that_binds_only_a_sampler_is_still_in_the_sampled_band() {
        let tex = |texture_ref| TextureBind {
            index: 0,
            texture_ref,
            ..Default::default()
        };
        let smp = |sampler_ref| SamplerBind {
            index: 0,
            sampler_ref,
            ..Default::default()
        };

        assert!(
            stage_uses_sampled_band(&[], &[smp(7)]),
            "a sampler with no texture occupies a binding number all the same"
        );
        assert!(stage_uses_sampled_band(&[tex(3)], &[]));
        assert!(stage_uses_sampled_band(&[tex(3)], &[smp(7)]));
        assert!(
            !stage_uses_sampled_band(&[], &[]),
            "a stage that binds nothing is not in the band"
        );
        assert!(
            !stage_uses_sampled_band(&[tex(0)], &[smp(0)]),
            "a zero ref is the guest leaving a slot empty, not a bind — the same \
             reading `push_smp`'s callers take"
        );
    }

    /// The `LOAD` seed's host-cache rung must refuse a surface the hypervisor
    /// watched the guest repaint, exactly as the sampled path's read of the same
    /// map does.
    ///
    /// The two are one cache with two readers, and only one of them asked. The
    /// damage is not a single wrong frame: this rung sits above the only rung
    /// that reads the guest's pages, so nothing below corrects it, and the pass
    /// that seeds from it composites onto the stale bytes and Stores them back
    /// over the pages the guest just wrote. The guest's repaint is then gone
    /// from both copies and the next frame loads what this one stored.
    #[test]
    fn the_type11_load_seed_cache_rung_refuses_a_surface_the_guest_rewrote() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::host::HostOps;
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        // Not shared with any other test's: `first_sight` latches per
        // `(reason, discriminant)` for the life of the process.
        let mid = 913u32;
        let pfn = 0x27u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(gpa, 0x4000, 0);
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).expect("mapped above");
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let (w, h) = (4u32, 2u32);
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));

        // What the guest painted, on the wire in BGRA and distinct per channel so
        // a swizzle cannot pass for a rung.
        let mut pages = vec![0u8; (w * h * 4) as usize];
        for px in pages.chunks_exact_mut(4) {
            px.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]);
        }
        assert!(write_bgra8(&mut state, &mut host, mid, &pages, w * 4, w, h));

        // What this device last published for the same surface — an older frame,
        // and the one the rung under test would serve.
        let mut cached = vec![0u8; (w * h * 4) as usize];
        for px in cached.chunks_exact_mut(4) {
            px.copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xFF]);
        }
        crate::runtime::surface_cache::store(&mut state, mid, w, h, cached);

        // Armed and stamped: with nothing written since, the cache is the
        // fresher copy and must still win. Asserting this first is what stops
        // the fix from being "refuse always", which would pass the real
        // assertion below while costing every seed a guest read.
        let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, mid)
            .expect("FakeHost observes guest writes");
        state
            .mappings
            .get_mut(&mid)
            .expect("mapped above")
            .guest_write_gen_at_store = host.guest_write_gen(token).expect("a live token has one");
        let Type11LoadSeed::Host(bytes, order) = resolve_type11_load_seed(
            &mut state,
            &mut host,
            mid,
            w,
            h,
            gva_resident_format(MTL_FORMAT_BGRA8_UNORM),
        )
        .expect("an unwritten surface may be seeded from the cache") else {
            panic!("the fresh cache must stay above the guest-page rung");
        };
        assert_eq!(order, crate::backend::vulkan::engine::SeedOrder::Bgra8);
        assert_eq!(&bytes[..4], &[0xAA, 0xBB, 0xCC, 0xFF]);

        // The guest CPU repaints the surface. No device operation, so the
        // content epoch does not move and this is the only witness that sees it.
        host.guest_wrote_page(gpa);
        assert_eq!(
            guest_write_site(&state, &host, mid, w, h),
            // Whole-page, because the hypervisor's witness has page granularity
            // and the surface's one page is the whole of its mapping offsets.
            GuestWriteSite::Pixels(vec![(0, 1u64 << PAGE_SHIFT_X86)]),
            "the write has to land inside the sampled window, or the rung under \
             test is being asked the wrong question"
        );

        let Type11LoadSeed::Guest(seed) = resolve_type11_load_seed(
            &mut state,
            &mut host,
            mid,
            w,
            h,
            gva_resident_format(MTL_FORMAT_BGRA8_UNORM),
        )
        .expect("the guest's own pages are still a seed") else {
            panic!("a repainted surface must bypass the stale host cache");
        };
        assert_eq!(seed.format, ash::vk::Format::B8G8R8A8_UNORM);
        let (_, bpr, _) = crate::runtime::mapping_write::type11_sample_window(
            &state.mappings[&mid],
            w,
            h,
            MTL_FORMAT_BGRA8_UNORM,
        )
        .expect("the seed used this mapping window");
        let (span, row_length_texels) = strided_window_extent(w, h, 4, u64::from(bpr))
            .expect("the native row pitch describes this image");
        assert_eq!(seed.source.total_len, span);
        assert_eq!(seed.source.row_length_texels, row_length_texels);
    }

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
        let served = resolve_type11_load_seed(
            &mut state,
            &mut host,
            mid,
            w,
            h,
            gva_resident_format(MTL_FORMAT_BGRA8_UNORM),
        );
        let seed = served.unwrap_or_else(|| {
            panic!(
                "a cold cache must not lose the guest's LOAD; sink said {:?}",
                cap.lines()
            )
        });
        drop(cap);
        let Type11LoadSeed::Guest(seed) = seed else {
            panic!("a cold cache should preserve the native guest-page source");
        };
        assert_eq!(seed.format, ash::vk::Format::B8G8R8A8_UNORM);
        let (_, bpr, _) = crate::runtime::mapping_write::type11_sample_window(
            &state.mappings[&mid],
            w,
            h,
            MTL_FORMAT_BGRA8_UNORM,
        )
        .expect("the seed used this mapping window");
        let (span, row_length_texels) = strided_window_extent(w, h, 4, u64::from(bpr))
            .expect("the native row pitch describes this image");
        assert_eq!(seed.source.total_len, span);
        assert_eq!(seed.source.row_length_texels, row_length_texels);
        assert_eq!(
            seed.source.runs.iter().map(|run| run.len).sum::<u64>(),
            seed.source.total_len
        );

        // A live cache entry still wins: it is the fresher copy (the last Store's
        // output) and the fallback must stay a fallback.
        let mut cached = vec![0u8; (w * h * 4) as usize];
        for px in cached.chunks_exact_mut(4) {
            px.copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xFF]);
        }
        crate::runtime::surface_cache::store(&mut state, mid, w, h, cached);
        let Type11LoadSeed::Host(bytes, order) = resolve_type11_load_seed(
            &mut state,
            &mut host,
            mid,
            w,
            h,
            gva_resident_format(MTL_FORMAT_BGRA8_UNORM),
        )
        .expect("a warm cache must serve") else {
            panic!("a live cache is the freshest rung");
        };
        assert_eq!(order, crate::backend::vulkan::engine::SeedOrder::Bgra8);
        assert_eq!(&bytes[..4], &[0xAA, 0xBB, 0xCC, 0xFF]);

        // An extent the surface is not latched at cannot be served by either rung,
        // and refusing is right: a seed of the wrong length is rejected by the
        // engine anyway, and the decline names both geometries.
        assert!(
            resolve_type11_load_seed(
                &mut state,
                &mut host,
                mid,
                w,
                h + 1,
                gva_resident_format(MTL_FORMAT_BGRA8_UNORM),
            )
            .is_none(),
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
        let cap = crate::observe::sink::FailCapture::resume();
        note_type11_load_seed(&state, mid, 4, 4, Some(Type11SeedRung::GuestPages));
        let pages = only(&cap);
        assert!(pages.contains("outcome=guest_pages"), "{pages}");
        drop(cap);

        // Same mapping, a geometry the cache does not hold: the entry's own
        // geometry is the load-bearing field, since it says a Store at another
        // extent orphaned every window still living at this one.
        let cap = crate::observe::sink::FailCapture::resume();
        note_type11_load_seed(&state, mid, 8, 1, None);
        let geom = only(&cap);
        assert!(geom.contains("reason=type11_seed_cache_geom"), "{geom}");
        assert!(geom.contains("have=8x4"), "{geom}");
        assert!(geom.contains("want=8x1"), "{geom}");
        drop(cap);

        // A mapping the cache has never held reports absence, not a geometry.
        let cap = crate::observe::sink::FailCapture::resume();
        note_type11_load_seed(&state, 910, 8, 4, None);
        let absent = only(&cap);
        assert!(
            absent.contains("reason=type11_seed_cache_absent"),
            "{absent}"
        );
        assert!(!absent.contains("have="), "{absent}");
        drop(cap);

        // Latched per (mapping, geometry, outcome): a repeat of any of the three
        // above emits nothing, so the branch is safe to leave on forever. Every
        // window after the first is a `resume`, because the claims the earlier
        // ones made are exactly what this last one asserts, and `start` would
        // clear them and see all four lines again.
        let cap = crate::observe::sink::FailCapture::resume();
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
             viewports=[] scissors=[]"
        );
    }

    /// A bind past its class's table refuses the whole draw, before the pipeline
    /// is even resolved.
    ///
    /// The order is the assertion. `pipeline_ref` here names nothing an empty
    /// state can resolve, so the sibling test above gets
    /// `draw_prepare_pipeline_missing` from the identical request — and this one
    /// must not, or the check has drifted below the resolves it is supposed to
    /// stand in front of. The reported class is the texture table's, not a
    /// shared bound: 31 buffer slots would have refused this index too.
    #[test]
    fn a_bind_past_its_table_refuses_the_draw_before_anything_resolves() {
        use crate::observe::Decline as _;
        use crate::runtime::draw::MAX_TEXTURE_BIND_SLOTS;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mut req = DrawEncodeRequest {
            pipeline_ref: 41,
            fragment_textures: vec![TextureBind {
                index: MAX_TEXTURE_BIND_SLOTS,
                texture_ref: 9,
                ..Default::default()
            }]
            .into(),
            ..DrawEncodeRequest::default()
        };

        let err = match try_metal2vulkan_draw(&mut state, &mut host, &mut req, true) {
            Err(err) => err,
            Ok(_) => panic!("a texture bind past the table cannot encode"),
        };
        assert_eq!(err.slug(), "draw_prepare_bind_slot_past_table");
        assert_eq!(
            err.fields(),
            vec![
                ("pipeline_ref", "41".to_string()),
                ("class", "texture".to_string()),
                ("stage", "fragment".to_string()),
                ("index", MAX_TEXTURE_BIND_SLOTS.to_string()),
                ("table", MAX_TEXTURE_BIND_SLOTS.to_string()),
                ("ref", "9".to_string()),
            ]
        );

        // The same request with the slot cleared reaches the pipeline resolve,
        // which is what says the refusal is about live guest work and not about
        // the index alone.
        std::sync::Arc::make_mut(&mut req.fragment_textures)[0].texture_ref = 0;
        let err = match try_metal2vulkan_draw(&mut state, &mut host, &mut req, true) {
            Err(err) => err,
            Ok(_) => panic!("an empty state cannot resolve pipeline 41"),
        };
        assert_eq!(err.slug(), "draw_prepare_pipeline_missing");
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

    /// Every kind reaches its whole flag set, not just its 1D-ness.
    ///
    /// This asserted `!one_dim` and nothing else, which left `D2Array`'s
    /// `arrayed` and `D3`'s `volume` unpinned — and those two are adjacent
    /// `bool`s of a five-field shape, so swapping them compiles and would have
    /// bound a 3D image for an array and vice versa with the test still green.
    #[test]
    fn sampled_image_shape_gives_each_kind_its_whole_flag_set() {
        use crate::runtime::spirv_bind::SampledImageKind;

        // (kind, arrayed, volume, one_dim). `cube` is false throughout — the
        // cube kinds decline instead of producing a shape.
        for (kind, arrayed, volume, one_dim) in [
            (SampledImageKind::D2, false, false, false),
            (SampledImageKind::D2Array, true, false, false),
            (SampledImageKind::D3, false, true, false),
            (SampledImageKind::D1, false, false, true),
            (SampledImageKind::D1Array, true, false, true),
        ] {
            let shape = sampled_image_shape(kind).expect("expressible shape");
            assert_eq!(
                (shape.arrayed, shape.volume, shape.cube, shape.one_dim),
                (arrayed, volume, false, one_dim),
                "{kind:?} did not map to its own flags"
            );
            assert_eq!(shape.layers, 1, "{kind:?} layers");
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
        state.define_task(1, 0x1_0000, 2);

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
            format: crate::backend::vulkan::translate::pixel::RESIDENT_RGBA_FORMAT,
        };

        let mut gen_of = |host: &mut FakeHost| {
            let secs = super::build_secondary_targets(
                &mut state, host, 1, &colors, &pipeline, &primary, 8, 8, [0.0; 4],
            )
            .expect("slot 1 is a resolvable secondary");
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

        // The live secondary resource retains its host texture across a task
        // page-table change.
        map_one_gva_page(&mut host, 5);
        assert_eq!(gen_of(&mut host), gen_a);
    }
}

/// Whether the CLEAR-seed Store at the head of a draw chain runs, for this
/// process.
///
/// Read once. The arms differ in what reaches the guest's pages, so a boot that
/// flipped it midway would be two devices in one log — and the arm that does not
/// write is an ablation whose damage is only visible in a photograph, so it has
/// to hold for the whole boot the photograph is of.
fn clear_seed_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::CLEAR_SEED);
        let on = !matches!(state, crate::env::Switch::Off);
        crate::observe::off(format!(
            "clear_seed on={on} switch={state:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        on
    })
}
