//! Resolved draw execution: sampled-source resolution, zero-copy load paths,
//! executor submission, and host-authoritative GVA/surface Store ownership.
//!
//! [`super`] re-exports these items flat so callers keep addressing
//! them as `crate::runtime::draw::<name>`. `use super::*` pulls in the
//! parent's imports, which this half shares.

use super::resident::*;
use super::sampled_source::*;
use super::*;
use crate::runtime::executor::ExecutorDiagnostic;
use reims_vgpu_core::pixel_format::solid_bgra8;
use reims_vgpu_core::{sampled_image_shape, SampledImageShape};

mod bind_plan;
mod load_plan;
mod pipeline_plan;
mod request_plan;
mod resource_plan;
mod sampler_plan;
mod shader_resource_plan;
mod texture_plan;

/// Linux / non-Apple product rail: metal2vulkan + Vulkan offscreen, then Store.
///
/// `writeback_guest` is the archive multi-draw store plan (only the last record
/// of a serialized render-pass chain writes guest memory). Intermediate records **must still
/// encode** and return color0 for chaining — returning `BackendUnavailable` when
/// `!writeback_guest` aborted every multi-draw stream after the first
/// record (live `draw_fail_clear_fallback backend_unavailable=1` on clear+draw packets).
#[derive(Debug)]
pub struct DrawChainResult {
    pub status: EncodeStatus,
    pub chain_rgba: Option<Vec<u8>>,
    pub visibility_samples: Option<u64>,
    pub resident_identity: Option<crate::model::TargetIdentity>,
}

impl DrawChainResult {
    fn new(
        status: EncodeStatus,
        chain_rgba: Option<Vec<u8>>,
        visibility_samples: Option<u64>,
        resident_identity: Option<crate::model::TargetIdentity>,
    ) -> Self {
        Self {
            status,
            chain_rgba,
            visibility_samples,
            resident_identity,
        }
    }
}

pub fn encode_draw_chain<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
    // Inert on this arm, and by construction rather than by omission: the Metal
    // arm consults it in `store_seed_policy` to suppress a scissor-local store,
    // and this rail has no scissor-local store to suppress — `req.scissor` only
    // ever reaches the pipeline scissor rect, never the Store extent.
    _force_full_store: bool,
) -> DrawChainResult {
    // Charges this chain to one phase at a time all the way down, including the
    // parts of it that live inside `try_metal2vulkan_draw`. Held here rather
    // than there because the Store routing below the engine is on the same
    // clock: `drain_duty`'s `draw_us` brackets exactly this function, and the
    // whole reading is that the phases sum to it.
    let _phase = crate::runtime::chain_phase::ChainTimer::start();
    let colors: Vec<ColorRtRequest> = req.colors.clone();
    let Some((pass_w, pass_h)) = colors.first().map(|c0| (c0.width, c0.height)) else {
        return DrawChainResult::new(
            EncodeStatus::BadArgs("draw_vk_no_color_target"),
            None,
            None,
            None,
        );
    };

    let mut any_store = false;
    // The seeded solid colour of attachment 0, as its recipe rather than as its
    // pixels.
    //
    // This used to be the `Vec<u8>` itself, built by `solid_rgba8` inside the
    // seed loop. Only one of this function's five exits returns it — the
    // clear-only one at the bottom, where the record encoded no draw — and the
    // three that matter under compositing (a returned resident identity, an armed
    // Store, and a real `draw_rgba`) each return something else and dropped it
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
            if !c.publishes_single_sample() {
                continue;
            }
            if !load_action_has_clear_seed(c.load_action) {
                // Load/composite needs real encode (metal2vulkan) — skip Store.
                continue;
            }
            if c.width == 0 || c.height == 0 {
                continue;
            }
            // Neither writer below takes a full-surface RGBA copy — the GVA
            // landing repeats a single row and the IOSurface texture landing builds its own
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
            let ok = if c.target_gva() != 0 {
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
                    c.target_gva(),
                    c.width,
                    c.height,
                    c.row_stride(),
                    c.format,
                    &c.clear_color,
                )
                .is_ok()
            } else if c.mapping_id() != 0 {
                // IOSurface texture CLEAR. `write_bgra8` takes guest scanout order and
                // converts to the mapping's native format per row; it handles a
                // fragmented mapping too, staging native rows and landing them
                // through `mapper::write_mapping_bytes`. (A comment here used to
                // call it contig-only, which it has not been.)
                //
                // Built from the swapped *pixel* rather than by exchanging the
                // channels of the RGBA image: a solid image is one repeated
                // word, so the exchange belongs to the word and doing it per
                // texel cost an allocation and two passes over the whole
                // surface. See `reims_vgpu_core::pixel_format::solid_bgra8`.
                let _span = StoreCostSpan::new("clear_seed_iosurface_us");
                crate::runtime::drain::note_store_route("clear_seed_iosurface");
                crate::runtime::drain::note_store_route_n("clear_seed_iosurface_kb", seed_kb);
                let bgra = solid_bgra8(c.width, c.height, &c.clear_color);
                let stride = c.width.saturating_mul(RGBA8_BPP);
                mapping_write::write_bgra8(
                    state,
                    host,
                    c.mapping_id(),
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
                    c.mapping_id(),
                    c.target_gva(),
                    c.width,
                    c.height,
                    req.pipeline_ref,
                    c.load_action
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
    // Physical order of `draw_rgba`. An IOSurface texture composite Store renders into a
    // BGRA `Surface` resident, so its readback is already in guest scanout
    // order; the pooled and GVA targets stay RGBA. Carried instead of assumed —
    // which of those a record hit depends on whether an identity resolved, and
    // that is not a condition the Store block can re-derive.
    let mut draw_bgra = false;
    // Set only by a validated executor receipt. Every content transition below
    // cites this identity instead of recovering one from mutable ambient state.
    let mut completed_submission = None;
    // Completion output belongs to the encode result, never to the immutable
    // request that caused it.
    let mut visibility_samples = None;
    let mut chain_resident_identity = None;
    // GVA render Store: a copied resident leaves a resource-scoped transfer
    // debt, while a guest-backed resident publishes the attachment write it
    // already performed.
    let mut gva_store_armed = false;
    let mut effects_only_completed = false;
    if req.pipeline_ref != 0 && (req.vertex_count > 0 || req.indexed.is_some()) {
        match try_metal2vulkan_draw(state, host, req, writeback_guest) {
            Ok(M2vDrawSpan::Pixels {
                submission,
                bytes,
                bgra,
                visibility_samples: samples,
            }) => {
                completed_submission = Some(submission);
                visibility_samples = samples;
                draw_rgba = Some(bytes);
                draw_bgra = bgra;
                crate::observe::line(format!(
                    "linux_m2v_draw ok pipe={} {}x{} vtx={}",
                    req.pipeline_ref, pass_w, pass_h, req.vertex_count
                ));
            }
            Ok(M2vDrawSpan::EffectsOnly {
                submission,
                visibility_samples: samples,
            }) => {
                completed_submission = Some(submission);
                visibility_samples = samples;
                effects_only_completed = true;
            }
            Ok(M2vDrawSpan::ResidentChain {
                submission,
                identity,
                visibility_samples: samples,
            }) => {
                completed_submission = Some(submission);
                visibility_samples = samples;
                chain_resident_identity = Some(identity);
                crate::observe::line(format!(
                    "linux_m2v_draw ok resident_chain pipe={} {}x{} mid={} gva={:#x}",
                    req.pipeline_ref,
                    pass_w,
                    pass_h,
                    req.colors.first().map(|c| c.mapping_id()).unwrap_or(0),
                    req.colors.first().map(|c| c.target_gva()).unwrap_or(0)
                ));
            }
            Ok(M2vDrawSpan::ResidentGvaReadback {
                submission,
                identity,
                visibility_samples: samples,
            }) => {
                completed_submission = Some(submission);
                visibility_samples = samples;
                let _store_span = StoreCostSpan::new("gva_store_us");
                note_iosurface_texture_store_route("gva_store_sync");
                draw_rgba = read_resident_chain(state.executor.as_ref(), req, &identity);
                crate::observe::line(format!(
                    "linux_m2v_draw ok resident_gva_readback pipe={} {}x{} gva={:#x} rgba={}",
                    req.pipeline_ref,
                    pass_w,
                    pass_h,
                    req.colors.first().map(|c| c.target_gva()).unwrap_or(0),
                    draw_rgba.is_some() as u8
                ));
            }
            Ok(M2vDrawSpan::ResidentGvaStore {
                submission,
                identity,
                guest_store_pages,
                visibility_samples: samples,
            }) => {
                completed_submission = Some(submission);
                visibility_samples = samples;
                let _store_span = StoreCostSpan::new("gva_store_us");
                note_iosurface_texture_store_route("gva_flush");
                let directly_landed = req.colors.first().is_some_and(|c0| {
                    guest_store_pages.as_ref().is_some_and(|pages| {
                        state.note_host_wrote_pages(pages.pages().to_vec());
                        crate::runtime::render_writeback::forget_gva_host_copies(
                            state,
                            req.task_id,
                            c0.target_gva(),
                            c0.texture_ref,
                        );
                        // The Store is recorded first because recording it is
                        // what advances the resource's content version, and the
                        // witness has to be stamped with the version this Store
                        // leaves behind rather than the one it replaced. Stamped
                        // the other way round the entry is stale the moment it is
                        // written and `reach` answers `GuestWrote` forever.
                        let stored = record_materialized_store(
                            state,
                            req.task_id,
                            c0.texture_ref,
                            submission,
                        );
                        match stored {
                            Some(version) => {
                                if let Some(key) = crate::runtime::writeback_debt::resource_key(
                                    state,
                                    req.task_id,
                                    c0.texture_ref,
                                )
                                .and_then(|resource| {
                                    crate::runtime::gva_store_witness::GvaTargetKey::of(
                                        resource, &identity,
                                    )
                                }) {
                                    let guest_write =
                                        reims_vgpu_core::ResourceWriteStamp::Resolved {
                                            resource: key.resource,
                                            version,
                                        };
                                    crate::runtime::gva_store_witness::note_store(
                                        state,
                                        key,
                                        pages.pages(),
                                        guest_write,
                                    );
                                }
                            }
                            // No version means no Store was recorded against the
                            // resource, so there is nothing for a witness entry to
                            // vouch for. Named rather than dropped, because a rail
                            // that stops stamping stops eliding and that reads as a
                            // slowdown with no cause.
                            None => {
                                crate::runtime::drain::note_store_route("gvaw_no_store_version")
                            }
                        }
                        true
                    })
                });
                if directly_landed {
                    note_iosurface_texture_store_route("gva_guest_backed");
                    gva_store_armed = true;
                } else {
                    // Planning selected a guest-backed target, but execution
                    // did not report guest pages. Completion cannot turn that
                    // missing materialization into deferred authority: publish
                    // synchronously from the exact resident instead.
                    note_iosurface_texture_store_route("gva_store_sync");
                    draw_rgba = read_resident_chain(state.executor.as_ref(), req, &identity);
                    crate::observe::line(format!(
                        "linux_m2v_draw ok resident_gva_store pipe={} {}x{} gva={:#x} rgba={}",
                        req.pipeline_ref,
                        pass_w,
                        pass_h,
                        req.colors.first().map(|c| c.target_gva()).unwrap_or(0),
                        draw_rgba.is_some() as u8
                    ));
                }
            }
            Ok(M2vDrawSpan::ResidentSurfaceStore {
                submission,
                identity,
                guest_store_pages,
                guest_store_window,
                visibility_samples: samples,
            }) => {
                return complete_surface_store(
                    state,
                    host,
                    req,
                    submission,
                    identity,
                    guest_store_pages,
                    guest_store_window,
                    samples,
                )
            }
            Ok(M2vDrawSpan::None) => {
                crate::observe::line(format!(
                    "linux_m2v_draw skip pipe={} (no color0 geom)",
                    req.pipeline_ref
                ));
            }
            Err(e) => {
                // Always-on + latched: a rejected engine draw falls to the
                // clear-store fallback and surfaces as a bare `backend_unavailable`
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
                let slug = crate::observe::Decline::slug(&e);
                // The latch below fires once per (reason, pipeline), which is
                // what keeps a persistent reject from flooding — and it is also
                // why the fail log can say *which* draws were refused and never
                // *how many*. A counter does not dedupe, so this is the only
                // reading that sizes a refusal: one increment per lost draw,
                // under the decline's own name, banded per census window like
                // every other `store_routes` field. Without it a refusal that
                // costs the guest one draw and one that costs it an entire
                // layer every frame are the same two log lines.
                crate::runtime::drain::note_store_route(slug);
                engine_refusal = Some(slug);
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
    // A resident render-pass chain intermediate tells the exec loop to arm the
    // next record's LoadFromTarget through this result value.
    if chain_resident_identity.is_some() {
        return DrawChainResult::new(
            EncodeStatus::Ok,
            None,
            visibility_samples,
            chain_resident_identity,
        );
    }

    // Deferred IOSurface texture composite Store: the window names the pinned resident and
    // the guest write lands on first access. `None`, not the frame, for the same
    // reason the `Owned` route returns `None` — `writeback_guest` is granted only
    // to the last record of a packet, so there is no record N+1 to seed.
    if gva_store_armed {
        return DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None);
    }

    // Taken, not borrowed. Every exit from this block returns the frame, and
    // borrowing forced each of them to `rgba.clone()` a whole framebuffer — 8 MB
    // at 1080p, at the 28-111 Stores/s `store_routes` measures, on the drain
    // worker `drain_duty` shows at duty 0.93-0.99. The deferred IOSurface texture arm is
    // the hot one and it cloned purely to hand back the buffer it already owned.
    if let Some(mut rgba) = draw_rgba.take() {
        // Intermediate multi-draw GVA records: return color0 for chaining without
        // guest Store (archive store plan). Resident IOSurface texture intermediates
        // returned above without materializing CPU pixels.
        if !writeback_guest {
            // A chain value seeds the next record, and `DrawRequest` states a
            // seed's order as `SeedOrder::Rgba8` for this rail.
            reorder_rb_in_place(&mut rgba, draw_bgra, false);
            return DrawChainResult::new(EncodeStatus::Ok, Some(rgba), visibility_samples, None);
        }
        // Store draw result into primary color RT.
        if let Some(c0) = colors.first() {
            // `rgb_nz`/`max_rgb` are diagnostic fields of the Store lines below,
            // and producing them is an O(w*h) pass over a whole framebuffer
            // readback — 2 073 600 pixels per Store at 1080p, at the 28-111
            // Stores/s `store_routes` measures under load. Computing it here
            // paid that on every route, including the IOSurface texture one whose only
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
            let ok = if c0.mapping_id() != 0 {
                // Unconditional. This used to be `if
                // iosurface_texture_cpu_store_fallback_allowed(import_allowed)`, where
                // `import_allowed` asked whether the device could import a host
                // pointer over the mapping's guest pages; when it could, the
                // draw took the import rail and landing here was a fail-closed
                // error (`rgba_not_import`) that preserved the zero-copy
                // invariant. There is no invariant left to preserve, and the
                // else arm was a refusal for a rail that cannot be chosen.
                {
                    // Brackets the whole IOSurface texture arm, into the same per-second
                    // window it divides into. `draw_phase` stops at the engine
                    // boundary, so this arm — the cache publish, the window arm,
                    // the guest scatter — is the bulk of the ~245 ms/s (28 % of
                    // `draw_us`) that no phase claimed.
                    let _span = StoreCostSpan::new("iosurface_store_us");
                    // Every consumer below wants guest scanout order: the
                    // deferred window's `write_bgra8`, `surface_cache`, and the
                    // synchronous route. A `Surface` resident reads back in that
                    // order already, so this is a no-op on the hot path and the
                    // ~152 ms/s whole-frame swizzle it replaces is gone. It still
                    // has to be written, because a record whose identity did not
                    // resolve rendered into a pooled RGBA target.
                    let mut bgra = rgba;
                    {
                        let _span = StoreCostSpan::new("iosurface_convert_us");
                        reorder_rb_in_place(&mut bgra, draw_bgra, true);
                    }
                    note_iosurface_texture_store_route("cpu_portability");
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
                        c0.mapping_id(),
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
                        .then(|| state.surfaces.mappings.get(&c0.mapping_id()))
                        .flatten()
                        .map(|m| m.content.surface_epoch);
                    if ok {
                        // Full-frame publish: same completeness proof as the
                        // import-present scatter paths — the write verified
                        // geometry (mw==w, mh==h) and landed the complete
                        // frame into the mapping's guest pages. Without it the
                        // `present_unbacked` gate is structurally dead on the
                        // CPU-portability Store path: no mapping's
                        // full-frame backing evidence would ever advance.
                        {
                            let _span = StoreCostSpan::new("iosurface_publish_us");
                            publish_surface_store(
                                state,
                                host,
                                c0.mapping_id(),
                                c0.width,
                                c0.height,
                                c0.format,
                            );
                        }
                        if let Some(epoch) = sync_epoch {
                            stamp_iosurface_texture_resident(state, req, writeback_guest, epoch);
                        }
                        if crate::observe::draw_log_enabled() {
                            // Order-independent: both fields reduce over the three
                            // colour channels, so an R/B exchange cannot move them.
                            let (rgb_nz, max_rgb) = rgb_stats(&bgra);
                            crate::observe::line(format!(
                                "linux_m2v_store mid={} {}x{} pipe={} import=0 reason=cpu_portability pages=1 rgb_nz={} max={}",
                                c0.mapping_id(),
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
                            c0.mapping_id(),
                            c0.width,
                            c0.height,
                            req.pipeline_ref,
                            rgb_nz,
                            max_rgb,
                            c0.format
                        ));
                    }
                    if ok {
                        record_materialized_store(
                            state,
                            req.task_id,
                            c0.texture_ref,
                            completed_submission
                                .expect("materialized engine pixels carry a completion identity"),
                        );
                        // `None`, for the same reason the deferred arm above
                        // returns it: this whole block runs only under
                        // `writeback_guest`, which `multi_draw_store_plan` grants
                        // solely to the **last** record of a packet, so there is no
                        // record N+1 for the chain value to seed. Returning it also
                        // could not be done honestly here — the frame is in guest
                        // scanout order and a chain seed is declared RGBA — so the
                        // alternative is a whole-frame exchange for a buffer with
                        // no reader.
                        return DrawChainResult::new(
                            EncodeStatus::Ok,
                            None,
                            visibility_samples,
                            None,
                        );
                    }
                    false
                }
            } else if c0.target_gva() != 0 {
                // Executor readback uses the render target's declared texel
                // order. Guest publication and the host replica helpers below
                // take semantic RGBA, so normalize once at this boundary.
                normalize_gva_store_pixels(&mut rgba, draw_bgra);
                // What this Store would cost if it were served the way the
                // IOSurface texture surface Store is served.
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
                        c0.format,
                        c0.width,
                        c0.height,
                        c0.row_stride()
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
                    c0.target_gva(),
                    c0.width,
                    c0.height,
                    c0.row_stride(),
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
                            .map(|entry| entry.wire_tag())
                            .unwrap_or(0);
                    host_cache_store_gva_layer(
                        state,
                        host,
                        req.task_id,
                        c0.texture_ref,
                        producer_object_type,
                        c0.target_gva(),
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
                    c0.target_gva(),
                    c0.width,
                    c0.height,
                    req.pipeline_ref,
                    c0.texture_ref,
                    c0.load_action,
                    gva_ok as u8,
                    rgb_nz,
                    max_rgb,
                    c0.row_stride()
                ));
                if !gva_ok {
                    crate::observe::fail(format!(
                        "linux_m2v_store lost gva={:#x} {}x{} pipe={} tex_ref={} rgb_nz={} max={} bpr={}",
                        c0.target_gva(),
                        c0.width,
                        c0.height,
                        req.pipeline_ref,
                        c0.texture_ref,
                        rgb_nz,
                        max_rgb,
                        c0.row_stride()
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
                record_materialized_store(
                    state,
                    req.task_id,
                    c0.texture_ref,
                    completed_submission
                        .expect("materialized engine pixels carry a completion identity"),
                );
                // `None`: everything from here up is under `writeback_guest`,
                // which `multi_draw_store_plan` grants only to `di == last_i`, so
                // the chain value has no record N+1 to seed and every other
                // reader of it in `exec` sits inside the record loop that just
                // ended. The intermediate handoff that *is* live returned above,
                // before the Store arms. Returning the frame here handed a whole
                // framebuffer to a binding that is dropped unread.
                return DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None);
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
        // `clear_seed_gva` + `clear_seed_iosurface`, which count the seeds, and the
        // difference is the full-surface images that used to be built and
        // dropped. A boot where the two are equal has nothing to save here.
        DrawChainResult::new(
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
            visibility_samples,
            None,
        )
    } else if effects_only_completed {
        DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None)
    } else {
        DrawChainResult::new(
            EncodeStatus::BackendUnavailable("draw_vk_nothing_stored"),
            None,
            visibility_samples,
            None,
        )
    }
}

pub(crate) fn complete_surface_store<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    submission: reims_vgpu_protocol::SubmissionId,
    identity: crate::model::TargetIdentity,
    guest_store_pages: Option<reims_vgpu_memory::GuestWritePages>,
    guest_store_window: Option<std::ops::Range<u64>>,
    visibility_samples: Option<u64>,
) -> DrawChainResult {
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Store);
    let _store_span = StoreCostSpan::new("iosurface_store_us");
    let Some(c0) = req.colors.first() else {
        return DrawChainResult::new(
            EncodeStatus::BadArgs("surface_store_without_color0"),
            None,
            visibility_samples,
            None,
        );
    };
    let target = (
        c0.mapping_id(),
        c0.width,
        c0.height,
        c0.format,
        c0.texture_ref,
    );
    let directly_landed = guest_store_pages
        .as_ref()
        .zip(guest_store_window.as_ref())
        .and_then(|(pages, window)| {
            state.note_host_wrote_pages(pages.pages().to_vec());
            let epoch = crate::runtime::mapping_write::note_iosurface_texture_landed(
                state,
                target.0,
                window.start,
                window.end,
            )?;
            if !state
                .executor
                .stamp_resident_content_epoch(&identity, epoch)
            {
                crate::observe::fail(format!(
                    "resident_surface_store_fail reason=epoch_stamp_refused mapping={} epoch={epoch}",
                    target.0
                ));
            }
            publish_surface_store(state, host, target.0, target.1, target.2, target.3);
            record_materialized_store(state, req.task_id, target.4, submission);
            Some(())
        });
    if directly_landed.is_some() {
        note_pass_scissor_union(target.1, target.2);
        note_iosurface_texture_store_route("surface_guest_backed");
        return DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None);
    }

    if store_surface_resident(state, host, &identity, target.0, target.1, target.2) {
        note_iosurface_texture_store_route("surface_resident");
        {
            let _span = StoreCostSpan::new("iosurface_publish_us");
            publish_surface_store(state, host, target.0, target.1, target.2, target.3);
        }
        record_materialized_store(state, req.task_id, target.4, submission);
        return DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None);
    }

    note_iosurface_texture_store_route("surface_resident_sync");
    let Some(mut bgra) = read_resident_chain(state.executor.as_ref(), req, &identity) else {
        return DrawChainResult::new(
            EncodeStatus::BackendUnavailable("draw_vk_nothing_stored"),
            None,
            visibility_samples,
            None,
        );
    };
    reorder_rb_in_place(&mut bgra, false, true);
    note_iosurface_texture_store_route("cpu_portability");
    let stored = mapping_write::write_bgra8(
        state,
        host,
        target.0,
        &bgra,
        target.1.saturating_mul(RGBA8_BPP),
        target.1,
        target.2,
    );
    if !stored {
        crate::observe::fail(format!(
            "linux_m2v_store mid={} {}x{} pipe={} reason=cpu_portability_write_fail fmt={:#x}",
            target.0, target.1, target.2, req.pipeline_ref, target.3
        ));
        return DrawChainResult::new(
            EncodeStatus::BackendUnavailable("draw_vk_nothing_stored"),
            None,
            visibility_samples,
            None,
        );
    }
    let epoch = state
        .surfaces
        .mappings
        .get(&target.0)
        .map(|mapping| mapping.content.surface_epoch);
    {
        let _span = StoreCostSpan::new("iosurface_publish_us");
        publish_surface_store(state, host, target.0, target.1, target.2, target.3);
    }
    if let Some(epoch) = epoch {
        stamp_iosurface_texture_resident(state, req, true, epoch);
    }
    record_materialized_store(state, req.task_id, target.4, submission);
    DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None)
}

/// Publish the one semantic outcome shared by ordered Store rails.
///
/// Vulkan may reach this through a resident surface, a host readback, or a
/// direct guest write. Those choices affect transfer cost only; all have a
/// same bytes in the guest replica. A direct imported write may still be owned
/// by the executor's fence ledger; all access to those bytes settles that debt.
///
/// The content version this Store produced is returned rather than discarded,
/// because [`crate::runtime::gva_store_witness::note_store`] has to stamp its
/// entry with the version the Store leaves behind and this is the only place
/// that knows it. Recording the Store is what advances that version, so a
/// caller that stamps the witness first captures the version this replaced.
fn record_materialized_store(
    state: &mut Device,
    task_id: u32,
    texture_ref: u32,
    submission: reims_vgpu_protocol::SubmissionId,
) -> Option<reims_vgpu_protocol::ContentVersion> {
    state
        .task_objects
        .resources
        .record_ordered_materialized_store(task_id, texture_ref, submission)
        .map(|(_resource, version)| version)
}

/// Guest pages the Vulkan draw's eager GVA fallback may write.
///
/// The Vulkan rail writes back only color attachment 0, so this narrows
/// [`sync_store_target_pages`] to that record and to the case where this record
/// owns guest writeback at all.
pub(super) fn sync_store_allowed_pages<M: HostMemory>(
    state: &Device,
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

/// Record the extent of IOSurface texture LOADs served from a current resident.
///
/// The mapping/resident epoch equality is the decision. These bands only price
/// the saved guest-memory transfer and never feed behavior.
fn note_iosurface_texture_elision_extent(w: u32, h: u32) {
    let texels = (w as u64).saturating_mul(h as u64);
    crate::runtime::drain::note_store_route(match texels {
        0..=4_096 => "iosurfaceelide_le_64x64",
        4_097..=65_536 => "iosurfaceelide_le_256x256",
        65_537..=262_144 => "iosurfaceelide_le_512x512",
        262_145..=1_048_576 => "iosurfaceelide_le_1024x1024",
        _ => "iosurfaceelide_display",
    });
    // The bytes, so the buckets can be priced without assuming a distribution
    // inside each one. RGBA8 is the seed's own upload format.
    crate::runtime::drain::note_store_route_n("iosurfaceelide_texels", texels);
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
/// `iosurface_texture_seed_elided` and `draw_partial_load_from_target` are all simply
/// proportional to round length, and the set of decline names is identical on
/// both sides. A defect that is stable on screen for minutes and leaves no
/// trace in a census this large will not be found by adding another counter to
/// these rails; the next instrument has to observe surface *content* across the
/// transition, not the routes taken to produce it.
fn note_draw_coverage(
    scissor: ScissorRect,
    target_w: u32,
    target_h: u32,
    load_action: Option<reims_vgpu_protocol::pass_action::LoadAction>,
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
        Some(reims_vgpu_protocol::pass_action::LoadAction::Load) if from_target => {
            "draw_partial_load_from_target"
        }
        Some(reims_vgpu_protocol::pass_action::LoadAction::Load) if seeded => {
            "draw_partial_load_seeded"
        }
        Some(reims_vgpu_protocol::pass_action::LoadAction::Load) => "draw_partial_load_unseeded",
        Some(reims_vgpu_protocol::pass_action::LoadAction::Clear) => "draw_partial_clear",
        Some(reims_vgpu_protocol::pass_action::LoadAction::DontCare) => "draw_partial_dontcare",
        None => "draw_partial_load_unknown",
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
/// module belongs to Vulkan draw preparation; the test that calls this is what stops the
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
        ("gva_guest", true) => "load_seed_ok_gva_guest",
        ("gva_guest", false) => "load_seed_lost_gva_guest",
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
        c0.target_gva() ^ ((c0.texture_ref as u64) << 40) ^ ((c0.mapping_id() as u64) << 20),
    ) {
        crate::observe::fail(format!(
            "load_seed_lost door={door} gva={:#x} ref={} mid={} {w}x{h} bpr={} fmt={:#x}",
            c0.target_gva(),
            c0.texture_ref,
            c0.mapping_id(),
            c0.row_stride(),
            c0.format
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
    state: &mut Device,
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

/// Convert executor readback into the semantic RGBA order used by GVA
/// publication and host replicas.
fn normalize_gva_store_pixels(pixels: &mut [u8], executor_reported_bgra: bool) {
    reorder_rb_in_place(pixels, executor_reported_bgra, false);
}

/// Result of a Linux metal2vulkan draw.
pub(crate) enum M2vDrawSpan {
    /// No drawable color0 geom.
    None,
    /// CPU-side pixels (readback path), in the order the engine reports.
    ///
    /// The order is carried rather than normalized because an IOSurface texture composite
    /// Store's consumers — `surface_cache`, the deferred window, the guest-page
    /// writeback — all want guest scanout order, and a BGRA resident hands them
    /// exactly that. Normalizing to RGBA here would restate a whole framebuffer
    /// per Store purely to have the Store restate it back.
    Pixels {
        submission: reims_vgpu_protocol::SubmissionId,
        bytes: Vec<u8>,
        bgra: bool,
        visibility_samples: Option<u64>,
    },
    /// The accepted draw published depth or stencil but no colour attachment.
    /// No CPU readback or guest colour Store belongs to this completion.
    EffectsOnly {
        submission: reims_vgpu_protocol::SubmissionId,
        visibility_samples: Option<u64>,
    },
    /// Intermediate record of a resident render-pass chain: content stays on
    /// the protocol-keyed engine target (no CPU pixels, no fence wait, no guest
    /// Store this record). The final record reads back and performs the
    /// contract Store on portability devices.
    ResidentChain {
        submission: reims_vgpu_protocol::SubmissionId,
        identity: crate::model::TargetIdentity,
        visibility_samples: Option<u64>,
    },
    /// Final GVA Store rendered into a copied registry resident. The draw may
    /// remain in its command-buffer chain, but completion synchronously reads
    /// this exact identity before the caller publishes pixels to guest pages.
    ResidentGvaReadback {
        submission: reims_vgpu_protocol::SubmissionId,
        identity: crate::model::TargetIdentity,
        visibility_samples: Option<u64>,
    },
    /// Final/single record of a GVA render Store planned against guest-backed
    /// target memory. A successful direct Store publishes its retained page
    /// footprint. If execution cannot report those pages, the caller reads this
    /// exact resident synchronously rather than deferring missing content.
    ///
    /// `identity` is the key the draw registered, carried rather than re-derived
    /// — see [`Self::ResidentSurfaceStore`] for what a second derivation costs.
    ResidentGvaStore {
        submission: reims_vgpu_protocol::SubmissionId,
        identity: crate::model::TargetIdentity,
        guest_store_pages: Option<reims_vgpu_memory::GuestWritePages>,
        visibility_samples: Option<u64>,
    },
    /// IOSurface texture composite Store executed into its registry resident with
    /// `skip_readback`. When the resident is the mapping's exact imported
    /// allocation, the Store is already in guest pages; otherwise the caller
    /// copies the device-local image there through
    /// [`crate::runtime::render_writeback`] without bringing the frame across
    /// host memory.
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
    /// construction. The Store used to call `iosurface_texture_store_identity` a second
    /// time instead, on the grounds that it is the same function that produced
    /// `DrawRequest::target_identity` — but it is not the same *value*. That
    /// identity carries the mapping lifecycle generation, the draw mutates
    /// `Device` between the two calls, and any writer that revalidates the
    /// mapping advances the generation. The Store then asked the registry for a
    /// key one generation ahead of the one the draw registered, which is
    /// `read_target_unknown_identity diverges=generation asked_gen=N
    /// held_gen=N-1` — the whole Maps frame lost, on the only render-target
    /// rail a host without `VK_EXT_external_memory_host` has.
    ResidentSurfaceStore {
        submission: reims_vgpu_protocol::SubmissionId,
        identity: crate::model::TargetIdentity,
        guest_store_pages: Option<reims_vgpu_memory::GuestWritePages>,
        guest_store_window: Option<std::ops::Range<u64>>,
        visibility_samples: Option<u64>,
    },
}

/// Failure before or after the immutable draw request crosses the executor
/// boundary. Semantic preparation never has to inhabit the backend error type;
/// backend execution remains opaque to orchestration beyond its decline.
#[derive(Debug)]
enum DrawAttemptError {
    Preparation(DrawPreparationDecline),
    Executor(ExecutorDiagnostic),
}

impl From<DrawPreparationDecline> for DrawAttemptError {
    fn from(value: DrawPreparationDecline) -> Self {
        Self::Preparation(value)
    }
}

impl From<ExecutorDiagnostic> for DrawAttemptError {
    fn from(value: ExecutorDiagnostic) -> Self {
        Self::Executor(value)
    }
}

impl crate::observe::Decline for DrawAttemptError {
    fn slug(&self) -> &'static str {
        match self {
            Self::Preparation(decline) => crate::observe::Decline::slug(decline),
            Self::Executor(decline) => crate::observe::Decline::slug(decline),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Preparation(decline) => crate::observe::Decline::fields(decline),
            Self::Executor(decline) => crate::observe::Decline::fields(decline),
        }
    }
}

impl std::fmt::Display for DrawAttemptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(decline) => decline.fmt(f),
            Self::Executor(decline) => decline.fmt(f),
        }
    }
}

impl std::error::Error for DrawAttemptError {}

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

/// Name which of the six routes an IOSurface texture Store took: counted every time, and
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
/// [`reims_vgpu_vulkan::engine::deferred_gpu_only_content_allowed`] — a **capability** gate,
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
fn note_iosurface_texture_store_route(route: &'static str) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<&'static str>>> = Mutex::new(None);
    crate::runtime::drain::note_store_route(route);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        if !guard.get_or_insert_with(Default::default).insert(route) {
            return;
        }
    }
    crate::observe::fail(format!("iosurface_texture_store_route route={route}"));
}

/// Build the engine's secondary MRT attachments (slot 1..) from a draw's color
/// list. `Ok(empty)` ⇒ the guest declared a single render target and this is the
/// classic single-RT path. A fragment shader that writes `location` 1.. has
/// those outputs rendered rather than discarded; each secondary persists as a
/// registry resident keyed by its protocol identity, exactly as slot 0 does.
///
/// Strict by construction — any ambiguity is an `Err` and the caller refuses the
/// draw, rather than a guessed attachment: requires a resident primary,
/// contiguous slots (0,1,2,… matching the shader's `location`s), a known
/// color-renderable format, and a resolvable identity. Attachment geometry is
/// carried independently; the render extent is the per-axis minimum while
/// load/store operations retain each attachment's full-image scope.
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
/// this same colour list at its own slot number and has never degraded, so the
/// two arms disagreed about one wire form and only one of them was silent.
#[allow(
    clippy::too_many_arguments,
    reason = "every argument is a distinct wire-derived input to the attachment set"
)]
pub(super) fn build_secondary_targets<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    colors: &[ColorRtRequest],
    pipeline: &crate::runtime::decode::resource::RenderPipelineDescriptor,
    primary: &crate::model::TargetIdentity,
    blend_states: &[(u32, reims_vgpu_protocol::BlendStateResource)],
) -> Result<
    Vec<reims_vgpu_core::SecondaryColorTarget>,
    crate::runtime::census::present_proxy::SecondaryMrtRefusal,
> {
    use crate::runtime::census::present_proxy::SecondaryMrtRefusal;
    use reims_vgpu_core::SecondaryColorTarget;
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
        // Unknown wire format stays unknown — never guess a secondary layout —
        // and a known format whose sRGB qualifier this attachment cannot carry
        // says so instead of folding silently.
        let format = match pixel_format::color_attachment_format_checked(c.format) {
            Ok(format) => format,
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
        // IOSurface texture surface.
        //
        // A secondary GVA is named by its own backing pages, exactly like color0
        // — the primary's generation describes a different address (a secondary
        // equal to the primary is rejected above), so it takes its own walk.
        // Without one this attachment is keyed on `(gva, width, height)` alone
        // and two guest allocations reusing that address at that geometry share
        // one GPU image — the wrong-content class `74748d2` closed for color0.
        let Some(identity) = color_target_identity(state, host, task_id, c, format.layout(), None)
        else {
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
        let load_action = match c.load_action {
            reims_vgpu_protocol::pass_action::LoadAction::Load => {
                reims_vgpu_core::ColorLoadAction::Load
            }
            reims_vgpu_protocol::pass_action::LoadAction::Clear => {
                reims_vgpu_core::ColorLoadAction::Clear
            }
            reims_vgpu_protocol::pass_action::LoadAction::DontCare => {
                reims_vgpu_core::ColorLoadAction::DontCare
            }
        };
        let clear = [
            c.clear_color[0] as f32,
            c.clear_color[1] as f32,
            c.clear_color[2] as f32,
            c.clear_color[3] as f32,
        ];
        // Find the pipeline's attachment entry for this guest slot. No
        // `or_else(first())` fallback: a secondary slot with no entry of its own
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
        let blend = blend_states
            .iter()
            .find(|(slot, _)| *slot == c.slot)
            .map(|(_, state)| *state);
        let target_guest =
            color_target_guest_backing(state, host, task_id, c, Some(identity.resident_layout()));
        out.push(SecondaryColorTarget {
            identity,
            target_guest,
            width: c.width,
            height: c.height,
            format,
            clear,
            load_action,
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
) -> Result<reims_vgpu_core::VertexAttributeFormat, DrawPreparationDecline> {
    reims_vgpu_protocol::decode_vertex_attribute_format(attribute.format).map_err(|reason| {
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
) -> Result<reims_vgpu_core::VertexStepFunction, DrawPreparationDecline> {
    reims_vgpu_protocol::decode_vertex_step_function(attribute.declared_step_function).map_err(
        |reason| DrawPreparationDecline::VertexStepFunctionUnsupported {
            location: attribute.location,
            buffer_index: attribute.buffer_index,
            reason,
        },
    )
}

#[derive(Default)]
pub(super) struct GvaLoadResolution {
    pub identity: Option<crate::model::TargetIdentity>,
    pub guest_seed: Option<reims_vgpu_memory::GuestTargetSeed>,
    pub cpu_seed: Option<Vec<u8>>,
}

/// Discharge colour0's typed GVA LOAD source after its packed allocation has
/// been resolved.
///
/// A current resident chains directly. Otherwise the same retained allocation
/// supplies a bounded guest-page seed for the engine's import path. CPU bytes
/// are the capability/lifetime fallback only when no stable allocation can
/// describe the plane. Every non-`None` input therefore produces one of those
/// three sources; leaving all of them absent would load undefined content.
pub(super) fn resolve_gva_load_source<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    gva_alloc_generation: u64,
    guest_backing: Option<&reims_vgpu_memory::GuestTargetMemory>,
    chain_load_from_target: &mut bool,
) -> GvaLoadResolution {
    use super::GvaLoadSource;

    if req.gva_load_source == GvaLoadSource::None || *chain_load_from_target {
        return GvaLoadResolution::default();
    }
    let identity = gva_chain_identity(state.executor.as_ref(), req, gva_alloc_generation);
    if req.gva_load_source == GvaLoadSource::Resident {
        let ready = identity.clone().filter(|identity| {
            let texture_ref = req.colors.first().map(|c0| c0.texture_ref).unwrap_or(0);
            gva_resident_ready(state, req.task_id, texture_ref, identity)
        });
        if let Some(identity) = ready {
            crate::runtime::drain::note_store_route("gvaseed_chained");
            *chain_load_from_target = true;
            return GvaLoadResolution {
                identity: Some(identity),
                guest_seed: None,
                cpu_seed: None,
            };
        }
        crate::runtime::drain::note_store_route("gvaseed_reseeded");
    }

    let Some(c0) = req.colors.first() else {
        return GvaLoadResolution::default();
    };
    let (tex_ref, gva, width, height) = (c0.texture_ref, c0.target_gva(), c0.width, c0.height);
    let target_format = identity.as_ref().map(|identity| identity.resident_layout());
    if let Some(seed) = guest_backing.and_then(|backing| {
        target_format
            .and_then(|format| reims_vgpu_memory::guest_target_seed(backing, width, height, format))
    }) {
        crate::runtime::drain::note_store_route("gvaseed_guest_pages");
        return GvaLoadResolution {
            identity: None,
            guest_seed: Some(seed),
            cpu_seed: None,
        };
    }

    crate::runtime::drain::note_store_route("gvaseed_guest_cpu_fallback");
    let seed = crate::runtime::draw::seed_color_load(
        state,
        host,
        req.task_id,
        tex_ref,
        gva,
        width,
        height,
    );
    if seed.is_none() {
        crate::observe::fail(format!(
            "gvaseed_resolve_miss ref={tex_ref} {width}x{height} gva={gva:#x} \
             (no resident or stable guest allocation supplied this LOAD)"
        ));
    }
    GvaLoadResolution {
        identity: None,
        guest_seed: None,
        cpu_seed: seed,
    }
}

pub(crate) struct PreparedM2vDraw {
    draw: PreparedDraw,
    task_id: u32,
    pipeline_ref: u32,
    width: u32,
    height: u32,
    census_verbose: bool,
}

impl PreparedM2vDraw {
    fn into_submission_parts(self) -> (PreparedDraw, PreparedM2vMetadata) {
        (
            self.draw,
            PreparedM2vMetadata {
                task_id: self.task_id,
                pipeline_ref: self.pipeline_ref,
                width: self.width,
                height: self.height,
                census_verbose: self.census_verbose,
            },
        )
    }

    fn execute(self, state: &mut Device) -> Result<M2vDrawSpan, ExecutorDiagnostic> {
        let (draw, meta) = self.into_submission_parts();
        let completed = draw.execute(state, meta.task_id)?;
        Ok(finish_m2v_draw(
            meta.pipeline_ref,
            meta.width,
            meta.height,
            meta.census_verbose,
            completed,
        ))
    }

    pub(crate) fn resident_chain_identity(&self) -> Option<&crate::model::TargetIdentity> {
        self.draw.resident_chain_identity()
    }

    pub(crate) fn is_surface_store(&self) -> bool {
        self.draw.is_surface_store()
    }

    pub(crate) fn execute_intermediate(self, state: &mut Device) -> DrawChainResult {
        let pipeline_ref = self.pipeline_ref;
        match self.execute(state) {
            Ok(span) => intermediate_chain_result(span),
            Err(error) => {
                crate::observe::fail(format!(
                    "linux_m2v_draw reason={} pipe={} detail={error}",
                    crate::observe::Decline::slug(&error),
                    pipeline_ref
                ));
                DrawChainResult::new(
                    EncodeStatus::BackendUnavailable(crate::observe::Decline::slug(&error)),
                    None,
                    None,
                    None,
                )
            }
        }
    }
}

struct PreparedM2vMetadata {
    task_id: u32,
    pipeline_ref: u32,
    width: u32,
    height: u32,
    census_verbose: bool,
}

fn finish_m2v_draw(
    pipeline_ref: u32,
    width: u32,
    height: u32,
    census_verbose: bool,
    completed: CompletedDraw,
) -> M2vDrawSpan {
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Store);
    report_completed_pixels(
        census_verbose,
        pipeline_ref,
        width,
        height,
        &completed.output,
    );
    let visibility_samples = completed.output.occlusion_samples;
    let submission = completed.submission;
    match completed.route {
        DrawCompletionRoute::Pixels => M2vDrawSpan::Pixels {
            submission,
            bytes: completed.output.pixels,
            bgra: completed.output.pixels_bgra,
            visibility_samples,
        },
        DrawCompletionRoute::EffectsOnly => M2vDrawSpan::EffectsOnly {
            submission,
            visibility_samples,
        },
        DrawCompletionRoute::ResidentChain(identity) => M2vDrawSpan::ResidentChain {
            submission,
            identity,
            visibility_samples,
        },
        DrawCompletionRoute::ResidentGvaReadback(identity) => M2vDrawSpan::ResidentGvaReadback {
            submission,
            identity,
            visibility_samples,
        },
        DrawCompletionRoute::ResidentGvaStore(identity) => M2vDrawSpan::ResidentGvaStore {
            submission,
            identity,
            guest_store_pages: completed.output.guest_store_pages,
            visibility_samples,
        },
        DrawCompletionRoute::ResidentSurfaceStore(identity) => M2vDrawSpan::ResidentSurfaceStore {
            submission,
            identity,
            guest_store_pages: completed.output.guest_store_pages,
            guest_store_window: completed.output.guest_store_window,
            visibility_samples,
        },
    }
}

/// One EXEC-owned sequence of prepared Vulkan draws and the metadata needed to
/// turn its exact successful prefix back into guest-ordered draw completions.
pub(crate) struct PreparedM2vSubmission {
    context: reims_vgpu_core::SubmissionContext,
    draws: Vec<PreparedM2vDraw>,
}

pub(crate) struct PreparedM2vProgress {
    pub(crate) completed: Vec<M2vDrawSpan>,
    pub(crate) failure: Option<ExecutorDiagnostic>,
}

impl PreparedM2vSubmission {
    pub(crate) fn new(context: reims_vgpu_core::SubmissionContext) -> Self {
        Self {
            context,
            draws: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, draw: PreparedM2vDraw) {
        self.draws.push(draw);
    }

    pub(crate) fn execute(
        self,
        state: &mut Device,
    ) -> Result<PreparedM2vProgress, ExecutorDiagnostic> {
        let mut submission = PreparedDrawSubmission::new(self.context);
        let mut metadata = Vec::with_capacity(self.draws.len());
        for prepared in self.draws {
            let (draw, meta) = prepared.into_submission_parts();
            submission.push(draw);
            metadata.push(meta);
        }
        let progress = submission.execute(state)?;
        let completed = metadata
            .into_iter()
            .zip(progress.completed)
            .map(|(meta, completed)| {
                finish_m2v_draw(
                    meta.pipeline_ref,
                    meta.width,
                    meta.height,
                    meta.census_verbose,
                    completed,
                )
            })
            .collect();
        Ok(PreparedM2vProgress {
            completed,
            failure: progress.failure,
        })
    }
}

pub(crate) fn intermediate_chain_result(span: M2vDrawSpan) -> DrawChainResult {
    match span {
        M2vDrawSpan::Pixels {
            mut bytes,
            bgra,
            visibility_samples,
            ..
        } => {
            reorder_rb_in_place(&mut bytes, bgra, false);
            DrawChainResult::new(EncodeStatus::Ok, Some(bytes), visibility_samples, None)
        }
        M2vDrawSpan::ResidentChain {
            identity,
            visibility_samples,
            ..
        } => DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, Some(identity)),
        M2vDrawSpan::EffectsOnly {
            visibility_samples, ..
        } => DrawChainResult::new(EncodeStatus::Ok, None, visibility_samples, None),
        M2vDrawSpan::ResidentGvaReadback {
            visibility_samples, ..
        }
        | M2vDrawSpan::ResidentGvaStore {
            visibility_samples, ..
        }
        | M2vDrawSpan::ResidentSurfaceStore {
            visibility_samples, ..
        } => DrawChainResult::new(
            EncodeStatus::BadArgs("intermediate_draw_selected_final_store_route"),
            None,
            visibility_samples,
            None,
        ),
        M2vDrawSpan::None => DrawChainResult::new(
            EncodeStatus::BackendUnavailable("draw_vk_nothing_stored"),
            None,
            None,
            None,
        ),
    }
}

fn try_metal2vulkan_draw<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
) -> Result<M2vDrawSpan, DrawAttemptError> {
    match prepare_metal2vulkan_draw(state, host, req, writeback_guest)? {
        Some(prepared) => prepared.execute(state).map_err(DrawAttemptError::from),
        None => Ok(M2vDrawSpan::None),
    }
}

fn prepare_metal2vulkan_draw<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
) -> Result<Option<PreparedM2vDraw>, DrawAttemptError> {
    // Only the final record of a portability render-pass chain reads back CPU
    // pixels; used by the resident-chain rail below (harmless on other paths).
    let _ = &writeback_guest;
    // Before anything is resolved or uploaded: a bind naming a slot past its
    // argument table refuses the draw, once, for all three classes and both
    // stages. Every consumer below therefore takes the slot as in-range and
    // spells no bound of its own.
    if let Some(bind) = crate::runtime::draw::first_bind_past_table(req) {
        return Err(DrawPreparationDecline::BindSlotPastTable {
            pipeline_ref: req.pipeline_ref,
            bind,
        }
        .into());
    }
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::PipelineGen);
    // Name the color0 GVA target's allocation before anything can render into
    // it, and once: the pinned Store identity, the cross-pass Load identity and
    // the deferred window's stored copy are all keyed on this value, and two
    // walks of one address across a submit are two answers.
    let gva_alloc_generation = gva_alloc_generation(state, host, req);
    // One call for the pipeline descriptor and both translated shaders. It is
    // memoized on the three objects' list entries — see
    // `crate::runtime::pipeline_resolve` for what that identity is and what it
    // does not cover — so the object-list walks, descriptor reads, MTLB reads,
    // AIR carves and content hashes behind it happen once per pipeline object
    // rather than once per draw. The sub-phases below still bracket the parts,
    // so a boot's `chain_phase` line says how much of the span survived.
    let Some(pipeline_plan::PipelinePlan {
        resolved,
        blend_states,
        width: w,
        height: h,
    }) = pipeline_plan::plan_pipeline(state, host, req)?
    else {
        return Ok(None);
    };
    {
        let resource_plan::DrawResourcePlan {
            attributes: attrs,
            storage_buffers: storage,
            sampled_images: images,
            samplers,
            sampler_provenance: sampler_origin,
            vertex_variant,
            fragment_variant,
            fragment_color_input: frag_color_input,
        } = resource_plan::plan_draw_resources(
            state,
            host,
            req,
            &resolved,
            gva_alloc_generation,
            w,
            h,
        )?;
        let v_variant = &vertex_variant;
        let f_variant = &fragment_variant;
        let load_plan::LoadPlan {
            target_rgba8,
            target_guest,
            target_clear,
            color_load_action,
            target_seed_order: seed_order,
            surface_target: iosurface_texture_resident_target,
            load_from_target: chain_load_from_target,
            gva_load_identity,
        } = load_plan::plan_load(
            state,
            host,
            req,
            gva_alloc_generation,
            writeback_guest,
            w,
            h,
        );
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
        crate::runtime::drain::note_store_route(if req.colors.len() > 1 {
            "mrt_draw_multi"
        } else {
            "mrt_draw_single"
        });
        let planned = request_plan::plan_executor_request(
            state,
            host,
            req,
            &resolved,
            request_plan::RequestPlanInputs {
                blend_states,
                attributes: attrs,
                storage_buffers: storage,
                sampled_images: images,
                samplers,
                target_rgba8,
                target_guest,
                target_clear,
                color_load_action,
                target_seed_order: seed_order,
                color_input: frag_color_input,
                gva_alloc_generation,
                writeback_guest,
                iosurface_texture_resident_target,
                chain_load_from_target,
                gva_load_identity,
                width: w,
                height: h,
                program: reims_vgpu_core::PreparedRenderProgram {
                    vertex: v_variant.program.clone(),
                    fragment: f_variant.program.clone(),
                },
            },
        );
        if matches!(
            &planned,
            Err(DrawPreparationDecline::SecondaryTargetUnbuildable { .. })
        ) {
            crate::runtime::drain::note_store_route("mrt_secondary_refused");
        }
        let request_plan::ExecutorRequestPlan {
            request: resources,
            completion_route,
            vertex_count,
            secondary_targets_built,
        } = planned?;
        if secondary_targets_built {
            crate::runtime::drain::note_store_route("mrt_secondary_built");
        }
        let census_verbose = observe_prepared_resources(
            state,
            req,
            &resources,
            v_variant,
            f_variant,
            &sampler_origin,
            w,
            h,
            vertex_count,
        );
        let render_target_resource = req
            .colors
            .first()
            .and_then(|color| color.resource.as_ref())
            .cloned();
        Ok(Some(PreparedM2vDraw {
            draw: PreparedDraw::new(resources, completion_route, render_target_resource),
            task_id: req.task_id,
            pipeline_ref: req.pipeline_ref,
            width: w,
            height: h,
            census_verbose,
        }))
    }
}

pub(crate) fn prepare_intermediate_draw_chain<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> Result<Option<PreparedM2vDraw>, DrawChainResult> {
    let _phase = crate::runtime::chain_phase::ChainTimer::start();
    let result = prepare_metal2vulkan_draw(state, host, req, false);
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Engine);
    match result {
        Ok(prepared) => Ok(prepared),
        Err(error) => {
            let slug = crate::observe::Decline::slug(&error);
            crate::runtime::drain::note_store_route(slug);
            linux_m2v_draw_failure(&error, req).fail_once(req.pipeline_ref as u64);
            Err(DrawChainResult::new(
                EncodeStatus::BackendUnavailable("draw_vk_nothing_stored"),
                None,
                None,
                None,
            ))
        }
    }
}

/// Prepare a final mapping Store for the EXEC-owned draw submission when no
/// CPU-side CLEAR publication must precede it.
pub(crate) fn prepare_surface_store_draw_chain<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> Result<Option<PreparedM2vDraw>, DrawChainResult> {
    let clear_store_precedes_draw = clear_seed_enabled()
        && req.colors.iter().any(|color| {
            color.publishes_single_sample()
                && load_action_has_clear_seed(color.load_action)
                && color.width != 0
                && color.height != 0
        });
    if clear_store_precedes_draw {
        return Ok(None);
    }
    let _phase = crate::runtime::chain_phase::ChainTimer::start();
    let result = prepare_metal2vulkan_draw(state, host, req, true);
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Engine);
    match result {
        Ok(Some(prepared)) if prepared.is_surface_store() => Ok(Some(prepared)),
        Ok(_) => Ok(None),
        Err(error) => {
            let slug = crate::observe::Decline::slug(&error);
            crate::runtime::drain::note_store_route(slug);
            linux_m2v_draw_failure(&error, req).fail_once(req.pipeline_ref as u64);
            Err(DrawChainResult::new(
                EncodeStatus::BackendUnavailable("draw_vk_nothing_stored"),
                None,
                None,
                None,
            ))
        }
    }
}

/// Land a multi-draw chain image into guest color targets (full-frame store).
/// Used when a later draw in the packet fails after earlier encodes succeeded.
/// Engine-resident identity for a color0 render-pass chain.
///
/// This identity lives only from the first serialized record through its final
/// Store. IOSurface texture targets use their current protocol mapping identity; linear
/// type-2/3 targets use the GVA identity below. Unlike deferred writeback, this
/// lifetime is safe on portability-subset devices because the final record
/// materializes guest bytes before the packet completes.
/// The registry resident this draw's depth/stencil attachment renders into, if
/// the guest's pass descriptor named either texture.
///
/// The attachment is a guest resource with a guest lifetime. The decoded ref is
/// task-local construction input, so it is replaced by the resource graph's
/// canonical index and generation before it becomes an executor identity. Two
/// tasks may legally use the same ref concurrently, and deleting then recreating
/// one ref starts a distinct lifetime; neither may inherit the other's resident
/// depth/stencil contents.
///
/// Geometry and aspect changes still recreate the image, through
/// `ResidentTargetSlot::reusable_for` and the `stencil` field of the key.
pub(super) fn depth_stencil_chain_identity(
    req: &DrawEncodeRequest,
    attachment_ref: u32,
    with_stencil: bool,
    resource: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
) -> Option<crate::model::TargetIdentity> {
    if attachment_ref == 0 {
        return None;
    }
    let c0 = req.colors.first()?;
    let (width, height) = (c0.width, c0.height);
    if width == 0 || height == 0 {
        return None;
    }
    Some(crate::model::TargetIdentity::Texture {
        ref_: resource.index(),
        width,
        height,
        generation: u64::from(resource.generation()),
        stencil: with_stencil,
    })
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

/// Whether a render-pass load action supplies a defined solid seed.
///
/// `DontCare` explicitly leaves the prior attachment contents undefined. It
/// must reach the backend as discard semantics; manufacturing the clear value
/// here changes the guest contract even when that happens to look preferable.
fn load_action_has_clear_seed(action: reims_vgpu_protocol::pass_action::LoadAction) -> bool {
    action == reims_vgpu_protocol::pass_action::LoadAction::Clear
}

#[cfg(test)]
mod execution_split_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn only_clear_has_a_defined_solid_seed() {
        use reims_vgpu_protocol::pass_action::LoadAction;

        assert!(load_action_has_clear_seed(LoadAction::Clear));
        assert!(!load_action_has_clear_seed(LoadAction::Load));
        assert!(!load_action_has_clear_seed(LoadAction::DontCare));
    }

    #[test]
    fn a_gva_store_normalizes_the_executor_reported_texel_order() {
        let mut bgra = [0x33, 0x22, 0x11, 0xff];
        normalize_gva_store_pixels(&mut bgra, true);
        assert_eq!(bgra, [0x11, 0x22, 0x33, 0xff]);

        let mut rgba = [0x11, 0x22, 0x33, 0xff];
        normalize_gva_store_pixels(&mut rgba, false);
        assert_eq!(rgba, [0x11, 0x22, 0x33, 0xff]);
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
        let mut state = Device::new(DeviceId(0), PAGE_SHIFT_X86);
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
        note_guest_rung_blank(
            &state,
            &host,
            1,
            9,
            (gva, w, h),
            &blank,
            SampledByteFormat::synthesised(TexelLayout::Rgba8),
        );
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
        let mut state = Device::new(DeviceId(0), PAGE_SHIFT_X86);
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
        use crate::runtime::host::HostMemory;
        use reims_vgpu_core::endian::st32;
        use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
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
                storage: crate::runtime::draw::ColorTargetStorage::Linear(
                    crate::runtime::draw::LinearColorTarget::whole(0x1000, 32, 8),
                ),
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        state.define_task(1, 0x1_0000, 2);
        state.register_test_resource(1, 7);

        let req = one_page_gva_request();
        let gen_a = super::gva_alloc_generation(&mut state, &mut host, &req);
        assert_ne!(gen_a, 0, "a fully walked GVA span must name its allocation");
        let id_a = super::gva_chain_identity(state.executor.as_ref(), &req, gen_a)
            .expect("a GVA color0 has a chain identity");

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
        let id_b = super::gva_chain_identity(state.executor.as_ref(), &req, gen_b)
            .expect("a GVA color0 has a chain identity");
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
        let identity = |generation: u64| crate::model::TargetIdentity::Gva {
            gva: 0x8000,
            width: 64,
            height: 64,
            generation,
            format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
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
        assert!(!iosurface_texture_resident_is_current(None, None));
        assert!(!iosurface_texture_resident_is_current(None, Some(0)));
        assert!(!iosurface_texture_resident_is_current(Some(7), None));
    }

    #[test]
    fn an_engine_owned_load_uses_copy_currency() {
        for copied_answer in [false, true] {
            assert_eq!(
                iosurface_texture_load_resident_is_current(|| copied_answer),
                copied_answer
            );
        }
    }

    /// Epoch 0 is a legal mapping value — "nothing published since attach" —
    /// and must not be matchable by a slot's unstamped default. It is only
    /// current against a resident explicitly stamped with 0.
    #[test]
    fn epoch_zero_is_current_only_against_an_explicit_stamp() {
        assert!(iosurface_texture_resident_is_current(Some(0), Some(0)));
        assert!(!iosurface_texture_resident_is_current(Some(0), None));
    }

    /// The elision is exact equality, not "at least as new". A resident stamped
    /// at an older epoch has been overtaken by some writer — a blit, a compute
    /// writeback, a guest CPU write, a sibling geometry's publish — and must
    /// fall back to the CPU seed.
    #[test]
    fn any_epoch_movement_since_the_stamp_refuses_the_elision() {
        assert!(iosurface_texture_resident_is_current(Some(4), Some(4)));
        assert!(!iosurface_texture_resident_is_current(Some(5), Some(4)));
        assert!(!iosurface_texture_resident_is_current(Some(4), Some(5)));
    }

    /// Every guest-page writer in this crate goes through
    /// `mark_mapping_written`, so making that advance the surface epoch is what
    /// closes the writer set without enumerating it. A blit or a guest CPU
    /// write invalidates a stamp without knowing this rail exists.
    #[test]
    fn a_guest_page_write_advances_the_surface_epoch() {
        let mut state = Device::new(DeviceId(0), PAGE_SHIFT_X86);
        state.set_mapping_geom(7, 8, 4, 0x1e);
        let before = state
            .surfaces
            .mappings
            .get(&7)
            .unwrap()
            .content
            .surface_epoch;

        let published = state.mark_mapping_written(7);
        let after = state
            .surfaces
            .mappings
            .get(&7)
            .unwrap()
            .content
            .surface_epoch;

        assert!(published > 0, "content_generation still advances");
        assert_ne!(
            before, after,
            "a guest-page write must invalidate any resident stamp"
        );
    }

    /// The deferred IOSurface texture publish writes only the host shadow — no guest
    /// page, so `mark_mapping_written` never runs — and it is the one writer
    /// that would otherwise change the mapping's pixels invisibly to the epoch.
    /// `surface_cache` holds one entry per mapping, so a sibling Store at
    /// another geometry replaces the entry an older geometry is compared
    /// against; without this bump that sibling is silent.
    #[test]
    fn a_deferred_publish_advances_the_epoch_without_a_guest_write() {
        let mut state = Device::new(DeviceId(0), PAGE_SHIFT_X86);
        state.set_mapping_geom(7, 8, 4, 0x1e);
        let gen_before = state
            .surfaces
            .mappings
            .get(&7)
            .unwrap()
            .content
            .guest_page_generation;

        let first = state.note_surface_content_published(7);
        let second = state.note_surface_content_published(7);

        assert_ne!(first, second, "each publish is a distinct epoch");
        assert_eq!(
            gen_before,
            state
                .surfaces
                .mappings
                .get(&7)
                .unwrap()
                .content
                .guest_page_generation,
            "a deferred publish touched no guest page, so content_generation \
             must not move — the compute rail reads it and would re-seed"
        );
    }

    /// Re-attaching a mapping resets the epoch to 0, and 0 is unstampable-by-
    /// default on the resident side, so no resident carried over from the
    /// previous incarnation can vouch for the new one's pixels.
    #[test]
    fn reattaching_a_mapping_resets_the_surface_epoch() {
        let mut state = Device::new(DeviceId(0), PAGE_SHIFT_X86);
        state.set_mapping_geom(7, 8, 4, 0x1e);
        state.mark_mapping_written(7);
        assert_ne!(
            state
                .surfaces
                .mappings
                .get(&7)
                .unwrap()
                .content
                .surface_epoch,
            0
        );

        // A geometry change is a new surface identity; the same reset guards
        // the MAP/UNMAP/reattach paths beside it.
        state.set_mapping_geom(7, 16, 8, 0x1e);
        assert_eq!(
            state
                .surfaces
                .mappings
                .get(&7)
                .unwrap()
                .content
                .surface_epoch,
            0
        );
    }

    /// A record carrying an explicit RT-provenance seed is not a candidate:
    /// that seed was selected for a reason the resident cannot know about, and
    /// the gate must not silently outvote it.
    #[test]
    fn an_explicitly_seeded_load_is_not_an_elision_candidate() {
        let mut c0 = ColorRtRequest {
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
            ..Default::default()
        };
        assert!(iosurface_texture_load_is_a_seed_candidate(&c0));

        c0.target_seed_rgba = Some(vec![0u8; 4]);
        assert!(!iosurface_texture_load_is_a_seed_candidate(&c0));
    }

    /// A CLEAR is not a LOAD. Eliding a seed there would replace the guest's
    /// explicit clear with whatever the resident happened to hold.
    #[test]
    fn a_clear_is_not_an_elision_candidate() {
        let c0 = ColorRtRequest {
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Clear,
            ..Default::default()
        };
        assert!(!iosurface_texture_load_is_a_seed_candidate(&c0));
    }

    #[test]
    fn a_iosurface_texture_load_seed_falls_back_to_the_surfaces_own_guest_pages() {
        use crate::runtime::mapping_write::write_bgra8;
        use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use reims_vgpu_paging::geometry::{
            MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
            MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
        };

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        let mid = 911u32;
        let pfn = 0x21u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(gpa, 0x4000, 0);
        state.map_surface(mid);
        {
            let m = state.surfaces.mappings.get_mut(&mid).unwrap();
            m.lifecycle.active = true;
            m.lifecycle.internal_kva = 1;
            m.pages.entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let (w, h) = (4u32, 2u32);
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
        crate::runtime::guest_ram::latch_import_limits(0x1000, 1 << 30, 1 << 30);

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
        let resident_format = gva_resident_format(state.executor.as_ref(), MTL_FORMAT_BGRA8_UNORM);
        let target = try_iosurface_texture_target_guest_memory(
            &mut state,
            &mut host,
            mid,
            w,
            h,
            resident_format,
        )
        .expect("the stable surface allocation is an importable target");
        assert_eq!(target.backing.allocation_len, 0x1000);
        assert_eq!(target.backing.resource_offset, 0);
        assert_eq!(target.backing.resource_len, 0x1000);
        assert_eq!(target.backing.plane_offset, 0);
        let (_, bpr, _) = crate::runtime::mapping_write::iosurface_texture_sample_window(
            &state.surfaces.mappings[&mid],
            w,
            h,
            MTL_FORMAT_BGRA8_UNORM,
        )
        .expect("the target used this mapping window");
        assert_eq!(target.backing.row_pitch, u64::from(bpr));
        assert_eq!(target.footprint.pages(), &[gpa]);
        let served =
            resolve_iosurface_texture_load_seed(&mut state, &mut host, mid, w, h, resident_format);
        let seed = served.unwrap_or_else(|| {
            panic!(
                "a cold cache must not lose the guest's LOAD; sink said {:?}",
                cap.lines()
            )
        });
        drop(cap);
        let IOSurfaceLoadSeed::Guest(seed) = seed else {
            panic!("a cold cache should preserve the native guest-page source");
        };
        assert_eq!(seed.format, reims_vgpu_protocol::TexelLayout::Bgra8);
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
        let IOSurfaceLoadSeed::Host(bytes, order) =
            resolve_iosurface_texture_load_seed(&mut state, &mut host, mid, w, h, resident_format)
                .expect("a warm cache must serve")
        else {
            panic!("a live cache is the freshest rung");
        };
        assert_eq!(order, reims_vgpu_core::SeedOrder::Bgra8);
        assert_eq!(&bytes[..4], &[0xAA, 0xBB, 0xCC, 0xFF]);

        // An extent the surface is not latched at cannot be served by either rung,
        // and refusing is right: a seed of the wrong length is rejected by the
        // engine anyway, and the decline names both geometries.
        assert!(
            resolve_iosurface_texture_load_seed(
                &mut state,
                &mut host,
                mid,
                w,
                h + 1,
                resident_format,
            )
            .is_none(),
            "a mismatched extent must refuse by name, not seed something else"
        );
    }

    /// The IOSurface texture `LOAD` seed branch reports both ways, and the miss arm names
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
    fn the_iosurface_texture_load_seed_branch_reports_both_ways() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mid = 909u32;
        state.map_surface(mid);
        {
            let m = state.surfaces.mappings.get_mut(&mid).unwrap();
            m.lifecycle.active = true;
            m.publish_geometry_for_test(8, 4, 0);
            m.lifecycle.generation = 3;
        }
        crate::runtime::surface_cache::store(&mut state, mid, 8, 4, vec![0u8; 8 * 4 * 4]);

        // The captured lines carry the sink's `OFF `/`FAIL ` severity prefix, so
        // match on the event token rather than on the first word.
        let only = |cap: &crate::observe::sink::FailCapture| -> String {
            let hits: Vec<String> = cap
                .lines()
                .into_iter()
                .filter(|l| l.contains("iosurface_texture_load_seed"))
                .collect();
            assert_eq!(hits.len(), 1, "expected exactly one line, got {hits:?}");
            hits.into_iter().next().unwrap_or_default()
        };

        let cap = crate::observe::sink::FailCapture::start();
        note_iosurface_texture_load_seed(&state, mid, 8, 4, Some(IOSurfaceSeedRung::Cache));
        let hit = only(&cap);
        assert!(hit.contains("outcome=cache_hit"), "{hit}");
        assert!(hit.contains("mapgeom=8x4"), "{hit}");
        assert!(hit.contains("mapgen=3"), "{hit}");
        drop(cap);

        // The recovered arm is its own outcome, not folded into `cache_hit`: its
        // rate is the only thing that prices the guest-pages fallback, and fusing
        // it would make the fix unmeasurable the moment it worked.
        let cap = crate::observe::sink::FailCapture::resume();
        note_iosurface_texture_load_seed(&state, mid, 4, 4, Some(IOSurfaceSeedRung::GuestPages));
        let pages = only(&cap);
        assert!(pages.contains("outcome=guest_pages"), "{pages}");
        drop(cap);

        // Same mapping, a geometry the cache does not hold: the entry's own
        // geometry is the load-bearing field, since it says a Store at another
        // extent orphaned every window still living at this one.
        let cap = crate::observe::sink::FailCapture::resume();
        note_iosurface_texture_load_seed(&state, mid, 8, 1, None);
        let geom = only(&cap);
        assert!(
            geom.contains("reason=iosurface_texture_seed_cache_geom"),
            "{geom}"
        );
        assert!(geom.contains("have=8x4"), "{geom}");
        assert!(geom.contains("want=8x1"), "{geom}");
        drop(cap);

        // A mapping the cache has never held reports absence, not a geometry.
        let cap = crate::observe::sink::FailCapture::resume();
        note_iosurface_texture_load_seed(&state, 910, 8, 4, None);
        let absent = only(&cap);
        assert!(
            absent.contains("reason=iosurface_texture_seed_cache_absent"),
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
        note_iosurface_texture_load_seed(&state, mid, 8, 4, Some(IOSurfaceSeedRung::Cache));
        note_iosurface_texture_load_seed(&state, mid, 4, 4, Some(IOSurfaceSeedRung::GuestPages));
        note_iosurface_texture_load_seed(&state, mid, 8, 1, None);
        note_iosurface_texture_load_seed(&state, 910, 8, 4, None);
        assert!(
            cap.lines().is_empty(),
            "second sighting must be latched: {:?}",
            cap.lines()
        );
    }

    #[test]
    fn m2v_draw_runtime_failure_returns_a_typed_decline() {
        use crate::observe::Decline as _;

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let req = DrawEncodeRequest {
            pipeline_ref: 41,
            ..DrawEncodeRequest::default()
        };

        let err = match try_metal2vulkan_draw(&mut state, &mut host, &req, true) {
            Err(err) => err,
            Ok(_) => panic!("an empty state cannot resolve pipeline 41"),
        };
        assert_eq!(err.slug(), "draw_prepare_pipeline_missing");
        assert_eq!(
            linux_m2v_draw_failure(&err, &req).render(),
            "linux_m2v_draw reason=draw_prepare_pipeline_missing \
             task_id=0 pipeline_ref=41 pipe=41 task=0 geom=0x0 vtx=0 inst=0 \
             prim=3 first=0 idx=0 colors=[] vbuf=[] fbuf=[] vtex=[] ftex=[] \
             viewports=[] scissors=[]"
        );
    }

    #[test]
    fn executor_failure_projection_preserves_the_native_diagnostic() {
        use crate::observe::Decline as _;

        let native = reims_vgpu_vulkan::engine::vk_call::exec_submit_device_lost_fixture();
        let expected_slug = native.slug();
        let expected_fields = native.fields();
        let expected_detail = native.to_string();
        let error = DrawAttemptError::from(ExecutorDiagnostic::from_decline(&native));

        assert_eq!(error.slug(), expected_slug);
        assert_eq!(error.fields(), expected_fields);
        assert_eq!(error.to_string(), expected_detail);
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

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
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

        let err = match try_metal2vulkan_draw(&mut state, &mut host, &req, true) {
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
        let err = match try_metal2vulkan_draw(&mut state, &mut host, &req, true) {
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
        note_iosurface_texture_store_route("test_route_a");
        note_iosurface_texture_store_route("test_route_a");
        note_iosurface_texture_store_route("test_route_a");
        note_iosurface_texture_store_route("test_route_b");

        let whole = std::fs::read_to_string(path).expect("fail log");
        let appended = &whole[mark.min(whole.len())..];
        let count = |route: &str| {
            appended
                .lines()
                .filter(|l| l.contains(&format!("iosurface_texture_store_route route={route}")))
                .count()
        };
        assert_eq!(count("test_route_a"), 1, "three calls, one line");
        assert_eq!(count("test_route_b"), 1, "a second route still reports");
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
        use crate::model::TargetIdentity;
        use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        state.define_task(1, 0x1_0000, 2);
        state.register_test_resource(1, 11);

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
                storage: crate::runtime::draw::ColorTargetStorage::Mapping(9),
                width: 8,
                height: 8,
                format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                ..ColorRtRequest::default()
            },
            ColorRtRequest {
                slot: 1,
                texture_ref: 11,
                storage: crate::runtime::draw::ColorTargetStorage::Linear(
                    crate::runtime::draw::LinearColorTarget::whole(0x1000, 32, 8),
                ),
                width: 8,
                height: 8,
                format: reims_vgpu_core::pixel_format::MTL_FORMAT_RG16_FLOAT,
                ..ColorRtRequest::default()
            },
        ];
        // Any identity that is not the secondary's; the function only compares.
        let primary = TargetIdentity::Gva {
            gva: 0xdead_0000,
            width: 8,
            height: 8,
            generation: 0,
            format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
        };

        let mut gen_of = |host: &mut FakeHost| {
            let secs = super::build_secondary_targets(
                &mut state,
                host,
                1,
                &colors,
                &pipeline,
                &primary,
                &[],
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
