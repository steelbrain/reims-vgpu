//! The Vulkan rail's half of the writeback: a resident's pixels into the
//! guest's own pages, with no CPU round trip.
//!
//! [`super`] owns what every rail owes the guest — the settle sites, and the
//! wait that has to happen before any host reader touches pages a GPU may still
//! be writing. That half is neutral: it asks
//! [`crate::backend::Backend::guest_writes_outstanding`] and reads the answer.
//!
//! This half is the copy itself, and it is Vulkan's alone because a resident is:
//! every entry point here takes a
//! [`crate::backend::vulkan::engine::TargetIdentity`], and a rail that holds no
//! residents has nothing to name. The three destinations are the three
//! namespaces the guest renders into — a mapper-ref-texture mapping, a guest-backed
//! surface, and a raw GVA plane — and each is reached only from that rail's own
//! draw, blit or compute path.
//!
//! Split out rather than left as eighteen `cfg`-ed items below the neutral
//! settle. The gate is the module's, once.

use super::*;
use crate::runtime::host::{HostMemory, HostOps};

/// Copy `identity`'s pixels into `mapping_id`'s guest pages.
///
/// `true` when the guest's pages hold the frame. `false` is a real loss and is
/// reported on the failure channel by the arm that refused — the caller has no
/// second copy to fall back to, because this rail never made one.
pub fn store_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    // The GPU writes the guest's pages directly. Tried first because when it
    // works there is nothing left to do: no staging buffer is mapped and no
    // host pass over the frame happens at all.
    match crate::runtime::mapping_write::vulkan::write_bgra8_from_resident_gpu(
        state, host, mapping_id, identity, width, height,
    ) {
        Ok(bytes) => {
            crate::runtime::drain::note_store_route("render_flush_gpu_direct");
            finish(state, mapping_id, identity, bytes as usize, started, false);
            return true;
        }
        Err(decline) => {
            // Latched per mapping as well as per reason: a host without
            // `VK_EXT_external_memory_host` declines every Store of every
            // surface, and a line each would drown the channel.
            crate::observe::Emit::decline("render_flush_gpu_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .fail_once(u64::from(mapping_id));
            crate::runtime::drain::note_store_route("render_flush_gpu_declined");
        }
    }
    // The copying arms. These are the only arms on a host that cannot import
    // guest RAM, and the arm a discrete GPU takes regardless.
    //
    // Borrow the readback where it needs no transformation. The writer below is
    // declared in guest scanout order, so a resident reporting semantic RGBA8
    // owes an R/B exchange first — a whole-frame pass, and `into_bgra8` on an
    // owned copy is its home, so a non-BGRA resident takes the copy rather than
    // teaching the lease to rewrite memory it does not own.
    let bpr = width.saturating_mul(4);
    let write_started = std::time::Instant::now();
    // A refused lease is a *routing* answer, never a loss. The lease is an
    // elision of one whole-frame copy and nothing else, so whatever it declines
    // for, the copying rail below can still serve — and it is strictly the
    // better place to find out that the frame is unreadable, because it is the
    // rail that owns the loss report.
    //
    // Collapsing the refusal into `None` rather than enumerating which refusals
    // are recoverable. That enumeration is what would go stale: the lease
    // refuses a resident wider than four bytes a texel (it hands out a pointer
    // into the slot and so has nowhere to narrow one), and reading that
    // particular `Err` as fatal would lose exactly the frames `read_target`
    // exists to quantize. Any future refusal of a rail that is an optimisation
    // has the same answer.
    //
    // Measured, and only reachable one way. On a host that can import guest RAM
    // every Store lands GPU-direct and this whole tail runs zero times — six
    // driven rails, `render_flush_copied` and `render_flush_leased` exactly zero
    // on all of them. With `REIMS_VGPU_GUEST_IMPORT=off`, macos-26 produces
    // **34 lease refusals and 34 copies, one to one, and zero
    // `render_store_lost`**: every one a Surface resident too wide for the lease
    // to lend. Reading the `Err` as fatal cost 34 frames a boot on exactly the
    // host class that has no second rail, and no boot of a capable host could
    // have shown it.
    let leased = match crate::backend::vulkan::engine::read_target_leased(identity) {
        Ok(leased) => leased,
        Err(decline) => {
            // The typed fields, not `Display`. `TargetReadDecline::UnknownIdentity`
            // carries `diverges`/`asked_gen`/`held_gen` precisely because "not in
            // the registry" is two findings with opposite repairs, and formatting
            // the error dropped all three: a `REIMS_VGPU_GUEST_IMPORT=off` boot
            // produced 80 of these and 80 `render_store_lost` beside them with
            // nothing in the log saying which of the two it was.
            crate::observe::Emit::decline("render_flush_lease_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .off();
            None
        }
    };
    let (ok, frame_len) = match leased {
        Some(leased) if leased.bgra => {
            crate::runtime::drain::note_store_route("render_flush_leased");
            let len = leased.bytes().len();
            let ok = crate::runtime::mapping_write::write_bgra8_uncached(
                state,
                host,
                mapping_id,
                leased.bytes(),
                bpr,
                width,
                height,
            );
            // End the lease before anything below reaches the engine again: the
            // re-stamp in `finish` does, and a holder blocking on the engine
            // lock while a teardown waits for this lease is the deadlock
            // `LeasedFrame` forbids.
            drop(leased);
            (ok, len)
        }
        // The pool declined the lease (uncached readback memory, where reading
        // the mapping in place is the slower shape), or it refused outright, or
        // the resident is not in scanout order. Drop any leased frame first so
        // its slot is back in the pool before the second readback asks for one.
        leased => {
            drop(leased);
            crate::runtime::drain::note_store_route("render_flush_copied");
            match crate::backend::vulkan::engine::read_target(identity) {
                Ok(rb) => {
                    // A mapping is scanout-ordered eight-bit colour, so a native
                    // readback has nothing this rail can land. Named rather than
                    // reinterpreted — the bytes are a texel the mapping cannot
                    // mean.
                    let texel = rb.texel;
                    match rb.into_bgra8() {
                        Some(scanout) => {
                            // Shared rather than owned outright: the write's
                            // tail publishes this frame to the surface cache,
                            // and a cache entry holds its frame behind an `Arc`
                            // precisely so the two can name one allocation
                            // instead of copying it.
                            let bytes = std::sync::Arc::new(scanout);
                            let len = bytes.len();
                            let ok = crate::runtime::mapping_write::write_bgra8_owned(
                                state, host, mapping_id, &bytes, bpr, width, height,
                            );
                            (ok, len)
                        }
                        None => {
                            crate::observe::fail(format!(
                                "render_store_lost reason=readback_texel_not_scanout \
                                 mapping={mapping_id} geom={width}x{height} texel={texel:?}"
                            ));
                            (false, 0)
                        }
                    }
                }
                Err(e) => {
                    // Typed, for the reason the lease decline above is: this is
                    // the one line that says *why* the frame was unreadable, and
                    // `Display` on a `DrawError` prints the slug alone.
                    crate::observe::Emit::decline("render_store_lost", &e)
                        .field("mapping", mapping_id)
                        .field("geom", format!("{width}x{height}"))
                        .field("at", "resident_read")
                        .fail();
                    return false;
                }
            }
        }
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    if !ok {
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=write_refused"
        ));
        return false;
    }
    finish(state, mapping_id, identity, frame_len, started, false);
    true
}

/// Publish a Store for a resident whose attachment memory is the guest mapping.
pub fn store_guest_backed_frame(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    guest_store_recorded: bool,
    guest_store_footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
) -> Result<(), crate::runtime::mapping_write::vulkan::GpuWritebackDecline> {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    let bytes = crate::runtime::mapping_write::vulkan::synchronize_guest_backed_resident(
        state,
        mapping_id,
        identity,
        width,
        height,
        guest_store_recorded,
        guest_store_footprint,
    )?;
    crate::runtime::drain::note_store_route("render_flush_gpu_direct");
    finish(state, mapping_id, identity, bytes as usize, started, true);
    Ok(())
}

fn finish_needs_registry_handoff(guest_backed: bool) -> bool {
    !guest_backed
}

/// Hand the currency witness back to the image the frame came out of, and score
/// the write.
///
/// `write_bgra8_*` ends in `mark_mapping_written`, which advances the mapping's
/// `surface_content_epoch` — correctly, because its guest pages did change. But
/// the *pixels* did not: they are this resident's, copied out of it one
/// statement ago. Leaving the stamp behind invalidates a resident that holds
/// exactly the mapping's content, which costs the next Load its elision and
/// sends it to a CPU seed for bytes it already has.
fn finish(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    frame_len: usize,
    started: std::time::Instant,
    guest_backed: bool,
) {
    // A copied resident needs two pieces of registry state handed back: the
    // mapping epoch says its mirror is current, and clearing sole-copy permits
    // reclaim now that the guest pages hold the pixels. An imported resident is
    // the guest allocation itself. It has no mirror epoch to stamp, and draw
    // completion already leaves it non-sole-copy, so neither registry mutation
    // applies.
    if finish_needs_registry_handoff(guest_backed) {
        if let Some(epoch) = state
            .mappings
            .get(&mapping_id)
            .map(|m| m.surface_content_epoch)
        {
            crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch);
        }
        crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    } else {
        crate::runtime::drain::note_store_route("shared_store_registry_handoff_elided");
    }
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::line(format!(
        "render_store mapping={mapping_id} bytes={frame_len} us={}",
        started.elapsed().as_micros()
    ));
}

#[cfg(test)]
mod guest_backed_finish_tests {
    use super::finish_needs_registry_handoff;

    #[test]
    fn shared_storage_has_no_mirror_or_sole_copy_handoff() {
        assert!(!finish_needs_registry_handoff(true));
        assert!(
            finish_needs_registry_handoff(false),
            "a copied resident still needs its epoch and reclaimability published"
        );
    }
}

/// Why a GVA render Store could not hand its resident straight to the guest's
/// pages, so it fell back to reading the frame back and converting it row by
/// row.
///
/// Every one of these is a routing answer and not a loss — the copying rail
/// still lands the frame — but each costs a blocking GPU→host readback of a
/// whole framebuffer plus a host pass over it, which is the largest single cost
/// this device pays. They are named individually so a boot says which check is
/// holding the volume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GvaWritebackDecline {
    /// The guest declared a destination format that no host texel reproduces
    /// verbatim, so no image→buffer copy can produce it and `convert_rgba8_to_row`
    /// is the only route.
    ///
    /// This used to say "not four bytes of colour", and to name `RGBA16_FLOAT`
    /// as landing here *always*. Both went stale together: a resident now
    /// carries the format the guest declared rather than always being eight bits
    /// per channel, so a half-float destination can be the same bytes as the
    /// image. The rule was never a width — it is whether the destination's texel
    /// and the resident's are one layout, which is
    /// [`crate::backend::vulkan::translate::pixel::verbatim_texel`]'s question
    /// and not this doc's to restate.
    ///
    /// `R16_FLOAT` is what lands here now, twice on a driven macos-26 boot: it
    /// is renderable but deliberately not a byte-copy destination, for the
    /// reason `store_texel_order`'s own doc gives for `RG16_FLOAT`, and no
    /// compute selector names it either.
    FormatNeedsConversion { format: u16 },
    /// The resident's format is not the format the destination stores, so a
    /// byte copy would land the wrong texel. Distinct from the engine's own
    /// check of the same pair: this one is asked before the walk, so a mismatch
    /// costs no page-table work.
    ///
    /// Whole formats and not channel orders. The two differ once a render
    /// target may be wider than eight bits per channel: `RGBA16_FLOAT` and
    /// `RGBA8_UNORM` share an order and are four bytes per texel apart, and this
    /// is the arm that catches a half-float destination whose resident fell back
    /// to eight bits because the host cannot render to the wider format.
    ResidentFormatMismatch {
        held: ash::vk::Format,
        want: ash::vk::Format,
    },
    /// The guest's row pitch is not a whole number of texels, or is narrower
    /// than the frame, so there is no `bufferRowLength` that describes it.
    PitchNotTexels { row_stride: u32 },
    /// The frame's first texel does not start on a texel boundary within its
    /// page. `VkBufferImageCopy::bufferOffset` must be a multiple of the texel
    /// block size, and a copy that ignored this is undefined rather than
    /// misaligned.
    OffsetNotTexelAligned { in_page: u64 },
    /// The command resolved no destination pages before it was submitted, so
    /// there is nothing this rail is authorised to write. The copying rail
    /// treats the same answer as "unbounded" and writes anyway through its own
    /// re-walk; this rail has no second walk to fail closed on, so it declines.
    Unlicensed,
    /// The pre-submit walk did not resolve every page of the destination span,
    /// so its page list cannot be read positionally.
    /// [`crate::runtime::draw::StoreTargetPages::ordered_complete`] states what
    /// a short list would land.
    SpanIncomplete,
    /// The destination span did not become a guest-RAM reference; the inner
    /// refusal names the check, restated here for the reason
    /// `GpuWritebackDecline::GuestRefRefused` restates its own.
    GuestRefRefused {
        refusal: crate::runtime::guest_ram_map::MapRefusal,
    },
    /// The engine declined or the copy failed; the inner error names which.
    Engine {
        inner: crate::backend::vulkan::engine::DrawError,
    },
    /// The **copying** arm could not read the resident back, so neither arm
    /// landed the frame. Unlike every variant above this one, it is a loss: the
    /// direct arm has already declined by the time it can be produced, and there
    /// is no third rail.
    CopiedReadRefused {
        inner: crate::backend::vulkan::engine::DrawError,
    },
    /// The **copying** arm read the frame and could not write it into the guest
    /// pages its licence names. A loss, for the same reason as above.
    CopiedWriteRefused { err: crate::runtime::host::MemError },
}

impl crate::observe::Decline for GvaWritebackDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::FormatNeedsConversion { .. } => "gvawb_format_needs_conversion",
            Self::ResidentFormatMismatch { .. } => "gvawb_resident_format_mismatch",
            Self::PitchNotTexels { .. } => "gvawb_pitch_not_texels",
            Self::OffsetNotTexelAligned { .. } => "gvawb_offset_not_texel_aligned",
            Self::Unlicensed => "gvawb_unlicensed",
            Self::SpanIncomplete => "gvawb_span_incomplete",
            Self::GuestRefRefused { .. } => "gvawb_guest_ref_refused",
            Self::Engine { inner } => crate::observe::Decline::slug(inner),
            Self::CopiedReadRefused { .. } => "gvawb_copied_read_refused",
            Self::CopiedWriteRefused { .. } => "gvawb_copied_write_refused",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unlicensed | Self::SpanIncomplete => Vec::new(),
            Self::FormatNeedsConversion { format } => vec![("fmt", format!("{format:#x}"))],
            Self::ResidentFormatMismatch { held, want } => vec![
                ("resident", format!("{held:?}")),
                ("want", format!("{want:?}")),
            ],
            Self::PitchNotTexels { row_stride } => vec![("bpr", row_stride.to_string())],
            Self::OffsetNotTexelAligned { in_page } => vec![("in_page", in_page.to_string())],
            Self::GuestRefRefused { refusal } => {
                let mut f = vec![("via", crate::observe::Decline::slug(refusal).to_string())];
                f.extend(crate::observe::Decline::fields(refusal));
                f
            }
            Self::Engine { inner } => crate::observe::Decline::fields(inner),
            Self::CopiedReadRefused { inner } => {
                let mut f = vec![("via", crate::observe::Decline::slug(inner).to_string())];
                f.extend(crate::observe::Decline::fields(inner));
                f
            }
            Self::CopiedWriteRefused { err } => vec![("err", format!("{err:?}"))],
        }
    }
}

crate::observe::decline::decline_display!(GvaWritebackDecline);

/// Land a GVA render Store's frame in the guest's pages, GPU-direct where the
/// host allows it and by reading the resident back where it does not.
///
/// # The ladder, and why it is here rather than at one call site
///
/// The GPU-direct arm below needs a guest-RAM reference over the destination
/// pages, which is `VK_EXT_external_memory_host` or nothing. On a host without
/// that extension — and on every discrete GPU, where `linear_target_import`
/// refuses `UnsupportedTopology` — it declines every Store of every GVA target
/// with `gvawb_guest_ref_refused via=guest_ram_map_no_backend_import`.
///
/// That decline used to be a **lost frame** for one of the two callers. The
/// eager site in `draw::vulkan` carries its own copying rail (`gva_store_sync`:
/// read the resident back, then `write_gva_rgba8_within`), so it never sees
/// one; the deferred site — [`crate::runtime::writeback_debt::pay_for_texture`],
/// which pays a resource's debt at the moment the guest samples it — had
/// nothing behind the decline and reported `gvadebt_pay_lost`. A driven
/// macos-13 boot under `REIMS_VGPU_GUEST_IMPORT=off` read **27
/// `gvadebt_paid_named` and 27 `gvadebt_pay_lost`**: every deferred payment on
/// that host class lost its frame, and because the failure also released the
/// debt, the sampled rail then read guest pages the frame had never reached.
/// Thirteen of the twenty-seven were one texture of one task re-rendered over
/// and over, which on screen is a window whose content stays blank while its
/// chrome draws.
///
/// So the copying arm lives here, in the function that owns "land this GVA
/// Store", and both callers get it. A decline of the direct arm is a *cost* —
/// a blocking readback plus a host pass over the frame — and never a loss.
///
/// # What the licence is
///
/// A mapping carries its own page list and a page-table vouch licenses it; a
/// GVA carries neither, so the licence is the exact page list supplied in
/// `pages`, which **neither** arm may widen. The eager fallback captures it
/// before draw submission; deferred payment gets it from the live resource's
/// transfer backing. `pages == None` is therefore still `Unlicensed` on both
/// arms: there is no authorisation to write anywhere, and a copy is not a
/// second opinion about that.
/// # What `skip` is
///
/// Bytes from `c0.target_gva` this store may not write, because the guest's own
/// memory holds something newer there. Every eager caller passes `&[]`: a Store
/// landing at the moment it is issued has nothing to make room for. The
/// deferred caller can have something, and the empty slice is the wrong answer
/// for it — see `writeback_debt::pay_gva`.
///
/// A non-empty `skip` forces the copying rail. The GPU-direct arm copies the
/// resident into the plane as one image region and has no way to exclude pages
/// from it, so a caller that must exclude some pays the readback. That is the
/// right price: the alternative is losing one of the two writers, and it is
/// paid only when the hypervisor has actually reported a guest write into a
/// plane this device still owes a frame for.
#[allow(
    clippy::too_many_arguments,
    reason = "the store's own parameters, plus the bytes its owner may not overwrite"
)]
pub(crate) fn store_gva_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
    skip: crate::runtime::mapping_write::SkipRanges<'_>,
) -> Result<u64, GvaWritebackDecline> {
    let direct = if skip.is_empty() {
        store_gva_frame_direct(state, host, task_id, identity, c0, texture_ref, pages)
    } else {
        crate::runtime::drain::note_store_route("gva_flush_skipping");
        Err(GvaWritebackDecline::Unlicensed)
    };
    let decline = match direct {
        Ok(extent) => {
            crate::runtime::drain::note_store_route("gva_flush_gpu_direct");
            return Ok(extent);
        }
        Err(decline) => decline,
    };
    // Latched per target GVA as well as per reason, for the reason
    // `store_render_frame`'s twin latch gives: a host without the import
    // declines every Store of every target, and a line each would drown the
    // channel.
    crate::observe::Emit::decline("gva_flush_gpu_declined", &decline)
        .field("gva", format!("{:#x}", c0.target_gva))
        .field("geom", format!("{}x{}", c0.width, c0.height))
        // The destination's declared format and the resident's own. The copying
        // arm below converts by `into_rgba8` and then
        // `convert_rgba8_to_row(c0.format)`, and whether those two cancel is
        // decided entirely by this pair — so without it a channel-order defect
        // on this rail is a screenshot and no reading.
        .field("fmt", format!("{:#x}", c0.format))
        .field("resident", format!("{:?}", identity.resident_format()))
        .fail_once(c0.target_gva);
    crate::runtime::drain::note_store_route("gva_flush_gpu_declined");
    let Some(pages) = pages else {
        return Err(GvaWritebackDecline::Unlicensed);
    };
    // The blocking readback the direct arm exists to avoid. `into_rgba8` is the
    // order every GVA guest writer takes, and it exchanges or not according to
    // the order the engine reports for the image it copied — which for a target
    // the guest declared in BGRA order, as most of them are, is a whole-frame
    // pass and not the no-op this comment used to claim. Both spellings of that
    // declaration must reach the same answer; `ResidentReadSnapshot::bgra` is
    // where they do, and where they did not.
    // Ask for the resident's own texel only where this destination can land
    // one. `store_texel_order` is the same table the verbatim arm below checks,
    // so the two cannot disagree: a format it does not name — `R8Unorm` is one —
    // has no native landing and must take the narrowed frame and the CPU row
    // converter, exactly as it did before. Asking for native there turned every
    // deferred payment of such a target into `gvadebt_pay_lost`.
    let destination = crate::protocol::pixel_format::store_texel_order(c0.format);
    let readback = match destination.filter(|layout| !layout.is_four_byte_color()) {
        Some(_) => crate::backend::vulkan::engine::read_target_native(identity),
        None => crate::backend::vulkan::engine::read_target(identity),
    }
    .map_err(|inner| GvaWritebackDecline::CopiedReadRefused { inner })?;
    // Two ways to land a frame, and which one is decided by what the readback
    // actually holds rather than by what this rail wishes it held.
    //
    // A native frame is the resident's own texel, carried out because no
    // eight-bit narrowing of it exists. It lands verbatim **only** when the
    // destination is that same layout — `store_texel_order`'s question, the one
    // the GPU-direct arm has always asked. That is what lets a destination with
    // no eight-bit form take the copying rail at all, so a host without the
    // guest-RAM import serves it instead of losing every frame of it.
    let extent = match readback.texel.native_layout() {
        Some(layout) => {
            if destination != Some(layout) {
                return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
            }
            crate::runtime::drain::note_store_route("gva_flush_copied_native");
            land_gva_frame_bytes(
                state,
                host,
                task_id,
                c0,
                texture_ref,
                crate::runtime::draw::FrameRows::Native(&readback.pixels),
                pages,
                skip,
            )?
        }
        None => {
            let Some(rgba) = readback.into_rgba8() else {
                return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
            };
            land_gva_frame_bytes(
                state,
                host,
                task_id,
                c0,
                texture_ref,
                crate::runtime::draw::FrameRows::Rgba8(&rgba),
                pages,
                skip,
            )?
        }
    };
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    crate::runtime::drain::note_store_route("gva_flush_copied");
    Ok(extent)
}

/// The copying arm's body: convert `rgba` into the destination's own texel
/// order and write it into the guest pages `pages` licenses.
///
/// Split out from [`store_gva_frame`] at the readback boundary so it can be
/// tested against real guest memory without a device — the readback is the only
/// part of the arm that needs one.
///
/// It forgets both GVA pixel caches on the way out, exactly as the direct arm
/// does and for the same reason: after this call the guest's own pages hold the
/// frame, and a stale entry left behind would serve a later sample the previous
/// Store's bytes. `write_gva_rgba8_within` records the host write itself, from
/// inside `gva_view`, so the two arms leave the same witness behind and a
/// decline is invisible to every reader of `gather_witness`.
#[allow(
    clippy::too_many_arguments,
    reason = "the destination's geometry, plus the bytes its owner may not overwrite"
)]
fn land_gva_frame_bytes<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    frame: crate::runtime::draw::FrameRows<'_>,
    pages: &crate::runtime::draw::StoreTargetPages,
    skip: crate::runtime::mapping_write::SkipRanges<'_>,
) -> Result<u64, GvaWritebackDecline> {
    // The destination extent in the destination's own bytes, which is what the
    // direct arm returns too — the two rails must agree about how much guest
    // memory a Store of this geometry lands, or a caller could tell them apart.
    let Some(tight) = crate::protocol::pixel_format::tight_row_bytes(c0.width, c0.format) else {
        return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
    };
    let extent =
        u64::from(c0.height.saturating_sub(1)) * u64::from(c0.row_stride) + u64::from(tight);
    crate::runtime::draw::write_gva_frame_within_skipping(
        state,
        host,
        task_id,
        c0.target_gva,
        c0.width,
        c0.height,
        c0.row_stride,
        c0.format,
        frame,
        Some(pages.membership()),
        skip,
    )
    .map_err(|err| GvaWritebackDecline::CopiedWriteRefused { err })?;
    crate::runtime::surface_cache::forget_gva_copies(state, task_id, c0.target_gva, texture_ref);
    Ok(extent)
}

/// Copy `identity`'s pixels into the guest pages behind a normal-texture render
/// target's `target_gva`, with no host copy of the frame at any point.
///
/// The GVA twin of [`store_render_frame`]'s first arm, and worth diffing
/// against it: both end in `copy_target_to_guest_pages` and they differ only in
/// how the destination pages are named.
///
/// # Why this is the whole cost of a GVA synchronization
///
/// The arm behind this one reads the resident back to the host (a blocking
/// fence) and then writes it out again a row at a time through
/// `convert_rgba8_to_row`. On a driven desktop-compositing boot that is 59 % of
/// render Stores and most of the time the device spends waiting on a fence.
/// Everything this call declines on pays that instead.
///
/// # What it does not do
///
/// It publishes no host-side copy of the frame. The two GVA pixel caches are
/// *dropped* instead, for the same reason `store_render_frame` forgets the
/// mapping's: after this call the guest's own pages are the only place the
/// frame exists, and an entry left behind would serve a later sample the
/// previous Store's bytes. The sampled rail re-reads them from guest memory,
/// which is what `store_gva_owned`'s `guest_holds_bytes` already promised.
fn store_gva_frame_direct<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<u64, GvaWritebackDecline> {
    copy_resident_into_gva_plane(
        state,
        host,
        task_id,
        identity,
        &GvaPlaneDestination {
            target_gva: c0.target_gva,
            width: c0.width,
            height: c0.height,
            row_stride: c0.row_stride,
            format: c0.format,
            texture_ref,
        },
        pages,
    )
}

/// A guest-linear plane a resident's pixels may be copied into.
///
/// The five terms [`copy_resident_into_gva_plane`] needs and nothing else. It
/// exists because that rail used to take a [`crate::runtime::draw::ColorRtRequest`],
/// which describes a *render attachment* — a slot, load and store actions, a
/// clear colour, a multisample source — and a second caller with the same
/// destination and no render pass behind it could not honestly fill one in.
///
/// `texture_ref` is the resource whose host-side pixel caches this copy
/// invalidates, and is `0` where the caller has none to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GvaPlaneDestination {
    pub target_gva: u64,
    pub width: u32,
    pub height: u32,
    pub row_stride: u32,
    /// The guest's own declaration for these bytes, which is what it will read
    /// them back as. A copy converts nothing, so this is the format the
    /// resident must already hold.
    pub format: u16,
    pub texture_ref: u32,
}

/// What a [`GvaPlaneDestination`]'s own terms imply about the bytes it names.
///
/// `extent` is the reason this is a type rather than three lines inside the copy
/// below. A caller has to walk the guest's page table before it can hand over a
/// licence, and the span it walks must be the span the copy writes — a walk one
/// row short resolves fewer pages than `ordered_complete` demands and the copy
/// declines, while a walk that is longer authorises pages the frame never
/// reaches. Deriving it in one pure place is the only way the two cannot drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GvaPlaneGeometry {
    /// The Vulkan format the resident must already hold.
    pub want: ash::vk::Format,
    pub bpt: u64,
    pub row_stride: u64,
    /// The bytes from `target_gva` this plane occupies — the extent the copy
    /// names and nothing past it. Padding after the final row belongs to the
    /// allocation but is not texels this destination was given, and the copying
    /// rail leaves it alone too. The two rails must land identical guest memory
    /// or a fallback would be visible.
    pub extent: u64,
}

impl GvaPlaneDestination {
    /// The geometry, or the typed reason this destination has none.
    ///
    /// Pure, so a caller may ask for the span it must walk and
    /// [`copy_resident_into_gva_plane`] may ask again for the copy it issues,
    /// without either restating the rule.
    pub(crate) fn geometry(&self) -> Result<GvaPlaneGeometry, GvaWritebackDecline> {
        // The destination's texel, and the whole reason this rail can exist at
        // all: a copy converts nothing, so the guest must already read these
        // bytes exactly as the resident holds them.
        //
        // Asked of every rail that creates images, not of the render Store's
        // table alone — this destination serves a compute storage output as
        // readily as a Store, and a storage image is a thing the guest neither
        // renders into nor samples. See
        // [`crate::backend::vulkan::translate::pixel::verbatim_texel`].
        let Some((want, bpt)) =
            crate::backend::vulkan::translate::pixel::verbatim_texel(self.format)
        else {
            return Err(GvaWritebackDecline::FormatNeedsConversion {
                format: self.format,
            });
        };
        let bpt = u64::from(bpt);
        let row_stride = u64::from(self.row_stride);
        if row_stride == 0
            || !row_stride.is_multiple_of(bpt)
            || row_stride < u64::from(self.width) * bpt
        {
            return Err(GvaWritebackDecline::PitchNotTexels {
                row_stride: self.row_stride,
            });
        }
        Ok(GvaPlaneGeometry {
            want,
            bpt,
            row_stride,
            extent: u64::from(self.height.saturating_sub(1)) * row_stride
                + u64::from(self.width) * bpt,
        })
    }
}

/// Copy `identity`'s pixels into the guest pages behind a linear destination,
/// with no host copy of the frame at any point.
///
/// The body of [`store_gva_frame_direct`], reachable by any caller holding a
/// resident and a licensed guest-linear plane. A render Store is one such
/// caller; a `copyFromTexture:toTexture:` whose source is an IOSurface the GPU
/// already holds and whose destination is a linear guest allocation is another,
/// and it is the same copy — the two differ only in where the geometry and the
/// page licence came from.
/// A licensed direct-to-guest-pages destination: where the bytes go, which
/// pages the licence covers, and how many bytes land.
///
/// The two rails that write guest pages from the GPU — a render Store and a
/// compute storage-image output — differ only in the image they copy *from* and
/// in which command buffer records the copy. Everything about the destination is
/// this, so it is derived once and the second rail cannot spell any of it
/// differently.
pub(crate) struct GvaPlaneLicence {
    pub target: crate::backend::vulkan::engine::GuestPageTarget,
    /// The pages walked, in guest-virtual order — what the copy is licensed
    /// over and what every witness on this path is armed against.
    pub gpas: Vec<u64>,
    /// The bytes from `target_gva` the copy will land. The caller's return
    /// value, and what the counters on this path are charged.
    pub extent: u64,
}

/// Resolve a guest-linear plane into a destination the GPU may copy into, or
/// the typed reason it may not.
///
/// `held` is the format the source image actually holds. **A copy converts
/// nothing** — neither `vkCmdCopyImageToBuffer` nor the blit behind the
/// rectangle plan performs a channel swap or a texel resize — so a source whose
/// format is not the one the guest will read these bytes back as must be refused
/// here rather than landed. That is the whole of the format contract on this
/// rail, and it is stated once so neither caller can state it differently.
///
/// The comparison is between whole formats, not channel orders. While every
/// resident was eight bits per channel an order was a complete description of
/// one, so `is_bgra() != (order == Bgra8)` decided this. It is not any more:
/// `RGBA16_FLOAT` and `RGBA8_UNORM` are both RGBA-ordered and are four bytes per
/// texel apart, so an order comparison would admit a half-float destination over
/// an eight-bit source and copy a frame of the wrong size and the wrong texel
/// into the guest's pages.
pub(crate) fn licence_gva_plane<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    held: ash::vk::Format,
    c0: &GvaPlaneDestination,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<GvaPlaneLicence, GvaWritebackDecline> {
    let GvaPlaneGeometry {
        want,
        bpt,
        row_stride,
        extent,
    } = c0.geometry()?;
    // On the render rail this is a healthy zero as things stand:
    // `gva_chain_identity` builds the key from this same `c0.format`, so the two
    // agree by construction and this arm is the alarm for an identity that came
    // from somewhere else. It is also the arm that catches the honest
    // disagreement — a guest format whose resident fell back to eight bits
    // because the host cannot render to it — and sends that Store down the CPU
    // conversion rail where it belongs. On the compute rail it is not a healthy
    // zero: a storage image takes its format from the specialized SPIR-V texel
    // format, which can legitimately differ from the guest surface's, and this
    // is what stops those bytes being landed raw.
    if held != want {
        return Err(GvaWritebackDecline::ResidentFormatMismatch { held, want });
    }
    let Some(pages) = pages else {
        return Err(GvaWritebackDecline::Unlicensed);
    };
    let page_size = state.page_size();
    let Some(gpas) = pages.ordered_complete(c0.target_gva, page_size) else {
        return Err(GvaWritebackDecline::SpanIncomplete);
    };
    let in_page = c0.target_gva % page_size;
    if !in_page.is_multiple_of(bpt) {
        return Err(GvaWritebackDecline::OffsetNotTexelAligned { in_page });
    }
    let runs =
        crate::runtime::guest_ram_map::references_for_runs(host, gpas, page_size, in_page, extent)
            .map_err(|refusal| GvaWritebackDecline::GuestRefRefused { refusal })?;
    let target = crate::backend::vulkan::engine::GuestPageTarget {
        runs,
        // Checked above to divide exactly, so this is the guest's pitch and not
        // a rounding of it.
        row_length_texels: (row_stride / bpt) as u32,
        width: c0.width,
        height: c0.height,
        format: want,
        shared_backing: None,
    };
    // This device is about to write these guest pages, and the hypervisor's
    // dirty bitmap is defined not to see it. Without this record a reader
    // holding a gathered image over the same pages
    // (`crate::runtime::gather_witness`) cannot tell "nobody wrote them" from
    // "we wrote them ourselves", and vouches a retained image that no longer
    // matches the pages — a wrong frame that then persists.
    //
    // The copying rail this stands in front of records the identical fact from
    // inside `gva_view`, so the two rails leave the same witness behind and a
    // decline is invisible to every reader. Before the submit and not after it,
    // and over the whole walked span rather than the copy's extent: a spurious
    // bump costs a re-read of bytes that did not change, and the opposite error
    // hands out a stale copy. It is armed here, past the last refusal, so a
    // caller that goes on to fail its submit has over-recorded rather than
    // under-recorded — the direction that costs a re-read instead of a wrong
    // frame.
    state.note_host_wrote_pages(gpas.to_vec());
    // The boot-wide write footprint, so a panic's page number can be looked up
    // against this rail too. The mapper-ref licence marks it; this one did not,
    // and a page this rail wrote read as "never written by this device".
    // Marked over the copy's extent within the walked pages, before the
    // submit — the over-recording direction, as the mapper's own note says.
    crate::runtime::mapper::note_page_write_footprint(page_size, in_page, extent, |i| {
        gpas.get(i).map(|&gpa| Some(gpa))
    });
    Ok(GvaPlaneLicence {
        target,
        gpas: gpas.to_vec(),
        extent,
    })
}

pub(crate) fn copy_resident_into_gva_plane<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &GvaPlaneDestination,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<u64, GvaWritebackDecline> {
    let licence = licence_gva_plane(state, host, identity.resident_format(), c0, pages)?;
    let GvaPlaneLicence {
        target,
        gpas,
        extent,
    } = &licence;
    crate::backend::vulkan::engine::copy_target_to_guest_pages(identity, target, gpas)
        .map_err(|inner| GvaWritebackDecline::Engine { inner })?;
    let extent = *extent;
    // Nothing here leaves a host copy of the frame, so neither GVA-keyed cache
    // may go on naming one.
    crate::runtime::surface_cache::forget_gva_copies(state, task_id, c0.target_gva, c0.texture_ref);
    // The copy means this image has stopped being the only place these pixels
    // exist, so the reclaim paths may take it — the same handover
    // `store_render_frame` performs in `finish`.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    // Arm the GVA write witness over the pages this Store just published, the
    // twin of `mapper::stamp_guest_write_gen` on the mapper-ref-texture rail. It is what
    // lets a later reader ask whether these pages still hold this frame without
    // reading them — see `crate::runtime::gva_store_witness`.
    //
    // After the submit, not before it: a stamp taken ahead of a copy that then
    // declines would claim the guest's pages hold a frame that never reached
    // them. And after `note_host_wrote_pages` above, because the epoch the
    // witness records is compared against that same ring — capturing it first
    // would have every target permanently invalidated by its own Store.
    //
    // Only this rail stamps. Both copying arms — the eager `gva_store_sync` and
    // [`land_gva_frame_bytes`] behind this call — leave no witness, so their
    // targets never read quiet and never take the shortcut this arms. That is
    // safe and deliberate rather than an oversight: it is the arm a host without
    // the guest-RAM import takes, and it already pays a blocking readback per
    // Store, so the shortcut is worth less there and the rails stay easier to
    // tell apart. The frame is in the guest's pages either way; what a missing
    // stamp costs is a re-read, never a wrong image.
    if let Some(key) = crate::backend::vulkan::gva_witness_key(identity) {
        crate::runtime::gva_store_witness::note_store(state, host, key, gpas);
    }
    Ok(extent)
}

#[cfg(test)]
mod gva_copying_arm_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::draw::{ColorRtRequest, StoreTargetPages};
    use crate::runtime::gva_mem::define_task_pages_arm64e;
    use crate::runtime::host::FakeHost;

    /// The first data page `define_task_pages_arm64e` installs, as a PFN. The
    /// task's GVA page `i` is this plus `i`, which is what lets a test name a
    /// destination page and the licence for it from one number.
    const DATA_BASE_PFN: u64 = 4;
    const PAGES: u32 = 8;
    /// GVA page 1, so the destination is neither the null address
    /// `write_gva_rows_within` rejects nor the first page of the walk.
    const TARGET_GVA: u64 = 1 << PAGE_SHIFT_ARM64E;
    const W: u32 = 4;
    const H: u32 = 2;
    const BPR: u32 = 16;

    fn fixture() -> (FakeHost, DeviceState) {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        define_task_pages_arm64e(&mut host, &mut state, DATA_BASE_PFN as u32, PAGES);
        (host, state)
    }

    fn request() -> ColorRtRequest {
        ColorRtRequest {
            texture_ref: 0,
            target_gva: TARGET_GVA,
            row_stride: BPR,
            width: W,
            height: H,
            // Byte-identical with the RGBA8 the readback hands over, so the
            // assertion below is about where the bytes landed and not about the
            // conversion, which has its own tests.
            format: crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
            ..Default::default()
        }
    }

    fn frame() -> Vec<u8> {
        (0..(W * H * 4) as u8).map(|b| b.wrapping_add(1)).collect()
    }

    /// The page the destination lives in, as a one-entry licence.
    fn licence_for(page_index: u64) -> StoreTargetPages {
        let span = u64::from(BPR) * u64::from(H);
        StoreTargetPages::from_ordered(&[(DATA_BASE_PFN + page_index) << PAGE_SHIFT_ARM64E], span)
    }

    /// The whole point of the arm: a GVA render Store whose GPU-direct rail
    /// cannot run still reaches the guest's pages.
    ///
    /// Before this arm existed the deferred payer had nothing behind the direct
    /// decline and reported `gvadebt_pay_lost`, so these bytes never left the
    /// resident on any host without `VK_EXT_external_memory_host`.
    #[test]
    fn the_copying_arm_lands_the_frame_in_the_guests_own_pages() {
        let (mut host, mut state) = fixture();
        let rgba = frame();
        let pages = licence_for(1);
        let extent = land_gva_frame_bytes(
            &mut state,
            &mut host,
            1,
            &request(),
            0,
            crate::runtime::draw::FrameRows::Rgba8(&rgba),
            &pages,
            &[],
        )
        .expect("the licensed page is writable");
        assert_eq!(
            extent,
            u64::from(H - 1) * u64::from(BPR) + u64::from(W) * 4,
            "the extent must be the destination's own bytes, as the direct arm reports"
        );
        let gpa = (DATA_BASE_PFN + 1) << PAGE_SHIFT_ARM64E;
        for y in 0..H as usize {
            let mut got = [0u8; (W * 4) as usize];
            crate::runtime::host::HostMemory::read_gpa(
                &host,
                gpa + (y as u64) * u64::from(BPR),
                &mut got,
            )
            .expect("the destination page is guest RAM");
            let at = y * (W * 4) as usize;
            assert_eq!(
                &got[..],
                &rgba[at..at + (W * 4) as usize],
                "row {y} of the frame did not reach the guest"
            );
        }
    }

    /// The copying arm may not widen the licence the direct arm was given.
    ///
    /// A GVA carries no page list of its own, so the supplied one is the entire
    /// authorisation. A second rail that re-walked and wrote wherever the
    /// address points now would be the stale-view class this device already
    /// bounds — so the arm refuses by name and the guest's memory is untouched.
    #[test]
    fn the_copying_arm_refuses_a_page_its_licence_does_not_name() {
        let (mut host, mut state) = fixture();
        let rgba = frame();
        // A licence naming some other page of the same task.
        let pages = licence_for(5);
        let refusal = land_gva_frame_bytes(
            &mut state,
            &mut host,
            1,
            &request(),
            0,
            crate::runtime::draw::FrameRows::Rgba8(&rgba),
            &pages,
            &[],
        )
        .expect_err("the destination page is outside the licence");
        assert_eq!(
            crate::observe::Decline::slug(&refusal),
            "gvawb_copied_write_refused",
            "an unlicensed write is a named refusal, not a silent skip"
        );
        let gpa = (DATA_BASE_PFN + 1) << PAGE_SHIFT_ARM64E;
        let mut got = [0u8; (W * 4) as usize];
        crate::runtime::host::HostMemory::read_gpa(&host, gpa, &mut got)
            .expect("the destination page is guest RAM");
        assert_eq!(
            got,
            [0u8; (W * 4) as usize],
            "nothing may have been written"
        );
    }

    /// A deferred frame landing over pages the guest CPU wrote in between keeps
    /// both writers.
    ///
    /// This is the external relation `cpu_write_after_render` asks for. A
    /// `.shared` texture's storage is guest RAM and the GPU and the guest CPU
    /// are both writers of it; Metal's guarantee is per region, so a CPU write
    /// into one part of a layer the GPU rendered leaves the rest of the GPU's
    /// work standing. A writeback with no third answer had only two, and both
    /// destroy a writer: land the whole frame and the guest's bytes are gone,
    /// drop it and everything the GPU rendered is gone.
    #[test]
    fn a_landed_frame_leaves_the_bytes_the_guest_wrote_alone() {
        let (mut host, mut state) = fixture();
        let gpa = (DATA_BASE_PFN + 1) << PAGE_SHIFT_ARM64E;
        // What the guest CPU put in row 1 after the render and before this
        // payment. Distinct from every byte of the frame.
        let guest_row = [0xEFu8; (W * 4) as usize];
        crate::runtime::host::HostMemory::write_gpa(&mut host, gpa + u64::from(BPR), &guest_row)
            .expect("the destination page is guest RAM");

        let rgba = frame();
        let pages = licence_for(1);
        // Row 1's bytes, in the same coordinate system the row offsets are in.
        let skip = [(u64::from(BPR), u64::from(BPR) + u64::from(W) * 4)];
        land_gva_frame_bytes(
            &mut state,
            &mut host,
            1,
            &request(),
            0,
            crate::runtime::draw::FrameRows::Rgba8(&rgba),
            &pages,
            &skip,
        )
        .expect("the licensed page is writable");

        let mut got0 = [0u8; (W * 4) as usize];
        crate::runtime::host::HostMemory::read_gpa(&host, gpa, &mut got0)
            .expect("the destination page is guest RAM");
        assert_eq!(
            &got0[..],
            &rgba[..(W * 4) as usize],
            "the device's own row must still land"
        );

        let mut got1 = [0u8; (W * 4) as usize];
        crate::runtime::host::HostMemory::read_gpa(&host, gpa + u64::from(BPR), &mut got1)
            .expect("the destination page is guest RAM");
        assert_eq!(
            got1, guest_row,
            "the guest's own row must survive the payment"
        );
    }
}
