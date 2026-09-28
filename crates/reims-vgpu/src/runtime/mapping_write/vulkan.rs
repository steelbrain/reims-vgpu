//! The Vulkan rail's half of the mapping write: a resident's pixels straight
//! into a mapper-ref-texture mapping's guest pages, with the frame never existing on the
//! host.
//!
//! [`super`] owns the neutral half — where a mapping's plane is, what geometry
//! and format a write into it must use, and the CPU write that lands bytes a
//! caller already holds. Every one of those answers is the guest's own
//! declaration and reads the same on any rail.
//!
//! This half is the GPU copy, and it is Vulkan's alone because the thing being
//! copied is: `write_bgra8_from_resident_gpu` takes a `TargetIdentity`, and
//! `licence_mapper_ref_texture_surface` hands back a
//! [`crate::backend::vulkan::engine::GuestPageTarget`] — a permission to write
//! guest pages that only a rail holding an import of them can grant.
//!
//! The gate is the module's, once, rather than thirteen items' each.

use super::*;

/// A check that stopped a resident's frame from reaching the guest's pages
/// without a host copy, so the flush owes the copying rail instead.
///
/// Every variant is a routing answer and not a loss — the caller still lands the
/// frame — but each one is a whole frame's worth of memcpy the device paid twice
/// over, on the rail that is 69% of the drain worker's time. So they are named
/// individually: "the GPU writeback declined" cannot tell a host whose GPU
/// cannot import guest RAM from a surface whose row pitch is not a whole texel,
/// and those have different fixes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuWritebackDecline {
    /// A zero or over-large rect, or a mapping that is gone or unmapped. The
    /// copying rail refuses these too, by its own [`SurfaceWriteRefusal`]; this
    /// path declines before making the guest pay two refusals for one cause.
    NotWritable,
    /// The mapping's declared geometry is not the frame's. The copying rail
    /// reports this as `GeometryMoved`; here it means the same thing and the
    /// same rail will say so.
    GeometryMoved {
        latched_width: u32,
        latched_height: u32,
        frame_width: u32,
        frame_height: u32,
    },
    /// No sample window resolves for this geometry, so there is no destination
    /// offset or row pitch to copy against.
    WindowUnresolved {
        width: u32,
        height: u32,
        format: u16,
    },
    /// The mapping's pixel format is not the one the resident holds, so landing
    /// it needs a per-row conversion. A buffer→image copy performs none, which
    /// is why this is a routing answer rather than something to work around.
    FormatNeedsConversion { format: u16 },
    /// The mapping's declared format has a linear texel to name, and the source
    /// image does not hold it. Same consequence as the variant above and a
    /// different cause: there the guest declared something no copy can express,
    /// here the two sides simply disagree.
    ///
    /// Expected on the compute rail, where a storage image's format comes from
    /// the specialized SPIR-V texel format and owes a surface mapping's
    /// declaration nothing. Whole formats, not channel orders: `RGBA16_FLOAT`
    /// and `RGBA8_UNORM` are both RGBA-ordered and four bytes per texel apart,
    /// so an order comparison would admit a half-float source over an eight-bit
    /// destination and land a frame of the wrong size.
    ResidentFormatMismatch {
        held: ash::vk::Format,
        want: ash::vk::Format,
    },
    /// The guest's row pitch is not a whole number of texels, so it cannot be
    /// expressed as `bufferRowLength`.
    PitchNotTexels { bpr: u32 },
    /// The frame's first texel does not start on a 4-byte boundary within its
    /// page. `VkBufferImageCopy::bufferOffset` must be a multiple of the texel
    /// block size, and a copy that ignored this is undefined rather than
    /// misaligned.
    OffsetNotTexelAligned { in_page: u64 },
    /// The mapping's page list does not cover the sample window.
    PageListShort { need: usize, have: usize },
    /// A page in the window carries no valid entry, so there is no guest page
    /// to resolve a reference against.
    PageUnbacked { index: usize },
    /// The page walk refused: these are no longer provably the mapping's pages.
    /// The copying rail refuses for the same reason and reports it.
    PagesNotOurs,
    /// This window's pages did not become a reference. Carries the check
    /// [`crate::runtime::guest_ram_map`] refused on — including the one that
    /// says nothing about the window at all, that this host cannot import guest
    /// RAM. There is deliberately no separate variant for that: the early-out
    /// above and the walk below now ask one function, so they cannot name the
    /// same host two ways, and `via=` says which check answered either way.
    ///
    /// It carries it rather than pointing at it. That module reports each
    /// distinct refusal **once per boot** — `report_once` latches on
    /// `first_sight` — while this decline is reached once per flush, so a boot
    /// where every 1080p writeback is refused prints twenty of these against a
    /// single `guest_ram_map` line elsewhere in the log. Nothing relates the
    /// two, and the reader's obvious move — ranking `reason=` on the fail
    /// channel — puts the twenty at the top under a name that says the host
    /// cannot import, on a host whose `vk_caps` says `supported`. Restating the
    /// inner check is the cheaper error.
    GuestRefRefused {
        refusal: crate::runtime::guest_ram_map::MapRefusal,
    },
    /// The engine declined or the copy failed; the inner error names which.
    Engine {
        inner: crate::backend::vulkan::engine::DrawError,
    },
}

impl crate::observe::Decline for GpuWritebackDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NotWritable => "gpuwb_not_writable",
            Self::GeometryMoved { .. } => "gpuwb_geometry_moved",
            Self::WindowUnresolved { .. } => "gpuwb_window_unresolved",
            Self::FormatNeedsConversion { .. } => "gpuwb_format_needs_conversion",
            Self::ResidentFormatMismatch { .. } => "gpuwb_resident_format_mismatch",
            Self::PitchNotTexels { .. } => "gpuwb_pitch_not_texels",
            Self::OffsetNotTexelAligned { .. } => "gpuwb_offset_not_texel_aligned",
            Self::PageListShort { .. } => "gpuwb_page_list_short",
            Self::PageUnbacked { .. } => "gpuwb_page_unbacked",
            Self::PagesNotOurs => "gpuwb_pages_not_ours",
            Self::GuestRefRefused { .. } => "gpuwb_guest_ref_refused",
            // The engine's own slug, so a driver that refuses the pointer and a
            // resident in the wrong channel order stay as distinguishable here
            // as they are where they were decided.
            Self::Engine { inner } => crate::observe::Decline::slug(inner),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NotWritable | Self::PagesNotOurs => Vec::new(),
            // `via` before the inner fields, so the check that refused reads
            // first and its own `pages=` / `first=` qualify it rather than
            // looking like this rail's own numbers.
            Self::GuestRefRefused { refusal } => {
                let mut f = vec![("via", crate::observe::Decline::slug(refusal).to_string())];
                f.extend(crate::observe::Decline::fields(refusal));
                f
            }
            Self::GeometryMoved {
                latched_width,
                latched_height,
                frame_width,
                frame_height,
            } => vec![
                ("latched", format!("{latched_width}x{latched_height}")),
                ("frame", format!("{frame_width}x{frame_height}")),
            ],
            Self::WindowUnresolved {
                width,
                height,
                format,
            } => vec![
                ("geom", format!("{width}x{height}")),
                ("fmt", format!("{format:#x}")),
            ],
            Self::FormatNeedsConversion { format } => vec![("fmt", format!("{format:#x}"))],
            Self::ResidentFormatMismatch { held, want } => {
                vec![("held", format!("{held:?}")), ("want", format!("{want:?}"))]
            }
            Self::PitchNotTexels { bpr } => vec![("bpr", bpr.to_string())],
            Self::OffsetNotTexelAligned { in_page } => vec![("in_page", in_page.to_string())],
            Self::PageListShort { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::PageUnbacked { index } => vec![("page", index.to_string())],
            Self::Engine { inner } => crate::observe::Decline::fields(inner),
        }
    }
}

crate::observe::decline::decline_display!(GpuWritebackDecline);

/// Which of a mapping's pages a writeback's texels live in, and where inside
/// them the first one is.
///
/// The guest reference this rail binds names whole pages and starts at a page
/// boundary; a sample window starts wherever the guest's plane descriptor put
/// it. This is the translation between the two, and getting it wrong lands a
/// frame at the wrong offset in the guest's memory — which is a visibly shifted
/// surface at best and another allocation's bytes at worst.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GuestWindowPlan {
    /// Indices into `page_entries` of the first and last page the frame touches.
    first_page: usize,
    last_page: usize,
    /// Byte offset of the frame's first texel within page `first_page`, which is
    /// therefore its offset within the guest reference the copy binds.
    in_page: u64,
    /// Guest row pitch in texels (`bufferRowLength`).
    row_length_texels: u32,
}

impl GuestWindowPlan {
    fn pages(&self) -> usize {
        self.last_page - self.first_page + 1
    }
}

/// Resolve a sample window against a mapping's page list.
///
/// Pure, and separate from its one caller for that reason: every value it
/// produces feeds a `VkBufferImageCopy` whose failure mode is silent — Vulkan
/// will happily write a frame at the wrong offset — and none of the surrounding
/// device state is needed to decide any of them.
fn plan_guest_window(
    page_entries: usize,
    page_size: u64,
    base_off: u64,
    span_end: u64,
    bpr: u32,
    width: u32,
    texel: u32,
) -> Result<GuestWindowPlan, GpuWritebackDecline> {
    // `bufferRowLength` is in texels, so a pitch that is not a whole number of
    // them has no spelling. Checked rather than assumed: the value comes from
    // the guest's own device descriptor.
    //
    // In *this destination's* texels and not in four bytes. A half-float plane
    // states its pitch in eight-byte texels, so dividing by four would report
    // twice as many of them — a `bufferRowLength` that reads as a valid padded
    // stride and lands every row at half its true spacing.
    //
    // The second half is a Vulkan validity rule rather than an arithmetic one:
    // `bufferRowLength` must be zero or at least `imageExtent.width`, so a pitch
    // narrower than the frame is an invalid copy and not a tight one. It cannot
    // happen for a well-formed plane, which is exactly why nothing would notice
    // if it did.
    if texel == 0 || !bpr.is_multiple_of(texel) || bpr / texel < width {
        return Err(GpuWritebackDecline::PitchNotTexels { bpr });
    }
    if span_end <= base_off || page_size == 0 {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let first_page = (base_off / page_size) as usize;
    let last_page = ((span_end - 1) / page_size) as usize;
    if page_entries <= last_page {
        return Err(GpuWritebackDecline::PageListShort {
            need: last_page + 1,
            have: page_entries,
        });
    }
    // The guest reference starts at a page boundary, so the frame's first texel
    // sits this far into it. Whole texels only, which is what `bufferOffset`
    // requires and what a guest pitch in texels already implies for every row
    // but the first.
    let in_page = base_off % page_size;
    if !in_page.is_multiple_of(u64::from(texel)) {
        return Err(GpuWritebackDecline::OffsetNotTexelAligned { in_page });
    }
    Ok(GuestWindowPlan {
        first_page,
        last_page,
        in_page,
        row_length_texels: bpr / texel,
    })
}

/// Copy a resident target straight into the guest's pages, with the frame never
/// existing on the host.
///
/// # What this is for
///
/// The copying rail this replaces moves the frame twice after the GPU has
/// already written it once: the resident is read into a `HOST_VISIBLE` staging
/// buffer, and the CPU then scatters that buffer into guest RAM row by row.
/// `readback_split` prices the pair at 0.83 ms of staging map plus 2.68 ms of
/// guest-page write inside a 6.9 ms flush, and the flush rail is 69% of the
/// drain worker's second. Making the guest's own pages the copy's destination
/// leaves only the copy that always had to happen.
///
/// # What it still owes
///
/// Everything `super::write_bgra8_inner` does *besides* moving bytes, because
/// those obligations are about the guest's pages having changed and not about
/// who changed them. In particular the guest-write witness
/// ([`DeviceState::note_host_wrote_mapping`]) and the page footprint: a rail that
/// lands frames without recording that it did makes
/// [`crate::runtime::gather_witness`] attribute its own writes to the guest, and
/// the mapper-ref-texture resident rung above it then refuses residents and gathers whole
/// surfaces per bind. That failure is measured and it costs more than this rail
/// saves.
///
/// # Errors
///
/// Every decline is a routing answer: the caller still owes the frame and takes
/// the copying rail. `Ok(())` means the pixels are in the guest's pages and the
/// GPU has finished writing them.
pub fn write_bgra8_from_resident_gpu<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> Result<u64, GpuWritebackDecline> {
    // This rail's own window, and the reason the licence does not resolve one:
    // a Store's destination *is* the surface, so a frame whose extent is not
    // the mapping's latched geometry is a frame for a mapping that has moved
    // under it. That is a scanout rule and not a property of a mapper-ref-texture
    // destination — a compute dispatch writing a sub-rectangle is ordinary — so
    // it is asked here, by the caller it belongs to.
    if !scanout_extent_ok(width, height) {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Err(GpuWritebackDecline::NotWritable);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return Err(GpuWritebackDecline::GeometryMoved {
            latched_width: mw,
            latched_height: mh,
            frame_width: width,
            frame_height: height,
        });
    }
    let Some((base_off, bpr, span_end)) = mapper_ref_texture_sample_window(m, mw, mh, format)
    else {
        return Err(GpuWritebackDecline::WindowUnresolved {
            width: mw,
            height: mh,
            format,
        });
    };
    let licence = licence_mapper_ref_texture_surface(
        state,
        host,
        identity.resident_format(),
        &MapperRefTextureSurfaceDestination {
            mapping_id,
            base_off,
            bpr,
            span_end,
            width: mw,
            height: mh,
            format,
        },
    )?;
    crate::backend::vulkan::engine::copy_target_to_guest_pages(
        identity,
        &licence.target,
        &licence.gpas,
    )
    .map_err(|inner| GpuWritebackDecline::Engine { inner })?;
    note_mapper_ref_texture_landed(state, mapping_id, licence.base_off, licence.span_end);
    Ok(licence.span_end - licence.base_off)
}

/// A window of a mapper-ref-texture surface mapping the GPU is asked to write.
///
/// The mapper-ref-texture counterpart of
/// [`crate::runtime::render_writeback::vulkan::GvaPlaneDestination`], and it exists for
/// the same reason: the licence must not resolve its own destination. Which
/// window of which plane a copy lands in is the *caller's* knowledge, and the
/// two callers here come by it differently — a render Store's destination is the
/// whole surface, while a compute bind resolves a window at stage time, which
/// may be a sub-rectangle and, for a ref-texture view, names its IOSurface plane on
/// the wire.
///
/// Resolving it inside the licence instead served the Store and silently
/// mis-served everything else. It refused every sub-rectangle — 15 of the 19
/// remaining compute readbacks on a driven macos-13 boot, all
/// [`GpuWritebackDecline::GeometryMoved`] — and behind that refusal sat a worse
/// failure it was hiding: [`mapper_ref_texture_sample_window`] takes no plane index and
/// matches by geometry, so a ref-texture bind's frame would have landed in whichever
/// plane happened to share its dimensions. [`resident_gpu_plane`]'s doc states
/// the cost of exactly that disagreement, which is that there is no error — the
/// frame lands in the wrong plane of the right surface and the symptom is the
/// next plane's pixels.
///
/// `format` is what the guest will read these bytes back as, and must be the
/// format the window was resolved against; the licence derives the bytes per
/// texel from it rather than taking one, so there is no second answer to carry.
pub(crate) struct MapperRefTextureSurfaceDestination {
    pub mapping_id: u32,
    /// Byte offset of the window's first texel within the mapping.
    pub base_off: u64,
    /// Bytes per row of the surface, which is not `width * bpp`.
    pub bpr: u32,
    /// One past the last byte the window may touch.
    pub span_end: u64,
    pub width: u32,
    pub height: u32,
    pub format: u16,
}

/// A licensed direct-to-guest-pages destination over a mapper-ref-texture surface mapping.
///
/// The mapper-ref-texture counterpart of
/// [`crate::runtime::render_writeback::vulkan::GvaPlaneLicence`], and it exists for the
/// same reason: the two rails that write a guest surface from the GPU — a render
/// Store and a compute storage-image output — differ only in the image they copy
/// *from* and in which command buffer records the copy. Everything about the
/// destination is this, so it is derived once and the second rail cannot spell
/// any of it differently.
///
/// `base_off` and `span_end` are carried because the landing note needs them and
/// deriving them a second time would be a second answer to a question the
/// licence already asked. See [`note_mapper_ref_texture_landed`].
pub(crate) struct MapperRefTextureSurfaceLicence {
    pub target: crate::backend::vulkan::engine::GuestPageTarget,
    /// The pages walked, in the surface's own order — what the copy is licensed
    /// over and what every witness on this path is armed against.
    pub gpas: Vec<u64>,
    /// The window within the mapping the copy names, and nothing past it.
    pub base_off: u64,
    pub span_end: u64,
}

/// Licence a caller's mapper-ref-texture surface window as a destination the GPU may copy
/// into, or give the typed reason it may not.
///
/// Takes the window rather than resolving one — see
/// [`MapperRefTextureSurfaceDestination`] for why that division is where it is. What is
/// shared between the two rails, and therefore lives here, is the format rule,
/// the page-list plan, the page walk, the guest-RAM references and the two
/// guest-write witnesses.
///
/// `held` is the format the source image actually holds. **A copy converts
/// nothing**, so a source whose format is not the one the guest will read these
/// bytes back as must be refused here rather than landed.
///
/// # Why this asks about the format and the render caller used not to
///
/// On the render rail the question is very nearly a tautology — a target's
/// identity takes its format from [`mapping_store_format`], the same function
/// that caller resolved its own window through — and the comparison that can
/// actually fail is made downstream by `copy_target_to_guest_pages`, against the
/// resident *image's* own format, which a mapping may have redeclared
/// underneath. Both of those are still true and that check is still there.
///
/// The compute rail has no such downstream. Its dispatch writes the guest's
/// pages itself, so there is no second pair to compare and this is the only
/// place the question can be asked. A storage image takes its format from the
/// specialized SPIR-V texel format, which has no reason to match the format the
/// bind staged its window against — banded at 3 of 35 mapper-ref-texture windows on a
/// driven macos-13 boot — so on that rail it is not a healthy zero either.
///
/// Asking it for both callers rather than for the one that needs it keeps the
/// rule in the licence, where a third writer of a guest surface would meet it
/// without knowing to look.
pub(crate) fn licence_mapper_ref_texture_surface<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    held: ash::vk::Format,
    dst: &MapperRefTextureSurfaceDestination,
) -> Result<MapperRefTextureSurfaceLicence, GpuWritebackDecline> {
    let &MapperRefTextureSurfaceDestination {
        mapping_id,
        base_off,
        bpr,
        span_end,
        width: mw,
        height: mh,
        format,
    } = dst;
    // The mapping still has to be here and still have to be backed — the caller
    // resolved a window against it, and may have done so before this call.
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Err(GpuWritebackDecline::NotWritable);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return Err(GpuWritebackDecline::NotWritable);
    }
    // An image→buffer copy moves bytes and converts nothing, so this rail can
    // only serve a window whose declared format is a host texel verbatim. A
    // compressed or planar declaration is not, and the copying rail's row
    // converter is the only thing that can land it.
    //
    // Asked of every rail that creates images rather than of the render Store's
    // table alone: this licence serves a compute storage output as well as a
    // Store, and a storage image is a thing the guest neither renders into nor
    // samples. See
    // [`crate::backend::vulkan::translate::pixel::verbatim_texel`].
    let Some((dst_format, texel)) =
        crate::backend::vulkan::translate::pixel::verbatim_texel(format)
    else {
        return Err(GpuWritebackDecline::FormatNeedsConversion { format });
    };
    // And that the source holds exactly it. See this function's doc for why the
    // render caller's own downstream check stays where it is: the two compare
    // different pairs, and this is the only one the compute rail has.
    if held != dst_format {
        return Err(GpuWritebackDecline::ResidentFormatMismatch {
            held,
            want: dst_format,
        });
    }
    let shared_backing = if host.map_pages_stable() {
        mapper::ensure_contig_view(state, host, mapping_id).map(|(ptr, len)| {
            crate::backend::vulkan::engine::GuestTargetBacking {
                allocation_host_ptr: ptr,
                allocation_len: len as u64,
                plane_offset: base_off,
                row_pitch: u64::from(bpr),
            }
        })
    } else {
        None
    };
    // One live window per physical format is enough to answer the remaining
    // device-level question: whether the driver's actual linear layout and
    // memory requirements agree with a guest plane on this host. The packed
    // view is the allocation a direct image retains. The report remains useful
    // because creation declines are per target and this gives the full binding
    // equation once per physical format.
    let probe_key = dst_format.as_raw() as u32 as u64;
    if crate::observe::first_sight("vk_linear_target_window_probe", probe_key) {
        if let Some(backing) = shared_backing {
            crate::backend::vulkan::engine::probe_guest_backed_target(
                backing.allocation_host_ptr,
                backing.allocation_len,
                backing.plane_offset,
                backing.row_pitch,
                mw,
                mh,
                dst_format,
            );
        } else {
            crate::observe::off(format!(
                "vk_linear_target_window verdict=no_packed_alias format={dst_format:?} {mw}x{mh} plane_offset={base_off} guest_row_pitch={bpr}"
            ));
        }
    }
    // No settle here, and the twin rail is why. `render_writeback::vulkan::store_gva_frame`
    // does exactly this for a GVA-addressed destination — vouch, resolve runs,
    // submit a buffer copy — and takes no settle at all, because nothing between
    // here and the submit reads the pixel bytes a pending writeback would land
    // in: the page list comes from `page_entries` (device state), the vouch
    // walks the guest's page tables, and `references_for_runs` resolves host
    // pointers. The copy itself is a GPU command on the same single queue as
    // any outstanding writeback, so queue order already puts the older write
    // ahead of this one and a CPU fence buys an ordering that holds without it.
    // That is the same argument `try_linear_sample_zero_copy` states for its own
    // gather.
    //
    // What used to stand here was the deferred rail's flush-on-access, whose
    // justification was that landing a pending window "can invalidate the
    // mapping". There are no windows: a mapper-ref-texture render Store lands its frame at
    // the Store (see `render_writeback`'s module doc). Measured at **5 204
    // settles and 7.29 s blocked** on a driven Safari-drag boot — 42 % of every
    // wait in the device — for an ordering the queue already had.
    //
    // Nothing below can land a frame on a host whose GPU cannot import guest
    // RAM, so the walks below are skipped rather than run and discarded.
    //
    // Not a second gate — it is the *same* gate, asked earlier.
    // `references_for_runs` below opens with the identical question and would
    // decline these same pages a few hundred microseconds later; this only
    // declines sooner, and on a pathway where the answer never changes that
    // saves a page-table walk per flush for the life of the process.
    //
    // Asked through `standing_refusal` rather than by re-reading the
    // granularity latch, because the latch is only one of the four things that
    // refuse here — a shim that cannot say where guest RAM lives and a machine
    // whose every span failed the bound both leave a granularity published, and
    // used to walk the whole page list before finding that out.
    if let Some(refusal) = crate::runtime::guest_ram_map::standing_refusal(host)
        .or_else(crate::runtime::guest_ram_map::write_refusal)
        .or_else(crate::runtime::guest_ram_map::surface_write_refusal)
    {
        return Err(GpuWritebackDecline::GuestRefRefused { refusal });
    }
    // Timed on its own because it is the largest `O(pages)` step left and its
    // fix is not the other one's. `vouch_for_write` re-walks every page of the
    // mapping through the guest's page table — the check that licenses writing
    // to them at all — and until the host copies were removed that cost was
    // hidden inside a millisecond of memcpy.
    use crate::runtime::drain::{note_readback_phase, ReadbackPhase};
    let vouch_started = std::time::Instant::now();
    let vouched = vouch_for_write(state, host, mapping_id, "gpu_writeback");
    note_readback_phase(
        ReadbackPhase::Vouch,
        vouch_started.elapsed().as_micros() as u64,
    );
    if vouched.is_none() {
        return Err(GpuWritebackDecline::PagesNotOurs);
    }
    let resolve_started = std::time::Instant::now();
    let page_size = state.page_size();
    let page_shift = state.page_shift;
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Err(GpuWritebackDecline::NotWritable);
    };
    let plan = plan_guest_window(
        m.page_entries.len(),
        page_size,
        base_off,
        span_end,
        bpr,
        mw,
        texel,
    )?;
    let mut gpas = Vec::with_capacity(plan.pages());
    for (i, &entry) in m.page_entries[plan.first_page..=plan.last_page]
        .iter()
        .enumerate()
    {
        let Some(gpa) = crate::protocol::iosurface_pages::entry_gpa_shift(entry, page_shift) else {
            return Err(GpuWritebackDecline::PageUnbacked {
                index: plan.first_page + i,
            });
        };
        gpas.push(gpa);
    }
    // The extent this copy names, and nothing past it. Padding after the final
    // row belongs to the surface's plane but is not texels this call was given,
    // and the copying rail does not write it either — a request that included
    // it would let the two rails land different guest memory for one frame.
    let pitch = u64::from(plan.row_length_texels.max(mw)) * u64::from(texel);
    let extent = u64::from(mh.saturating_sub(1)) * pitch + u64::from(mw) * u64::from(texel);
    // Every run, not one range. The guest backs a surface in 16 KiB granules
    // with no relation to each other, so a full-screen window is ~507 stretches
    // and asking for a single contiguous reference refused every 1080p flush of
    // a driven boot.
    let runs = crate::runtime::guest_ram_map::references_for_runs(
        host,
        &gpas,
        page_size,
        plan.in_page,
        extent,
    )
    .map_err(|refusal| GpuWritebackDecline::GuestRefRefused { refusal })?;
    let target = crate::backend::vulkan::engine::GuestPageTarget {
        runs,
        row_length_texels: plan.row_length_texels,
        width: mw,
        height: mh,
        // The guest's own declaration for this plane, which is what the guest
        // will read these bytes back as. Every byte offset planned above came
        // from its width, and the engine refuses the copy outright if the
        // resident does not hold exactly this.
        format: dst_format,
        shared_backing,
    };
    // Both witnesses before the copy rather than after it, matching
    // `contig_for_write`: a refused write costs a spurious bump, which makes a
    // reader re-read bytes that did not change, while the opposite error hands
    // out a stale copy as fresh.
    //
    // Marked over `[base_off, span_end)` — the extent the copy names — rather
    // than from zero, because unlike the contig view this rail knows its own
    // rows and has no pointer whose coverage it is restating.
    mapper::note_mapping_write_footprint(state, mapping_id, base_off, span_end - base_off);
    state.note_host_wrote_mapping(mapping_id);
    note_readback_phase(
        ReadbackPhase::Resolve,
        resolve_started.elapsed().as_micros() as u64,
    );
    Ok(MapperRefTextureSurfaceLicence {
        target,
        gpas,
        base_off,
        span_end,
    })
}

/// What a landed mapper-ref-texture GPU write owes the rest of the device.
///
/// Called once the copy is *issued*, not once it has completed, and by both
/// rails that issue one — the render Store, which submits and waits, and the
/// compute storage-image output, whose copy rides its dispatch's own command
/// buffer and lands at the fence. Neither leaves a host copy of the frame, so
/// nothing on the host may go on naming one, and that is true from the moment
/// the copy is queued rather than from the moment it retires.
///
/// Every one of these errs in the same direction on purpose: a cache forgotten
/// early costs a re-read of bytes that are about to change anyway, while one
/// forgotten late hands out a stale frame as fresh. The same argument the
/// witnesses in [`licence_mapper_ref_texture_surface`] are armed on.
pub(crate) fn note_mapper_ref_texture_landed(
    state: &mut DeviceState,
    mapping_id: u32,
    base_off: u64,
    span_end: u64,
) {
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // The guest's pages are now the only place this frame exists, and the
    // surface cache's entry, if any, is a previous flush's bytes. Same reason
    // `write_bgra8_uncached` invalidates rather than publishes.
    crate::runtime::surface_cache::forget(state, mapping_id);
}

/// Publish a Store from an attachment already backed by this mapping.
///
/// Import admission retained the mapping's bounded allocation and physical-page
/// footprint in the resident. Synchronization therefore names that resident;
/// it does not reconstruct a `GuestPageTarget`, re-walk the page table, or
/// reacquire one guest reference per page.  Those operations describe a copy
/// destination, and this path has no copy destination—the attachment is the
/// guest allocation.
pub fn synchronize_guest_backed_resident(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    guest_store_recorded: bool,
    guest_store_footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
) -> Result<u64, GpuWritebackDecline> {
    if !scanout_extent_ok(width, height) {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Err(GpuWritebackDecline::NotWritable);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return Err(GpuWritebackDecline::GeometryMoved {
            latched_width: mw,
            latched_height: mh,
            frame_width: width,
            frame_height: height,
        });
    }
    let Some((base_off, _bpr, span_end)) = mapper_ref_texture_sample_window(m, mw, mh, format)
    else {
        return Err(GpuWritebackDecline::WindowUnresolved {
            width: mw,
            height: mh,
            format,
        });
    };
    let footprint = if guest_store_needs_separate_sync(guest_store_recorded) {
        crate::backend::vulkan::engine::synchronize_guest_backed_target(identity)
            .map_err(|inner| GpuWritebackDecline::Engine { inner })?
    } else {
        guest_store_footprint.ok_or(GpuWritebackDecline::Engine {
            inner: crate::backend::vulkan::engine::DrawError::GuestPageWrite(
                crate::backend::vulkan::engine::GuestWriteDecline::NoSharedBacking,
            ),
        })?
    };

    mapper::note_physical_page_write_footprint(&footprint, base_off, span_end - base_off);
    state.host_writes.note_footprint(&footprint);
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    crate::runtime::surface_cache::forget(state, mapping_id);
    Ok(span_end - base_off)
}

fn guest_store_needs_separate_sync(recorded_in_draw: bool) -> bool {
    !recorded_in_draw
}

/// This rail's own tests, here rather than beside [`super`]'s because what they
/// exercise is here: `plan_guest_window` and `guest_store_needs_separate_sync`
/// are private to this module, and the geometry refusals they assert are this
/// module's `GpuWritebackDecline` rather than the neutral write's.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DeviceId;
    use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::FakeHost;

    #[test]
    fn a_store_recorded_by_its_draw_needs_no_second_engine_sync() {
        assert!(!guest_store_needs_separate_sync(true));
        assert!(
            guest_store_needs_separate_sync(false),
            "an older or fallback engine result must retain the synchronization transaction"
        );
    }

    /// A tight full-page-aligned surface names exactly the pages its bytes
    /// occupy, and no more.
    ///
    /// The last page is the one holding the last *texel*, not the one holding
    /// `bpr * height`. A plan that rounded up to the row pitch would hand the GPU
    /// write access to a page past the surface on every flush of a padded layout,
    /// and the guest owns whatever is in it.
    #[test]
    fn a_tight_window_names_the_pages_its_texels_occupy() {
        // 1920x1080 BGRA8, tight, starting at offset 0 of a 4 KiB-page guest.
        let (page, bpr) = (4096u64, 1920 * 4u32);
        let span = u64::from(bpr) * 1080;
        let plan = plan_guest_window(usize::MAX, page, 0, span, bpr, 1920, RGBA8_BPP)
            .expect("a tight window plans");
        assert_eq!(plan.first_page, 0);
        assert_eq!(plan.last_page, ((span - 1) / page) as usize);
        assert_eq!(plan.in_page, 0);
        assert_eq!(plan.row_length_texels, 1920);
        // Exactly the pages the bytes are in: 1920*4*1080 is a whole number of
        // 4 KiB pages, so the last texel is the last byte of the last one.
        assert_eq!(plan.pages() as u64, span / page);
    }

    /// A window starting part-way into a page reports that offset, and the page
    /// it starts in is the first the guest reference names.
    ///
    /// This is the whole reason the plan exists. The reference starts at a page
    /// boundary and the sample window does not, so a copy that took the window's
    /// mapping offset as its `bufferOffset` would land the frame `first_page *
    /// page_size` bytes early — off the front of the reference entirely for any
    /// surface past the first page.
    #[test]
    fn a_window_starting_inside_a_page_carries_the_offset_and_not_the_mapping_one() {
        let (page, bpr) = (4096u64, 256 * 4u32);
        let base = 3 * page + 512;
        let span = base + u64::from(bpr) * 8;
        let plan =
            plan_guest_window(usize::MAX, page, base, span, bpr, 256, RGBA8_BPP).expect("plans");
        assert_eq!(plan.first_page, 3);
        assert_eq!(plan.in_page, 512);
        // Not the mapping offset: that is the bug this asserts against.
        assert_ne!(plan.in_page, base);
    }

    /// Page shift is explicit, so the same window plans differently on the two
    /// guests. A helper that assumed 4 KiB would name four times too many pages
    /// on arm64 and expose three quarters of a surface it was never asked for.
    #[test]
    fn the_same_window_spans_fewer_pages_on_a_sixteen_kilobyte_guest() {
        let bpr = 1024 * 4u32;
        let span = u64::from(bpr) * 64;
        let x86 = plan_guest_window(usize::MAX, 4096, 0, span, bpr, 1024, RGBA8_BPP)
            .expect("plans on x86");
        let arm = plan_guest_window(usize::MAX, 16384, 0, span, bpr, 1024, RGBA8_BPP)
            .expect("plans on arm64");
        assert_eq!(x86.pages(), arm.pages() * 4);
    }

    /// A padded guest pitch travels as texels, because that is what
    /// `bufferRowLength` is. The inter-row bytes are never named, so the guest's
    /// own content in the padding survives the flush — matching the copying
    /// rail, which writes row by row and skips it too.
    #[test]
    fn a_padded_pitch_becomes_a_row_length_in_texels() {
        let bpr = 2048 * 4u32;
        let plan = plan_guest_window(
            usize::MAX,
            4096,
            0,
            u64::from(bpr) * 4,
            bpr,
            1600,
            RGBA8_BPP,
        )
        .expect("plans");
        assert_eq!(plan.row_length_texels, 2048);
    }

    /// A plane's pitch is a count of **its own** texels, so a wider destination
    /// resolves the same byte pitch to fewer of them.
    ///
    /// `bufferRowLength` is what this number becomes, and Vulkan multiplies it by
    /// the image's texel size to space the rows. Dividing a half-float plane's byte
    /// pitch by four reports twice as many texels as the row holds — a value that
    /// passes every validity rule and lands every row after the first at half its
    /// true spacing, so the frame arrives sheared into the top half of the window
    /// with no refusal anywhere. Both spellings are asserted from one byte pitch,
    /// because the defect is the *relation* between them and either one alone reads
    /// as correct.
    #[test]
    fn a_pitch_resolves_to_the_destinations_own_texels() {
        use crate::protocol::pixel_format::RGBA16F_BPP;
        // One tightly-packed row of 256 half-float RGBA texels.
        let bpr = 256 * RGBA16F_BPP;
        let span = u64::from(bpr) * 4;
        let wide = plan_guest_window(usize::MAX, 4096, 0, span, bpr, 256, RGBA16F_BPP)
            .expect("a half-float plane plans");
        assert_eq!(
            wide.row_length_texels, 256,
            "a tight row is exactly the frame's width in the destination's texels"
        );
        let narrow = plan_guest_window(usize::MAX, 4096, 0, span, bpr, 256, RGBA8_BPP)
            .expect("the same bytes as an eight-bit plane plans");
        assert_eq!(
            narrow.row_length_texels,
            wide.row_length_texels * 2,
            "the same byte pitch is twice as many texels at half the width"
        );
        // And a pitch that is whole texels at four bytes but not at eight is
        // refused for the wide destination rather than truncated into one.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 0, span, bpr + RGBA8_BPP, 256, RGBA16F_BPP),
            Err(GpuWritebackDecline::PitchNotTexels {
                bpr: bpr + RGBA8_BPP
            })
        );
    }

    /// Every value a `VkBufferImageCopy` cannot express declines by name rather
    /// than being rounded into one it can.
    ///
    /// `bufferOffset` must be a multiple of the texel block size and
    /// `bufferRowLength` is counted in texels; a copy submitted with either one
    /// wrong is undefined behaviour, not a misplaced frame, so neither may be
    /// silently repaired.
    #[test]
    fn a_geometry_the_copy_cannot_express_declines_by_name() {
        // A row pitch that is not a whole number of texels.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 0, 4096, 1023, 1, RGBA8_BPP),
            Err(GpuWritebackDecline::PitchNotTexels { bpr: 1023 })
        );
        // A window starting on an odd byte inside its page.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 2, 4096, 4, 1, RGBA8_BPP),
            Err(GpuWritebackDecline::OffsetNotTexelAligned { in_page: 2 })
        );
        // A page list that stops before the window does. Writing anyway would
        // export whatever the shorter list's tail happens to name.
        assert_eq!(
            plan_guest_window(2, 4096, 0, 3 * 4096, 4, 1, RGBA8_BPP),
            Err(GpuWritebackDecline::PageListShort { need: 3, have: 2 })
        );
        // An empty or inverted window has no destination at all.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 100, 100, 4, 1, RGBA8_BPP),
            Err(GpuWritebackDecline::NotWritable)
        );
        // A pitch narrower than the frame. Vulkan requires `bufferRowLength` to
        // be zero or at least the extent's width, so this is an invalid copy
        // rather than a tight one — and a plan that let it through would submit
        // it, because nothing else in the path re-derives the row length.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 0, 4096, 4 * 8, 9, RGBA8_BPP),
            Err(GpuWritebackDecline::PitchNotTexels { bpr: 32 })
        );
    }

    /// "This host cannot import" and "these pages would not resolve" are different
    /// findings and must not share a name.
    ///
    /// They did, twice, from opposite directions, and the fix for the first is
    /// what made the second reachable.
    ///
    /// Originally both `granularity()` returning `None` and any refusal from
    /// `guest_ram_map` returned `NoGuestImport`, so a driven x86 boot printed
    /// twenty `gpuwb_no_guest_import` lines — one per 1080p mapping — on a host
    /// whose `vk_caps` said `host_pointer_import=supported`. The real cause was
    /// `Scattered`, reported by `guest_ram_map` exactly once for the whole boot
    /// because `report_once` latches on `first_sight`. Ranking the fail channel by
    /// `reason=`, the documented way to read that log, put the twenty at the top
    /// under a name that contradicted the capability line.
    ///
    /// Splitting them fixed that and left two spellings of "this host cannot
    /// import" — a granularity read here and the resolution over in
    /// `guest_ram_map` — which is the divergence class in its own right. So the
    /// distinction now rides on `via=` rather than on the slug: one variant, one
    /// authority (`guest_ram_map::standing_refusal`), and the inner check named on
    /// every record.
    ///
    /// The `assert_ne!`s are the regression. What must never come back is two
    /// records that a `reason=` ranking cannot tell apart — whichever field
    /// carries the difference.
    #[test]
    fn a_refused_page_list_does_not_report_itself_as_a_host_without_the_import() {
        use crate::observe::Decline;
        use crate::runtime::guest_ram_map::MapRefusal;

        let via = |d: &GpuWritebackDecline| {
            d.fields()
                .into_iter()
                .find(|(k, _)| *k == "via")
                .map(|(_, v)| v)
        };

        let scattered = GpuWritebackDecline::GuestRefRefused {
            refusal: MapRefusal::Scattered {
                pages: 32,
                runs: 9,
                first: 0x39bb_6a000,
            },
        };
        let no_import = GpuWritebackDecline::GuestRefRefused {
            refusal: MapRefusal::NoBackendImport,
        };
        assert_ne!(
            via(&scattered),
            via(&no_import),
            "a refused page list must not read as a host that cannot import"
        );
        assert_eq!(
            via(&no_import).as_deref(),
            Some("guest_ram_map_no_backend_import"),
            "the host-wide statement still names itself, on the record rather than \
             on one line elsewhere in the log"
        );

        // The check that refused, and its own numbers, on this record.
        let fields = scattered.fields();
        assert_eq!(via(&scattered).as_deref(), Some("guest_ram_map_scattered"));
        assert_eq!(
            fields
                .iter()
                .find(|(k, _)| *k == "pages")
                .map(|(_, v)| v.as_str()),
            Some("32")
        );
        // A host-wide fact has nothing per-record to carry beyond its own name.
        assert_eq!(no_import.fields().len(), 1);

        // A different inner check must reach the log differently, or carrying it
        // buys nothing.
        let not_in_import = GpuWritebackDecline::GuestRefRefused {
            refusal: MapRefusal::GpaNotInAnyImport { gpa: 0x1000 },
        };
        assert_ne!(via(&not_in_import), via(&scattered));
    }

    /// The mapper-ref-texture licence judges the window it is given, not the surface's extent.
    ///
    /// A render Store's destination *is* the surface, so that caller refuses a frame
    /// whose rect is not the mapping's latched geometry — the test above drives
    /// exactly that, and it stays where it belongs, in the caller. A compute
    /// dispatch's destination is not a scanout: writing a sub-rectangle of a surface
    /// is ordinary, and the licence resolving its own full-extent window refused
    /// every one of them. On a driven macos-13 boot that was 15 of the 19 remaining
    /// compute readbacks, all `GeometryMoved`, at extents like 44x26 of a 64x64
    /// surface and 128x512 of a 512x512 one.
    ///
    /// So the assertion is that extent is no longer a *term*: a sub-rectangle and a
    /// whole-surface destination over the same mapping must reach the same decline,
    /// and it must not be `GeometryMoved`. `FakeHost` publishes no guest-RAM import,
    /// so both stop at the reference gate — which is downstream of every rule the
    /// licence still owns, and therefore says both got through all of them.
    #[test]
    fn a_mapper_ref_texture_licence_judges_the_callers_window_and_not_the_surfaces_extent() {
        use crate::model::PAGE_SHIFT_X86;
        use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(9), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let base_pfn = 0x40u32;
        host.map_range(
            (base_pfn as u64) << PAGE_SHIFT_X86,
            16 * PAGE as usize,
            0x55,
        );
        state.map_surface(7);
        state.attach_mapping_internal(7, 0);
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapping_internal = 1;
        m.page_entries = (0..16)
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        assert!(state.set_mapping_geom(7, 64, 64, MTL_FORMAT_BGRA8_UNORM));

        let held = crate::backend::vulkan::translate::pixel::vk_texel_layout(
            pixel_format::store_texel_order(MTL_FORMAT_BGRA8_UNORM)
                .expect("BGRA8 has a linear texel"),
        );
        let dest = |width, height| MapperRefTextureSurfaceDestination {
            mapping_id: 7,
            base_off: 0,
            bpr: 64 * 4,
            span_end: u64::from(height) * 64 * 4,
            width,
            height,
            format: MTL_FORMAT_BGRA8_UNORM,
        };

        let whole = licence_mapper_ref_texture_surface(&mut state, &mut host, held, &dest(64, 64));
        let part = licence_mapper_ref_texture_surface(&mut state, &mut host, held, &dest(44, 26));
        for (what, got) in [("the whole surface", whole), ("a sub-rectangle", part)] {
            match got {
                Err(GpuWritebackDecline::GuestRefRefused { .. }) => {}
                other => panic!(
                    "{what} must reach the reference gate, and only that gate; got {:?}",
                    other.err()
                ),
            }
        }
    }
}
