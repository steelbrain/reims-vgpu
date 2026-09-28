//! Write host BGRA8 into a guest IOSurface mapping (render writeback).
//!
//! This half is the CPU write and the geometry every rail resolves the same
//! way: where a mapping's plane is, what extent and format a write into it must
//! use, and the bytes-in-hand copy. [`vulkan`] is the other half — the same
//! frame landed from a resident, without the bytes ever existing on the host.
//!
//! Product writes go **only** through a revalidated contiguous HostOps view
//! (`map_pages`) — never `write_gpa` fragment walks over cached PFNs (freelist
//! `0xff000000ff000000` class). Always bumps [`DeviceState::mark_mapping_written`]
//! on success.

use crate::model::{scanout_extent_ok, DeviceState, MappingEntry, MAX_SCANOUT_DIM};
use crate::protocol::iosurface_pages::{packed_span_estimate, sample_window_from_device_desc};
use crate::protocol::pixel_format::{self, RowToRgba8, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP};
use crate::runtime::changed_runs::ChangedRuns;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;

/// The rail that can copy a resident straight into a mapping's guest pages.
///
/// Gated on the build — whether this binary carries a Vulkan rail at all is a
/// fact about the build, and every signature in there names a `TargetIdentity`
/// or a `GuestPageTarget`. *Which rail is running* is never asked here.
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

/// Why one render writeback did not land its frame in the guest's pages.
///
/// [`write_bgra8_inner`] has fifteen refusal sites and used to answer all of them
/// with a bare `false`, which its caller rendered as a single
/// `deferred_flush_lost reason=write_refused`. That is the defect the decline
/// vocabulary exists to prevent: the composite surface is the largest frame this
/// device moves, and a reader watching that slug fire could tell that the
/// wallpaper had been dropped but not whether the mapping had gone, its geometry
/// had moved under the armed window, the page walk had refused, or the source
/// buffer was short. Those have four different fixes.
///
/// One variant per check, carrying the values that decide it. The class currently
/// reads **zero across every accumulated boot log**, so this is an instrument for
/// a failure that is not happening rather than a repair for one that is — which is
/// exactly when it is cheap to install and exactly when nobody remembers to.
#[derive(Debug)]
pub enum SurfaceWriteRefusal {
    /// A zero or over-large rect. `MAX_SCANOUT_DIM` is the bound.
    Geometry { width: u32, height: u32 },
    /// The source's row pitch cannot hold `width` BGRA8 texels.
    SourceStride { src_stride: u32, width: u32 },
    /// A native source row is narrower than the destination's own packed row.
    NativeSourceStride { src_stride: u32, row_bytes: u32 },
    /// No such mapping. The surface went away between the arm and the landing.
    MappingAbsent,
    /// The mapping is unmapped or has no page list, so there is nowhere to write.
    MappingNotResident,
    /// **The mapping's latched geometry is not the geometry of the frame being
    /// landed.** A deferred window carries the rect it was armed with, and a
    /// wallpaper or appearance change re-publishes the surface at another one.
    /// Landing the old frame at the new pitch would skew it, so it is refused.
    GeometryMoved {
        latched_width: u32,
        latched_height: u32,
        frame_width: u32,
        frame_height: u32,
    },
    /// The sample window could not be resolved from the surface descriptor.
    WindowUnresolved {
        width: u32,
        height: u32,
        format: u16,
    },
    /// The page walk refused to vouch for the mapping's page list.
    PagesNotOurs,
    /// The format has no packed row length, so there is no rect to write.
    FormatRowLength { format: u16 },
    /// Native bytes name a different format from the mapping window they would
    /// be copied into.
    NativeFormatMismatch { source: u16, mapping: u16 },
    /// The source buffer ends before the row this write is up to.
    SourceShort { need: usize, have: usize, row: u32 },
    /// A row would not convert into the mapping's pixel format.
    RowConvert { format: u16, row: u32 },
    /// The staged frame's extent overflowed, so the rows do not describe a buffer.
    FrameExtent { bpr: usize, height: u32 },
    /// The staged frame ends before the row being placed in it.
    StagedShort { need: usize, have: usize, row: u32 },
    /// The mapper refused to write a run of the frame into the guest's pages.
    MapperWrite { lo: u64, len: usize },
    /// The seed (previous-frame) buffer ends before the frame it must diff
    /// against. Distinct from [`Self::SourceShort`] because the two buffers come
    /// from different producers, and a log that conflated them could not say
    /// which one to go and look at.
    SeedShort { need: usize, have: usize },
    /// A seed row would not convert into the mapping's pixel format. Same
    /// distinction from [`Self::RowConvert`], and the same reason.
    SeedRowConvert { format: u16, row: u32 },
}

impl crate::observe::decline::Decline for SurfaceWriteRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Geometry { .. } => "surface_write_geometry",
            Self::SourceStride { .. } => "surface_write_source_stride",
            Self::NativeSourceStride { .. } => "surface_write_native_source_stride",
            Self::MappingAbsent => "surface_write_mapping_absent",
            Self::MappingNotResident => "surface_write_mapping_not_resident",
            Self::GeometryMoved { .. } => "surface_write_geometry_moved",
            Self::WindowUnresolved { .. } => "surface_write_window_unresolved",
            Self::PagesNotOurs => "surface_write_pages_not_ours",
            Self::FormatRowLength { .. } => "surface_write_format_row_length",
            Self::NativeFormatMismatch { .. } => "surface_write_native_format_mismatch",
            Self::SourceShort { .. } => "surface_write_source_short",
            Self::RowConvert { .. } => "surface_write_row_convert",
            Self::FrameExtent { .. } => "surface_write_frame_extent",
            Self::StagedShort { .. } => "surface_write_staged_short",
            Self::MapperWrite { .. } => "surface_write_mapper_write",
            Self::SeedShort { .. } => "surface_write_seed_short",
            Self::SeedRowConvert { .. } => "surface_write_seed_row_convert",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Geometry { width, height } => vec![
                ("geom", format!("{width}x{height}")),
                ("max", MAX_SCANOUT_DIM.to_string()),
            ],
            Self::SourceStride { src_stride, width } => vec![
                ("src_stride", src_stride.to_string()),
                ("need", (width.saturating_mul(RGBA8_BPP)).to_string()),
            ],
            Self::NativeSourceStride {
                src_stride,
                row_bytes,
            } => vec![
                ("src_stride", src_stride.to_string()),
                ("need", row_bytes.to_string()),
            ],
            Self::MappingAbsent | Self::MappingNotResident | Self::PagesNotOurs => Vec::new(),
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
            Self::FormatRowLength { format } => vec![("fmt", format!("{format:#x}"))],
            Self::NativeFormatMismatch { source, mapping } => vec![
                ("source_fmt", format!("{source:#x}")),
                ("mapping_fmt", format!("{mapping:#x}")),
            ],
            Self::SourceShort { need, have, row } => vec![
                ("need", need.to_string()),
                ("have", have.to_string()),
                ("row", row.to_string()),
            ],
            Self::RowConvert { format, row } => {
                vec![("fmt", format!("{format:#x}")), ("row", row.to_string())]
            }
            Self::FrameExtent { bpr, height } => {
                vec![("bpr", bpr.to_string()), ("height", height.to_string())]
            }
            Self::StagedShort { need, have, row } => vec![
                ("need", need.to_string()),
                ("have", have.to_string()),
                ("row", row.to_string()),
            ],
            Self::MapperWrite { lo, len } => {
                vec![("lo", format!("{lo:#x}")), ("len", len.to_string())]
            }
            Self::SeedShort { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::SeedRowConvert { format, row } => {
                vec![("fmt", format!("{format:#x}")), ("row", row.to_string())]
            }
        }
    }
}

/// Report one writeback refusal and answer `false` for the caller to return.
///
/// Latched per `(check, mapping)`: a surface whose geometry has moved refuses
/// every frame until something re-arms it, and the second line says nothing the
/// first did not. The route beside it carries the magnitude, which is what
/// [`crate::observe::emit::Emit::fail_once`]'s contract asks for.
fn refuse(mapping_id: u32, why: SurfaceWriteRefusal) -> bool {
    use crate::observe::decline::Decline;
    crate::runtime::drain::note_store_route(why.slug());
    crate::observe::emit::Emit::decline("surface_write", &why)
        .field("mid", mapping_id)
        .fail_once(u64::from(mapping_id));
    false
}

/// Resolve the sample window a texture of this geometry occupies inside its
/// mapping, for both wire families.
///
/// Two states the device used to answer identically, and the whole point of this
/// function is that they are not the same:
///
/// - **The mapping has published no descriptor.** `MappingInternal.descriptor`
///   reads zero until the guest fills it, which `mapper::resolve` documents as a
///   real state rather than a failure, and the geometry then comes from the
///   mapper-ref-texture object instead. There are no plane records to confuse here;
///   the single unknown is the pitch, and [`packed_span_estimate`]'s aligned row
///   stands in for it over a surface starting at offset 0.
/// - **The descriptor is published and resolves nothing.** Here the guest *has*
///   told us the layout and the texture cannot be placed in it: its geometry
///   matched no plane record, or — the case that matters — it matched more than
///   one. A v0a8 surface's Y and alpha planes are both R8 at the luma geometry,
///   so the scan cannot tell them apart *by construction*, and the packed window
///   over plane 0 is a coin flip that reads as success at every layer above.
///
/// The second case declines, and callers answer it with a named refusal. That is
/// the difference between a bind that is lost visibly and one that samples luma
/// for alpha with nothing in the device able to say so.
///
/// Neither case is reached on a healthy x86 desktop. Measured on driven Ventura
/// boots with a Safari window drag, both with the host-pointer import available
/// and with `REIMS_VGPU_GUEST_IMPORT=off` — which is the run that matters here,
/// because a capable host takes the import for every guest window and leaves the
/// copying rails at zero. With the gate closed
/// (`host_pointer_import=disabled_by_env`, nothing reporting a bound import) the
/// copying rails carried the whole workload — every ref-texture view, every mapper-ref-texture
/// resident rung and every surface flush of the drag — and **no window failed to
/// resolve**. Every bind came from a published descriptor, so the estimate above
/// is the state before the guest fills one rather than a rung this device leans
/// on.
fn sample_window(
    m: &MappingEntry,
    plane_index: Option<u32>,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    let Some(desc) = m.device_desc_complete() else {
        let end = packed_span_estimate(format, width, height)?;
        // The estimate is a whole number of aligned rows, so dividing it back
        // out is the row it was built from rather than a second derivation.
        return Some((0, (end / u64::from(height)) as u32, end));
    };
    sample_window_from_device_desc(Some(desc), plane_index, format, width, height)
}

/// Resolve the sample window for a mapper-ref-texture binding on a mapping.
///
/// Mapper-ref-texture is the case with **no wire plane index**: nothing on the wire names
/// which plane the texture wants, so a multi-plane surface is resolved by
/// matching width, height and bytes-per-element, and the plane is taken only
/// when exactly one matches. See [`sample_window`] for what each outcome means.
pub fn mapper_ref_texture_sample_window(
    m: &MappingEntry,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    sample_window(m, None, width, height, format)
}

/// Resolve the sample window for a ref-texture serialized view, which — unlike
/// mapper-ref-texture — carries the IOSurface plane index on the wire (ref-texture record
/// `+0x20`).
///
/// Every ref-texture consumer must come through here rather than through
/// [`mapper_ref_texture_sample_window`], and the distinction is not cosmetic: the wire index
/// names the plane record directly, and it is the only key that separates
/// same-geometry planes. Handing a ref-texture view's geometry to the mapper-ref-texture scan
/// drops that index, so a bind the wire said was alpha resolves against
/// whichever same-geometry plane the scan happens to reach.
pub fn ref_texture_sample_window(
    m: &MappingEntry,
    plane_index: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    sample_window(m, Some(plane_index), width, height, format)
}

/// Revalidate + packed contig host view covering at least `span_end` bytes.
///
/// Returns `None` when the mapping is fragmented on Linux (use
/// [`mapper::write_mapping_bytes`] / [`mapper::read_mapping_bytes`]).
fn contig_for_span<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    span_end: u64,
) -> Option<(usize, usize)> {
    let (ptr, len) = mapper::ensure_contig_view(state, host, mapping_id)?;
    if (len as u64) < span_end {
        crate::observe::fail(format!(
            "mapping_write contig mid={mapping_id} reason=short_view len={len} need={span_end}"
        ));
        return None;
    }
    Some((ptr, len))
}

/// Take the write proof at the head of a writer, naming the rail that wanted it.
///
/// [`mapper::vouch_mapping_pages_verdict`] already fail-logs *why* a walk refused, with
/// the page and both translations. This adds the one fact that line cannot
/// carry: which writer was about to use the list. Four rails write through
/// `page_entries` and they fail for different reasons at different rates, so a
/// single undifferentiated refusal total would not say which one to read next.
///
/// # Measured: this rail carries the traffic and none of the drift
///
/// One 300 s crash-hunt boot, x86 / Vulkan: `mapw_pages_vouched` 29 002,
/// `mapw_pages_refused` **0**, while the deferred flush rail on the same boot
/// scored `mapping_pages_ours` 25 741 and `mapping_pages_drifted` 9. So these
/// four writers do more writing than the flush rail does, and on this workload
/// not one of them found a contradicted list. The guard here is currently inert;
/// say so rather than counting it as the repair.
///
/// The split is not noise, and the reason is structural: a *deferred* frame is
/// armed at one time and landed at another, and the interval is precisely the
/// window in which the guest can re-point the surface underneath it. These
/// writers vouch and write in the same breath, so their window is nearly zero.
/// **Deferral is the exposure.** That predicts the measurement rather than
/// explaining it after the fact, and it says where to look next: shortening the
/// arm-to-land interval should move `mapping_pages_drifted`, and nothing else
/// here should.
///
/// The drift rate is also not stable boot to boot — 22 on the preceding boot, 9
/// on this one, same workload — so a single boot cannot score it and neither can
/// a pair.
fn vouch_for_write<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    writer: &'static str,
) -> Option<mapper::PagesVouched> {
    let (verdict, vouched) = mapper::vouch_mapping_pages_verdict(state, host, mapping_id);
    match verdict {
        mapper::PagesVerdict::Ours => {
            crate::runtime::drain::note_store_route("mapw_pages_vouched");
        }
        // The write proceeds exactly as for `Ours`; only the counter differs.
        // `mapw_pages_vouched` used to carry both, so its companion zero
        // (`mapw_pages_refused`) could not distinguish a guard that passed from
        // one that was never armed. Read the two together: `vouched` is the
        // guard's coverage and `unwitnessed` is the hole in it.
        mapper::PagesVerdict::Unwitnessed(why) => {
            crate::runtime::drain::note_store_route("mapw_pages_unwitnessed");
            crate::runtime::drain::note_store_route(match why {
                "no_walk" => "mapw_unwit_no_walk",
                "walk_superseded" => "mapw_unwit_superseded",
                "no_pages" => "mapw_unwit_no_pages",
                _ => "mapw_unwit_no_mapping",
            });
        }
        mapper::PagesVerdict::Drifted => {
            crate::observe::fail(format!(
                "mapping_write fail reason=pages_not_vouched mid={mapping_id} writer={writer}"
            ));
            crate::runtime::drain::note_store_route("mapw_pages_refused");
        }
    }
    vouched
}

/// [`contig_for_span`] for a caller that is about to write through the pointer.
///
/// The view `ensure_contig_view` hands back is a live `mach_vm_remap` of guest
/// PFNs, cached on the mapping and returned again on every later call. Its own
/// doc states the contract — "always revalidate first so a cached contig never
/// aliases PFNs after ReplacePhysical / guest recycle" — but the revalidation it
/// names cannot deliver that for a backing record: with no `MappingInternal` it
/// re-resolves nothing and answers "resolvable" on a non-empty list alone. So a
/// writer holding this pointer is holding whatever those PFNs became, and a
/// full white frame poked through it is the `0xff`-filled freed guest heap the
/// crash census reads back.
///
/// Reads keep [`contig_for_span`]: a read through a drifted view returns another
/// process's bytes, which is a wrong picture and not a corrupted guest, and the
/// two losses want separate slugs.
fn contig_for_write<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    span_end: u64,
    vouched: &mapper::PagesVouched,
) -> Option<(usize, usize)> {
    if !vouched.covers(state, mapping_id) {
        crate::observe::fail(format!(
            "mapping_write contig mid={mapping_id} reason=vouch_stale need={span_end} \
             (the page list was cleared or replaced between the walk and this write)"
        ));
        return None;
    }
    let view = contig_for_span(state, host, mapping_id, span_end)?;
    // Every raw-pointer write in this file goes through here, and none of them
    // goes through `mapper::write_mapping_bytes` — they poke rows straight into
    // the view. So this is where those writes enter `observe::footprint`, and
    // without it the *largest* guest-write rail in the device would be missing
    // from a set whose whole use is answering "did we write there?".
    //
    // Marked over `[0, span_end)` because that is the extent this function
    // guarantees and the callers' row offsets are not visible here. That
    // over-marks the pages before a rect's first row — the surface's own pages,
    // never anyone else's, since the marking walks this mapping's page list — and
    // over-marking can only turn a miss into a hit. Under-marking would
    // manufacture the clean "we never wrote there" the set must never invent.
    mapper::note_mapping_write_footprint(state, mapping_id, 0, span_end);
    // The other reader of these writes, and for the same reason: the hypervisor's
    // dirty bitmap witnesses guest stores only, so a copy vouched for by "the
    // guest has not written" is stale the moment this rail runs. Recorded beside
    // the footprint mark rather than in each caller, so the two cannot drift and
    // a new caller inherits both.
    state.note_host_wrote_mapping(mapping_id);
    Some(view)
}

/// One past the last mapping byte a rect transfer touches: the last texel of its
/// last row, at `bpr` pitch, `x_off` bytes into the row.
///
/// Both the raw-pointer read and the raw-pointer write below must compare this
/// against `span_end`, because `contig_for_span` guarantees the view covers
/// `span_end` and nothing more — past it a read takes unrelated QEMU heap and a
/// write smashes unrelated guest pages, both trace-lessly. Written once because
/// duplicated arithmetic is the only reason the two sides could disagree, and
/// they did: the write side was hardened for this bound and the read side
/// shipped without it. Each caller still names its own slug — `read_overrun` and
/// `writeback_overrun` are different losses.
fn rect_extent_end(
    base_off: u64,
    origin_y: u32,
    height: u32,
    bpr: usize,
    x_off: u64,
    rb: usize,
) -> u64 {
    base_off
        .saturating_add(
            (origin_y as u64)
                .saturating_add(height as u64)
                .saturating_sub(1)
                .saturating_mul(bpr as u64),
        )
        .saturating_add(x_off)
        .saturating_add(rb as u64)
}

/// Mapping byte ranges a writeback must leave alone, ascending and disjoint.
///
/// Offsets are from the mapping's page 0, the same space `base_off`/`span_end`
/// are in, so a caller holding guest *page* addresses converts once with
/// [`crate::runtime::mapper::mapping_offsets_of_pages`] and everything below
/// stays in one coordinate system.
pub type SkipRanges<'a> = &'a [(u64, u64)];

/// The sub-ranges of `[start, end)` that are not covered by `skip`.
///
/// `skip` is ascending and disjoint, so one forward walk answers it. Kept
/// separate from the two writers below because they lay their bytes out
/// differently — one pokes a host view in place, the other stages a frame and
/// hands runs to the mapper — and the only thing they must agree on is *which
/// bytes are excluded*. Two open-coded walks would be two chances to disagree.
pub(crate) fn unskipped(start: u64, end: u64, skip: SkipRanges<'_>) -> Vec<(u64, u64)> {
    if skip.is_empty() {
        return if start < end {
            vec![(start, end)]
        } else {
            vec![]
        };
    }
    let mut out = Vec::new();
    let mut cur = start;
    for &(s, e) in skip {
        if e <= cur {
            continue;
        }
        if s >= end {
            break;
        }
        if s > cur {
            out.push((cur, s.min(end)));
        }
        cur = cur.max(e);
        if cur >= end {
            return out;
        }
    }
    if cur < end {
        out.push((cur, end));
    }
    out
}

/// Write a tight BGRA8 image into the mapping's guest pages.
///
/// Packed contig HostOps view when possible; else multi-import maximal packed
/// page runs ([`mapper::write_mapping_bytes`]). Never `write_gpa`.
///
/// `src` is row-major BGRA8 with `src_stride` bytes/row. Geometry must match
/// the latched mapping size (or width/height args when has_geom is set).
pub fn write_bgra8<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    write_bgra8_skipping(state, host, mapping_id, src, src_stride, width, height, &[])
}

/// [`write_bgra8`], leaving `skip` untouched.
///
/// A deferred writeback holds a frame the device rendered and lands it in the
/// guest's pages later. If the guest CPU wrote some of those pages in between,
/// writing the whole frame loses the guest's stores and dropping the frame loses
/// the device's; `skip` is how the caller expresses the third answer, one page
/// at a time, from the hypervisor's own per-page report.
///
/// Everything else is unchanged, deliberately — including the cache refresh and
/// the epoch bump at the tail. The frame *is* what the device rendered, and the
/// host-side copies of it stay that; what `skip` decides is only which of those
/// bytes the guest's own memory is allowed to keep instead.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, plus the ranges its owner may not overwrite"
)]
pub fn write_bgra8_skipping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src,
        CacheOutcome::Publish(None),
        src_stride,
        width,
        height,
        skip,
    )
}

/// [`write_bgra8_skipping`] for a caller that owns its frame behind an `Arc`.
///
/// The tail of every non-skipping writeback publishes the frame to
/// [`crate::runtime::surface_cache`], and a caller holding only a borrow has to
/// build a second whole-frame buffer for it to keep. On the 8.29 MB composite
/// that copy costs 1.21 ms about 100 times a second — more than landing the same
/// bytes in the guest's own pages does. The cache already stores its frames
/// behind an `Arc` so that an entry and a deferred window can name one
/// allocation, so a caller that arrives holding one can publish it rather than
/// duplicate it.
///
/// The sharing conditions are checked rather than assumed, because the cache's
/// contract is a tight BGRA8 frame at the entry's geometry and an allocation that
/// is not one would be handed to every later reader as though it were: the
/// pointer has to be the one the rest of this write read from, the pitch has to
/// be the packed row length, and the allocation has to cover the whole frame.
/// Anything else takes the copying publish.
/// Writes the frame whole. Its one caller is the render writeback, which
/// preserves nothing by design: the witness a narrowing would rest on cannot
/// say which bytes the guest still wants. A caller that does
/// need to skip has [`write_bgra8_skipping`]; adding the parameter back here
/// belongs with the caller that can fill it.
pub fn write_bgra8_owned<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &std::sync::Arc<Vec<u8>>,
    src_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src.as_slice(),
        CacheOutcome::Publish(Some(src)),
        src_stride,
        width,
        height,
        &[],
    )
}

/// [`write_bgra8_skipping`] for a caller whose frame is about to stop existing.
///
/// The other writers end by publishing the frame to
/// [`crate::runtime::surface_cache`], which is nearly free when the caller
/// already owns an `Arc` and a whole-frame copy when it does not. A caller
/// holding borrowed bytes — the deferred render flush reading a resident
/// through `engine::LeasedFrame`, which is a Vulkan staging buffer it gives
/// back a moment later — would pay that copy purely to fill a cache entry, and
/// `render_flush_cache_used` prices those entries at 0.4 %: 15 reads against
/// 3751 that nothing touched before the next flush replaced them.
///
/// So this writer drops the entry instead of refreshing it, and dropping is the
/// only correct alternative. Leaving the previous frame behind would serve a
/// later reader an old frame with nothing saying so, which is the stale-tile
/// class the fence binding exists to close. Every reader that misses falls
/// through to a source that does hold this frame — the surface's own guest
/// pages, which this write has just landed, or the resident it came out of —
/// so the miss costs a slower route to the same pixels and never wrong ones.
/// Writes the frame whole, for the same reason [`write_bgra8_owned`] does.
pub fn write_bgra8_uncached<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src,
        CacheOutcome::Invalidate,
        src_stride,
        width,
        height,
        &[],
    )
}

/// [`write_bgra8_uncached`], leaving `skip` untouched: the borrowed-frame
/// writer for a caller that must also preserve the guest's own stores.
///
/// The resident merge in the sampled ladder is that caller. It reads a resident
/// back through `engine::LeasedFrame` and lands it around the pages the guest
/// wrote, so it holds a borrow of a staging buffer (no frame of its own for the
/// cache to share) and a skip list (which the whole-frame writers do not take).
/// The cache entry is dropped for the reason [`write_bgra8_uncached`] gives,
/// and the pages this write lands are what a later reader falls through to.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, plus the ranges its owner may not overwrite"
)]
pub fn write_bgra8_skipping_uncached<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src,
        CacheOutcome::Invalidate,
        src_stride,
        width,
        height,
        skip,
    )
}

/// The plane of `m` that [`vulkan::write_bgra8_from_resident_gpu`] would write a frame
/// of this extent into, as `(surface_offset, row_stride, pixel_format)`.
///
/// # Why a caller has to ask
///
/// That rail takes a mapping id and an extent and nothing else: it resolves the
/// plane itself, from the mapping's own declaration, through the same two steps
/// this function performs. A caller that already holds its own idea of the
/// destination plane — the blit rail resolves one out of the guest's texture
/// descriptor, and a ref-texture view carries a **wire plane index** this rail has no
/// parameter for — must compare the two before routing a copy here, because a
/// disagreement is not an error anywhere: the frame lands, in the wrong plane of
/// the right surface, and the only symptom is the next plane's pixels.
///
/// `None` means the rail would decline, so a caller that gets it owes the frame
/// to whatever path it was going to take anyway. The rail names *which* check
/// declined, through [`vulkan::GpuWritebackDecline`]; this collapses them, because a
/// pre-question only needs to know that the answer is not "yes".
pub fn resident_gpu_plane(m: &MappingEntry, width: u32, height: u32) -> Option<(u64, u32, u16)> {
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return None;
    }
    let (base_off, bpr, _span_end) = mapper_ref_texture_sample_window(m, mw, mh, format)?;
    Some((base_off, bpr, format))
}

/// The geometry and pixel format a writeback to this mapping must land in.
///
/// A mapping that has declared its own (`has_geom`) owns the answer, and a
/// zero format there means BGRA8; one that has not takes the caller's geometry.
/// Factored out because the writeback and the pre-flush above must resolve it
/// identically — a pre-flush computed at a different extent than the write
/// would leave exactly the windows the write is about to land on.
fn mapping_write_geometry(m: &MappingEntry, width: u32, height: u32) -> (u32, u32, u16) {
    if m.has_geom {
        (m.width, m.height, mapping_store_format(m))
    } else {
        (width, height, MTL_FORMAT_BGRA8_UNORM)
    }
}

/// The Metal pixel format a Store into this mapping's plane must land in.
///
/// A mapping that has declared its own geometry owns this, and a zero format
/// there means guest scanout order — the format every mapper-ref-texture plane had before
/// any of them declared otherwise, so it is the contract's default and not a
/// guess. A mapping that has declared no geometry has declared no format either.
///
/// Public because the *resident* has to agree with it. A render target's
/// identity carries the format its image is created with, and for this namespace
/// that answer is this one — so
/// [`crate::backend::vulkan::present_identity::surface_identity`] reads it here rather
/// than deriving a second copy from the same fields. Two derivations agreeing
/// only because they share an input is precisely how the primary attachment's
/// format used to be decided, and it did not stay agreed.
pub fn mapping_store_format(m: &MappingEntry) -> u16 {
    if m.has_geom && m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    }
}

/// What a writeback leaves in the host surface cache when it is done.
enum CacheOutcome<'a> {
    /// Publish this frame as the mapping's entry, sharing the caller's
    /// allocation when it is one the cache's contract allows sharing.
    Publish(Option<&'a std::sync::Arc<Vec<u8>>>),
    /// Drop the mapping's entry. For a caller that cannot leave the cache
    /// naming its frame, because the memory holding it is about to be reused.
    Invalidate,
}

#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, its owner when it has one, plus the \
              ranges that owner may not overwrite"
)]
fn write_bgra8_inner<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    cache: CacheOutcome<'_>,
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
) -> bool {
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    if src_stride < width.saturating_mul(RGBA8_BPP) {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceStride { src_stride, width },
        );
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: mw,
                latched_height: mh,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    let Some((base_off, bpr_u32, span_end)) = mapper_ref_texture_sample_window(m, mw, mh, format)
    else {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::WindowUnresolved {
                width: mw,
                height: mh,
                format,
            },
        );
    };
    // Deferred-writeback flush-on-access: land pending resident content in
    // these pages before touching them.
    //
    // Except when this write has ranges it must not touch. The payment writes the
    // owed resident over the *whole* window with no exclusions, and the one
    // caller that passes a skip list — `merge_guest_writes_into_pages` — passes
    // exactly the pages the guest painted under a live resident. Paying first
    // overwrites them, and the write below then restores everything *except*
    // them, so the guest's repaint is destroyed by the mechanism built to keep
    // it. What this write is about to land is the same surface at the same
    // geometry (`GeometryMoved` above refuses anything else), so the owed frame
    // is superseded rather than lost.
    if skip.is_empty() {
        crate::runtime::writeback_debt::settle_for_mapping(
            state,
            host,
            mapping_id,
            crate::runtime::render_writeback::SettleSite::MappingBgra8Write,
        );
    } else {
        crate::runtime::writeback_debt::supersede_for_mapping(
            state,
            mapping_id,
            crate::runtime::render_writeback::SettleSite::MappingBgra8Write,
        );
    }
    // Taken after the flush, because the flush can invalidate this mapping, and
    // once for the whole frame, because the loop below writes a row at a time.
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "bgra8") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    let bpr = bpr_u32 as usize;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return refuse(mapping_id, SurfaceWriteRefusal::FormatRowLength { format });
    };
    let tight = tight as usize;

    let mut row = vec![0u8; tight];
    let mut rgba = if format == MTL_FORMAT_BGRA8_UNORM
        || format == pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
    {
        None
    } else {
        Some(vec![0u8; (mw as usize) * (RGBA8_BPP as usize)])
    };
    // Whether a row can go straight from `src` to the guest without passing
    // through `row`.
    //
    // `row` is the *conversion* destination: when the mapping's format is
    // already BGRA8 there is nothing to convert, and staging through it copied
    // every byte of the frame a second time — an extra 8 MB memcpy per flush on
    // the composite surface, ~106 times a second.
    //
    // Removing it is strictly less work for identical bytes, but do not go
    // looking for it in `readback_split`. It was landed on the prediction that
    // `write_us` would drop by the ~0.8 ms the same byte count costs elsewhere,
    // and a live driven boot then measured 2.79 ms per flush against 2.68 ms
    // before — no change outside run-to-run noise. The prediction was wrong
    // about *which* copy is expensive: `row` is a few KiB and stays in L1, so
    // filling it is nearly free, while the copy into cold guest pages is the
    // one that costs and is still there. `write_us` runs at ~3 GB/s against
    // ~9 GB/s for the readback's own memcpy of the identical frame, which is
    // the shape of a cache-cold scattered write, not of an avoidable pass.
    //
    // The consequence for whoever shrinks this next: the cost is bytes landing
    // in guest RAM, so fewer bytes helps proportionally and fewer staging hops
    // does not.
    //
    // Only sound while the row is byte-identical, which is why `tight` is
    // compared rather than assumed: `tight_row_bytes` is the format's own
    // packed row length, and if it ever disagrees with the source's `mw * 4`
    // the staged path still runs. That also keeps `row`'s reuse across rows
    // safe — a short source row would otherwise leave the previous row's bytes
    // in its tail.
    let direct_rows = rgba.is_none() && tight == (mw as usize) * (RGBA8_BPP as usize);
    // Parsed once for both staged arms below. `Option` rather than an early
    // refusal because `direct_rows` converts nothing, so a format with no arm is
    // not that path's problem. See `pixel_format::Rgba8ToRow`.
    let store_rail = pixel_format::Rgba8ToRow::for_format(format);

    use crate::runtime::drain::{
        note_surface_write_path, note_surface_write_phase, SurfaceWritePhase,
    };
    let frame_bytes = (mh as u64).saturating_mul(tight as u64);

    // Fast path: one packed view, poke rows in place.
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched) {
        note_surface_write_path(true, frame_bytes);
        let land_started = std::time::Instant::now();
        // SAFETY: contig covers span_end; revalidated in ensure_contig_view.
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        for y in 0..mh {
            let src_off = (y as usize) * (src_stride as usize);
            let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
            if src_off + src_row_len > src.len() {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::SourceShort {
                        need: src_off + src_row_len,
                        have: src.len(),
                        row: y,
                    },
                );
            }
            let src_row = &src[src_off..src_off + src_row_len];
            let row_bytes: &[u8] = if direct_rows {
                &src_row[..tight]
            } else {
                if let Some(ref mut rgba_row) = rgba {
                    if !RowToRgba8::Bgra8.convert(src_row, mw, rgba_row)
                        || !store_rail.is_some_and(|rail| rail.convert(rgba_row, mw, &mut row))
                    {
                        return refuse(
                            mapping_id,
                            SurfaceWriteRefusal::RowConvert { format, row: y },
                        );
                    }
                } else {
                    let n = src_row_len.min(row.len());
                    row[..n].copy_from_slice(&src_row[..n]);
                }
                &row
            };
            // The row's destination in mapping-offset space, so the skip list —
            // which is in that space — is subtracted before any pointer exists.
            let row_off = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
            for (lo, hi) in unskipped(row_off, row_off.saturating_add(tight as u64), skip) {
                let within = (lo - row_off) as usize;
                let len = (hi - lo) as usize;
                let dst = unsafe { base.add((y as usize).saturating_mul(bpr) + within) };
                // SAFETY: `within + len <= tight <= row_bytes.len()`, and the
                // view covers span_end which is at or past this row's last byte.
                unsafe {
                    std::ptr::copy_nonoverlapping(row_bytes.as_ptr().add(within), dst, len);
                }
            }
        }
        note_surface_write_phase(
            SurfaceWritePhase::Land,
            land_started.elapsed().as_micros() as u64,
        );
    } else {
        note_surface_write_path(false, frame_bytes);
        let stage_started = std::time::Instant::now();
        // Fragmented: stage native rows then multi-import (one map_pages pass set).
        // The sample window ends at the final row's last texel, not at
        // `bpr * height`; padding after the final row is outside the texture
        // contract and may belong to another guest allocation.
        let Some(frame_len) = (mh as usize)
            .checked_sub(1)
            .and_then(|rows| bpr.checked_mul(rows))
            .and_then(|prefix| prefix.checked_add(tight))
        else {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::FrameExtent { bpr, height: mh },
            );
        };
        // The staged buffer is `src` itself whenever the layout it would be
        // built into is the layout `src` already has: no conversion (the rows go
        // through untouched) and a source pitch equal to the mapping's row pitch
        // (so row `y` is already at `y * bpr`). Under both, byte `i` of the
        // staged frame is byte `i` of `src` for every `i < frame_len`, and
        // building it copies 8 MB to produce a slice we are holding.
        //
        // What `src` has in the gaps between rows does not enter into it: the
        // store below names the texel runs only, so those bytes are never read
        // out of this buffer whichever way it was built.
        let staged: std::borrow::Cow<'_, [u8]> =
            if direct_rows && bpr == src_stride as usize && src.len() >= frame_len {
                std::borrow::Cow::Borrowed(&src[..frame_len])
            } else {
                let mut frame = vec![0u8; frame_len];
                for y in 0..mh {
                    let src_off = (y as usize) * (src_stride as usize);
                    let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
                    if src_off + src_row_len > src.len() {
                        return refuse(
                            mapping_id,
                            SurfaceWriteRefusal::SourceShort {
                                need: src_off + src_row_len,
                                have: src.len(),
                                row: y,
                            },
                        );
                    }
                    let src_row = &src[src_off..src_off + src_row_len];
                    let row_bytes: &[u8] = if direct_rows {
                        &src_row[..tight]
                    } else {
                        if let Some(ref mut rgba_row) = rgba {
                            if !RowToRgba8::Bgra8.convert(src_row, mw, rgba_row)
                                || !store_rail
                                    .is_some_and(|rail| rail.convert(rgba_row, mw, &mut row))
                            {
                                return refuse(
                                    mapping_id,
                                    SurfaceWriteRefusal::RowConvert { format, row: y },
                                );
                            }
                        } else {
                            let n = src_row_len.min(row.len());
                            row[..n].copy_from_slice(&src_row[..n]);
                        }
                        &row
                    };
                    let dst_off = (y as usize).saturating_mul(bpr);
                    if dst_off + tight > frame.len() {
                        return refuse(
                            mapping_id,
                            SurfaceWriteRefusal::StagedShort {
                                need: dst_off + tight,
                                have: frame.len(),
                                row: y,
                            },
                        );
                    }
                    frame[dst_off..dst_off + tight].copy_from_slice(&row_bytes[..tight]);
                }
                note_surface_write_phase(
                    SurfaceWritePhase::Stage,
                    stage_started.elapsed().as_micros() as u64,
                );
                std::borrow::Cow::Owned(frame)
            };
        let frame: &[u8] = staged.as_ref();
        let land_started = std::time::Instant::now();
        // One call for the whole frame, carrying the runs it should store,
        // rather than one call per surviving run: every call re-runs
        // `flush_intersecting` over the deferred windows and re-resolves the
        // mapping's page list, both `O(pages)`, so the per-run shape pays that
        // twice-over walk for each hole the skip list cuts. The selection
        // travels into the walk instead, so the resolution happens once and
        // each imported page run moves only the parts of itself the runs name.
        //
        // The runs are the `tight` bytes at the head of each of `mh` rows, not
        // the frame's whole extent. A row pitch wider than the packed row leaves
        // padding between rows, and that padding is not a texel this call was
        // given: the contig arm above writes row by row and never touches it,
        // so storing the staged frame entire would zero it here and leave it
        // alone there — the same call landing different guest memory depending
        // only on whether the guest's pages happened to be adjacent.
        //
        // Those bytes do belong to this plane (`sample_window_from_device_plane`
        // requires the plane's own `plane_size` to cover `bpr * (mh - 1) +
        // tight`), so this is not an overrun into a neighbouring allocation. It
        // is content the guest put there and the device was never asked to
        // replace.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for y in 0..mh {
            let row_lo = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
            for (lo, hi) in unskipped(row_lo, row_lo.saturating_add(tight as u64), skip) {
                match runs.last_mut() {
                    // A packed pitch makes consecutive rows adjacent; coalescing
                    // keeps that frame the single run it was before the split,
                    // which is the shape the hot 8 MB composite surface takes.
                    Some(last) if last.1 == lo => last.1 = hi,
                    _ => runs.push((lo, hi)),
                }
            }
        }
        if !mapper::write_mapping_bytes_only(
            state,
            host,
            mapping_id,
            base_off,
            frame,
            Some(&runs),
            &vouched,
        ) {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::MapperWrite {
                    lo: base_off,
                    len: frame.len(),
                },
            );
        }
        note_surface_write_phase(
            SurfaceWritePhase::Land,
            land_started.elapsed().as_micros() as u64,
        );
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    if !skip.is_empty() {
        // A skipping write leaves the guest's pages holding `src` everywhere
        // except the ranges the guest itself owns, so the pages are now the only
        // complete copy of this surface and no host-side copy of `src` is its
        // content any more. Both of them have to be told so.
        //
        // The byte cache would answer the mapper-ref-texture LOAD seed, which prefers it
        // over the surface's own pages, and hand back exactly the bytes the
        // guest's stores were preserved *from*.
        crate::runtime::surface_cache::forget(state, mapping_id);
        // The resident `src` was read out of does not have the guest's stores
        // either, but it is NOT retired here: it is also the source other
        // deferred windows on the same identity are still going to flush from,
        // and withdrawing its content mid-drain loses their frames outright
        // (`chain_resident_land_fail reason=read_target`, `deferred_flush_lost
        // reason=resident_epoch_drift live=None`, both on 1920x1080 scanout
        // surfaces — measured, and a black screen).
        //
        // What disqualifies it instead is the `mark_mapping_written` above,
        // which advances `surface_content_epoch` past the resident's stamp. Both
        // rails that would bind a resident in place of this surface compare that
        // pair — the attachment LOAD elision always did, and the sampled ladder's
        // resident rung now does too. The caller that produced `src` must
        // therefore not hand the stamp back after a skipping write.
        //
        // The guest-write stamp is re-taken, because the device has *adopted*
        // the guest's stores: they are in the pages it just wrote around.
        //
        // Withholding it was the defect this rail shipped with. The stamp is the
        // `since` every later `guest_written_pages` call is asked against, and
        // `page_gen` records the harvest that last saw each page written, never
        // resetting per consumer. So a stamp that does not move makes the skip
        // set grow monotonically: one full CPU repaint of a window marks every
        // page of it, and from then on every deferred flush of that surface
        // skips the entire extent and the device's composite never reaches guest
        // memory again. Measured live as a desktop that goes black and stays
        // black, at `render_flush_preserved_guest_write` ~65 a second, on a boot
        // whose sampled resident rung was gated off — so it was this rail and
        // not that one.
        //
        // It is honest as well as necessary: the stamp says "no host-side copy
        // is known stale relative to these pages", and after the two retirements
        // above there is no host-side copy at all.
        crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
        return true;
    }
    let cache_started = std::time::Instant::now();
    let tight_frame = (mw as usize)
        .saturating_mul(mh as usize)
        .saturating_mul(RGBA8_BPP as usize);
    match cache {
        CacheOutcome::Invalidate => crate::runtime::surface_cache::forget(state, mapping_id),
        CacheOutcome::Publish(shared) => match shared.filter(|owner| {
            src_stride == mw.saturating_mul(RGBA8_BPP)
                && owner.len() >= tight_frame
                && std::ptr::eq(owner.as_ptr(), src.as_ptr())
        }) {
            Some(owner) => crate::runtime::surface_cache::store_shared(
                state,
                mapping_id,
                mw,
                mh,
                owner.clone(),
            ),
            None => crate::runtime::surface_cache::store_rows(
                state, mapping_id, mw, mh, src, src_stride,
            ),
        },
    }
    note_surface_write_phase(
        SurfaceWritePhase::Cache,
        cache_started.elapsed().as_micros() as u64,
    );
    // This write just made the host copy and the guest pages agree, so it is the
    // moment the copy's currency can be pinned. Nothing else arms this mapping:
    // the backing sampled ladder's first census read `gw_no_stamp` 14 092 against
    // `gw_clean` 0 because only the Vulkan Store rails ever stamped, and the
    // copy that rung serves is written here. Unstamped, the reader cannot tell a
    // surface the guest has rewritten from one it has not, and must assume the
    // worst on every bind.
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

/// What becomes of this device's own copy of the frame a Store lands.
///
/// # Why the caller decides and not this writer
///
/// The host copy is not part of the write: the guest's pages hold the frame
/// either way, and every reader of the copy has another source. What the copy
/// buys is *speed* for two readers — the present capture and a sampled bind of
/// the surface — and what it costs is a second whole-frame pass over the source,
/// which a driven macos-13 Metal boot measured at 12.6 ms a flush, 43 % of that
/// rail's `store_us`.
///
/// Only the caller knows whether that pass is worth running, because only the
/// caller knows whether its rail is about to hold the same frame in a resident
/// render target it can read instead. Deriving it here would mean this writer
/// asking a rail a question about a texture, which is the wrong direction
/// through the layer.
///
/// Both arms leave one invariant true, and it is the one the readers depend on:
/// after either, [`crate::runtime::surface_cache::frame_generation`] names this
/// frame. They differ only in where its bytes are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FramePublication {
    /// Build the BGRA8 host copy and publish it. The frame is readable from
    /// [`crate::runtime::surface_cache::get`] and its siblings.
    HostCache,
    /// Skip the copy and cede the frame to the caller's rail resident
    /// ([`crate::runtime::surface_cache::cede_surface_to_resident`]).
    ///
    /// The caller is promising that a resident of its rail holds these same
    /// pixels and can be read back — on the Metal rail that promise is
    /// `backend::metal::resident::read_published_bgra8`, and it is kept by
    /// `runtime::draw::metal::publish_resident_target` running after this
    /// returns. A caller that cedes without a resident does not corrupt
    /// anything: the guest's pages are still written, and the two byte readers
    /// simply miss, which they already handle (the capture keeps the prior
    /// frame, the sampled bind reads the guest's pages).
    RailResident,
}

/// Write a tight RGBA8 image into a mapper-ref-texture mapping, optionally as changed-spans.
///
/// Archive `apple_pv_gpu_write_mapper_ref_texture_image_changed`: when `seed_rgba` is present
/// (same layout as `rgba`), only contiguous native-format spans that differ from
/// the seed are written. Equivalent to a full `storeAction=Store` when the seed
/// was the Metal Load attachment content (unchanged texels match guest), without
/// rewriting multi-MiB of identical bytes on every damage pass. `seed_rgba = None`
/// always writes every row (Clear / multi-draw final / force-full).
///
/// `publication` says what becomes of the host copy of the frame; see
/// [`FramePublication`], and note that it decides whether the second whole-frame
/// pass over `rgba` runs at all.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping writer keeps source rows, destination geometry and \
              frame publication explicit"
)]
pub fn write_rgba8_image_changed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    rgba: &[u8],
    seed_rgba: Option<&[u8]>,
    width: u32,
    height: u32,
    publication: FramePublication,
) -> bool {
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    let rgba_stride = width.saturating_mul(RGBA8_BPP);
    let need = (height as usize).saturating_mul(rgba_stride as usize);
    if rgba.len() < need {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceShort {
                need,
                have: rgba.len(),
                row: 0,
            },
        );
    }
    if let Some(seed) = seed_rgba {
        if seed.len() < need {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::SeedShort {
                    need,
                    have: seed.len(),
                },
            );
        }
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: mw,
                latched_height: mh,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    let Some((base_off, bpr_u32, span_end)) = mapper_ref_texture_sample_window(m, mw, mh, format)
    else {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::WindowUnresolved {
                width: mw,
                height: mh,
                format,
            },
        );
    };
    let bpr = bpr_u32 as u64;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return refuse(mapping_id, SurfaceWriteRefusal::FormatRowLength { format });
    };
    let bpr_usize = bpr as usize;
    let tight = tight as usize;
    let mut native = vec![0u8; tight];
    let mut seed_native = vec![0u8; tight];
    // Settle submitted guest-page writes, as at every other read/write entry in
    // this file. It has to be here rather than on one arm: the fragmented arm
    // ends in `mapper::write_mapping_bytes`, which settles, while the
    // `contig_for_write` arm is a raw `copy_nonoverlapping` into the mapped span
    // and settles nothing. Whether a submitted copy executed before or after this
    // write therefore depended on whether the guest's pages happened to be
    // contiguous, and landing after puts an older frame on top of this one —
    // which `mapper::write_mapping_bytes_only` states as its own reason for
    // flushing here.
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingRgba8Write,
    );
    // One proof for the whole image: the changed-span loop below writes each
    // differing row separately, and the walk is a translation per page.
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "rgba8_changed") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    let contig = contig_for_write(state, host, mapping_id, span_end, &vouched);
    // SAFETY: when Some, contig covers span_end.
    let base = contig.map(|(ptr, _)| unsafe { (ptr as *mut u8).add(base_off as usize) });

    // The host-cache frame, built once, up here, into the buffer the previous
    // frame is already in.
    //
    // It used to be built after the row loop, into a fresh `vec![0u8; need]`,
    // by a second whole-frame pass over `rgba` — and that was 45.9 % of the
    // Metal rail's `store_us`, 2 296 us a draw, for a conversion the row loop
    // below had already performed. `tight_row_is_bgra8` is when the two are the
    // same bytes: a BGRA8 mapping's native texel *is* the cache's, so the rows
    // are taken from here and converted once instead of twice.
    //
    // Building it before the write rather than after is what lets the rows read
    // from it, and it is safe in the direction that matters: `take_frame_buffer`
    // has already removed the entry, so a refusal below leaves this cache
    // holding nothing rather than holding a frame the guest's half-written pages
    // no longer match. Publishing happens at the bottom, on success only.
    // Empty when the frame is ceded: nothing below reads it, because
    // `cache_rows_are_native` is `&&`-ed with the same condition.
    //
    // Both arms *remove* this mapping's entry before the write, and that is the
    // half that matters for correctness rather than for allocation reuse: a
    // refusal below leaves the guest's pages half-written, and an entry that
    // survived would go on claiming to be a frame those pages no longer hold.
    // `take_frame_buffer` removes and hands back the allocation; the ceded arm
    // has no allocation to want, so it removes and keeps nothing. Either way the
    // claim is only put back at the bottom, on success.
    let mut cache = match publication {
        FramePublication::HostCache => {
            crate::runtime::surface_cache::take_frame_buffer(state, mapping_id, mw, mh)
        }
        FramePublication::RailResident => {
            crate::runtime::surface_cache::forget(state, mapping_id);
            Vec::new()
        }
    };
    if publication == FramePublication::HostCache {
        let _span_cache = crate::runtime::chain_phase::CostSpan::new("surface_changed_cache_us");
        // Converted straight into the cache buffer, overwriting the previous
        // frame. What that costs is measured: ~0.90 ms a flush moving 16.6 MB,
        // which is memory-bound.
        //
        // # The previous frame is under this write, and using it does not pay
        //
        // `take_frame_buffer` reuses the buffer the *previous* frame is in, so
        // the bytes being overwritten here are the last frame this device
        // published — and a BGRA8 row equals its predecessor exactly when the
        // RGBA8 row it came from does. That is a free damage map, in principle:
        // 1c81555a puts the row loop's convert at ~88 % of this writer and
        // `surface_row_no_seed` at 200 of 201 flushes, so every row is converted
        // and landed because nothing says which rows changed.
        //
        // It was built, measured on a driven macos-13 Metal boot, and removed.
        // **71 % of rows are unchanged under a full-screen drag and 96 % at
        // idle** — the win is real and large. Detecting it is not free, and the
        // cost lands in a place the offline numbers did not predict:
        //
        // ```text
        //                                     offline bench   live
        //   convert straight into the cache        0.15 ms    0.90 ms
        //   convert into a scratch, then bcmp      0.75 ms   11.57 ms
        // ```
        //
        // Split three ways live, the scratch version spends 10.05 ms in the
        // *conversion*, 1.15 ms in the comparison and 0.28 ms in the copy back.
        // The comparison is as cheap as the bench said; converting a row into a
        // small reused buffer instead of into its place in the frame is what
        // costs, by an order of magnitude, and no reading of the two loops
        // explains it. The cache loop's share of the row loop went from 0.14 to
        // 0.83 across the change, so this is not boot-to-boot drift.
        //
        // Net: the detection costs about what skipping the unchanged rows would
        // save. Two other shapes were tried and are worse — see the table in the
        // commit and note that both were written as `Rgba8ToRow` methods and
        // neither survived its measurement, so neither is in the tree.
        //
        // What is worth trying next is not a cheaper comparison. It is not
        // needing one: the guest's Load attachment already carries the previous
        // content as `seed_rgba`, and it arrived on 1 of 201 flushes. Why the
        // other 200 have no seed is a question about `seed_color_load` and the
        // pass's load action, not about this loop.
        for y in 0..mh as usize {
            let off = y * rgba_stride as usize;
            let src_row = &rgba[off..off + rgba_stride as usize];
            let dst_row = &mut cache[off..off + rgba_stride as usize];
            // `Rgba8ToRow::Bgra8` *is* this swap. It was the third hand-inlined
            // copy of it in this crate; the other two were the row loop's
            // `rgba8_row_to_native` and the staged writers'. Named as the
            // variant rather than parsed from an ordinal because the cache's
            // layout is this device's own choice and not the guest's — see
            // `cache_rows_are_native` below, which is the only place the two
            // meet.
            let swapped = pixel_format::Rgba8ToRow::Bgra8.convert(src_row, mw, dst_row);
            debug_assert!(swapped, "the cache row is exactly mw BGRA8 texels");
        }
    }
    // Whether a cache row is byte for byte this mapping's native texel row, so
    // the row loop can read it instead of converting again. Both halves must
    // hold: the format's tight row is BGRA8, and the tight row is the whole of
    // the cache's row — `tight_row_bytes` and `rgba_stride` are computed from
    // different places and a format that made them differ would slice the wrong
    // bytes.
    let cache_rows_are_native = publication == FramePublication::HostCache
        && matches!(
            format,
            MTL_FORMAT_BGRA8_UNORM | pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
        )
        && tight == rgba_stride as usize;

    // Parsed once above `surface_changed_rows_us`, which this loop is: it runs
    // `mh` times a flush and used to re-answer "is this BGRA8" per row and then
    // the format ordinal per texel. `Rgba8ToRow::Bgra8` *is* the hand-rolled
    // BGRA arm this replaced, vectorised. `Option` rather than an early refusal
    // because `cache_rows_are_native` converts nothing, so a format with no arm
    // is not that path's problem.
    let store_rail = pixel_format::Rgba8ToRow::for_format(format);
    let span_rows = crate::runtime::chain_phase::CostSpan::new("surface_changed_rows_us");
    // The three things inside `surface_changed_rows_us`, which was 90 % of the
    // Metal rail's `store_us` and had nothing dividing it. Accumulated in
    // locals and emitted once per flush rather than through
    // `chain_phase::CostSpan` per row, for two reasons `sampled_phase`'s doc
    // records: a `CostSpan` truncates to microseconds and a row's convert is
    // ~0.15 us at the rates `pixel_format::Rgba8ToRow` reaches, so every part
    // but the landing write would report as free; and a per-row commit would be
    // `mh` atomic map lookups a flush where three suffice.
    let (mut convert_ns, mut diff_ns, mut land_ns) = (0u64, 0u64, 0u64);
    for y in 0..mh as usize {
        let src_off = y * rgba_stride as usize;
        let src_row = &rgba[src_off..src_off + rgba_stride as usize];
        let row_started = std::time::Instant::now();
        if !cache_rows_are_native
            && !store_rail.is_some_and(|rail| rail.convert(src_row, mw, &mut native))
        {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::RowConvert {
                    format,
                    row: y as u32,
                },
            );
        }
        // One name for the row's native bytes, whichever produced them, so the
        // three readers below cannot disagree about which buffer they are in.
        let native: &[u8] = if cache_rows_are_native {
            &cache[src_off..src_off + tight]
        } else {
            &native
        };
        let seed_row = if let Some(seed) = seed_rgba {
            let s = &seed[src_off..src_off + rgba_stride as usize];
            if !store_rail.is_some_and(|rail| rail.convert(s, mw, &mut seed_native)) {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::SeedRowConvert {
                        format,
                        row: y as u32,
                    },
                );
            }
            Some(seed_native.as_slice())
        } else {
            None
        };
        convert_ns += row_started.elapsed().as_nanos() as u64;
        let diff_started = std::time::Instant::now();
        let row_unchanged = seed_row.is_some_and(|srow| srow == native);
        diff_ns += diff_started.elapsed().as_nanos() as u64;
        if row_unchanged {
            continue;
        }
        let row_moff = base_off.saturating_add((y as u64).saturating_mul(bpr));
        if let Some(base) = base {
            let dst = unsafe { base.add(y.saturating_mul(bpr_usize)) };
            if let Some(seed) = seed_row {
                // Changed spans only within the row. The scan and the copy are
                // charged apart because they are the two candidate answers for
                // this bar and they have opposite fixes.
                let mut runs = ChangedRuns::new(&native[..tight], &seed[..tight]);
                loop {
                    let scan_started = std::time::Instant::now();
                    let run = runs.next();
                    diff_ns += scan_started.elapsed().as_nanos() as u64;
                    let Some(run) = run else { break };
                    let land_started = std::time::Instant::now();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            native.as_ptr().add(run.start),
                            dst.add(run.start),
                            run.len(),
                        );
                    }
                    land_ns += land_started.elapsed().as_nanos() as u64;
                }
            } else {
                let land_started = std::time::Instant::now();
                unsafe {
                    std::ptr::copy_nonoverlapping(native.as_ptr(), dst, tight);
                }
                land_ns += land_started.elapsed().as_nanos() as u64;
            }
        } else if let Some(seed) = seed_row {
            let mut runs = ChangedRuns::new(&native[..tight], &seed[..tight]);
            loop {
                let scan_started = std::time::Instant::now();
                let run = runs.next();
                diff_ns += scan_started.elapsed().as_nanos() as u64;
                let Some(run) = run else { break };
                let land_started = std::time::Instant::now();
                let landed = mapper::write_mapping_bytes(
                    state,
                    host,
                    mapping_id,
                    row_moff.saturating_add(run.start as u64),
                    &native[run.clone()],
                    &vouched,
                );
                land_ns += land_started.elapsed().as_nanos() as u64;
                if !landed {
                    return refuse(
                        mapping_id,
                        SurfaceWriteRefusal::MapperWrite {
                            lo: row_moff.saturating_add(run.start as u64),
                            len: run.len(),
                        },
                    );
                }
            }
        } else {
            let land_started = std::time::Instant::now();
            let landed =
                mapper::write_mapping_bytes(state, host, mapping_id, row_moff, native, &vouched);
            land_ns += land_started.elapsed().as_nanos() as u64;
            if !landed {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::MapperWrite {
                        lo: row_moff,
                        len: native.len(),
                    },
                );
            }
        }
    }
    drop(span_rows);
    // Three counters a flush, not `mh` of them. Nanoseconds, so a part that is
    // sub-microsecond per row is not reported as free; divide by 1000 against
    // `surface_changed_rows_us` by hand.
    crate::runtime::drain::note_store_route_n("surface_row_convert_ns", convert_ns);
    crate::runtime::drain::note_store_route_n("surface_row_diff_ns", diff_ns);
    crate::runtime::drain::note_store_route_n("surface_row_land_ns", land_ns);
    // The denominators. A microsecond total answers "how long"; it cannot
    // answer "per what", and the three parts above have different per-whats.
    crate::runtime::drain::note_store_route("surface_row_flushes");
    crate::runtime::drain::note_store_route_n("surface_row_rows", mh as u64);
    crate::runtime::drain::note_store_route_n(
        "surface_row_dst_bytes",
        (mh as u64).saturating_mul(tight as u64),
    );
    // Which format the rows are converted *to*. The rate the convert reaches
    // has no meaning without it: this rail's arms differ by 40x and by a factor
    // of two in destination width, and the census cannot say which arm ran.
    if crate::observe::first_sight("surface_row_native_format", u64::from(format)) {
        // `publication` rides along because `cache_is_native` reads `false` for
        // two unrelated reasons — a format whose texel is not the cache's, and a
        // frame that has no cache at all — and the fix for each is the other's
        // no-op.
        crate::observe::fail(format!(
            "surface_row_native_format format={format:#x} tight={tight} \
             rgba_stride={rgba_stride} cache_is_native={cache_rows_are_native} \
             publication={publication:?} mapping={mapping_id}"
        ));
    }
    crate::runtime::drain::note_store_route_n(
        "surface_row_src_bytes",
        (mh as u64).saturating_mul(rgba_stride as u64),
    );
    crate::runtime::drain::note_store_route(if cache_rows_are_native {
        "surface_row_cache_is_native"
    } else {
        "surface_row_converts_frame"
    });
    crate::runtime::drain::note_store_route(if seed_rgba.is_some() {
        "surface_row_has_seed"
    } else {
        "surface_row_no_seed"
    });
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // Host render-cache (Linux §8.5): the frame, published now that the guest's
    // pages hold it too. Both arms publish a *generation* for this frame — see
    // `FramePublication` — and differ only in whether the bytes are here or in
    // the caller's rail resident.
    match publication {
        FramePublication::HostCache => {
            crate::runtime::surface_cache::store(state, mapping_id, mw, mh, cache)
        }
        FramePublication::RailResident => {
            crate::runtime::drain::note_store_route(
                if crate::runtime::surface_cache::cede_surface_to_resident(
                    state, mapping_id, mw, mh,
                ) {
                    "surface_row_ceded"
                } else {
                    // A geometry this cache would not have stored anyway. The
                    // guest's pages are written; nothing host-side claims the
                    // frame, which is the same state a refused store leaves.
                    "surface_row_cede_refused"
                },
            );
        }
    }
    // This write just made the host copy and the guest pages agree, so it is the
    // moment the copy's currency can be pinned. Nothing else arms this mapping:
    // the backing sampled ladder's first census read `gw_no_stamp` 14 092 against
    // `gw_clean` 0 because only the Vulkan Store rails ever stamped, and the
    // copy that rung serves is written here. Unstamped, the reader cannot tell a
    // surface the guest has rewritten from one it has not, and must assume the
    // worst on every bind.
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

/// Write rows already encoded as a mapper-ref-texture mapping's native pixel format.
///
/// Unlike [`write_raw_rows`], this resolves the texture's sample window inside
/// an IOSurface allocation: its base offset and row pitch come from the mapping
/// descriptor, and row padding is left untouched. Unlike
/// [`write_rgba8_image_changed`], it performs no colour conversion because the
/// source bytes already are the destination texels.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping writer keeps source rows and destination geometry explicit"
)]
pub fn write_native_image<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
    format: u16,
) -> bool {
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    let Some(tight) = pixel_format::tight_row_bytes(width, format) else {
        return refuse(mapping_id, SurfaceWriteRefusal::FormatRowLength { format });
    };
    if src_stride < tight {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::NativeSourceStride {
                src_stride,
                row_bytes: tight,
            },
        );
    }
    let Some(need) = (height as usize)
        .checked_sub(1)
        .and_then(|rows| (src_stride as usize).checked_mul(rows))
        .and_then(|prefix| prefix.checked_add(tight as usize))
    else {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::FrameExtent {
                bpr: src_stride as usize,
                height,
            },
        );
    };
    if src.len() < need {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceShort {
                need,
                have: src.len(),
                row: height.saturating_sub(1),
            },
        );
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    let (mw, mh, mapping_format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: mw,
                latched_height: mh,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    if mapping_format != format {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::NativeFormatMismatch {
                source: format,
                mapping: mapping_format,
            },
        );
    }
    let Some((base_off, bpr, span_end)) =
        mapper_ref_texture_sample_window(m, mw, mh, mapping_format)
    else {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::WindowUnresolved {
                width: mw,
                height: mh,
                format: mapping_format,
            },
        );
    };

    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingNativeImageWrite,
    );
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "native_image") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched) {
        // SAFETY: the revalidated contiguous view covers `span_end`.
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let dst = unsafe { base.add(y * bpr as usize) };
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_off), dst, tight as usize);
            }
        }
    } else {
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let row_off = base_off.saturating_add((y as u64).saturating_mul(u64::from(bpr)));
            let row = &src[src_off..src_off + tight as usize];
            if !mapper::write_mapping_bytes(state, host, mapping_id, row_off, row, &vouched) {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::MapperWrite {
                        lo: row_off,
                        len: row.len(),
                    },
                );
            }
        }
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // Native integer texels have no BGRA host-cache representation. Guest pages
    // are authoritative after this write, so an older cache entry must retire.
    crate::runtime::surface_cache::forget(state, mapping_id);
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

/// Write tightly packed raw rows into a mapping (depth32float / stencil8).
///
/// Contig HostOps view when possible; else multi-import (no write_gpa).
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API keeps source rows and destination geometry explicit"
)]
pub fn write_raw_rows<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    row_bytes: u32,
    width: u32,
    height: u32,
) -> bool {
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    if row_bytes == 0 || src_stride < row_bytes {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceStride { src_stride, width },
        );
    }
    let need = (height as u64).saturating_mul(src_stride as u64) as usize;
    if src.len() < need {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceShort {
                need,
                have: src.len(),
                row: 0,
            },
        );
    }
    // Deferred-writeback flush-on-access (coarse: whole mapping — this entry
    // resolves its window only later and is off the hot compute path).
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingRawRowsWrite,
    );
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    if m.has_geom && (m.width != width || m.height != height) {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: m.width,
                latched_height: m.height,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    let span_end = (row_bytes as u64).saturating_mul(height as u64);
    let rb = row_bytes as usize;
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "raw_rows") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched) {
        // SAFETY: contig covers span_end from offset 0.
        let base = ptr as *mut u8;
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let dst = unsafe { base.add(y * rb) };
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_off), dst, rb);
            }
        }
    } else {
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let moff = (y as u64).saturating_mul(row_bytes as u64);
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                moff,
                &src[src_off..src_off + rb],
                &vouched,
            ) {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::MapperWrite { lo: moff, len: rb },
                );
            }
        }
    }
    state.invalidate_storage_residency_window(mapping_id, 0, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // Depth and stencil rows, so there is usually no BGRA entry under this
    // mapping at all and this is a no-op. Kept anyway, and not as a formality:
    // a mapping's id is not typed by aspect, and the one thing that must never
    // happen is a writer of a mapping's pages leaving a host-side copy behind
    // that claims to be them. The rule belongs to every writer here or it is
    // not a rule.
    crate::runtime::surface_cache::forget(state, mapping_id);
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

/// Read tightly packed raw rows from a mapping (depth32float / stencil8 LOAD).
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API keeps source rows and destination geometry explicit"
)]
pub fn read_raw_rows<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    row_bytes: u32,
    width: u32,
    height: u32,
) -> bool {
    if !scanout_extent_ok(width, height) || row_bytes == 0 || dst_stride < row_bytes {
        return false;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return false;
    }
    // Deferred-writeback flush-on-access (coarse: whole mapping — this entry
    // resolves its window only later and is off the hot compute path).
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingRawRowsRead,
    );
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    if m.has_geom && (m.width != width || m.height != height) {
        return false;
    }
    let span_end = (row_bytes as u64).saturating_mul(height as u64);
    let rb = row_bytes as usize;
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
        // SAFETY: contig covers span_end from offset 0.
        let base = ptr as *const u8;
        for y in 0..height as usize {
            let dst_off = y * dst_stride as usize;
            let src = unsafe { base.add(y * rb) };
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst[dst_off..].as_mut_ptr(), rb);
            }
        }
    } else {
        for y in 0..height as usize {
            let dst_off = y * dst_stride as usize;
            let moff = (y as u64).saturating_mul(row_bytes as u64);
            if !mapper::read_mapping_bytes(
                state,
                host,
                mapping_id,
                moff,
                &mut dst[dst_off..dst_off + rb],
            ) {
                return false;
            }
        }
    }
    true
}

/// Resolve the window a mapping's own latched geometry names, for a rectangle
/// addressed in that geometry rather than in an explicit plane window.
///
/// The resolution [`read_rect_raw`] and [`write_rect_raw`] share: the latched
/// format (BGRA8 when the mapping never declared one), the plane window
/// [`mapper_ref_texture_sample_window`] decodes for it, and the texel size. Returns
/// `(base_offset, bytes_per_row, span_end, bytes_per_texel)`, or `None` when
/// the mapping is gone, carries no latched geometry, has no decodable window,
/// has an unknown format, or the rectangle leaves the surface.
/// Where a mapped mapper-ref-texture surface's texels sit, and how wide one is.
///
/// `mapping_geom_window` used to return this as `Option<(u64, u32, u64, u32)>`,
/// a shape whose meaning existed only in the destructuring patterns of the two
/// callers that unpacked it — and which they then splatted straight into four
/// parameters of the `_at` functions, split around the rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceWindow {
    /// Byte offset of the plane's first texel within the mapping.
    pub base_off: u64,
    /// Bytes per row of the surface, which is not `width * bpp`.
    pub bpr: u32,
    /// One past the last byte the window may touch.
    pub span_end: u64,
    /// Bytes per texel.
    pub bpp: u32,
}

/// A texel rectangle within a surface.
///
/// The four fields are `u32` and were adjacent in five signatures here, so
/// every permutation of them compiled and no call site could object. One test
/// call read `..., 0, 0, 4, 1, 1, ...` — five bare numbers spanning the origin,
/// the extent and the bytes per texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub origin_x: u32,
    pub origin_y: u32,
    pub width: u32,
    pub height: u32,
}

fn mapping_geom_window(state: &DeviceState, mapping_id: u32, rect: Rect) -> Option<SurfaceWindow> {
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    let m = state.mappings.get(&mapping_id)?;
    if !m.has_geom {
        return None;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let (base_off, bpr, span_end) = mapper_ref_texture_sample_window(m, m.width, m.height, format)?;
    let bpp = pixel_format::bytes_per_pixel(format)?;
    if origin_x.saturating_add(width) > m.width || origin_y.saturating_add(height) > m.height {
        return None;
    }
    Some(SurfaceWindow {
        base_off,
        bpr,
        span_end,
        bpp,
    })
}

/// Read a rectangular texel region from a mapped mapper-ref-texture IOSurface.
/// Contig HostOps view when possible; else multi-import.
#[cfg(test)]
pub fn read_rect_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    rect: Rect,
    dst: &mut [u8],
    dst_stride: u32,
) -> bool {
    let Some(window) = mapping_geom_window(state, mapping_id, rect) else {
        return false;
    };
    read_rect_raw_at(state, host, mapping_id, window, rect, dst, dst_stride)
}

/// Read a rect using an explicit sample window (plane base + bpr + span).
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the explicit-plane API mirrors its sample window and rectangle"
)]
pub fn read_rect_raw_at<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    window: SurfaceWindow,
    rect: Rect,
    dst: &mut [u8],
    dst_stride: u32,
) -> bool {
    let SurfaceWindow {
        base_off,
        bpr: surface_bpr,
        span_end,
        bpp,
    } = window;
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    if !scanout_extent_ok(width, height) || bpp == 0 {
        return false;
    }
    // Deferred-writeback flush-on-access, for the same reason
    // `mapper::read_mapping_bytes` does it: this read must observe the deferred
    // Store's pixels, not the stale pre-Store guest bytes.
    //
    // It has to be here rather than at the callers because only one of the two
    // paths below was ever covered. The fragmented path ends in
    // `read_mapping_bytes`, which flushes; the `contig_for_span` path is a raw
    // `copy_nonoverlapping` out of the mapped span and flushes nothing — so
    // whether a mapper-ref-texture surface read observed the deferred Store depended on
    // whether its guest pages happened to be contiguous. Three callers read
    // guest pages through here with no flush of their own: the ref-texture view
    // loader, a blit reading a mapper-ref-texture backing, and the compute sample
    // stage.
    //
    // `flush_intersecting` returns immediately when nothing is armed, so this
    // costs a map-empty check per read. It must also precede `contig_for_span`:
    // the flush writes through the mapping and can retire the cached view.
    let settle_started = std::time::Instant::now();
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingRectRead,
    );
    crate::runtime::drain::note_store_route_us(
        "rectrd_settle_us",
        settle_started.elapsed().as_micros() as u64,
    );
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    let Some(row_bytes) = width.checked_mul(bpp) else {
        return false;
    };
    if dst_stride < row_bytes {
        return false;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return false;
    }
    let x_off = (origin_x as u64).saturating_mul(bpp as u64);
    if x_off.saturating_add(row_bytes as u64) > surface_bpr as u64 {
        return false;
    }
    let rb = row_bytes as usize;
    let bpr = surface_bpr as usize;
    // The rect must end inside the sample window, and that is asked once for both
    // arms below rather than inside either. A correctly-sized read satisfies it
    // exactly (a dense tight read has `read_end == span_end`), so it drops only a
    // genuine overrun.
    //
    // It used to sit inside the contig arm, on the reasoning that the fragmented
    // arm was bounded anyway — which is true, but only by its own slice bounds:
    // that arm reads the window and then indexes rows into it, so an overrunning
    // rect came back as a bare `false` from a `get` that returned `None`. Both
    // callers do name that (`rd_row_t11_io`, `RefTextureViewDecline::Read`), so it was
    // never a silent loss, but neither can say the rect left the window, and the
    // fragmented arm is the one a driven x86 boot actually takes. One check above
    // the split gives both arms the same refusal and the same line.
    let read_end = rect_extent_end(base_off, origin_y, height, bpr, x_off, rb);
    if read_end > span_end {
        crate::observe::fail(format!(
            "mapping_read fail reason=read_overrun mid={mapping_id} base_off={base_off} origin_y={origin_y} height={height} bpr={surface_bpr} x_off={x_off} rb={rb} read_end={read_end} span_end={span_end}"
        ));
        return false;
    }
    // The rail's remaining cost is per *call*, so every phase of a call is
    // charged and the arms are disjoint: `rectrd_contig_us` covers the view
    // revalidation both arms need, then exactly one of `rectrd_copy_us` (the
    // packed arm's memcpy) and `rectrd_window_us` (the fragmented arm, which
    // materialises the whole sample window however small the rect) runs. With
    // `rectrd_settle_us` above them, the four sum to the call.
    let contig_started = std::time::Instant::now();
    let contig = contig_for_span(state, host, mapping_id, span_end);
    crate::runtime::drain::note_store_route_us(
        "rectrd_contig_us",
        contig_started.elapsed().as_micros() as u64,
    );
    if let Some((ptr, _)) = contig {
        let copy_started = std::time::Instant::now();
        // SAFETY: contig covers span_end, and read_end ≤ span_end (checked).
        let base = unsafe { (ptr as *const u8).add(base_off as usize) };
        if x_off == 0 && rb == bpr && dst_stride as usize == rb {
            // Dense rows: identical byte range as the loop, one copy.
            let src = unsafe { base.add((origin_y as usize).saturating_mul(bpr)) };
            let len = (height as usize) * rb;
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), len);
            }
        } else {
            for y in 0..height as usize {
                let dst_off = y * dst_stride as usize;
                let row_off = ((origin_y as usize) + y)
                    .saturating_mul(bpr)
                    .saturating_add(x_off as usize);
                let src = unsafe { base.add(row_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst[dst_off..].as_mut_ptr(), rb);
                }
            }
        }
        crate::runtime::drain::note_store_route_us(
            "rectrd_copy_us",
            copy_started.elapsed().as_micros() as u64,
        );
        crate::runtime::drain::note_store_route("rectrd_contig_n");
    } else {
        crate::runtime::drain::note_store_route("rectrd_frag_n");
        let window_started = std::time::Instant::now();
        // A packed destination is the rectangle the run walk speaks natively:
        // the rows are `bpr` apart in the mapping and back to back in `dst`, so
        // one walk over the rectangle's own span lands every row where it goes.
        // That subsumes what used to be a separate full-plane-tight special
        // case, and it reaches every sub-rectangle that case could not — those
        // paid a plane-sized zeroing allocation, a whole-window read, and a
        // second row-by-row copy out of it, for a rectangle that may be a
        // fraction of the plane.
        //
        // A padded destination (`dst_stride > row_bytes`) is a shape
        // [`RectStride`] cannot hold — it describes one stride, the guest's —
        // so it keeps the window materialisation below.
        if dst_stride as usize == rb {
            let rect_off = base_off
                .saturating_add((origin_y as u64).saturating_mul(bpr as u64))
                .saturating_add(x_off);
            let ok = mapper::RectStride::new(bpr as u64, rb as u64, height as u64).is_some_and(
                |shape| mapper::read_mapping_rect(state, host, mapping_id, rect_off, shape, dst),
            );
            crate::runtime::drain::note_store_route(if ok {
                "rectrd_rect_walk"
            } else {
                "rectrd_rect_refused"
            });
            crate::runtime::drain::note_store_route_us(
                "rectrd_window_us",
                window_started.elapsed().as_micros() as u64,
            );
            return ok;
        }
        crate::runtime::drain::note_store_route("rectrd_window_padded_dst");
        // Materialize the fragmented sample window once. Calling
        // read_mapping_bytes for every row revalidates every page and rebuilds
        // all packed GPA runs each time (O(height × pages)); fullscreen
        // compute textures then strand every channel behind staging.
        let window_len_u64 = span_end.saturating_sub(base_off);
        let Ok(window_len) = usize::try_from(window_len_u64) else {
            return false;
        };
        let mut window = vec![0u8; window_len];
        if !mapper::read_mapping_bytes(state, host, mapping_id, base_off, &mut window) {
            return false;
        }
        for y in 0..height as usize {
            let dst_off = y * dst_stride as usize;
            let row_off = ((origin_y as usize) + y)
                .saturating_mul(bpr)
                .saturating_add(x_off as usize);
            let row_end = row_off.saturating_add(rb);
            let Some(row) = window.get(row_off..row_end) else {
                return false;
            };
            dst[dst_off..dst_off + rb].copy_from_slice(row);
        }
        crate::runtime::drain::note_store_route_us(
            "rectrd_window_us",
            window_started.elapsed().as_micros() as u64,
        );
    }
    true
}

/// Write a rectangular texel region into a mapped mapper-ref-texture IOSurface.
///
/// Uses latched mapping geom + [`mapper_ref_texture_sample_window`]. Prefer
/// [`write_rect_raw_at`] for an explicit plane window.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API mirrors the decoded texture rectangle"
)]
pub fn write_rect_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    rect: Rect,
    src: &[u8],
    src_stride: u32,
) -> bool {
    let Some(window) = mapping_geom_window(state, mapping_id, rect) else {
        return false;
    };
    write_rect_raw_at(state, host, mapping_id, window, rect, src, src_stride)
}

/// Write a rect using an explicit sample window (plane base + bpr + span).
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the explicit-plane API mirrors its sample window and rectangle"
)]
pub fn write_rect_raw_at<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    window: SurfaceWindow,
    rect: Rect,
    src: &[u8],
    src_stride: u32,
) -> bool {
    let SurfaceWindow {
        base_off,
        bpr: surface_bpr,
        span_end,
        bpp,
    } = window;
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    write_rect_raw_at_impl(
        state,
        host,
        mapping_id,
        SurfaceWindow {
            base_off,
            bpr: surface_bpr,
            span_end,
            bpp,
        },
        Rect {
            origin_x,
            origin_y,
            width,
            height,
        },
        src,
        src_stride,
        false,
    )
}

/// Write a complete explicit texture plane. Fragmented mappings import each
/// maximal packed GPA run once instead of re-importing for every image row.
#[allow(
    clippy::too_many_arguments,
    reason = "the full-plane API mirrors its mapping window and row layout"
)]
pub fn write_full_rect_raw_at<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    base_off: u64,
    surface_bpr: u32,
    span_end: u64,
    width: u32,
    height: u32,
    bpp: u32,
    src: &[u8],
    src_stride: u32,
) -> bool {
    write_rect_raw_at_impl(
        state,
        host,
        mapping_id,
        SurfaceWindow {
            base_off,
            bpr: surface_bpr,
            span_end,
            bpp,
        },
        Rect {
            origin_x: 0,
            origin_y: 0,
            width,
            height,
        },
        src,
        src_stride,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_rect_raw_at_impl<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    window: SurfaceWindow,
    rect: Rect,
    src: &[u8],
    src_stride: u32,
    full_plane: bool,
) -> bool {
    let SurfaceWindow {
        base_off,
        bpr: surface_bpr,
        span_end,
        bpp,
    } = window;
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    if !scanout_extent_ok(width, height) || bpp == 0 {
        return false;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    let Some(row_bytes) = width.checked_mul(bpp) else {
        return false;
    };
    if src_stride < row_bytes {
        return false;
    }
    let need = (height as u64).saturating_mul(src_stride as u64) as usize;
    if src.len() < need {
        return false;
    }
    let x_off = (origin_x as u64).saturating_mul(bpp as u64);
    if x_off.saturating_add(row_bytes as u64) > surface_bpr as u64 {
        return false;
    }
    let rb = row_bytes as usize;
    let bpr = surface_bpr as usize;
    // The destination bound, before the branch, because all three arms below
    // write guest memory and only two of them used to check it. The per-row
    // fragmented arm went through `mapper::write_mapping_bytes`, which bounds
    // against the *whole mapping's* page span and not this plane's window, so an
    // over-tall rect landed in whatever follows the window — on a multi-plane
    // IOSurface that is the next plane's pixels — and said nothing.
    //
    // `rect_extent_end` is the shared expression for exactly this reason: its own
    // doc records that the read and write sides disagreed while each computed it
    // separately. A third caller computing its own variant is how that happens
    // again, so the bound is taken once here and the arms carry none of their own.
    // A correctly-sized writeback satisfies it exactly (a dense tight write gives
    // `write_end == span_end`), so this drops ONLY a genuine overrun — named,
    // never silent.
    let write_end = rect_extent_end(base_off, origin_y, height, bpr, x_off, rb);
    if write_end > span_end {
        crate::observe::fail(format!(
            "mapping_write fail reason=writeback_overrun mid={mapping_id} base_off={base_off} origin_y={origin_y} height={height} bpr={surface_bpr} x_off={x_off} rb={rb} write_end={write_end} span_end={span_end}"
        ));
        return false;
    }
    // Deferred-writeback flush-on-access, for the same reason and on the same
    // split as `write_rgba8_image_changed`: the fragmented arms below flush
    // through `mapper::write_mapping_bytes` and the contiguous one does not, so
    // without this an armed window could land on top of this rect on packed
    // surfaces only. Safe to call from inside a flush — the storage rail reaches
    // this function through `write_full_rect_raw_at`, and `flush_intersecting`
    // removes intersecting windows up front so the nested call finds nothing.
    // Charged in the same partition as the read side above: settle, vouch, and
    // view revalidation are per *call*, and this rail's remaining cost is per
    // call rather than per byte. See `rectrd_contig_us`.
    let settle_started = std::time::Instant::now();
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingRectWrite,
    );
    crate::runtime::drain::note_store_route_us(
        "rectwr_settle_us",
        settle_started.elapsed().as_micros() as u64,
    );
    let vouch_started = std::time::Instant::now();
    let vouched = vouch_for_write(state, host, mapping_id, "rect_raw");
    crate::runtime::drain::note_store_route_us(
        "rectwr_vouch_us",
        vouch_started.elapsed().as_micros() as u64,
    );
    let Some(vouched) = vouched else {
        return false;
    };
    let contig_started = std::time::Instant::now();
    let contig = contig_for_write(state, host, mapping_id, span_end, &vouched);
    crate::runtime::drain::note_store_route_us(
        "rectwr_contig_us",
        contig_started.elapsed().as_micros() as u64,
    );
    if let Some((ptr, _)) = contig {
        crate::runtime::drain::note_store_route("rectwr_contig_n");
        // SAFETY: contig covers span_end, and write_end ≤ span_end (checked).
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        if x_off == 0 && rb == bpr && src_stride as usize == rb {
            // Dense rows: identical byte range as the loop, one copy.
            let dst = unsafe { base.add((origin_y as usize).saturating_mul(bpr)) };
            let len = (height as usize) * rb;
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
            }
        } else {
            for y in 0..height as usize {
                let src_off = y * src_stride as usize;
                let row_off = ((origin_y as usize) + y)
                    .saturating_mul(bpr)
                    .saturating_add(x_off as usize);
                let dst = unsafe { base.add(row_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr().add(src_off), dst, rb);
                }
            }
        }
    } else if full_plane {
        crate::runtime::drain::note_store_route("rectwr_frag_full_n");
        // Fragmented full-plane write: stage the native row layout and import
        // each maximal packed GPA run once. Calling write_mapping_bytes once
        // per row turns a 1928-row storage-texture writeback into thousands of
        // QEMU memory-region imports (the live compute_writeback_amplification
        // class).
        //
        // The store names each row's `rb` texel bytes rather than the frame's
        // whole extent, for the reason `write_bgra8_inner` states at its own
        // staging branch: a row pitch wider than the packed row leaves padding
        // between rows, the contig arm twenty lines above writes row by row and
        // never touches it, so storing the staged frame entire would zero it
        // here and leave it alone there — the same call landing different guest
        // memory depending only on whether the guest's pages happened to be
        // adjacent. Those bytes belong to this plane, so it was never an overrun
        // into a neighbour; it is content the guest put there and this call was
        // not asked to replace.
        // `span_end` ends at the final row's last texel. It deliberately does
        // not include padding after the final row, so staging bpr * height
        // rejects every exact-span surface whose row pitch exceeds row_bytes.
        let frame_len = match (height as usize)
            .checked_sub(1)
            .and_then(|rows| (surface_bpr as usize).checked_mul(rows))
            .and_then(|prefix| prefix.checked_add(rb))
        {
            Some(v) => v,
            None => return false,
        };
        // No `frame_end > span_end` here: this arm used to take its own variant
        // of the bound, computed without `x_off` and so looser than the one
        // `rect_extent_end` gives above. The overflow check on `frame_len` stays
        // because it guards the allocation on the next lines.
        if base_off.checked_add(frame_len as u64).is_none() {
            return false;
        }
        // With no physical row padding, the engine's tight result is already
        // the exact mapping byte window. Write it through the fragmented-run
        // importer directly; a second frame allocation/copy is redundant.
        let window_len = span_end
            .checked_sub(base_off)
            .and_then(|len| usize::try_from(len).ok());
        if origin_x == 0
            && origin_y == 0
            && rb == bpr
            && src_stride == surface_bpr
            && Some(frame_len) == window_len
        {
            crate::observe::off(format!(
                "mapping_write full_tight_direct mid={mapping_id} bytes={frame_len} bpr={surface_bpr} rows={height}"
            ));
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                base_off,
                &src[..frame_len],
                &vouched,
            ) {
                return false;
            }
            let _ = state.mark_mapping_written(mapping_id);
            crate::runtime::surface_cache::forget(state, mapping_id);
            crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
            return true;
        }
        let mut frame = vec![0u8; frame_len];
        // Built alongside the fill so the two cannot describe different rows.
        // Adjacent runs coalesce, so a packed pitch collapses to the single run
        // it was before the split and moves exactly the same bytes.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let dst_off = ((origin_y as usize) + y)
                .saturating_mul(bpr)
                .saturating_add(x_off as usize);
            if dst_off + rb > frame.len() {
                return false;
            }
            frame[dst_off..dst_off + rb].copy_from_slice(&src[src_off..src_off + rb]);
            let lo = base_off.saturating_add(dst_off as u64);
            let hi = lo.saturating_add(rb as u64);
            match runs.last_mut() {
                Some(last) if last.1 == lo => last.1 = hi,
                _ => runs.push((lo, hi)),
            }
        }
        if !mapper::write_mapping_bytes_only(
            state,
            host,
            mapping_id,
            base_off,
            &frame,
            Some(&runs),
            &vouched,
        ) {
            return false;
        }
    } else {
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let moff = base_off
                .saturating_add(((origin_y as u64) + y as u64).saturating_mul(surface_bpr as u64))
                .saturating_add(x_off);
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                moff,
                &src[src_off..src_off + rb],
                &vouched,
            ) {
                return false;
            }
        }
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // Guest pages are authoritative after this write and no host-side copy
    // represents them, so an older cache entry must retire. The same two lines
    // `write_native_image` carries, for the same reason and against a sharper
    // consequence: unlike a native-texel write, this one leaves a *BGRA* entry
    // of exactly this geometry behind, which is the shape every reader of the
    // cache is looking for.
    //
    // # The defect these close
    //
    // This writer publishes nothing and used to stamp nothing, so a partial
    // store landed composite pixels in the guest's pages while
    // `host_surfaces[mapping_id]` kept the previous frame and
    // `mapping_guest_write_verdict` kept answering `Clean` — the device's own
    // writes go through a mapped host pointer and cannot set `DIRTY_MEMORY_VGA`
    // (`reims-vgpu-dirty.c`), so the hypervisor's witness cannot see them and
    // only the writer can. `draw::seed_from_published_surface` then served that
    // stale frame as the attachment's prior content, the pass composited onto
    // it, and its Store published the result back over the guest's pages: the
    // rect's pixels lost, and *held*, which is the exact class that door's
    // strict evidence standard exists to prevent.
    //
    // Retiring rather than repairing, because this writer holds rows in the
    // mapping's native layout and the cache holds a tight BGRA frame; building
    // the second from the first is the full-frame conversion the partial store
    // exists to avoid. A miss costs the next LOAD a guest read.
    crate::runtime::surface_cache::forget(state, mapping_id);
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

#[cfg(test)]
mod tests;
