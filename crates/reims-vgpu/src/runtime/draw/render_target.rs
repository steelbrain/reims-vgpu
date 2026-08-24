//! Which host or guest bytes a colour attachment's `texture_ref` names.
//!
//! Every render pass this device encodes begins by turning the guest's
//! `texture_ref` into somewhere to render: a host mapping id, or a linear guest
//! VA plus a row stride. [`lookup_render_target`] is the only implementation of
//! that question, and its three callers — the single-target request builder, the
//! MRT one, and the abandoned-chain writeback — all treat a refusal as the whole
//! pass being lost, because Metal will not form an encoder with a null colour
//! attachment.
//!
//! # The rungs, in the order the archive tries them
//!
//! 1. **Type-8 view → base.** The shared view resolver composes the exact
//!    mip/layer range into the final base namespace and carries the format
//!    override with it. A swizzled view is refused rather than resolved.
//! 2. **IOSurface.** Geometry comes from the live mapping, never from a
//!    sticky latch — the latch exists only for the window where the object-list
//!    entry is transiently missing, and preferring it has twice routed a
//!    dual-mapping composite onto one mapping.
//! 3. **Surface backing surface / IOSurface plane view `RefTextureHandle`.** The object-list index is
//!    the surface id; IOSurface plane view wraps surface backing and is what product colour targets
//!    actually bind.
//! 4. **Type-2/3 linear guest VA.** Wallpaper and background intermediates and
//!    UI intermediate render targets live here, so an IOSurface texture-only resolve drops
//!    those passes entirely.
//!
//! The order is live-type-driven at every step: the object list is re-read and
//! the current type decides the rung, because the guest recycles object refs and
//! a ref that was an IOSurface last frame can be a linear texture this one.
//!
//! # Why this is its own module
//!
//! It is 380 lines of one question with four answers, and it sat in the middle
//! of `runtime::draw`'s 4 700-line body between the ICB execute entry point and the
//! guest-page writers. Nothing in it is backend-specific — no `cfg` gate here,
//! for the same reason [`super::texture_view`] carries none — and both arms take
//! it on every draw.
//!
//! # Every refusal here is a healthy zero
//!
//! Measured so the next reader does not have to guess whether these paths are
//! exercised. One driven x86/Vulkan boot (Safari window drag, 2 541 posted
//! events, real motion from (320,180) to (678,124), ~38 Hz median present)
//! produced **zero** `rt_resolve` records — the fail channel's whole reason set
//! for the boot was five slugs, none of them this ladder's.
//!
//! A zero over an unstated amount of work is not a measurement, so: on the same
//! boot `mrt_draw_single` counted **179 123** single-attachment draws reaching
//! the Vulkan encode, every one of which is a colour attachment this ladder had
//! already resolved, and `rt_iosurface_plane_view_same` counted **23 951** attachments
//! that reached the *bottom* of the surface backing/5 rung. Both counters predate the
//! typed refusals and sit on the success side, which is what makes them usable
//! as a denominator here.
//!
//! So a green boot says the rungs still resolve and says nothing whatever about
//! [`RenderTargetCause`]'s arms. Those are held by tests, not by booting — an
//! `rt_resolve` line in the log is a real event worth reading rather than
//! background noise.

use super::*;
/// The colour render target's base format for a **surface backing** surface, or nothing.
///
/// On this arm `m.format_or_zero() == 0` is not "unset", it is a decoded refusal:
/// [`objects::apply_surface_backing`] is the only writer of it, and it stores 0 for
/// a multi-plane surface and for a single-plane one whose FourCC it does not
/// know, saying why — "stage/paint must not invent BGRA".
/// [`objects::iosurface_pixel_format_to_mtl`] repeats it twice more, and the
/// compute staging path honours it, declining with typed `multiplane` /
/// `fmt_unknown` reasons.
///
/// This resolve used to invent BGRA8 from it, so one surface was refused by the
/// compute path and rendered into as BGRA8 by this one. That is not a survivable
/// disagreement for a multi-plane surface: BGRA8 over a `'420f'` allocation
/// describes the wrong stride and the wrong bytes, and every downstream window is
/// built from what this returns.
///
/// It now declines. Refusing was held back because it drops a colour attachment
/// — a compositing layer going black — on a class nothing had counted, so the
/// class was counted first: `rt_base_fmt_invent` read **0 on two driven
/// x86/Vulkan boots** (Safari window drag plus the web-content probe). The arm is
/// unreached on this workload, so declining costs nothing measurable and stops
/// the device from silently rendering a format the guest did not declare. The
/// counter stays and the fail line stays, because "unreached on this workload" is
/// not "unreachable" — the first surface to take it will now be named and
/// refused rather than named and rendered wrong.
///
/// The **IOSurface texture** arm deliberately does not come through here. An IOSurface texture
/// mapping's format has other writers, so its 0 can mean "not latched yet" rather
/// than "refused", and BGRA8 is the display contract's stated default for that
/// case ([`crate::runtime::compute_exec`]'s `or_bgra8` writes the same rule down).
/// Those are different zeros and only this one is provably a refusal.
fn rt_surface_backing_base_format(format: u16, mapping_id: u32) -> Option<u16> {
    if format != 0 {
        return Some(format);
    }
    crate::runtime::drain::note_store_route("rt_base_fmt_declined");
    if crate::observe::first_sight("rt_base_fmt_declined", mapping_id as u64) {
        crate::observe::fail(format!(
            "rt_base_fmt_declined mapping={mapping_id} \
             (the mapping's format is the surface backing decoder's multi-plane / \
             unknown-FourCC refusal, so this surface is not a single-format \
             colour attachment and no format is invented for it)"
        ));
    }
    None
}

/// The format a surface backing colour attachment is **declared** in.
///
/// A IOSurface plane view object is a texture view over the surface allocation, and its format
/// is attachment state: the UNORM and sRGB spellings name identical stored bytes
/// and differ only in the fixed-function conversion the hardware applies on
/// render writes. The guest declared that view, so the view's format is the
/// contract and the base mapping's is what a surface bound without one falls
/// back to. A type-8 view is a further reinterpretation asked for on top, so it
/// outranks the IOSurface plane view record when both are present.
///
/// # Why this stopped answering the base mapping's format
///
/// It used to answer `base_fmt` for every IOSurface plane view view, on the ground that
/// honouring the view would fork the resident — the guest binds one surface
/// through both spellings, and a second spelling that missed would retire the
/// resident and recreate it empty, alternating two half-filled images frame to
/// frame. That was a real hazard and it is no longer one:
/// `translate::pixel::ResidentFormat` carries the allocation and the declaration
/// as one value, `ResidentTargetSlot::reusable_for` compares only its
/// `allocation()` (which folds `_SRGB` onto `_UNORM`), the image is created
/// `MUTABLE_FORMAT` in the allocation format, and the attachment gets its own
/// view in `declared()`. One allocation, a view per interpretation — which is
/// what Metal's contract describes. The type-2/3 linear GVA targets have been
/// attaching sRGB through that same machinery all along; only this resolve was
/// still throwing the qualifier away, so a surface the guest declared sRGB was
/// rendered into without the linear-to-sRGB encode on write.
///
/// # The geometry half is deliberately not honoured
///
/// A IOSurface plane view view is taken only where it agrees with the base *extent*, because
/// the resolve this feeds takes the base mapping's geometry. A view describing a
/// different grid is one this device is not honouring at all, and lifting its
/// format alone would attach a reinterpretation to a grid that is not its own —
/// the row-byte-equivalent quarter-width `RGBA32Uint` view over the desktop
/// target is exactly that shape. That population is
/// `rt_iosurface_plane_view_differs_geometry`; it has never been observed on any boot, and
/// it still resolves through the base.
///
/// A view format whose texel is a different *width* from the base's is not a
/// reinterpretation of one allocation at all, so [`effective_view_sample_format`]
/// refuses it and the base format stands.
fn rt_surface_backing_declared_format(
    base_fmt: u16,
    base_extent: (u32, u32),
    iosurface_plane_view: Option<objects::IOSurfacePlaneViewDescriptor>,
    view_fmt_override: Option<u16>,
) -> u16 {
    let (base_w, base_h) = base_extent;
    let iosurface_plane_view_declared = iosurface_plane_view
        .filter(|view| view.width == base_w && view.height == base_h)
        .map(|view| view.pixel_format)
        .filter(|&fmt| fmt != 0);
    effective_view_sample_format(
        base_fmt,
        view_fmt_override.or(iosurface_plane_view_declared),
    )
    .unwrap_or(base_fmt)
}

/// Report a IOSurface plane view colour attachment whose view record disagrees with the base
/// mapping it is resolved through.
///
/// This resolve reads only `surfaceID@0` out of a IOSurface plane view descriptor and takes
/// geometry and format from the mapping. [`objects::decode_iosurface_plane_view`]'s
/// own contract forbids that — "callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable" — and the
/// live case it names is real: the BGRA8 desktop target is also exposed as a
/// row-byte-equivalent quarter-width RGBA32Uint view. Every other IOSurface plane view
/// consumer binds the view's own geometry.
///
/// It is harmless exactly while view == base, so the question is how often that
/// holds for a *render target* specifically, which nothing has measured.
/// `rt_iosurface_plane_view_differs` against `rt_iosurface_plane_view_same` answers it. Reported
/// rather than repaired: taking the view's geometry here changes what every
/// IOSurface plane view colour attachment renders into, and that is not a change to make on an
/// unmeasured population.
///
/// **Read on two driven x86/Vulkan boots: `same` 20 273 and 24 360, `differs`
/// 0, `undecoded` 0.** So on this workload every IOSurface plane view colour attachment's view
/// agrees with the base mapping in width, height and format, and resolving
/// through the base loses nothing. The reinterpretation view the contract names
/// is real traffic elsewhere — the compute staging path sees it — but it is not
/// bound as a render target here.
///
/// That is a reason not to change the resolve, and not a reason to stop asking.
/// `differs` is a healthy zero: the first non-zero line names a surface being
/// rendered at the wrong geometry, which no other counter in this path could
/// report.
///
/// # It stopped being zero, and the format half is now honoured
///
/// A driven macos-11 boot on the Intel iGPU host, 2026-08-12, read `same` 1 271
/// and `differs` **1**: `sid=56 view=1024x768 fmt=0x51 base=1024x768 fmt=0x50`.
/// Geometry agrees; the *format* does not, and 0x50/0x51 are `BGRA8Unorm` and
/// `BGRA8Unorm_sRGB`. So this device rendered into a target the guest declared
/// as sRGB using a UNORM attachment, and the hardware's linear-to-sRGB encode on
/// write never happened — the stored pixels were the shader's linear values,
/// displayed as though they were sRGB.
///
/// A driven macos-13 boot, 2026-08-16, reproduced it twice at icon size:
/// `sid=64` and `sid=26`, both `view=300x300 fmt=0x51 base=300x300 fmt=0x50`,
/// with `same` 37 657, `..._geometry` 0 and `..._sid_both_ways` 0. The same
/// boot's guest is the one bug-03 is reported against, and a macos-26 boot on
/// the same tree declares no sRGB format at all — its window server composites
/// in extended-range half-float — which is the report's own "renders fine in
/// macOS 15".
///
/// **The resolve now honours the view's format**, so the format half of this
/// counter no longer measures a loss; it measures how often the two spellings
/// are used, which is worth keeping. The forked-resident hazard the repair was
/// waiting on is gone: `ResidentFormat` carries the allocation and declaration
/// as one value and `ResidentTargetSlot::reusable_for` compares only the
/// allocation, so both spellings reach one image with a view each.
///
/// # It is not what `bugs/bug-03` is, and that took a pixel comparison to say
///
/// The paragraph above reads like an identification and it is not one. Two
/// driven macos-13 boots from one snapshot, one on the tree carrying both halves
/// of the sRGB round trip and one on the tree before either, with System
/// Settings opened and photographed in **both appearances**:
///
/// ```text
///                    identical   >4 levels   max channel delta
/// light  icons        99.6 %       0.000 %          1
/// dark   icons        99.1 %       0.000 %          1
/// ```
///
/// A maximum channel difference of one level over the sidebar icons is
/// dithering, not a colour space. **The two commits change no pixel bug-03 is
/// about**, in either appearance, and the report still reproduces on the later
/// tree — the dark-mode icons are the reporter's washed-out pale squares on
/// both builds.
///
/// Neither is this counter in play on the boot that reproduces it: the dark
/// macos-13 boot reads `same` only, with `differs` absent entirely, and
/// `runtime::census::srgb_census` emits nothing across its six sites. So the
/// extra encode `bugs/bug-03` measures enters somewhere neither this rail nor
/// that census watches, and the IOSurface plane view view divergence — real, and worth
/// honouring on its own terms — is not its road.
///
/// **`..._geometry` is still a live healthy zero and is still a loss.** It has
/// never been observed on any boot, it is the population this doc's warning was
/// always really about — the quarter-width `RGBA32Uint` view over the desktop
/// target — and a view describing a different grid still resolves through the
/// base. `..._sid_both_ways` stays too: it no longer gates anything, but a
/// surface bound both ways is what exercises the view swap, so a non-zero there
/// is the reading that says the swap is being taken rather than merely
/// available.
fn note_rt_iosurface_plane_view(
    view: Option<objects::IOSurfacePlaneViewDescriptor>,
    surface_id: u32,
    base: (u32, u32, u16),
) {
    let Some(view) = view else {
        crate::runtime::drain::note_store_route("rt_iosurface_plane_view_undecoded");
        return;
    };
    let (base_w, base_h, base_fmt) = base;
    if view.width == base_w && view.height == base_h && view.pixel_format == base_fmt {
        crate::runtime::drain::note_store_route("rt_iosurface_plane_view_same");
        if differed_before(surface_id) {
            // The reading that decides whether the format repair above is safe.
            // A surface resolved both ways is one this device would key two
            // residents on if the view were honoured, and a frame alternating
            // between two images is worse than a frame in the wrong colour
            // space. A boot reporting `differs_format_only` with this at zero is
            // the one that licenses the repair.
            crate::runtime::drain::note_store_route("rt_iosurface_plane_view_sid_both_ways");
        }
        return;
    }
    crate::runtime::drain::note_store_route("rt_iosurface_plane_view_differs");
    // Which half diverged decides both the counter and what the fail line says
    // happened, so it is asked once. The two halves no longer have the same
    // answer — the format is honoured and the geometry is not — and a single
    // sentence covering both was accurate only while neither was.
    let (route, disposition) = if view.width == base_w && view.height == base_h {
        (
            "rt_iosurface_plane_view_differs_format_only",
            "the colour attachment is resolved in the view's format",
        )
    } else {
        (
            "rt_iosurface_plane_view_differs_geometry",
            "the colour attachment is resolved with the base mapping's geometry, not the view's",
        )
    };
    crate::runtime::drain::note_store_route(route);
    note_differed(surface_id);
    if crate::observe::first_sight("rt_iosurface_plane_view_differs", surface_id as u64) {
        crate::observe::fail(format!(
            "rt_iosurface_plane_view_differs sid={surface_id} view={}x{} fmt={:#x} plane={} \
             base={base_w}x{base_h} fmt={base_fmt:#x} ({disposition})",
            view.width, view.height, view.pixel_format, view.plane_index
        ));
    }
}

/// Surface ids this ladder has resolved a render target through a *differing*
/// IOSurface plane view view for.
///
/// Bounded, and the bound is the whole design: this exists to answer whether one
/// surface is bound both ways in one boot, and the population it watches was one
/// member on the boot that made it necessary. Past [`DIFFERED_MAX`] it stops
/// admitting rather than growing or evicting — an evicting set would answer
/// "not seen before" for a surface it had forgotten, which is the direction that
/// reports the repair as safe when it is not. `rt_iosurface_plane_view_differ_set_full`
/// says the bound bit, and a boot that reports it has not answered the question.
const DIFFERED_MAX: usize = 64;

static DIFFERED: std::sync::Mutex<std::collections::BTreeSet<u32>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

fn note_differed(surface_id: u32) {
    let mut set = DIFFERED.lock().unwrap_or_else(|e| e.into_inner());
    if set.len() >= DIFFERED_MAX && !set.contains(&surface_id) {
        crate::runtime::drain::note_store_route("rt_iosurface_plane_view_differ_set_full");
        return;
    }
    set.insert(surface_id);
}

fn differed_before(surface_id: u32) -> bool {
    let set = DIFFERED.lock().unwrap_or_else(|e| e.into_inner());
    !set.is_empty() && set.contains(&surface_id)
}

/// Where a colour attachment's `texture_ref` actually resolved to.
///
/// This was six loose positional values — `(u32, u64, u32, u32, u32, u16)` —
/// and three of them are `u32` in a row, so every call site accepted the
/// permutation that swaps width, height and row stride. The two sites that
/// destructure it do so in different orders from the one that builds a
/// [`ColorRtRequest`] out of it, which is where such a swap would have gone
/// unnoticed: all three orders type-check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResolvedRenderTarget {
    pub(super) storage: super::ColorTargetStorage,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: u16,
    /// Attachment samples requested for this encode. Linear texture resource
    /// dimensions do not retain the creation descriptor's sample count, so the
    /// Vulkan encode replaces this provisional single-sample value with the
    /// bound pipeline's raster sample count before constructing an image.
    pub(super) sample_count: u32,
}

/// Why a colour attachment's `texture_ref` could not be turned into somewhere
/// to render.
///
/// The ladder had ~30 exits and two of them said anything. The other 28 were a
/// bare `return None` or a `?`, and all three callers can only report that the
/// resolve failed: the writeback says `reason=render_target_unresolved`, and the
/// MRT builder re-derives a guess with [`super::sample_miss_detail`], a
/// *sampled*-texture diagnostic pointed at a render-target question. So a lost
/// pass named the caller, never the check — and the checks are not
/// interchangeable. "The guest has not populated this task's object list yet" and
/// "the level's rows run past the allocation the guest declared" are the same
/// black attachment on screen and completely different bugs.
///
/// [`RenderTargetCause`] carries which check, `base_ref` carries what it was
/// looking at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RenderTargetRefusal {
    /// The ref the ladder was working on: `texture_ref` itself, or the base a
    /// type-8 view resolved to.
    ///
    /// Reported next to the attachment's own ref rather than instead of it,
    /// because they differ exactly when a view is in play — and a refusal
    /// naming only one of the two cannot say whether the view chain was even
    /// followed.
    base_ref: u32,
    cause: RenderTargetCause,
}

/// The check that refused, and the values it refused on.
///
/// One variant per check, never shared, per [`crate::observe::Decline`]'s
/// contract. Grouped by rung in the order [`lookup_render_target`] tries them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RenderTargetCause {
    /// The opcode-9 buffer-backed texture declaration or its retained buffer
    /// relation could not name one complete linear image plane.
    BufferTexturePlacement {
        reason: objects::BufferTexturePlacementRefusal,
    },
    /// Buffer-backed render targets expose only their one declared level,
    /// slice, and depth plane.
    BufferTextureSubresource {
        level: u32,
        slice: u32,
        depth_plane: u32,
    },
    /// The buffer-backed texture's row stride cannot cross the core target
    /// boundary, whose contract carries strides as `u32`.
    BufferTextureStride {
        row_stride: u64,
    },
    /// A type-8 view whose swizzle is not the identity. The archive's
    /// `resolve_texture` requires `!has_swizzle` for a linear resolve, so the
    /// channel order this view asks for cannot be honoured by rendering into
    /// the base.
    ViewSwizzled,
    /// The type-8 view chain ended at ref 0 — the view names no base texture.
    ViewBaseUnbound,
    /// The pass selected a level or slice outside the view's declared range.
    ViewSubresourceOutOfRange {
        level: u32,
        slice: u32,
    },
    ArraySliceOutOfRange {
        slice: u64,
        layers: u32,
    },
    DepthPlaneOutOfRange {
        depth_plane: u32,
        depth: u32,
    },
    SubresourceShapeUnsupported {
        texture_type: u16,
        slice: u64,
        depth_plane: u32,
    },
    SubresourceOffsetOverflow,
    LinearPackingMismatch {
        texture_type: u16,
        declared_slices: u32,
        dimension_slices: u32,
        bytes_per_slice: u64,
        cube_faces: bool,
    },

    /// The view's own level base plus the level the pass named does not fit a
    /// `u32`. A healthy zero: both are mip indices and a real texture has at
    /// most [`TEXTURE_MAX_MIP_LEVELS`](crate::runtime::decode::resource::TEXTURE_MAX_MIP_LEVELS)
    /// of them. It exists so the sum is never a wrapped small level, which would
    /// render into the wrong plane of a real allocation.
    LevelOverflow {
        view_level: u32,
        attachment_level: u32,
    },
    /// A mip>0 view of an IOSurface. IOSurface texture sample windows carry planes, not
    /// mip levels, so this geometry has no contract behind it.
    IOSurfaceMipView {
        level: u32,
    },
    /// Neither the live descriptor nor the latch produced a mapping id.
    IOSurfaceUnresolved,
    /// The mapping id resolved and names no live mapping.
    IOSurfaceNoMapping {
        mapping_id: u32,
    },
    /// The mapping has no latched geometry, or a zero dimension.
    IOSurfaceGeometry {
        mapping_id: u32,
        has_geom: bool,
        width: u32,
        height: u32,
    },
    /// The mapping's format is not one Metal can render into.
    IOSurfaceFormat {
        mapping_id: u32,
        fmt: u16,
    },

    /// The IOSurface plane view entry's descriptor bytes could not be read.
    IOSurfacePlaneViewDescRead,
    /// The bytes were read and are not a IOSurface plane view header.
    IOSurfacePlaneViewDescDecode {
        len: usize,
    },
    /// The IOSurface plane view header names surface 0, so it wraps nothing.
    IOSurfacePlaneViewSurfaceZero,

    /// A mip>0 view of a surface backing surface. Colour RT materialization is level 0
    /// only.
    SurfaceBackingMipView {
        surface_id: u32,
        level: u32,
    },
    /// The surface's backing could not be resolved.
    SurfaceBackingUnresolved {
        surface_id: u32,
        live_type: Option<ObjectKind>,
    },
    /// The surface resolved and left no mapping under its id.
    SurfaceBackingNoMapping {
        surface_id: u32,
    },
    /// The surface's mapping has no geometry, a zero dimension, or no pages.
    SurfaceBackingGeometry {
        surface_id: u32,
        has_geom: bool,
        width: u32,
        height: u32,
        pages: usize,
    },
    /// The surface backing decoder refused this surface a format — multi-plane, or a
    /// FourCC it has no contract for. [`rt_surface_backing_base_format`] carries the
    /// argument for why no format is invented here; this is the ladder
    /// recording that it stopped there.
    SurfaceBackingBaseFormat {
        surface_id: u32,
        raw_fmt: u16,
    },
    /// The surface's format is not one Metal can render into.
    SurfaceBackingFormat {
        surface_id: u32,
        fmt: u16,
    },

    /// The guest has put nothing under this ref. Expected while a task's object
    /// list is still being populated, which is why it is reported and not
    /// treated as a decode error.
    NoListEntry,
    /// Something is under the ref and it is not a texture.
    WrongType {
        object_type: ObjectKind,
    },
    /// The texture entry's descriptor bytes could not be read.
    LinearDescRead,
    /// The bytes were read and are not a texture descriptor. Carries the
    /// resource decoder's own slug, so the specific malformation survives.
    LinearDescDecode {
        decode: &'static str,
    },
    /// The descriptor decoded and left one of format, extent or row stride
    /// undeclared.
    LinearDescIncomplete {
        format: u16,
        width: u32,
        height: u32,
        row_stride: u32,
    },
    /// The declared format is not one Metal can render into.
    LinearFormat {
        fmt: u16,
    },
    /// The view's mip level has no layout in the declared allocation.
    LinearLevelGva {
        level: u32,
    },
    /// The level's row stride does not fit a `u32`, so the bind is
    /// unrepresentable.
    LinearLevelStride {
        row_stride: u64,
    },
    /// `row_stride * height` overflowed for the level.
    LinearLevelSpan {
        row_stride: u64,
        height: u32,
    },
    /// The level's rows end past the allocation the guest declared. Rendering
    /// it would write over whatever guest memory follows.
    LinearLevelPastAllocation {
        offset: u64,
        span: u64,
        allocation_size: u64,
    },
    /// The descriptor names no backing: a zero allocation, a zero handle, or a
    /// page shift outside the geometry this device supports.
    LinearBackingGva {
        allocation_size: u64,
        handle: u32,
    },
    /// `row_stride * height` overflowed for the base level.
    LinearSpan {
        row_stride: u32,
        height: u32,
    },
    /// The declared width and format have no tight row length — a zero width,
    /// or a format with no bytes-per-texel.
    LinearTightRow {
        width: u32,
        fmt: u16,
    },
    /// The rows end past the allocation under both the padded and the
    /// exclusive-last-row measure.
    LinearPastAllocation {
        span: u64,
        alloc: u64,
        alt: u64,
    },

    /// The resolved target's width and format have no tight row length. Reached
    /// for a mip level, whose width is the level's rather than the declared one.
    RowTight {
        width: u32,
        fmt: u16,
    },
    /// The target's row stride is narrower than one tight row of its own
    /// format, so consecutive rows would overlap.
    RowStride {
        bpr: u32,
        tight: u32,
    },
    /// The resolved target has a zero dimension.
    ZeroExtent {
        width: u32,
        height: u32,
    },
}

impl RenderTargetCause {
    /// Name the ref this check refused on.
    fn at(self, base_ref: u32) -> RenderTargetRefusal {
        RenderTargetRefusal {
            base_ref,
            cause: self,
        }
    }
}

impl crate::observe::Decline for RenderTargetRefusal {
    fn slug(&self) -> &'static str {
        use RenderTargetCause as C;
        match self.cause {
            C::BufferTexturePlacement { reason } => match reason {
                objects::BufferTexturePlacementRefusal::Decode => "rt_buffer_tex_desc_decode",
                objects::BufferTexturePlacementRefusal::SemanticKind => {
                    "rt_buffer_tex_semantic_kind"
                }
                objects::BufferTexturePlacementRefusal::InvalidShape => "rt_buffer_tex_shape",
                objects::BufferTexturePlacementRefusal::MissingFormat => "rt_buffer_tex_no_fmt",
                objects::BufferTexturePlacementRefusal::UnsupportedFormat => {
                    "rt_buffer_tex_fmt_bytes"
                }
                objects::BufferTexturePlacementRefusal::Buffer(_) => "rt_buffer_tex_no_backing",
                objects::BufferTexturePlacementRefusal::RowStrideTooSmall => {
                    "rt_buffer_tex_bpr_short"
                }
                objects::BufferTexturePlacementRefusal::ReachOverflow => {
                    "rt_buffer_tex_reach_overflow"
                }
                objects::BufferTexturePlacementRefusal::PastAllocation => "rt_buffer_tex_span_oob",
                objects::BufferTexturePlacementRefusal::AddressOverflow => {
                    "rt_buffer_tex_offset_overflow"
                }
            },
            C::BufferTextureSubresource { .. } => "rt_buffer_tex_subresource",
            C::BufferTextureStride { .. } => "rt_buffer_tex_stride",
            C::ViewSwizzled => "rt_view_swizzled",
            C::ViewBaseUnbound => "rt_view_base_unbound",
            C::ViewSubresourceOutOfRange { .. } => "rt_view_subresource_out_of_range",
            C::ArraySliceOutOfRange { .. } => "rt_array_slice_out_of_range",
            C::DepthPlaneOutOfRange { .. } => "rt_depth_plane_out_of_range",
            C::SubresourceShapeUnsupported { .. } => "rt_subresource_shape_unsupported",
            C::SubresourceOffsetOverflow => "rt_subresource_offset_overflow",
            C::LinearPackingMismatch { .. } => "rt_linear_packing_mismatch",
            C::LevelOverflow { .. } => "rt_level_overflow",
            C::IOSurfaceMipView { .. } => "rt_iosurface_texture_mip_view",
            C::IOSurfaceUnresolved => "rt_iosurface_texture_unresolved",
            C::IOSurfaceNoMapping { .. } => "rt_iosurface_texture_no_mapping",
            C::IOSurfaceGeometry { .. } => "rt_iosurface_texture_geometry",
            C::IOSurfaceFormat { .. } => "rt_iosurface_texture_format",
            C::IOSurfacePlaneViewDescRead => {
                crate::observe::ladder_slug!("rt_iosurface_plane_view", desc_read)
            }
            C::IOSurfacePlaneViewDescDecode { .. } => {
                crate::observe::ladder_slug!("rt_iosurface_plane_view", desc_decode)
            }
            C::IOSurfacePlaneViewSurfaceZero => "rt_iosurface_plane_view_surface_zero",
            C::SurfaceBackingMipView { .. } => "rt_surface_backing_mip_view",
            C::SurfaceBackingUnresolved { .. } => "rt_surface_backing_unresolved",
            C::SurfaceBackingNoMapping { .. } => "rt_surface_backing_no_mapping",
            C::SurfaceBackingGeometry { .. } => "rt_surface_backing_geometry",
            C::SurfaceBackingBaseFormat { .. } => "rt_surface_backing_base_format",
            C::SurfaceBackingFormat { .. } => "rt_surface_backing_format",
            C::NoListEntry => crate::observe::ladder_slug!("rt", no_list_entry),
            C::WrongType { .. } => crate::observe::ladder_slug!("rt", wrong_type),
            C::LinearDescRead => crate::observe::ladder_slug!("rt_linear", desc_read),
            C::LinearDescDecode { .. } => crate::observe::ladder_slug!("rt_linear", desc_decode),
            C::LinearDescIncomplete { .. } => "rt_linear_desc_incomplete",
            C::LinearFormat { .. } => "rt_linear_format",
            C::LinearLevelGva { .. } => "rt_linear_level_gva",
            C::LinearLevelStride { .. } => "rt_linear_level_stride",
            C::LinearLevelSpan { .. } => "rt_linear_level_span",
            C::LinearLevelPastAllocation { .. } => "rt_linear_level_past_alloc",
            C::LinearBackingGva { .. } => "rt_linear_backing_gva",
            C::LinearSpan { .. } => "rt_linear_span",
            C::LinearTightRow { .. } => "rt_linear_tight_row",
            C::LinearPastAllocation { .. } => "rt_linear_past_alloc",
            C::RowTight { .. } => "rt_row_tight",
            C::RowStride { .. } => "rt_row_stride",
            C::ZeroExtent { .. } => "rt_zero_extent",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        use RenderTargetCause as C;
        let mut v = vec![("base", self.base_ref.to_string())];
        match self.cause {
            C::BufferTexturePlacement { reason } => {
                v.push(("reason", format!("{reason:?}")));
            }
            C::BufferTextureSubresource {
                level,
                slice,
                depth_plane,
            } => {
                v.push(("level", level.to_string()));
                v.push(("slice", slice.to_string()));
                v.push(("depth_plane", depth_plane.to_string()));
            }
            C::BufferTextureStride { row_stride } => {
                v.push(("row_stride", row_stride.to_string()));
            }
            C::ViewSubresourceOutOfRange { level, slice } => {
                v.push(("level", level.to_string()));
                v.push(("slice", slice.to_string()));
            }
            C::ArraySliceOutOfRange { slice, layers } => {
                v.push(("slice", slice.to_string()));
                v.push(("layers", layers.to_string()));
            }
            C::DepthPlaneOutOfRange { depth_plane, depth } => {
                v.push(("depth_plane", depth_plane.to_string()));
                v.push(("depth", depth.to_string()));
            }
            C::SubresourceShapeUnsupported {
                texture_type,
                slice,
                depth_plane,
            } => {
                v.push(("texture_type", texture_type.to_string()));
                v.push(("slice", slice.to_string()));
                v.push(("depth_plane", depth_plane.to_string()));
            }
            C::LinearPackingMismatch {
                texture_type,
                declared_slices,
                dimension_slices,
                bytes_per_slice,
                cube_faces,
            } => {
                v.push(("texture_type", texture_type.to_string()));
                v.push(("declared_slices", declared_slices.to_string()));
                v.push(("dimension_slices", dimension_slices.to_string()));
                v.push(("bytes_per_slice", bytes_per_slice.to_string()));
                v.push(("cube_faces", u8::from(cube_faces).to_string()));
            }
            C::IOSurfaceMipView { level } | C::LinearLevelGva { level } => {
                v.push(("level", level.to_string()))
            }
            C::LevelOverflow {
                view_level,
                attachment_level,
            } => {
                v.push(("view_level", view_level.to_string()));
                v.push(("attachment_level", attachment_level.to_string()));
            }
            C::IOSurfaceNoMapping { mapping_id } => v.push(("mid", mapping_id.to_string())),
            C::IOSurfaceGeometry {
                mapping_id,
                has_geom,
                width,
                height,
            } => {
                v.push(("mid", mapping_id.to_string()));
                v.push(("has_geom", has_geom.to_string()));
                v.push(("dims", format!("{width}x{height}")));
            }
            C::IOSurfaceFormat { mapping_id, fmt } => {
                v.push(("mid", mapping_id.to_string()));
                v.push(("fmt", format!("{fmt:#x}")));
            }
            C::IOSurfacePlaneViewDescDecode { len } => v.push(("desc_len", len.to_string())),
            C::SurfaceBackingMipView { surface_id, level } => {
                v.push(("sid", surface_id.to_string()));
                v.push(("level", level.to_string()));
            }
            C::SurfaceBackingUnresolved {
                surface_id,
                live_type,
            } => {
                v.push(("sid", surface_id.to_string()));
                v.push((
                    "live_type",
                    live_type.map_or_else(|| "none".to_string(), |t| t.to_string()),
                ));
            }
            C::SurfaceBackingNoMapping { surface_id } => v.push(("sid", surface_id.to_string())),
            C::SurfaceBackingGeometry {
                surface_id,
                has_geom,
                width,
                height,
                pages,
            } => {
                v.push(("sid", surface_id.to_string()));
                v.push(("has_geom", has_geom.to_string()));
                v.push(("dims", format!("{width}x{height}")));
                v.push(("pages", pages.to_string()));
            }
            C::SurfaceBackingBaseFormat {
                surface_id,
                raw_fmt,
            } => {
                v.push(("sid", surface_id.to_string()));
                v.push(("raw_fmt", format!("{raw_fmt:#x}")));
            }
            C::SurfaceBackingFormat { surface_id, fmt } => {
                v.push(("sid", surface_id.to_string()));
                v.push(("fmt", format!("{fmt:#x}")));
            }
            C::WrongType { object_type } => v.push(("object_type", object_type.to_string())),
            C::LinearDescDecode { decode } => v.push(("decode", decode.to_string())),
            C::LinearDescIncomplete {
                format,
                width,
                height,
                row_stride,
            } => {
                v.push(("fmt", format!("{format:#x}")));
                v.push(("dims", format!("{width}x{height}")));
                v.push(("bpr", row_stride.to_string()));
            }
            C::LinearFormat { fmt } => v.push(("fmt", format!("{fmt:#x}"))),
            C::LinearLevelStride { row_stride } => v.push(("bpr", row_stride.to_string())),
            C::LinearLevelSpan { row_stride, height } => {
                v.push(("bpr", row_stride.to_string()));
                v.push(("h", height.to_string()));
            }
            C::LinearLevelPastAllocation {
                offset,
                span,
                allocation_size,
            } => {
                v.push(("off", offset.to_string()));
                v.push(("span", span.to_string()));
                v.push(("alloc", allocation_size.to_string()));
            }
            C::LinearBackingGva {
                allocation_size,
                handle,
            } => {
                v.push(("alloc", allocation_size.to_string()));
                v.push(("handle", format!("{handle:#x}")));
            }
            C::LinearSpan { row_stride, height } => {
                v.push(("bpr", row_stride.to_string()));
                v.push(("h", height.to_string()));
            }
            C::LinearTightRow { width, fmt } | C::RowTight { width, fmt } => {
                v.push(("w", width.to_string()));
                v.push(("fmt", format!("{fmt:#x}")));
            }
            C::LinearPastAllocation { span, alloc, alt } => {
                v.push(("span", span.to_string()));
                v.push(("alloc", alloc.to_string()));
                v.push(("alt", alt.to_string()));
            }
            C::RowStride { bpr, tight } => {
                v.push(("bpr", bpr.to_string()));
                v.push(("tight", tight.to_string()));
            }
            C::ZeroExtent { width, height } => v.push(("dims", format!("{width}x{height}"))),
            C::ViewSwizzled
            | C::ViewBaseUnbound
            | C::SubresourceOffsetOverflow
            | C::IOSurfaceUnresolved
            | C::IOSurfacePlaneViewDescRead
            | C::IOSurfacePlaneViewSurfaceZero
            | C::NoListEntry
            | C::LinearDescRead => {}
        }
        v
    }
}

/// Archive `apple_pv_gpu_lookup_render_target`: IOSurface texture first, else type-2/3 GVA.
///
/// Wallpaper/background intermediates are type-2/3 guest-VA; IOSurface texture-only resolve
/// drops those passes (black wallpaper). Color RT formats are the Metal color-
/// renderable set admitted by [`pixel_format::render_target_bpp`] (RGBA8 family,
/// BGRA8 family, RGBA16Float) — bring-up only listed compositor BGRA8/0x73.
///
/// Type-8 texture views (archive `resource_resolve_texture` view chain): resolve
/// to the base texture. Swizzled views are rejected as RTs (archive
/// `resolve_texture` requires `!has_swizzle`). Level 0 only for color RT
/// materialization (mip RT not supported). Without this, UI passes that bind a
/// type-8 view as color attachment fail MRT (`mrt_request fail slots=[211]`) and
/// drop entire draws (blank App Store sidebar / missing chrome labels).
///
/// # An unbound ref is not a refusal
///
/// `texture_ref == 0` is the guest saying no attachment is bound in this slot,
/// which `AGENTS.md` names as the canonical thing that must stay quiet. It is
/// handled here rather than in [`resolve_render_target`] so that everything the
/// resolver can return is a genuine loss, and every one of them is reported.
pub(super) fn lookup_render_target<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &M,
    task_id: u32,
    att: crate::runtime::decode::render::ColorAttachment,
) -> Option<ResolvedRenderTarget> {
    let texture_ref = att.texture_ref;
    if texture_ref == 0 {
        return None;
    }
    match resolve_render_target(state, host, task_id, att) {
        Ok(rt) => Some(rt),
        Err(refusal) => {
            // Per attachment, per check: the guest re-issues the same pass every
            // frame, so an undeduped line here is a flood at draw rate. The two
            // ad-hoc lines this replaced had no latch at all.
            crate::observe::Emit::decline("rt_resolve", &refusal)
                .field("task", task_id)
                .field("ref", texture_ref)
                .fail_once(u64::from(task_id) << 32 | u64::from(texture_ref));
            None
        }
    }
}

/// Byte displacement of one colour-attachment slice/depth plane inside a
/// decoded mip record.
///
/// The texture declaration decides whether the third coordinate is an array
/// slice or a depth plane. Keeping this projection beside view-range
/// resolution prevents the render path from silently treating both as zero.
fn linear_attachment_subresource_offset(
    storage_type: u16,
    physical_slices: u32,
    bytes_per_slice: u64,
    level: u32,
    layout: Option<&crate::runtime::decode::resource::TextureLevelLayout>,
    slice: u64,
    depth_plane: u32,
) -> Result<u64, RenderTargetCause> {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_1D, TEXTURE_VIEW_MTL_TYPE_1D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE,
        TEXTURE_VIEW_MTL_TYPE_3D, TEXTURE_VIEW_MTL_TYPE_CUBE, TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY,
    };
    use RenderTargetCause as C;
    match storage_type {
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
            if depth_plane != 0 {
                return Err(C::SubresourceShapeUnsupported {
                    texture_type: storage_type,
                    slice,
                    depth_plane,
                });
            }
            let layers = physical_slices;
            if slice >= u64::from(layers) {
                return Err(C::ArraySliceOutOfRange { slice, layers });
            }
            bytes_per_slice
                .checked_mul(slice)
                .ok_or(C::SubresourceOffsetOverflow)
        }
        TEXTURE_VIEW_MTL_TYPE_CUBE | TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY => {
            if depth_plane != 0 {
                return Err(C::SubresourceShapeUnsupported {
                    texture_type: storage_type,
                    slice,
                    depth_plane,
                });
            }
            let layers = physical_slices;
            if slice >= u64::from(layers) {
                return Err(C::ArraySliceOutOfRange { slice, layers });
            }
            bytes_per_slice
                .checked_mul(slice)
                .ok_or(C::SubresourceOffsetOverflow)
        }
        TEXTURE_VIEW_MTL_TYPE_3D => {
            if slice != 0 {
                return Err(C::SubresourceShapeUnsupported {
                    texture_type: storage_type,
                    slice,
                    depth_plane,
                });
            }
            let layout = layout.ok_or(C::LinearLevelGva { level })?;
            let depth = layout.planes();
            if depth_plane >= depth {
                return Err(C::DepthPlaneOutOfRange { depth_plane, depth });
            }
            layout
                .row_stride
                .checked_mul(u64::from(layout.height))
                .and_then(|pitch| pitch.checked_mul(u64::from(depth_plane)))
                .ok_or(C::SubresourceOffsetOverflow)
        }
        TEXTURE_VIEW_MTL_TYPE_1D
        | TEXTURE_VIEW_MTL_TYPE_2D
        | TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE => {
            if slice != 0 || depth_plane != 0 {
                return Err(C::SubresourceShapeUnsupported {
                    texture_type: storage_type,
                    slice,
                    depth_plane,
                });
            }
            Ok(0)
        }
        _ => Err(C::SubresourceShapeUnsupported {
            texture_type: storage_type,
            slice,
            depth_plane,
        }),
    }
}

/// The ladder itself: four rungs, and a named refusal at every exit.
///
/// Returning `Result` rather than `Option` is what holds that open. A bare
/// `return None` does not compile here, and neither does a `?` on an `Option`
/// without an `ok_or` naming the check — so the property is the type, not a test
/// that has to be kept in step with the body. That matters for this function in
/// particular: it is 240 lines with ~30 exits, it grew every one of them one
/// bug at a time, and a source-scanning gate over it would have to understand
/// four rungs of control flow to say anything.
fn resolve_render_target<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &M,
    task_id: u32,
    att: crate::runtime::decode::render::ColorAttachment,
) -> Result<ResolvedRenderTarget, RenderTargetRefusal> {
    use RenderTargetCause as C;
    let texture_ref = att.texture_ref;
    if objects::lookup_list_entry(state, host, task_id, texture_ref)
        .is_some_and(|entry| entry.kind == ObjectKind::TextureView)
    {
        let resource = objects::resolve_resource(state, host, task_id, texture_ref)
            .map_err(|_| C::LinearDescRead.at(texture_ref))?;
        let buffer_texture =
            objects::resolve_buffer_texture_placement_from_resource(state, &resource)
                .map_err(|reason| C::BufferTexturePlacement { reason }.at(texture_ref))?;
        if let Some(level) = buffer_texture {
            if att.level != 0 || att.slice != 0 || att.depth_plane != 0 {
                return Err(C::BufferTextureSubresource {
                    level: att.level,
                    slice: att.slice,
                    depth_plane: att.depth_plane,
                }
                .at(texture_ref));
            }
            if pixel_format::render_target_bpp(level.pixel_format).is_none() {
                return Err(C::LinearFormat {
                    fmt: level.pixel_format,
                }
                .at(texture_ref));
            }
            let row_stride = u32::try_from(level.row_stride).map_err(|_| {
                C::BufferTextureStride {
                    row_stride: level.row_stride,
                }
                .at(texture_ref)
            })?;
            let storage = super::LinearColorTarget::new(
                level.base_gva,
                level.alloc_size,
                level.level_offset,
                row_stride,
            )
            .map(super::ColorTargetStorage::Linear)
            .ok_or(
                C::BufferTexturePlacement {
                    reason: objects::BufferTexturePlacementRefusal::AddressOverflow,
                }
                .at(texture_ref),
            )?;
            return Ok(ResolvedRenderTarget {
                storage,
                width: level.width,
                height: level.height,
                format: level.pixel_format,
                sample_count: 1,
            });
        }
    }
    // Type-8 view → base (archive resource_resolve_texture view chain).
    let (resolved_ref, view_fmt_override, level, slice) =
        if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
            // Archive resolve_texture rejects swizzled views for linear resolve.
            if let Some(plan) = view.swizzle.as_ref() {
                if !pixel_format::swizzle_is_identity(plan) {
                    return Err(C::ViewSwizzled.at(view.base_texture_ref));
                }
            }
            let (level, slice) = view
                .select(u64::from(att.level), u64::from(att.slice))
                .ok_or(
                    C::ViewSubresourceOutOfRange {
                        level: att.level,
                        slice: att.slice,
                    }
                    .at(view.base_texture_ref),
                )?;
            (view.base_texture_ref, view.pixel_format, level, slice)
        } else {
            (
                texture_ref,
                None,
                u64::from(att.level),
                u64::from(att.slice),
            )
        };
    // The level the pass names is relative to the texture it names, so a pass
    // rendering into level 1 of a view whose own range starts at level 2 lands
    // on the base texture's level 3. Both halves reach every rung below as one
    // number, which is what keeps the IOSurface texture and surface backing rungs — neither of
    // which has a mip layout — refusing an attachment level as loudly as they
    // already refuse a view level.
    let level = u32::try_from(level).map_err(|_| {
        C::LevelOverflow {
            view_level: u32::MAX,
            attachment_level: att.level,
        }
        .at(resolved_ref)
    })?;
    if resolved_ref == 0 {
        return Err(C::ViewBaseUnbound.at(resolved_ref));
    }
    // Archive lookup order is by **live** object-list type + descriptor, not a
    // sticky cache: IOSurface texture first, else type-2/3. Guest reuses object refs;
    // two failure modes for a stale IOSurface mapping relation:
    // 1) live type is now type-2/3 → must not force IOSurface texture (live residual
    //    mrt color RT resolve fail ref=199 type=2 fmt=0x73 480x64).
    // 2) live type is still IOSurface texture but descriptor mapping_id changed (or a
    //    recycled ref now names a different mid) → must re-read the live
    //    descriptor, not prefer the latch. Preferring latch routed dual-mid
    //    full-screen desktop composites onto only one mid (mid=3 nz=6M vs
    //    mid=4 stuck logo nz=1.97M; damage rects then preserved logo via Load).
    let live = objects::lookup_list_entry(state, host, task_id, resolved_ref);
    let live_type = live.as_ref().map(|e| e.kind);
    if let Some(ot) = live_type {
        if ot != ObjectKind::IOSurfaceTexture {
            // Synthetic fixtures may carry the retired side-map relation. A
            // product relation belongs to its retained resource and disappears
            // with that resource's explicit deletion.
            #[cfg(test)]
            state
                .fixtures
                .texture_to_mapping
                .remove(&(task_id, resolved_ref));
        }
    }
    let try_iosurface_texture = live_type == Some(ObjectKind::IOSurfaceTexture)
        || (live_type.is_none()
            && state
                .registered_texture_mapping(task_id, resolved_ref)
                .is_some());
    if try_iosurface_texture {
        // IOSurface texture sample windows carry planes, not mip levels — a mip>0 view
        // of an IOSurface has no contract-backed layout; fail visibly.
        if level != 0 {
            return Err(C::IOSurfaceMipView { level }.at(resolved_ref));
        }
        if slice != 0 || att.depth_plane != 0 {
            return Err(C::SubresourceShapeUnsupported {
                texture_type: crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_2D,
                slice,
                depth_plane: att.depth_plane,
            }
            .at(resolved_ref));
        }
        // Live list is source of truth for mapping_id when the entry is IOSurface texture.
        // Latch is only a fallback when the list entry is transiently missing
        // (resolve_iosurface_texture_ref refreshes the latch from the live descriptor).
        let mapping_id = if live_type == Some(ObjectKind::IOSurfaceTexture) {
            // A live mapper descriptor whose explicit mapper-service edge is
            // absent must refuse. Falling back to an older texture latch here
            // aliases two namespaces and can render into the previous surface
            // after a reference is recycled.
            objects::resolve_iosurface_texture_ref(state, host, task_id, resolved_ref)
        } else {
            state
                .registered_texture_mapping(task_id, resolved_ref)
                .or_else(|| {
                    objects::resolve_iosurface_texture_ref(state, host, task_id, resolved_ref)
                })
        }
        .ok_or(C::IOSurfaceUnresolved.at(resolved_ref))?;
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        // This rung is terminal either way, and both directions were already
        // true before it said so. A live IOSurface texture that fails geometry must not
        // be decoded as type-2/3 — that is the sticky-latch bug above. And in
        // the only other case that reaches here, `live_type` is `None`
        // (`try_iosurface_texture` admits nothing else), so every rung below ends at
        // `NoListEntry`: falling through reported the ladder's least specific
        // refusal for a failure the IOSurface texture rung had already diagnosed.
        let m = state
            .surfaces
            .mappings
            .get(&mapping_id)
            .ok_or(C::IOSurfaceNoMapping { mapping_id }.at(resolved_ref))?;
        if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
            return Err(C::IOSurfaceGeometry {
                mapping_id,
                has_geom: m.has_geometry(),
                width: m.width_or_zero(),
                height: m.height_or_zero(),
            }
            .at(resolved_ref));
        }
        // Not `rt_surface_backing_base_format`: an IOSurface texture mapping's format has
        // writers other than the surface backing decoder, so 0 here can mean "not
        // latched yet" rather than "refused", and BGRA8 is the display
        // contract's default for that case. See that function.
        let base_fmt = if m.format_or_zero() != 0 {
            m.format_or_zero()
        } else {
            MTL_FORMAT_BGRA8_UNORM
        };
        let fmt = effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
        if pixel_format::render_target_bpp(fmt).is_none() {
            return Err(C::IOSurfaceFormat { mapping_id, fmt }.at(resolved_ref));
        }
        return Ok(ResolvedRenderTarget {
            storage: super::ColorTargetStorage::Mapping(mapping_id),
            width: m.width_or_zero(),
            height: m.height_or_zero(),
            format: fmt,
            sample_count: 1,
        });
    }
    // x86 Ventura/Tahoe surface backing surface/backing (present IOSurface). Object-list
    // index == surface_id (ResourceHeap addObject type=4 objectId=getSurfaceID).
    // Without this, clear-only streams and Store writebacks never touch display
    // mids — guest pages stay empty and dual-mid thrash paints black.
    // Surface backing: object-list index is surface_id. IOSurface plane view RefTextureHandle: surfaceID@0
    // (allocateRefTextureHandle) — product color RTs are IOSurface plane view wrapping surface backing.
    let mut iosurface_plane_view: Option<objects::IOSurfacePlaneViewDescriptor> = None;
    let surface_backing_sid = match live.as_ref() {
        Some(e) if e.kind == ObjectKind::SurfaceBacking => Some(resolved_ref),
        Some(e) if e.kind == ObjectKind::IOSurfacePlaneView => {
            let resource = objects::resolve_resource(state, host, task_id, resolved_ref)
                .map_err(|_| C::IOSurfacePlaneViewDescRead.at(resolved_ref))?;
            let t5 = match objects::decoded_resource(&resource) {
                Ok(crate::runtime::decode::resource::Descriptor::IOSurfacePlaneView(view)) => view,
                _ => {
                    return Err(C::IOSurfacePlaneViewDescDecode {
                        len: resource.descriptor().len(),
                    }
                    .at(resolved_ref));
                }
            };
            let sid = t5.surface.get();
            if sid == 0 {
                return Err(C::IOSurfacePlaneViewSurfaceZero.at(resolved_ref));
            }
            iosurface_plane_view = t5.view;
            Some(sid)
        }
        _ => None,
    };
    if let Some(surface_id) = surface_backing_sid {
        if level != 0 {
            return Err(C::SurfaceBackingMipView { surface_id, level }.at(resolved_ref));
        }
        if slice != 0 || att.depth_plane != 0 {
            return Err(C::SubresourceShapeUnsupported {
                texture_type: crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_2D,
                slice,
                depth_plane: att.depth_plane,
            }
            .at(resolved_ref));
        }
        if !objects::resolve_surface_backing(state, host, surface_id) {
            return Err(C::SurfaceBackingUnresolved {
                surface_id,
                live_type,
            }
            .at(resolved_ref));
        }
        let m = state
            .surfaces
            .mappings
            .get(&surface_id)
            .ok_or(C::SurfaceBackingNoMapping { surface_id }.at(resolved_ref))?;
        if !m.has_geometry()
            || m.width_or_zero() == 0
            || m.height_or_zero() == 0
            || m.pages.entries.is_empty()
        {
            return Err(C::SurfaceBackingGeometry {
                surface_id,
                has_geom: m.has_geometry(),
                width: m.width_or_zero(),
                height: m.height_or_zero(),
                pages: m.pages.entries.len(),
            }
            .at(resolved_ref));
        }
        let (base_w, base_h, base_raw_fmt) =
            (m.width_or_zero(), m.height_or_zero(), m.format_or_zero());
        if live_type == Some(ObjectKind::IOSurfacePlaneView) {
            note_rt_iosurface_plane_view(
                iosurface_plane_view,
                surface_id,
                (base_w, base_h, base_raw_fmt),
            );
        }
        let base_fmt = rt_surface_backing_base_format(base_raw_fmt, surface_id).ok_or(
            C::SurfaceBackingBaseFormat {
                surface_id,
                raw_fmt: base_raw_fmt,
            }
            .at(resolved_ref),
        )?;
        let fmt = rt_surface_backing_declared_format(
            base_fmt,
            (base_w, base_h),
            iosurface_plane_view,
            view_fmt_override,
        );
        if pixel_format::render_target_bpp(fmt).is_none() {
            return Err(C::SurfaceBackingFormat { surface_id, fmt }.at(resolved_ref));
        }
        // mapping_id = surface_id; no linear GVA.
        return Ok(ResolvedRenderTarget {
            storage: super::ColorTargetStorage::Mapping(surface_id),
            width: m.width_or_zero(),
            height: m.height_or_zero(),
            format: fmt,
            sample_count: 1,
        });
    }
    // type-2/3 linear GVA (wallpaper/background layers, UI intermediate RTs).
    let entry = live.ok_or(C::NoListEntry.at(resolved_ref))?;
    if entry.kind != ObjectKind::Texture {
        return Err(C::WrongType {
            object_type: entry.kind,
        }
        .at(resolved_ref));
    }
    let resource = objects::resolve_resource(state, host, task_id, resolved_ref)
        .map_err(|_| C::LinearDescRead.at(resolved_ref))?;
    let tex = match objects::decoded_resource(&resource) {
        Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => tex,
        Err(status) => {
            return Err(C::LinearDescDecode {
                decode: status.slug(),
            }
            .at(resolved_ref));
        }
        Ok(_) => {
            return Err(C::WrongType {
                object_type: resource.entry().kind,
            }
            .at(resolved_ref));
        }
    };
    if tex.declared_pixel_format().is_none()
        || tex.extent().is_none()
        || tex.declared_row_stride().is_none()
    {
        return Err(C::LinearDescIncomplete {
            format: tex.declared_pixel_format().unwrap_or(0),
            width: tex.width,
            height: tex.height,
            row_stride: tex.row_stride,
        }
        .at(resolved_ref));
    }
    let base_fmt = tex
        .declared_pixel_format()
        .expect("the incomplete declaration returned above");
    let fmt = effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
    // Refuses a format with no known bytes-per-texel; the value is not needed.
    if pixel_format::render_target_bpp(fmt).is_none() {
        return Err(C::LinearFormat { fmt }.at(resolved_ref));
    }
    let declaration = tex.declaration.ok_or(
        C::LinearDescIncomplete {
            format: tex.declared_pixel_format().unwrap_or(0),
            width: tex.width,
            height: tex.height,
            row_stride: tex.row_stride,
        }
        .at(resolved_ref),
    )?;
    let storage_type = u16::from(declaration.texture_type);
    let storage_is_cube = matches!(
        storage_type,
        crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    );
    let level_layout = tex.level(level);
    if tex.slice_count != u32::from(declaration.array_length)
        || tex.cube_faces != storage_is_cube
        || !tex.declared_packing_fits_allocation()
        || !level_layout.is_some_and(|layout| tex.level_fits_slice(layout))
    {
        return Err(C::LinearPackingMismatch {
            texture_type: storage_type,
            declared_slices: u32::from(declaration.array_length),
            dimension_slices: tex.slice_count,
            bytes_per_slice: tex.bytes_per_slice,
            cube_faces: tex.cube_faces,
        }
        .at(resolved_ref));
    }
    let physical_slices = tex
        .physical_slice_count()
        .ok_or(C::SubresourceOffsetOverflow.at(resolved_ref))?;
    let subresource_offset = linear_attachment_subresource_offset(
        storage_type,
        physical_slices,
        tex.bytes_per_slice,
        level,
        level_layout,
        slice,
        att.depth_plane,
    )
    .map_err(|cause| cause.at(resolved_ref))?;
    // Mip>0 view of a linear texture: the RT is that level's plane inside the
    // base allocation (archive collapses view mip into linear geometry —
    // compositor blur/backdrop pyramids render into successive levels).
    let (gva, w, h, bpr) = if level != 0 || subresource_offset != 0 {
        let (level_gva, layout) = tex
            .level_gva(level, state.page_shift)
            .ok_or(C::LinearLevelGva { level }.at(resolved_ref))?;
        if layout.row_stride > u32::MAX as u64 {
            return Err(C::LinearLevelStride {
                row_stride: layout.row_stride,
            }
            .at(resolved_ref));
        }
        // Full level span must fit the allocation — writing rows past it would
        // corrupt adjacent guest memory.
        //
        // This is deliberately still `row_stride * height` and NOT
        // `TextureLevelLayout::read_span`, which every *reader* of a level uses.
        // The difference is one row of trailing padding, and whether this path
        // touches it depends on what the render-target store writes per row —
        // a question about the store, not about this bound. Until that is
        // measured, the wider span is the safe direction here: it can only
        // refuse a target, never let a write run past the allocation.
        let span = layout.row_stride.checked_mul(layout.height as u64).ok_or(
            C::LinearLevelSpan {
                row_stride: layout.row_stride,
                height: layout.height,
            }
            .at(resolved_ref),
        )?;
        let offset = tex
            .base_offset
            .checked_add(layout.offset)
            .and_then(|offset| offset.checked_add(subresource_offset))
            .ok_or(C::SubresourceOffsetOverflow.at(resolved_ref))?;
        if tex.allocation_size != 0 && offset.saturating_add(span) > tex.allocation_size {
            return Err(C::LinearLevelPastAllocation {
                offset,
                span,
                allocation_size: tex.allocation_size,
            }
            .at(resolved_ref));
        }
        (
            level_gva
                .checked_add(subresource_offset)
                .ok_or(C::SubresourceOffsetOverflow.at(resolved_ref))?,
            layout.width,
            layout.height,
            layout.row_stride as u32,
        )
    } else {
        let (gva, alloc) = tex.backing_gva_size(state.page_shift).ok_or(
            C::LinearBackingGva {
                allocation_size: tex.allocation_size,
                handle: tex.handle,
            }
            .at(resolved_ref),
        )?;
        let span = (tex.row_stride as u64)
            .checked_mul(tex.height as u64)
            .ok_or(
                C::LinearSpan {
                    row_stride: tex.row_stride,
                    height: tex.height,
                }
                .at(resolved_ref),
            )?;
        let tight0 = pixel_format::tight_row_bytes(tex.width, fmt).ok_or(
            C::LinearTightRow {
                width: tex.width,
                fmt,
            }
            .at(resolved_ref),
        )?;
        // Exclusive last-row end (archive): height-1 * bpr + tight may fit
        // tighter allocs; accept if bpr*height fits allocation.
        if alloc > 0 && span > alloc {
            let alt = if tex.height > 0 {
                (tex.row_stride as u64)
                    .saturating_mul((tex.height - 1) as u64)
                    .saturating_add(tight0 as u64)
            } else {
                0
            };
            if alt > alloc {
                return Err(C::LinearPastAllocation { span, alloc, alt }.at(resolved_ref));
            }
        }
        (gva, tex.width, tex.height, tex.row_stride)
    };
    // Ordered before the tight-row check on purpose. `tight_row_bytes` refuses a
    // zero width too, so the old single `bpr < tight || w == 0 || h == 0` exit
    // could only ever be reached for a zero *height* — a zero width was already
    // gone, reported as whatever the tight-row rung would have said. Asking the
    // extent question first keeps both of these checks reachable and lets each
    // report what it actually found.
    if w == 0 || h == 0 {
        return Err(C::ZeroExtent {
            width: w,
            height: h,
        }
        .at(resolved_ref));
    }
    let tight = pixel_format::tight_row_bytes(w, fmt)
        .ok_or(C::RowTight { width: w, fmt }.at(resolved_ref))?;
    if bpr < tight {
        return Err(C::RowStride { bpr, tight }.at(resolved_ref));
    }
    let allocation_gva = tex.allocation_base_gva(state.page_shift).ok_or(
        C::LinearBackingGva {
            allocation_size: tex.allocation_size,
            handle: tex.handle,
        }
        .at(resolved_ref),
    )?;
    let plane_offset = gva.checked_sub(allocation_gva).ok_or(
        C::LinearBackingGva {
            allocation_size: tex.allocation_size,
            handle: tex.handle,
        }
        .at(resolved_ref),
    )?;
    let storage =
        super::LinearColorTarget::new(allocation_gva, tex.allocation_size, plane_offset, bpr)
            .map(super::ColorTargetStorage::Linear)
            .ok_or(
                C::LinearBackingGva {
                    allocation_size: tex.allocation_size,
                    handle: tex.handle,
                }
                .at(resolved_ref),
            )?;
    Ok(ResolvedRenderTarget {
        storage,
        width: w,
        height: h,
        format: fmt,
        sample_count: 1,
    })
}

#[cfg(test)]
mod subresource_tests {
    use super::*;
    use crate::runtime::decode::resource::{
        TextureLevelLayout, TEXTURE_VIEW_MTL_TYPE_2D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE,
        TEXTURE_VIEW_MTL_TYPE_3D,
    };

    #[test]
    fn array_attachments_use_the_declared_full_chain_slice_pitch() {
        let layout = TextureLevelLayout {
            size: 0x4000,
            row_stride: 0x100,
            width: 64,
            height: 64,
            depth: 1,
            ..Default::default()
        };
        assert_eq!(
            linear_attachment_subresource_offset(
                TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
                4,
                0x4000,
                0,
                Some(&layout),
                3,
                0,
            ),
            Ok(0xc000)
        );
        assert_eq!(
            linear_attachment_subresource_offset(
                TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
                4,
                0x4000,
                0,
                Some(&layout),
                4,
                0,
            ),
            Err(RenderTargetCause::ArraySliceOutOfRange {
                slice: 4,
                layers: 4,
            })
        );
    }

    #[test]
    fn multisample_source_is_a_single_2d_attachment_subresource() {
        assert_eq!(
            linear_attachment_subresource_offset(
                TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE,
                1,
                0x4000,
                0,
                None,
                0,
                0,
            ),
            Ok(0)
        );
        assert_eq!(
            linear_attachment_subresource_offset(
                TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE,
                1,
                0x4000,
                0,
                None,
                1,
                0,
            ),
            Err(RenderTargetCause::SubresourceShapeUnsupported {
                texture_type: TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE,
                slice: 1,
                depth_plane: 0,
            })
        );
    }

    #[test]
    fn volume_attachments_use_depth_planes_not_array_slices() {
        let layout = TextureLevelLayout {
            size: 0x6000,
            row_stride: 0x100,
            width: 64,
            height: 32,
            depth: 3,
            ..Default::default()
        };
        assert_eq!(
            linear_attachment_subresource_offset(
                TEXTURE_VIEW_MTL_TYPE_3D,
                1,
                0x6000,
                0,
                Some(&layout),
                0,
                2,
            ),
            Ok(0x4000)
        );
        assert_eq!(
            linear_attachment_subresource_offset(
                TEXTURE_VIEW_MTL_TYPE_3D,
                1,
                0x6000,
                0,
                Some(&layout),
                1,
                0,
            ),
            Err(RenderTargetCause::SubresourceShapeUnsupported {
                texture_type: TEXTURE_VIEW_MTL_TYPE_3D,
                slice: 1,
                depth_plane: 0,
            })
        );
    }
}

/// The two report helpers above, tested where they live.
///
/// Both are pure given their arguments — one maps a decoded format to a decision
/// and one scores a view against a base — so neither needs a device, a mapping,
/// or a boot to hold. They moved here with the code they describe; they were
/// written against it in `runtime::draw`'s 4 700-line colocated test module, which
/// is the file the plan wants to stop growing.
#[cfg(test)]
mod tests {
    use crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE;

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    /// The both-ways watch answers "have I seen this surface through a
    /// *differing* view", and it must not answer "no" for a surface it merely
    /// stopped tracking. Forgetting one is the direction that reports the
    /// format repair as safe on a surface it would break.
    #[test]
    fn the_differing_view_watch_stops_admitting_rather_than_forgetting() {
        // Ids of its own, because the set is process-wide and shared with every
        // other test in this binary.
        let base = 0x0100_0000u32;
        assert!(!differed_before(base), "nothing has been noted for this id");
        note_differed(base);
        assert!(differed_before(base));

        // Fill past the bound with ids nothing else uses, then confirm the
        // first one is still known: a set that evicted to make room would have
        // dropped it, and that is the answer this watch may not give.
        for i in 1..=(DIFFERED_MAX as u32 * 2) {
            note_differed(base + i);
        }
        assert!(
            differed_before(base),
            "the bound must stop admissions, not evict what is already recorded"
        );
        assert!(
            DIFFERED.lock().unwrap().len() <= DIFFERED_MAX,
            "the set is bounded"
        );
    }

    /// A surface backing colour attachment whose mapping carries the decoder's format
    /// refusal must be declined, and every decline must be counted.
    ///
    /// `m.format_or_zero() == 0` on a surface backing mapping has exactly one writer,
    /// `apply_surface_backing`, and it means multi-plane or unknown FourCC — a surface
    /// that is not a single-format colour attachment. Inventing BGRA8 from it
    /// describes the wrong stride over the wrong bytes and every downstream window
    /// is built from the answer. The counter has to fire on the refusal and only on
    /// it: one that also fired on ordinary formats would answer a different question
    /// and read identically.
    #[test]
    fn a_surface_backing_render_target_declines_the_decoders_format_refusal() {
        use crate::runtime::drain::store_route_count;
        use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        let before = store_route_count("rt_base_fmt_declined");
        // A format the decoder resolved is passed through untouched and uncounted.
        assert_eq!(
            rt_surface_backing_base_format(MTL_FORMAT_BGRA8_UNORM, 11),
            Some(MTL_FORMAT_BGRA8_UNORM)
        );
        assert_eq!(store_route_count("rt_base_fmt_declined"), before);
        // The refusal declines, and is counted per occurrence — the fail line is
        // deduped per mapping, the counter is not.
        assert_eq!(rt_surface_backing_base_format(0, 11), None);
        assert_eq!(rt_surface_backing_base_format(0, 12), None);
        assert_eq!(rt_surface_backing_base_format(0, 11), None);
        assert_eq!(store_route_count("rt_base_fmt_declined"), before + 3);
    }

    /// A IOSurface plane view view's declared format reaches the colour attachment, so the
    /// hardware performs the linear-to-sRGB encode the guest asked for.
    ///
    /// This is the write half of the sRGB round trip. Metal stores `E(L)` into an
    /// sRGB-declared attachment and returns `L` when the same surface is sampled
    /// through an sRGB view; this device used to store `L` and — once the sampled
    /// rails learned to decode — return `D(L)`, which is darker. Both halves have
    /// to agree, and the base mapping's format is not the declaration.
    ///
    /// The live shape is `view=300x300 fmt=0x51 base=300x300 fmt=0x50`, seen twice
    /// on a driven macos-13 boot at icon size.
    #[test]
    fn a_iosurface_plane_views_declared_format_is_what_the_colour_attachment_attaches() {
        use crate::runtime::objects::IOSurfacePlaneViewDescriptor;
        use reims_vgpu_core::pixel_format::{
            MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA16_FLOAT,
        };

        let extent = (300u32, 300u32);
        let view = |w, h, fmt| {
            Some(IOSurfacePlaneViewDescriptor {
                pixel_format: fmt,
                width: w,
                height: h,
                depth: 1,
                plane_index: 0,
            })
        };

        // The defect: an sRGB view over a linear base must attach sRGB.
        assert_eq!(
            rt_surface_backing_declared_format(
                MTL_FORMAT_BGRA8_UNORM,
                extent,
                view(300, 300, MTL_FORMAT_BGRA8_UNORM_SRGB),
                None
            ),
            MTL_FORMAT_BGRA8_UNORM_SRGB
        );
        // A surface bound without a view keeps the mapping's own format.
        assert_eq!(
            rt_surface_backing_declared_format(MTL_FORMAT_BGRA8_UNORM, extent, None, None),
            MTL_FORMAT_BGRA8_UNORM
        );
        // A view that agrees says nothing new.
        assert_eq!(
            rt_surface_backing_declared_format(
                MTL_FORMAT_BGRA8_UNORM,
                extent,
                view(300, 300, MTL_FORMAT_BGRA8_UNORM),
                None
            ),
            MTL_FORMAT_BGRA8_UNORM
        );
        // A view describing a different grid is not honoured at all: this resolve
        // takes the base mapping's geometry, so lifting the format alone would
        // attach a reinterpretation to a grid that is not its own.
        assert_eq!(
            rt_surface_backing_declared_format(
                MTL_FORMAT_BGRA8_UNORM,
                extent,
                view(75, 300, MTL_FORMAT_RGBA16_FLOAT),
                None
            ),
            MTL_FORMAT_BGRA8_UNORM
        );
        // A same-extent view whose texel is a different width is not a
        // reinterpretation of one allocation, and the base format stands.
        assert_eq!(
            rt_surface_backing_declared_format(
                MTL_FORMAT_BGRA8_UNORM,
                extent,
                view(300, 300, MTL_FORMAT_RGBA16_FLOAT),
                None
            ),
            MTL_FORMAT_BGRA8_UNORM
        );
        // A zero format is the IOSurface plane view decoder saying it has none, not a
        // declaration of format zero.
        assert_eq!(
            rt_surface_backing_declared_format(
                MTL_FORMAT_BGRA8_UNORM,
                extent,
                view(300, 300, 0),
                None
            ),
            MTL_FORMAT_BGRA8_UNORM
        );
        // A type-8 view is a further reinterpretation the guest asked for on top,
        // so it outranks the IOSurface plane view record.
        assert_eq!(
            rt_surface_backing_declared_format(
                MTL_FORMAT_BGRA8_UNORM,
                extent,
                view(300, 300, MTL_FORMAT_BGRA8_UNORM_SRGB),
                Some(MTL_FORMAT_BGRA8_UNORM)
            ),
            MTL_FORMAT_BGRA8_UNORM
        );
    }

    /// A IOSurface plane view colour attachment must be scored on whether its view agrees with
    /// the base mapping, and "no view decoded" must not read as agreement.
    ///
    /// The resolve takes geometry from the base mapping either way, so the counter
    /// is the only thing that can say whether that is lossless. Folding an
    /// undecoded record into `same` would report the ambiguous case as the healthy
    /// one, which is the failure mode that makes a census worthless.
    #[test]
    fn a_iosurface_plane_view_render_target_view_is_scored_against_the_base_it_resolves_through() {
        use crate::runtime::drain::store_route_count;
        use crate::runtime::objects::IOSurfacePlaneViewDescriptor;

        let base = (64u32, 32u32, 0x50u16);
        let view = |w, h, fmt| {
            Some(IOSurfacePlaneViewDescriptor {
                pixel_format: fmt,
                width: w,
                height: h,
                depth: 1,
                plane_index: 0,
            })
        };
        let (same0, diff0, und0) = (
            store_route_count("rt_iosurface_plane_view_same"),
            store_route_count("rt_iosurface_plane_view_differs"),
            store_route_count("rt_iosurface_plane_view_undecoded"),
        );

        note_rt_iosurface_plane_view(view(64, 32, 0x50), 5, base);
        assert_eq!(store_route_count("rt_iosurface_plane_view_same"), same0 + 1);

        // The live case the contract names: a row-byte-equivalent reinterpretation
        // at a different width and format over the same bytes.
        note_rt_iosurface_plane_view(view(16, 32, 0x73), 6, base);
        assert_eq!(
            store_route_count("rt_iosurface_plane_view_differs"),
            diff0 + 1
        );
        // Geometry alone is not the test — a format-only view is still a different
        // view, and it is the one this resolve would silently render as BGRA8.
        note_rt_iosurface_plane_view(view(64, 32, 0x73), 7, base);
        assert_eq!(
            store_route_count("rt_iosurface_plane_view_differs"),
            diff0 + 2
        );

        note_rt_iosurface_plane_view(None, 8, base);
        assert_eq!(
            store_route_count("rt_iosurface_plane_view_undecoded"),
            und0 + 1
        );
        assert_eq!(
            store_route_count("rt_iosurface_plane_view_same"),
            same0 + 1,
            "an undecoded record must not be scored as agreement"
        );
    }

    /// An unbound colour attachment spends no line, and every other exit of the
    /// ladder names the check that refused.
    ///
    /// The two halves are one test because they are one rule with two
    /// directions. `texture_ref == 0` is the guest saying nothing is bound in
    /// this slot — `AGENTS.md`'s canonical quiet case, and the one an
    /// over-eager emitter turns into a per-draw flood. Everything else is a
    /// pass being dropped, and the ladder used to report ~30 distinct reasons
    /// for that as one bare `None`.
    #[test]
    fn an_unbound_attachment_stays_quiet_and_a_missing_one_names_its_rung() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();

        let cap = crate::observe::FailCapture::start();
        assert!(lookup_render_target(&mut state, &host, /*task*/ 4, attach(0)).is_none());
        assert!(
            cap.lines().is_empty(),
            "an unbound colour attachment must spend no line: {:?}",
            cap.lines()
        );
        drop(cap);

        // Nothing under the ref: no object list on this task at all, which is
        // the ladder's least specific refusal and still names itself.
        let cap = crate::observe::FailCapture::start();
        assert!(lookup_render_target(&mut state, &host, 4, attach(0x5c1)).is_none());
        assert_eq!(
            cap.one("rt_resolve"),
            "rt_resolve reason=rt_no_list_entry base=1473 task=4 ref=1473"
        );
    }

    /// An IOSurface texture latch whose mapping has no geometry is reported as that, not
    /// as the missing object-list entry three rungs further down.
    ///
    /// This is the terminal-rung property. The IOSurface texture arm used to return only
    /// when the *live* list said IOSurface; a latch-only attempt that failed
    /// geometry fell through to the surface backing and linear rungs instead. It could
    /// not resolve there — `live_type` is `None` in that arm by construction,
    /// so `surface_backing_sid` is `None` and the linear rung's first act is to unwrap
    /// the entry that is not there. The fall-through therefore changed nothing
    /// about the outcome and everything about the diagnosis: the ladder
    /// reported `no_list_entry` for a surface whose mapping it had already
    /// found and measured.
    #[test]
    fn a_iosurface_texture_latch_without_geometry_names_the_geometry_check() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        let tex_ref = 0x5c2u32;

        // A mapping exists and is latched for this ref, and the guest has not
        // declared its geometry yet.
        assert!(state.map_surface(77));
        state.fixtures.texture_to_mapping.insert((4, tex_ref), 77);

        let cap = crate::observe::FailCapture::start();
        assert!(lookup_render_target(&mut state, &host, 4, attach(tex_ref)).is_none());
        let line = cap.one("rt_resolve");
        assert!(
            line.starts_with("rt_resolve reason=rt_iosurface_texture_geometry "),
            "the rung that found the mapping must be the one that reports: {line}"
        );
        assert!(
            line.contains(" mid=77 has_geom=false dims=0x0"),
            "the refusal must carry the mapping it measured: {line}"
        );
    }

    /// A colour attachment naming `texture_ref` at level 0 — the shape every
    /// case below is about, so that a case that means to vary the subresource
    /// has to say so.
    fn attach(texture_ref: u32) -> crate::runtime::decode::render::ColorAttachment {
        crate::runtime::decode::render::ColorAttachment {
            texture_ref,
            ..Default::default()
        }
    }

    /// Opcode 9 shares the texture-view wire family but semantically declares
    /// a texture over a buffer allocation. Render-target resolution must stop
    /// at that terminal placement instead of treating its buffer ref as a view
    /// base and rejecting the attachment as the wrong object type.
    #[test]
    fn a_buffer_backed_texture_is_a_linear_render_target() {
        use crate::model::PAGE_SHIFT_ARM64E;
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
            OBJECT_TYPE_TEXTURE_VIEW,
        };
        use crate::runtime::gva_mem::{self, write_task_gva_arm64e};
        use reims_vgpu_core::endian::{st32, st64};
        use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use reims_vgpu_wire::ops::backed_texture::{
            BufferTextureBody, BUFFER_TEXTURE_TOTAL_LEN, OPCODE_BUFFER_TEXTURE,
        };

        const TASK: u32 = 1;
        const BUFFER_REF: u32 = 7;
        const TEXTURE_REF: u32 = 10;
        const BUFFER_HANDLE: u32 = 5;
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 16;
        const OFFSET: u64 = 0x100;
        const BPR: u64 = 0x120;
        const ALLOC: u64 = 0x4000;

        let mut host = FakeHost::new();
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(TASK, 0, 32));

        let mut buffer_descriptor = [0u8; 16];
        st64(&mut buffer_descriptor, ALLOC);
        st32(&mut buffer_descriptor[8..], BUFFER_HANDLE);
        let buffer_descriptor_gva = 0x180;
        write_task_gva_arm64e(
            &mut host,
            &state.tasks[TASK],
            buffer_descriptor_gva,
            &buffer_descriptor,
        );
        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            (OBJECT_TYPE_BUFFER as u32) | ((buffer_descriptor.len() as u32) << 8),
        );
        st64(&mut entry[4..], buffer_descriptor_gva);
        write_task_gva_arm64e(
            &mut host,
            &state.tasks[TASK],
            list_object_entry_offset(BUFFER_REF, 32).unwrap(),
            &entry,
        );

        let mut record = vec![0u8; BUFFER_TEXTURE_TOTAL_LEN as usize];
        st32(&mut record, OPCODE_BUFFER_TEXTURE);
        st32(&mut record[4..], BUFFER_TEXTURE_TOTAL_LEN);
        st32(&mut record[8..], TEXTURE_REF);
        let body = reims_vgpu_wire::OP_HEADER_LEN;
        let buffer_ref_at = body + core::mem::offset_of!(BufferTextureBody, buffer_ref);
        let offset_at = body + core::mem::offset_of!(BufferTextureBody, offset);
        let bpr_at = body + core::mem::offset_of!(BufferTextureBody, bytes_per_row);
        let desc_at = body + core::mem::offset_of!(BufferTextureBody, desc);
        st32(&mut record[buffer_ref_at..], BUFFER_REF);
        st64(&mut record[offset_at..], OFFSET);
        st64(&mut record[bpr_at..], BPR);
        st32(
            &mut record[desc_at..],
            2 | (3 << 8) | (u32::from(MTL_FORMAT_BGRA8_UNORM) << 16),
        );
        st32(&mut record[desc_at + 4..], WIDTH);
        st32(&mut record[desc_at + 8..], HEIGHT);
        st32(&mut record[desc_at + 12..], 1);
        for (field, value) in [(16usize, 1u16), (18, 1), (20, 1), (22, 0)] {
            record[desc_at + field..desc_at + field + 2].copy_from_slice(&value.to_le_bytes());
        }
        let record_gva = 6u64 << PAGE_SHIFT_ARM64E;
        write_task_gva_arm64e(&mut host, &state.tasks[TASK], record_gva, &record);
        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            (OBJECT_TYPE_TEXTURE_VIEW as u32) | (BUFFER_TEXTURE_TOTAL_LEN << 8),
        );
        st64(&mut entry[4..], record_gva);
        write_task_gva_arm64e(
            &mut host,
            &state.tasks[TASK],
            list_object_entry_offset(TEXTURE_REF, 32).unwrap(),
            &entry,
        );

        let target = resolve_render_target(&mut state, &host, TASK, attach(TEXTURE_REF))
            .expect("the buffer texture has one complete linear render plane");
        let linear = target.storage.linear().expect("linear target");
        assert_eq!(
            linear.allocation_gva,
            u64::from(BUFFER_HANDLE) << PAGE_SHIFT_ARM64E
        );
        assert_eq!(linear.allocation_size, ALLOC);
        assert_eq!(linear.plane_offset, OFFSET);
        assert_eq!(linear.row_stride, BPR as u32);
        assert_eq!((target.width, target.height), (WIDTH, HEIGHT));
        assert_eq!(target.format, MTL_FORMAT_BGRA8_UNORM);
    }

    /// A refusal is reported once per attachment per check, not once per draw.
    ///
    /// The guest re-issues the same pass every frame, so this path is entered
    /// at draw rate. The two ad-hoc lines this ladder used to emit had no latch
    /// at all — one of them on the surface backing rung, which a compositing workload
    /// takes for every desktop surface.
    #[test]
    fn a_repeated_refusal_on_the_same_attachment_reports_once() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        let tex_ref = 0x5c3u32;

        let cap = crate::observe::FailCapture::start();
        for _ in 0..8 {
            assert!(lookup_render_target(&mut state, &host, 4, attach(tex_ref)).is_none());
        }
        assert_eq!(
            cap.lines().len(),
            1,
            "eight draws against one unresolvable attachment must spend one line: {:?}",
            cap.lines()
        );
    }

    /// A colour attachment naming mip level 1 resolves to that level's own
    /// plane — its guest VA, its stride and its geometry — not to level 0's.
    ///
    /// macOS 26's compositor renders a blur pyramid level by level, and until
    /// this the whole class was refused at decode as an unbindable subresource.
    /// The rung it reaches was already here for a type-8 *view* that carries a
    /// level; what was missing was the pass's own `level` field reaching it.
    ///
    /// Every assertion names a number that differs between the two levels, so a
    /// resolve that quietly answered for level 0 fails all four rather than
    /// returning a plausible target. That is the failure this replaces: level 1
    /// used to land on level 0's bytes, overwriting the image the guest samples
    /// at LOD 0.
    #[test]
    fn a_colour_attachment_at_mip_one_resolves_that_levels_own_plane() {
        use crate::model::PAGE_SHIFT_ARM64E;
        use crate::runtime::decode::resource::{
            list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
            TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_BASE_OFFSET, TEXTURE_DESC_BYTES_PER_SLICE,
            TEXTURE_DESC_DECLARATION, TEXTURE_DESC_HEIGHT, TEXTURE_DESC_LEVEL_RECORDS,
            TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_MIP_LEVEL_RECORD_LEN,
            TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_SLICE_COUNT, TEXTURE_DESC_WIDTH,
            TEXTURE_LEVEL_HEIGHT, TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE,
            TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH,
        };
        use crate::runtime::gva_mem::{self, write_task_gva_arm64e};
        use reims_vgpu_core::endian::{st16, st32, st64};
        use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 16);
        assert!(state.set_object_list(1, 0, 256));

        // A two-level BGRA8 pyramid: 64x32 at stride 256, then 32x16 at stride
        // 128 starting one level-0 span in.
        let (tex_ref, w0, h0, bpr0) = (200u32, 64u32, 32u32, 256u32);
        let (w1, h1, bpr1) = (32u32, 16u32, 128u32);
        let l1_offset = (bpr0 as u64) * (h0 as u64);
        let alloc = l1_offset + (bpr1 as u64) * (h1 as u64);
        let handle = 8u32;

        // The complete serialized declaration shifts one record along for a
        // two-level body; a payload ending after only its format word is not a
        // texture declaration.
        let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let mut desc = vec![0u8; body];
        st64(&mut desc[LINEAR_DESC_SIZE..], alloc);
        st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr0);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], w0);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], h0);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 2);
        st16(&mut desc[TEXTURE_DESC_SLICE_COUNT..], 1);
        st64(&mut desc[TEXTURE_DESC_BASE_OFFSET..], 0);
        st64(&mut desc[TEXTURE_DESC_BYTES_PER_SLICE..], alloc);
        let rec = TEXTURE_DESC_LEVEL_RECORDS;
        st64(&mut desc[rec + TEXTURE_LEVEL_OFFSET..], l1_offset);
        st64(
            &mut desc[rec + TEXTURE_LEVEL_SIZE..],
            (bpr1 as u64) * (h1 as u64),
        );
        st64(&mut desc[rec + TEXTURE_LEVEL_ROW_STRIDE..], bpr1 as u64);
        st32(&mut desc[rec + TEXTURE_LEVEL_WIDTH..], w1);
        st32(&mut desc[rec + TEXTURE_LEVEL_HEIGHT..], h1);
        let declaration = TEXTURE_DESC_DECLARATION + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let array_length = declaration
            + core::mem::offset_of!(
                reims_vgpu_wire::ops::texture::TextureDescriptorBody,
                array_length
            );
        st16(&mut desc[array_length..], 1);
        st32(
            &mut desc[declaration..],
            2 | (4 << 8) | (u32::from(MTL_FORMAT_BGRA8_UNORM) << 16),
        );

        let desc_gva = 0x280u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
        let off = list_object_entry_offset(tex_ref, 256).expect("ref is inside the list");
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut list_entry[0..],
            (OBJECT_TYPE_TEXTURE as u32) | ((desc.len() as u32) << 8),
        );
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

        let base = (handle as u64) << PAGE_SHIFT_ARM64E;
        let cap = crate::observe::FailCapture::start();
        let level0 =
            lookup_render_target(&mut state, &host, 1, attach(tex_ref)).unwrap_or_else(|| {
                panic!(
                    "level 0 of a two-level texture still resolves: {:?}",
                    cap.lines()
                )
            });
        assert_eq!(
            (
                level0.storage.target_gva(),
                level0.width,
                level0.height,
                level0.storage.row_stride()
            ),
            (base, w0, h0, bpr0),
            "level 0 must be unchanged by the pyramid above it"
        );

        let mut at_level_1 = attach(tex_ref);
        at_level_1.level = 1;
        let level1 = lookup_render_target(&mut state, &host, 1, at_level_1)
            .expect("a pass naming mip level 1 must resolve, not be refused");
        assert!(
            cap.lines().is_empty(),
            "resolving a named level is not a refusal: {:?}",
            cap.lines()
        );
        assert_eq!(
            (
                level1.storage.target_gva(),
                level1.width,
                level1.height,
                level1.storage.row_stride()
            ),
            (base + l1_offset, w1, h1, bpr1),
            "level 1 must render into its own plane, at its own geometry"
        );
    }

    /// A linear target whose declared row stride is narrower than one tight row
    /// of its own format names the stride, and says both numbers.
    ///
    /// The deepest rung, reached only after the descriptor decoded and the
    /// format was accepted, and the one where a bare `None` was least
    /// recoverable: everything above it succeeded, so the caller's
    /// "unresolved" said nothing at all. Rendering it would have consecutive
    /// rows overlapping in guest memory.
    #[test]
    fn a_linear_target_with_a_stride_narrower_than_its_own_row_names_both_numbers() {
        use crate::model::PAGE_SHIFT_ARM64E;
        use crate::runtime::decode::resource::{
            list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
            TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_BASE_OFFSET, TEXTURE_DESC_BYTES_PER_SLICE,
            TEXTURE_DESC_DECLARATION, TEXTURE_DESC_HEIGHT, TEXTURE_DESC_MIPMAP_LEVEL_COUNT,
            TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_SLICE_COUNT,
            TEXTURE_DESC_WIDTH,
        };
        use crate::runtime::gva_mem::{self, write_task_gva_arm64e};
        use reims_vgpu_core::endian::{st16, st32, st64};
        use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA16_FLOAT;

        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 16);
        assert!(state.set_object_list(1, 0, 256));

        // 64 texels of RGBA16Float need 512 bytes a row; the guest declared 256.
        let (tex_ref, w, h, bpr) = (199u32, 64u32, 32u32, 256u32);
        let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
        st64(&mut desc[LINEAR_DESC_SIZE..], (bpr as u64) * (h as u64));
        st32(&mut desc[LINEAR_DESC_HANDLE..], 8);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], w);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], h);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
        st16(&mut desc[TEXTURE_DESC_SLICE_COUNT..], 1);
        st64(&mut desc[TEXTURE_DESC_BASE_OFFSET..], 0);
        st64(
            &mut desc[TEXTURE_DESC_BYTES_PER_SLICE..],
            u64::from(bpr) * u64::from(h),
        );
        let array_length = TEXTURE_DESC_DECLARATION
            + core::mem::offset_of!(
                reims_vgpu_wire::ops::texture::TextureDescriptorBody,
                array_length
            );
        st16(&mut desc[array_length..], 1);
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_RGBA16_FLOAT,
        );
        let desc_gva = 0x280u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
        let off = list_object_entry_offset(tex_ref, 256).expect("ref is inside the list");
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut list_entry[0..],
            (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8),
        );
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

        let cap = crate::observe::FailCapture::start();
        assert!(lookup_render_target(&mut state, &host, 1, attach(tex_ref)).is_none());
        let line = cap.one("rt_resolve");
        assert!(
            line.starts_with("rt_resolve reason=rt_row_stride "),
            "the stride check must be the one that reports: {line}"
        );
        assert!(
            line.contains(" bpr=256 tight=512"),
            "the refusal must carry the stride it compared against: {line}"
        );
    }
}
