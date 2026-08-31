//! Extents the guest's API defines: a three-dimensional compute extent, the
//! size of one mip level of a texture, and the byte length of a tightly-packed
//! image.

/// A grid or threadgroup size, as three dimensions that travel together.
///
/// Its own type rather than three `u32`s because a dispatch carries **two** of
/// these side by side, built from sources that look alike — three consecutive
/// little-endian words for the indirect arms — and a transposition between them
/// dispatches a valid grid of the wrong shape, which nothing downstream can tell
/// from the right one.
///
/// It lives in `contract` rather than beside the decoder that first needed it
/// because the hazard is at the *boundary*, not at construction. The decoder
/// built two of these correctly and then destructured both back into six loose
/// `u32` parameters to reach the backend, where every one of the 720 orderings
/// compiles again — so the type protected the half of the journey that was
/// already safe and stopped exactly where the two extents become
/// interchangeable. Both sides of that call now name it.
///
/// What this does **not** close: two `Extent3` arguments are still the same
/// type, so passing the threadgroup where the grid belongs compiles. That is
/// the one remaining transposition of the 720, and it is the one the callers'
/// own `grid` / `threadgroup` bindings name at every site. Closing it needs two
/// newtypes, and the argument for them is a measurement nobody has: it has not
/// happened, whereas a `grid_y`/`grid_z` slip in a six-argument run is the kind
/// that has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Metal's dimension for mip `level` of an axis whose level-0 size is `base`.
///
/// Each level halves and floors, and the chain stops at 1 rather than reaching
/// 0 — `MTLTexture`'s level sizes are `max(1, base >> level)`, which is why a
/// 100-wide texture's levels run 100, 50, 25, 12, ... and its last level is 1
/// and not 0. `base == 0` is the one case that is not a clamp: an axis with no
/// texels has no levels, and answering 1 there would size a read of a texture
/// that does not exist.
///
/// Here rather than in either backend because both rails need the same answer
/// and each used to hold its own copy — `backend::metal::mipmap` for the
/// filtered generator, and a cfg-forked `metal_mip_extent_local` in
/// `runtime::mipmap` whose Vulkan arm reimplemented the line the Metal arm
/// called. That fork also decided nothing: the two arms were identical, so the
/// only thing the `#[cfg]` could ever change was which copy ran.
///
/// The runtime resolver rejects any stored mip layout whose extent disagrees
/// with this, so a wrong formula either refuses valid mip chains or accepts a
/// mismatched layout that then samples out of bounds.
pub fn mip_extent(base: u32, level: u32) -> u32 {
    if base == 0 {
        return 0;
    }
    // `base >> level` is a panic in a debug build for any level at or past 32,
    // and a *masked* shift in a release one — Rust does not trap an over-wide
    // shift, so `mip_extent(1024, 32)` answered 1024 where 1 is right. The
    // `checked_shr` is not a guard bolted on to survive a hostile level: past
    // the width of the type the chain has bottomed out, so shifting to zero and
    // then clamping to one is the same formula continued, and it agrees with
    // every level below. The level is a decoded guest field —
    // `TEXTURE_MAX_MIP_LEVELS` bounds the ones the decoder admits, but this is
    // `pub` in `contract` and the Metal generator reaches it directly, so it
    // states its own domain rather than borrowing the decoder's.
    base.checked_shr(level).unwrap_or(0).max(1)
}

/// Bytes of a tightly-packed image of this geometry: `width * height * bpp`,
/// with no row alignment.
///
/// "Tight" is the whole contract. Anything the guest has told us a pitch for
/// must not come through here — [`crate::iosurface_pages::packed_span_estimate`]
/// is the row-aligned estimate for sizing a page table, and the two differ by
/// exactly the alignment slack. Mixing them up reads short.
///
/// Zero on either axis, or a zero pixel size, returns `None` rather than 0. The
/// callers are all length checks of the form "does the guest's buffer hold a
/// whole image", and `0` would pass every one of them.
///
/// Here rather than in `backend::metal`, where it lived, for the reason
/// [`mip_extent`] gives: it is arithmetic both rails need, `backend::metal` is
/// behind a feature gate, and `runtime::compute_session` was already reaching
/// through that gate to borrow it.
pub fn tight_image_bytes(width: u32, height: u32, bytes_per_pixel: usize) -> Option<usize> {
    if width == 0 || height == 0 || bytes_per_pixel == 0 {
        return None;
    }
    (width as u64)
        .checked_mul(height as u64)?
        .checked_mul(bytes_per_pixel as u64)?
        .try_into()
        .ok()
}

/// [`tight_image_bytes`] for an array texture: `layers` slices, tightly packed.
///
/// Separate from [`tight_image_bytes`] rather than a `layers` parameter on it
/// because most callers size a single 2D image and a `1` threaded through every
/// one of them is a value that will eventually be wrong at one site. `layers ==
/// 0` is `None` for the reason the other axes are: an array texture with no
/// slices is a geometry no caller here can act on, and a `0` length passes every
/// "does the guest's buffer hold this" check there is.
///
/// The checked multiply is the point. `width as usize * height as usize` already
/// widens its operands, which reads as safe and is — but only just: two `u32`s
/// at their maximum multiply to a hair under `u64::MAX`, so **one more factor
/// overflows**, and both a layer count and a bytes-per-texel are exactly that
/// third factor. `backend::vulkan::engine::exec`'s `validate_v1` had both, where
/// an overflow panic would have aborted the process from inside the function
/// whose whole job is to survive a malformed request.
pub fn tight_layered_image_bytes(
    width: u32,
    height: u32,
    layers: u32,
    bytes_per_texel: usize,
) -> Option<usize> {
    if layers == 0 {
        return None;
    }
    (tight_image_bytes(width, height, bytes_per_texel)? as u64)
        .checked_mul(u64::from(layers))?
        .try_into()
        .ok()
}

/// [`tight_layered_image_bytes`] over a storage **block** grid.
///
/// The same product with the grid stated, so it covers a block-compressed image
/// as well as an uncompressed one: an uncompressed block is 1x1 and its `bytes`
/// is the bytes-per-texel, so this reduces to the sibling above for every format
/// that has one. Blocks round *up* on both axes, which is the contract — a 2x2
/// BC3 level still occupies one whole sixteen-byte block.
///
/// Zero on either axis is `None` for [`tight_image_bytes`]'s reason: a zero
/// length passes every "does the guest's buffer hold this" check there is.
pub fn tight_layered_block_bytes(
    width: u32,
    height: u32,
    layers: u32,
    block: crate::pixel_format::BlockGeometry,
) -> Option<usize> {
    if layers == 0 || width == 0 || height == 0 || block.bytes == 0 {
        return None;
    }
    let across = u64::from(block.blocks_across(width));
    let down = u64::from(block.block_rows(height));
    across
        .checked_mul(down)?
        .checked_mul(u64::from(block.bytes))?
        .checked_mul(u64::from(layers))?
        .try_into()
        .ok()
}

/// [`tight_image_bytes`] together with the row stride it implies, as one pair.
///
/// For the callers that hand a buffer to a texel-copy API. Such a call is given
/// a row stride and a region, and reads `stride * height` bytes from whatever
/// pointer it receives — so the length of that allocation and the stride passed
/// alongside it are not two facts but one, and deriving them separately is what
/// lets them disagree.
///
/// They did. `runtime::draw::metal_icb` sized an ICB colour attachment's
/// staging from the *guest* attachment's bytes-per-pixel while creating a
/// BGRA8Unorm texture and passing `width * 4` as the stride, so for any format
/// narrower than four bytes ([`crate::pixel_format::R8_BPP`],
/// [`crate::pixel_format::RG8_BPP`]) Metal read past the end of the
/// buffer and copied whatever followed it on the host heap into a render target
/// the guest reads back. The writeback half of that same function computed the
/// length correctly, ten lines away. One quantity, two derivations, and only
/// one of them right.
///
/// Same `None` contract as [`tight_image_bytes`], and for the same reason: a
/// zero on any axis is not a zero-length image, it is a geometry no caller here
/// can act on.
pub fn tight_image_layout(width: u32, height: u32, bytes_per_pixel: u32) -> Option<(u32, usize)> {
    let total = tight_image_bytes(width, height, bytes_per_pixel as usize)?;
    Some((width.checked_mul(bytes_per_pixel)?, total))
}

/// One level of a tightly-packed mip pyramid: where it starts, how long it is,
/// and the extent it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipLevelSpan {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    /// Byte offset of this level from the start of the packed pyramid.
    pub offset: usize,
    /// [`tight_image_bytes`] of this level's extent.
    pub len: usize,
}

/// Every level of a tightly-packed mip pyramid, base first.
///
/// A pyramid staged for a host image is *two* facts that must agree: the byte
/// range each level occupies in one upload allocation, and the extent the copy
/// into that level names. Deriving them at the two ends of that journey is how
/// they come apart — the producer packs level 3 at one offset and the consumer
/// copies from another, and the result is a level holding a neighbour's texels,
/// which reads exactly like a texture whose upper levels were never written.
/// So both ends call this and neither computes an offset of its own.
///
/// Extents are [`mip_extent`], which is Metal's own rule, so a chain built here
/// matches the one a guest descriptor declares level for level. `None` for a
/// zero extent, a zero pixel size, no levels, or any overflow — the same
/// contract [`tight_image_bytes`] states, and for its reason.
pub fn tight_pyramid_spans(
    width: u32,
    height: u32,
    levels: u32,
    bytes_per_pixel: usize,
) -> Option<Vec<MipLevelSpan>> {
    if levels == 0 {
        return None;
    }
    let mut spans = Vec::with_capacity(levels as usize);
    let mut offset: usize = 0;
    for level in 0..levels {
        let w = mip_extent(width, level);
        let h = mip_extent(height, level);
        let len = tight_image_bytes(w, h, bytes_per_pixel)?;
        spans.push(MipLevelSpan {
            level,
            width: w,
            height: h,
            offset,
            len,
        });
        offset = offset.checked_add(len)?;
    }
    Some(spans)
}

/// Total bytes of [`tight_pyramid_spans`], without building the level list.
pub fn tight_pyramid_bytes(
    width: u32,
    height: u32,
    levels: u32,
    bytes_per_pixel: usize,
) -> Option<usize> {
    let spans = tight_pyramid_spans(width, height, levels, bytes_per_pixel)?;
    let last = spans.last()?;
    last.offset.checked_add(last.len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel_format::{R8_BPP, RG8_BPP, RGBA8_BPP};

    /// The pyramid's byte layout and its level extents come from one call, so a
    /// producer and a consumer of the same staged upload cannot disagree about
    /// where level `n` begins.
    #[test]
    fn a_packed_pyramid_places_each_level_after_the_one_above_it() {
        let spans = tight_pyramid_spans(64, 64, 7, RGBA8_BPP as usize).expect("7 levels of 64x64");
        let dims: Vec<(u32, u32)> = spans.iter().map(|s| (s.width, s.height)).collect();
        assert_eq!(
            dims,
            vec![(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
        );
        let mut want_offset = 0usize;
        for span in &spans {
            assert_eq!(span.offset, want_offset, "level {} offset", span.level);
            assert_eq!(
                span.len,
                span.width as usize * span.height as usize * RGBA8_BPP as usize
            );
            want_offset += span.len;
        }
        assert_eq!(
            tight_pyramid_bytes(64, 64, 7, RGBA8_BPP as usize),
            Some(want_offset)
        );
    }

    /// A one-level pyramid is exactly the single image it describes, so the
    /// packed form is not a second layout for the overwhelmingly common case.
    #[test]
    fn a_single_level_pyramid_is_one_tight_image() {
        assert_eq!(
            tight_pyramid_bytes(37, 11, 1, RG8_BPP as usize),
            tight_image_bytes(37, 11, RG8_BPP as usize)
        );
    }

    /// A non-power-of-two chain floors and stops at 1 on each axis
    /// independently — the rule `mip_extent` states, carried into the layout.
    #[test]
    fn a_non_square_chain_bottoms_out_on_each_axis_separately() {
        let spans = tight_pyramid_spans(5, 1, 4, R8_BPP as usize).expect("4 levels");
        assert_eq!(
            spans
                .iter()
                .map(|s| (s.width, s.height))
                .collect::<Vec<_>>(),
            vec![(5, 1), (2, 1), (1, 1), (1, 1)]
        );
    }

    /// Zero levels is not an empty pyramid; it is a geometry no caller can act
    /// on, and a zero length would pass every "does the guest's buffer hold
    /// this" check there is.
    #[test]
    fn a_pyramid_of_no_levels_has_no_layout() {
        assert_eq!(tight_pyramid_spans(8, 8, 0, RGBA8_BPP as usize), None);
        assert_eq!(tight_pyramid_bytes(8, 8, 0, RGBA8_BPP as usize), None);
        assert_eq!(tight_pyramid_spans(0, 8, 3, RGBA8_BPP as usize), None);
    }

    /// The length a texel copy is allowed to read is exactly `stride * height`,
    /// and both come from one call, so no caller can pair one format's stride
    /// with another format's length.
    ///
    /// The regression this pins: an ICB colour attachment at `R8_BPP` sized its
    /// staging `width * height * 1` while the texture was BGRA8Unorm and the
    /// copy was told `width * 4`. At 1920x1080 that is a 2,073,600-byte
    /// allocation read as 8,294,400 bytes — 6.2 MB of whatever followed it on
    /// the host heap, copied into a surface the guest reads back.
    #[test]
    fn a_tight_image_layout_pairs_a_length_with_the_stride_that_reads_it() {
        for bpp in [R8_BPP, RG8_BPP, RGBA8_BPP] {
            let (stride, len) = tight_image_layout(1920, 1080, bpp).expect("valid geometry");
            assert_eq!(stride, 1920 * bpp);
            assert_eq!(
                len,
                stride as usize * 1080,
                "the length must be exactly what a copy at this stride reads"
            );
        }
        // The defect in one line: a BGRA8 copy's stride against an R8 length.
        let (bgra_stride, _) = tight_image_layout(1920, 1080, RGBA8_BPP).expect("valid");
        let (_, r8_len) = tight_image_layout(1920, 1080, R8_BPP).expect("valid");
        assert!(
            bgra_stride as usize * 1080 > r8_len,
            "pairing one format's stride with another's length reads past the end"
        );
    }

    /// Two `u32` maxima all but exhaust a `u64`, so any third factor overflows.
    ///
    /// The measurement behind both this function and
    /// [`tight_image_bytes`]: `u32::MAX * u32::MAX` is 18446744065119617025
    /// against a `u64::MAX` of 18446744073709551615 — under it by less than nine
    /// billion, which is to say by nothing. That is why widening the operands is
    /// not the fix here and a checked multiply is: `w as usize * h as usize`
    /// looks careful and survives only because nothing else is multiplied in.
    /// A layer count or a bytes-per-texel is that something else.
    #[test]
    fn a_third_factor_is_refused_rather_than_wrapped() {
        assert_eq!(
            u64::from(u32::MAX) * u64::from(u32::MAX),
            18_446_744_065_119_617_025,
            "the headroom this function exists because of"
        );
        // Two axes alone still fit, at one byte per texel.
        assert!(tight_image_bytes(u32::MAX, u32::MAX, 1).is_some());
        // A third factor does not, however it arrives.
        assert_eq!(tight_image_bytes(u32::MAX, u32::MAX, 2), None);
        assert_eq!(
            tight_layered_image_bytes(u32::MAX, u32::MAX, 2, 1),
            None,
            "a second layer is the same third factor"
        );
        assert_eq!(
            tight_layered_image_bytes(65536, 65536, 1, 4),
            Some(65536 * 65536 * 4),
            "the geometry that wraps a u32 product is representable here"
        );
    }

    /// A zero on any axis, including the layer count, is not a zero-length image.
    ///
    /// Every caller is a check of the form "does the guest's buffer hold this",
    /// and a `Some(0)` passes all of them.
    #[test]
    fn a_layered_image_with_no_slices_has_no_length() {
        assert_eq!(tight_layered_image_bytes(64, 64, 0, 4), None);
        assert_eq!(tight_layered_image_bytes(0, 64, 6, 4), None);
        assert_eq!(tight_layered_image_bytes(64, 0, 6, 4), None);
        assert_eq!(tight_layered_image_bytes(64, 64, 6, 0), None);
        assert_eq!(
            tight_layered_image_bytes(64, 64, 6, 4),
            Some(64 * 64 * 6 * 4)
        );
    }

    /// A degenerate geometry is `None`, not a zero-length pair.
    ///
    /// Inherited from [`tight_image_bytes`] deliberately: a caller allocating a
    /// staging buffer and a caller length-checking a guest buffer both need a
    /// zero here to be unusable rather than trivially satisfied.
    #[test]
    fn a_tight_image_layout_refuses_a_degenerate_geometry() {
        assert_eq!(tight_image_layout(0, 1080, RGBA8_BPP), None);
        assert_eq!(tight_image_layout(1920, 0, RGBA8_BPP), None);
        assert_eq!(tight_image_layout(1920, 1080, 0), None);
        // A stride that does not fit a u32 is refused rather than wrapped:
        // `1 << 30` texels at four bytes is exactly `1 << 32`, and the wrapping
        // answer is a stride of *zero* — an allocation of nothing handed to a
        // copy that still reads a full image.
        assert_eq!(
            (1u32 << 30).wrapping_mul(RGBA8_BPP),
            0,
            "the wrapping answer here is zero, which is what makes this the case to pin"
        );
        assert_eq!(tight_image_layout(1 << 30, 16, RGBA8_BPP), None);
    }

    #[test]
    fn a_mip_level_halves_and_floors_at_one() {
        // An axis with no texels has no levels.
        assert_eq!(mip_extent(0, 0), 0);
        assert_eq!(mip_extent(0, 3), 0);

        // Power-of-two base halves each level and floors at 1, never 0.
        assert_eq!(mip_extent(8, 0), 8);
        assert_eq!(mip_extent(8, 1), 4);
        assert_eq!(mip_extent(8, 2), 2);
        assert_eq!(mip_extent(8, 3), 1);
        assert_eq!(mip_extent(8, 4), 1, "past the last level clamps to 1");
        assert_eq!(mip_extent(8, 20), 1, "huge level never underflows to 0");

        // Past the width of the type. `20` above is inside it and says nothing
        // about these: `base >> 32` is a panic in a debug build and answers
        // `base` unchanged in a release one, so a level here used to give the
        // *base* extent — a full-size read charged to the smallest level of the
        // chain. The three boundary values, plus a base wide enough that a
        // masked shift would be obvious.
        assert_eq!(mip_extent(1024, 31), 1);
        assert_eq!(mip_extent(1024, 32), 1, "at the width, not through it");
        assert_eq!(mip_extent(1024, 33), 1);
        assert_eq!(mip_extent(u32::MAX, u32::MAX), 1);
        assert_eq!(mip_extent(0, u32::MAX), 0, "no texels, still no levels");

        // Non-power-of-two base right-shifts (floors), matching Metal.
        assert_eq!(mip_extent(5, 1), 2);
        assert_eq!(mip_extent(5, 2), 1);
        assert_eq!(mip_extent(3, 1), 1);
        assert_eq!(mip_extent(100, 1), 50);
        assert_eq!(mip_extent(100, 2), 25);
        assert_eq!(mip_extent(100, 3), 12);
    }

    /// Moved here from `backend::metal::util`, where it could only run on an
    /// Apple host. Every case is one the callers actually guard against.
    #[test]
    fn a_tight_image_is_the_product_and_an_empty_one_has_no_length() {
        assert_eq!(tight_image_bytes(2, 3, 4), Some(24));
        assert_eq!(tight_image_bytes(2, 3, 8), Some(48));
        assert_eq!(tight_image_bytes(1, 1, 1), Some(1));

        // None, not Some(0) — a zero would satisfy every `len >= expected`
        // check the callers make.
        assert_eq!(tight_image_bytes(0, 3, 4), None);
        assert_eq!(tight_image_bytes(2, 0, 4), None);
        assert_eq!(tight_image_bytes(2, 3, 0), None);

        // The product is taken in u64 and only then narrowed, so a geometry
        // that overflows `usize` declines instead of wrapping.
        assert_eq!(tight_image_bytes(u32::MAX, u32::MAX, usize::MAX), None);
    }
}
