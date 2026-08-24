//! Turning a scattered guest window into image-copy rectangles.
//!
//! # Why this is a module and not four lines at the copy site
//!
//! A guest surface's pages are not one contiguous stretch. A driven x86 boot
//! measured every window at four 4 KiB pages per run — the guest backs a
//! surface in 16 KiB physically-contiguous granules — so a 1920x1080 writeback
//! is 2025 pages in 507 runs, and that is the steady state rather than a
//! fragmentation artifact. See the device's `runtime::guest_ram_map` and its
//! `MapRefusal::Scattered` reporting.
//!
//! Each run is bindable on its own, but a run is a range of *bytes* and
//! `vkCmdCopyImageToBuffer` names *rectangles*. A 16 KiB run against a
//! 7680-byte row pitch covers two-and-a-bit rows starting part-way along one,
//! which is not a rectangle. Converting between the two is the whole content
//! of this module.
//!
//! It names nothing from `ash` and nothing from `metal` on purpose: a region
//! here is four `u32`s and an offset, and the backend can build its own
//! descriptor from one. Pure logic in a gated backend tree is logic nobody on
//! a Linux host ever runs, and this crate is what lets the tests below run on
//! every arm instead of on none.
//!
//! # What a driven boot measured
//!
//! x86 PCI attach, Safari window-drag probe, 25 s, quiesced machine, one
//! `vk_caps` so one boot. Against the same probe on the build that refused
//! scattered windows outright:
//!
//! - `render_flush_gpu_declined` 23 → **0**, and `render_flush_leased` → 0. The
//!   writeback no longer falls back for a full-screen surface, which is what
//!   this module exists to make possible.
//! - Armed deferred-writeback windows peaked at 656 → **288**. §5's target is
//!   zero on a UMA host; this is progress toward it and not arrival.
//! - `gpu_writeback_too_many_regions` never fired. That decline and the
//!   `MAX_GUEST_COPY_REGIONS` behind it are now retired: a region ceiling on
//!   this rail refused whole frames rather than degrading them, and the count
//!   it bounded is one `plan_regions` derives from the window's own geometry.
//!   `guest_write_regions` is what reports the width instead.
//! - The desktop renders correctly — wallpaper, dock and a Safari window, no
//!   banding or tearing. One screenshot of a near-static desktop is evidence
//!   against gross corruption and is *not* a regression gate.
//!
//! **It did not pay in throughput**, and that is what moved the work off this
//! module. Present cadence fell (median 18.95 → 16.50 Hz) and `fence_us` rose
//! (592 → 945 µs) — roughly what ~1500 copy rectangles per flush cost the GPU
//! against one. The answer was to stop making the bus-crossing pass detile as
//! well: the device's Vulkan engine now copies a dense frame into a
//! device-local scratch as one rectangle and scatters it with one plain
//! `VkBufferCopy` per stretch, which took the same probe to 24.50 Hz and 480 µs
//! with the declines still at zero.
//!
//! # What still comes here, and why it is not dead
//!
//! That linear form cannot express a window whose rows carry padding: a run's
//! bytes then include inter-row bytes the copy must not write, and a byte range
//! has no way to skip them. Rectangles do, so a padded window still plans here.
//! A driven boot measured **35 such windows against 9186 dense ones** — 0.4 %,
//! and not zero. Deleting this because the common case moved would lose those
//! frames, or write padding the copying rail leaves alone, which is the
//! two-rails-disagree divergence the section below exists to prevent.
//!
//! Timings are wall clock on a shared machine and are upper bounds; the counts
//! are not.
//!
//! # What the caller still owes
//!
//! Padding is skipped, not written. A guest row pitch wider than the frame
//! leaves inter-row bytes that belong to the surface's plane but are not texels
//! this copy was given, and the copying rail does not write them either. Two
//! rails landing different guest memory for one frame is the divergence that
//! matters here, so the rule is stated once, in [`plan_regions`], and both
//! rails inherit it from the same walk.

use alloc::vec::Vec;

/// One rectangle of the frame, and where its first byte lives in the window.
///
/// `window_offset` is a byte offset from the first byte of the requested
/// window — not from a page, not from the import, and not from the resource.
/// The caller adds whatever base its own binding uses; nothing here knows one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRegion {
    /// Byte offset of this rectangle's first texel within the window.
    pub window_offset: u64,
    /// Texel column of the rectangle's left edge.
    pub x: u32,
    /// Texel row of the rectangle's top edge.
    pub y: u32,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

/// Geometry a window's bytes are laid out under, in the frame's own terms.
///
/// `pitch_bytes` is the guest's row stride and `width_texels` the frame's
/// width; they differ exactly when the guest's rows carry padding. Both are
/// taken rather than derived so this module never has to guess which of the two
/// a caller meant — the two have been conflated before, and a copy that reads
/// `pitch` as `width` writes the padding.
#[derive(Clone, Copy, Debug)]
pub struct WindowGeometry {
    pub pitch_bytes: u64,
    pub width_texels: u32,
    pub height_texels: u32,
}

/// Bytes per texel. This rail is BGRA8 only — the device's `mapping_write`
/// refuses any other format by name before reaching here, so widening this is
/// a format decision made there and not a constant to generalise on spec.
const BYTES_PER_TEXEL: u64 = 4;

impl WindowGeometry {
    /// One past the last byte the frame's texels occupy: the last texel of the
    /// last row, with no trailing padding.
    pub fn extent_end(&self) -> u64 {
        if self.height_texels == 0 || self.width_texels == 0 {
            return 0;
        }
        u64::from(self.height_texels - 1) * self.pitch_bytes
            + u64::from(self.width_texels) * BYTES_PER_TEXEL
    }
}

/// Rectangles covering the texels of `[start, end)`, in ascending window order.
///
/// `start` and `end` are byte offsets within the window. Bytes that fall in a
/// row's padding are **not** covered, so the returned rectangles can total less
/// than `end - start`; that is the point rather than a shortfall.
///
/// Consecutive whole rows are merged into one rectangle. That is not a
/// micro-optimisation: a 16 KiB run against a 7680-byte pitch is two whole rows
/// and two fragments, so merging is the difference between three rectangles per
/// run and four, over five hundred runs. It is only valid because the copy
/// descriptor carries the row stride separately — the caller must set its
/// buffer row length to `pitch_bytes / 4` for a merged rectangle to name the
/// bytes this function thinks it names.
pub fn plan_regions(geom: &WindowGeometry, start: u64, end: u64) -> Vec<WindowRegion> {
    let mut out = Vec::new();
    if geom.pitch_bytes == 0 || geom.width_texels == 0 || geom.height_texels == 0 {
        return out;
    }
    let row_texel_bytes = u64::from(geom.width_texels) * BYTES_PER_TEXEL;
    // Never past the frame: a window longer than the texels it describes must
    // not turn its tail into rows that do not exist.
    let end = end.min(geom.extent_end());
    if start >= end {
        return out;
    }
    let first_row = start / geom.pitch_bytes;
    let last_row = (end - 1) / geom.pitch_bytes;

    for row in first_row..=last_row {
        if row >= u64::from(geom.height_texels) {
            break;
        }
        let row_start = row * geom.pitch_bytes;
        // Intersect the requested byte range with this row's *texels*, which is
        // what drops the padding without a second branch for it.
        let seg_start = start.max(row_start);
        let seg_end = end.min(row_start + row_texel_bytes);
        if seg_start >= seg_end {
            continue;
        }
        let x = (seg_start - row_start) / BYTES_PER_TEXEL;
        let width = (seg_end - seg_start) / BYTES_PER_TEXEL;
        if width == 0 {
            continue;
        }
        // Extend the previous rectangle when this row continues it: both must
        // be whole rows, and they must be adjacent.
        //
        // The adjacency term is redundant *today* and is kept deliberately.
        // This loop visits rows consecutively and a middle row can never be
        // skipped — a contiguous byte range that covered row N in full and
        // continues past it necessarily intersects row N+1's texels — so
        // `prev.y + prev.height == row` always holds where the other two terms
        // do. Removing it was probed and broke no test, which is precisely why
        // the note is here rather than the term being dropped: the day someone
        // adds a `continue` to this loop, merging across the gap would silently
        // claim rows the range never covered, and no coverage test built from
        // contiguous ranges would see it.
        if let Some(prev) = out.last_mut() {
            let prev: &mut WindowRegion = prev;
            let whole_row = x == 0 && width == u64::from(geom.width_texels);
            let prev_whole_row = prev.x == 0 && prev.width == geom.width_texels;
            let adjacent = u64::from(prev.y) + u64::from(prev.height) == row;
            if whole_row && prev_whole_row && adjacent {
                prev.height += 1;
                continue;
            }
        }
        out.push(WindowRegion {
            window_offset: seg_start,
            x: x as u32,
            y: row as u32,
            width: width as u32,
            height: 1,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn tight(width: u32, height: u32) -> WindowGeometry {
        WindowGeometry {
            pitch_bytes: u64::from(width) * BYTES_PER_TEXEL,
            width_texels: width,
            height_texels: height,
        }
    }

    /// The whole frame in one range is one rectangle, because every row is
    /// whole and adjacent.
    ///
    /// This is the case that already worked before scattering was handled, and
    /// it must not become five hundred rectangles just because the code that
    /// can produce them now exists.
    #[test]
    fn a_full_tight_frame_is_a_single_rectangle() {
        let g = tight(1920, 1080);
        let regions = plan_regions(&g, 0, g.extent_end());
        assert_eq!(
            regions,
            vec![WindowRegion {
                window_offset: 0,
                x: 0,
                y: 0,
                width: 1920,
                height: 1080
            }]
        );
    }

    /// Every texel of the frame is covered exactly once, whatever the runs.
    ///
    /// The invariant, walked rather than spelled out as expected tuples. A
    /// texel covered twice is written from two source offsets and the winner is
    /// whichever copy retires last; a texel covered zero times keeps the
    /// previous frame's content while its neighbours update, which is the
    /// banding this rail exists to avoid. Both are silent.
    ///
    /// Driven at the measured shape — 16 KiB runs against a 1920-wide frame —
    /// because that is the geometry the guest actually produces, and a
    /// hand-picked one would be the fixture agreeing with itself.
    #[test]
    fn the_measured_run_shape_covers_every_texel_exactly_once() {
        const RUN: u64 = 16 * 1024;
        for &(w, h) in &[(1920u32, 1080u32), (800, 600), (17, 5), (1, 1)] {
            let g = tight(w, h);
            let mut seen = vec![0u32; (w as usize) * (h as usize)];
            let mut offset = 0u64;
            while offset < g.extent_end() {
                let end = (offset + RUN).min(g.extent_end());
                for r in plan_regions(&g, offset, end) {
                    for row in r.y..r.y + r.height {
                        for col in r.x..r.x + r.width {
                            seen[(row as usize) * (w as usize) + col as usize] += 1;
                        }
                    }
                }
                offset = end;
            }
            assert!(
                seen.iter().all(|&n| n == 1),
                "{w}x{h}: every texel must be covered exactly once, found {:?}",
                seen.iter().filter(|&&n| n != 1).count()
            );
        }
    }

    /// A window offset maps back to the byte the rectangle's first texel came
    /// from, for every rectangle.
    ///
    /// The offset and the (x, y) are computed together here and consumed apart
    /// — one becomes the copy's buffer offset, the other its image offset — so
    /// a disagreement between them lands the right pixels at the wrong place,
    /// which no coverage count above would notice.
    #[test]
    fn each_rectangles_offset_names_the_byte_its_first_texel_sits_at() {
        let g = WindowGeometry {
            pitch_bytes: 7808, // 1920 texels plus 128 bytes of padding
            width_texels: 1920,
            height_texels: 64,
        };
        for r in plan_regions(&g, 5000, 40_000) {
            let expected = u64::from(r.y) * g.pitch_bytes + u64::from(r.x) * BYTES_PER_TEXEL;
            assert_eq!(
                r.window_offset, expected,
                "rectangle at ({}, {}) claims offset {}",
                r.x, r.y, r.window_offset
            );
        }
    }

    /// Padding is skipped rather than written.
    ///
    /// The guest's own content in a padded row's tail must survive the flush —
    /// the copying rail writes row by row and skips it, and two rails landing
    /// different guest memory for one frame is the divergence that matters. So
    /// no rectangle may name a texel past the frame's width.
    #[test]
    fn a_padded_pitch_never_covers_the_padding() {
        let g = WindowGeometry {
            pitch_bytes: 4096,
            width_texels: 1000, // 4000 bytes of texels, 96 bytes of padding
            height_texels: 8,
        };
        let regions = plan_regions(&g, 0, g.extent_end());
        assert!(!regions.is_empty());
        for r in &regions {
            assert!(
                r.x + r.width <= g.width_texels,
                "{r:?} reaches into the padding"
            );
        }
        // A range lying wholly inside one row's padding covers nothing at all,
        // rather than covering a zero-width rectangle the driver would reject.
        assert!(plan_regions(&g, 4000, 4096).is_empty());
    }

    /// A range past the frame's last texel contributes nothing.
    ///
    /// The window's final page reaches past the last row whenever the extent is
    /// not a whole page, which is the normal case. Turning that tail into a row
    /// that does not exist would write past the resource.
    #[test]
    fn the_tail_past_the_last_row_is_not_turned_into_a_row() {
        let g = tight(64, 4);
        let past = g.extent_end();
        assert!(plan_regions(&g, past, past + 4096).is_empty());
        // And a range straddling the end stops at it.
        let regions = plan_regions(&g, past - 8, past + 4096);
        let covered: u64 = regions
            .iter()
            .map(|r| u64::from(r.width) * u64::from(r.height) * BYTES_PER_TEXEL)
            .sum();
        assert_eq!(covered, 8);
    }

    /// Degenerate geometry produces no rectangles rather than a panic or a
    /// rectangle of nothing.
    #[test]
    fn degenerate_geometry_plans_nothing() {
        let zero_pitch = WindowGeometry {
            pitch_bytes: 0,
            width_texels: 4,
            height_texels: 4,
        };
        assert!(plan_regions(&zero_pitch, 0, 64).is_empty());
        assert!(plan_regions(&tight(0, 4), 0, 64).is_empty());
        assert!(plan_regions(&tight(4, 0), 0, 64).is_empty());
        // An inverted or empty range is not a copy.
        let g = tight(16, 16);
        assert!(plan_regions(&g, 100, 100).is_empty());
        assert!(plan_regions(&g, 200, 100).is_empty());
    }
}
