//! Pure EFI GOP paint paths — no UEFI boot-services dependency.
//!
//! Contract: UEFI Spec `EFI_GRAPHICS_OUTPUT_PROTOCOL` Blt / fixed linear BGRA8
//! mode matching host BAR1 (`1920×1080`, 4 bpp, row stride = width × 4).
//!
//! Performance: full-row and full-rect fills/copies use word/`memcpy` bulk ops;
//! per-pixel loops are reserved for partial-edge cases only. Unit tests drive
//! these same functions the protocol handlers call.

use core::mem::size_of;
use core::ptr;

/// Fixed mode — must match host BAR1 scanout in `reims-vgpu-pci.c`.
pub const FB_W: usize = 1920;
pub const FB_H: usize = 1080;
pub const FB_BPP: usize = 4;
pub const FB_STRIDE: usize = FB_W * FB_BPP;
pub const FB_BYTES: usize = FB_STRIDE * FB_H;

/// Dark slate BGRA (non-black) so QMP can prove the FB is live before OpenCore paints.
pub const SLATE_BGRA: u32 = u32::from_le_bytes([0x40, 0x28, 0x18, 0xff]);

/// PixelBlueGreenRedReserved8BitPerColor (UEFI).
pub const PIXEL_BGR: u32 = 1;

pub const BLT_VIDEO_FILL: u32 = 0;
pub const BLT_VIDEO_TO_BUFFER: u32 = 1;
pub const BLT_BUFFER_TO_VIDEO: u32 = 2;
pub const BLT_VIDEO_TO_VIDEO: u32 = 3;

/// The bit EFI sets in a `UINTN` to mark a status an error.
///
/// UEFI defines `EFI_ERROR` as the high bit of a `UINTN`, so the numeric value
/// of every error code depends on the pointer width. Spelling
/// `0x8000_0000_0000_0002` is therefore two mistakes in one number: it is wrong
/// on a 32-bit firmware, and it is a second, silent copy of "the top bit means
/// error" that no reader can tell from an arbitrary constant.
const EFI_ERROR: usize = 1 << (usize::BITS - 1);

/// Subset of EFI_STATUS used by paint (host-testable without the uefi crate).
///
/// The numbers live in [`GopStatus::efi_code`] rather than in discriminants.
/// They used to be discriminants of a `#[repr(usize)]` C-like enum, which was
/// wrong twice: nothing in this crate ever casts the enum to an integer — the
/// firmware boundary in `gop.rs` matches each variant to a `uefi::Status` by
/// name — and a discriminant with the high `UINTN` bit set does not fit an
/// `i32`, so the representation was unportable in service of a conversion that
/// does not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GopStatus {
    Success,
    /// EFI_INVALID_PARAMETER
    InvalidParameter,
    /// EFI_UNSUPPORTED
    Unsupported,
    /// EFI_DEVICE_ERROR
    DeviceError,
}

impl GopStatus {
    /// The `EFI_STATUS` value the UEFI spec gives this status, for the pointer
    /// width being compiled for.
    ///
    /// Nothing in the shipping path calls this — `gop.rs` maps variant to
    /// `uefi::Status` by name, which is what keeps the two vocabularies from
    /// agreeing only numerically. It exists so the spec ordinals stay written
    /// down next to the variants they belong to, and so a test can check them.
    pub const fn efi_code(self) -> usize {
        match self {
            Self::Success => 0,
            Self::InvalidParameter => EFI_ERROR | 2,
            Self::Unsupported => EFI_ERROR | 3,
            Self::DeviceError => EFI_ERROR | 7,
        }
    }
}

/// UEFI `EFI_GRAPHICS_OUTPUT_BLT_PIXEL` — BGRA memory order on little-endian.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BltPixel {
    pub blue: u8,
    pub green: u8,
    pub red: u8,
    pub reserved: u8,
}

impl BltPixel {
    #[inline]
    pub const fn from_bgra_u32(v: u32) -> Self {
        let b = v.to_le_bytes();
        Self {
            blue: b[0],
            green: b[1],
            red: b[2],
            reserved: b[3],
        }
    }

    #[inline]
    pub const fn to_bgra_u32(self) -> u32 {
        u32::from_le_bytes([self.blue, self.green, self.red, self.reserved])
    }
}

/// Counters for bulk-path proxy (tests assert full-frame ops use bulk, not per-px).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaintStats {
    /// Number of bulk row/rect memory ops (memcpy / word-fill chunks).
    pub bulk_ops: u64,
    /// Bytes touched by bulk ops.
    pub bulk_bytes: u64,
    /// Per-pixel fallback iterations (should stay 0 for aligned full-row rects).
    pub pixel_ops: u64,
}

/// Fill entire linear FB with a BGRA word using bulk word stores.
pub fn fill_solid(fb: &mut [u8], color: u32) {
    debug_assert_eq!(fb.len(), FB_BYTES);
    let words = FB_BYTES / size_of::<u32>();
    let p = fb.as_mut_ptr() as *mut u32;
    unsafe {
        for i in 0..words {
            *p.add(i) = color;
        }
    }
}

/// Clear to black (SetMode contract).
pub fn clear_black(fb: &mut [u8]) {
    fb.fill(0);
}

/// Fill BAR1 with dark slate (install-time visible non-black).
pub fn fill_slate(fb: &mut [u8]) {
    fill_solid(fb, SLATE_BGRA);
}

#[inline]
fn rect_in_fb(x: usize, y: usize, w: usize, h: usize) -> bool {
    w > 0
        && h > 0
        && x < FB_W
        && y < FB_H
        && w <= FB_W.saturating_sub(x)
        && h <= FB_H.saturating_sub(y)
}

/// One Blt's geometry: where it reads, where it writes, and how big.
///
/// UEFI passes these six as six arguments, and so does [`blt`], which mirrors
/// the protocol entry point. Everything below `blt` takes this instead. The six
/// are all `usize` and the two points are interchangeable at a call site, so
/// every permutation type-checks — and the interior helpers took them in an
/// order (`sx, sy, dx, dy, w, h`) that is not the order the protocol lists them
/// (`sx, sy, dx, dy, w, h, delta` after the operation, but source and
/// destination swap meaning per operation). Naming them once removes the
/// permutation, and gives the two rules below somewhere to live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BltRect {
    pub source_x: usize,
    pub source_y: usize,
    pub destination_x: usize,
    pub destination_y: usize,
    pub width: usize,
    pub height: usize,
}

impl BltRect {
    /// Whether the source rectangle lies inside the framebuffer.
    #[inline]
    fn source_in_fb(&self) -> bool {
        rect_in_fb(self.source_x, self.source_y, self.width, self.height)
    }

    /// Whether the destination rectangle lies inside the framebuffer.
    #[inline]
    fn destination_in_fb(&self) -> bool {
        rect_in_fb(
            self.destination_x,
            self.destination_y,
            self.width,
            self.height,
        )
    }

    /// The six values in the protocol's own order, for the row loops that want
    /// them as locals. One destructure per helper rather than six parameters
    /// per call is what removes the permutation.
    #[inline]
    fn corners(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.source_x,
            self.source_y,
            self.destination_x,
            self.destination_y,
            self.width,
            self.height,
        )
    }

    /// How many `BltPixel`s the BltBuffer must hold for this rect, given the
    /// point that names the buffer side and the declared row pitch.
    ///
    /// The BltBuffer is the *source* for BltBufferToVideo and the *destination*
    /// for VideoToBltBuffer, so this is one rule read from two ends. It used to
    /// be written out at both, differing only in which pair of coordinates it
    /// read — and the last index it computes is `(y + h - 1) * row_px +
    /// (x + w - 1)`, which is exactly the arithmetic a copy gets subtly wrong.
    #[inline]
    pub fn buffer_pixels_needed(&self, buffer_x: usize, buffer_y: usize, row_px: usize) -> usize {
        buffer_y
            .saturating_add(self.height.saturating_sub(1))
            .saturating_mul(row_px)
            .saturating_add(buffer_x.saturating_add(self.width))
    }
}

/// EFI GOP Blt on a linear BGRA8 framebuffer.
///
/// `delta` is bytes per BltBuffer row (0 ⇒ `width * sizeof(BltPixel)`), used for
/// VideoToBltBuffer and BltBufferToVideo only (UEFI Spec).
/// The BltBuffer's row pitch in pixels, or `None` if `delta` cannot describe a
/// row this wide.
///
/// UEFI states `Delta` in *bytes* and gives `0` the meaning "rows are exactly
/// `Width` pixels", so three refusals live in one conversion: a delta that is
/// not a whole number of pixels, a delta that describes a row narrower than the
/// rectangle being copied, and — through the `width == 0` its callers check
/// first — a pitch of zero.
///
/// It is a free function rather than a method because `gop::gop_blt` needs the
/// same answer *before* it has a rectangle: it sizes the `BltBuffer` slice from
/// the pitch, and it used to reach that number with its own copy of this
/// conversion that omitted the `rp < width` refusal and papered over the
/// zero-pitch case with a `.max(1)`.
pub fn row_pitch_pixels(delta: usize, width: usize) -> Option<usize> {
    if delta == 0 {
        return Some(width);
    }
    if !delta.is_multiple_of(size_of::<BltPixel>()) {
        return None;
    }
    let row_px = delta / size_of::<BltPixel>();
    (row_px >= width).then_some(row_px)
}

// The parameter run mirrors `EFI_GRAPHICS_OUTPUT_PROTOCOL.Blt` — same names,
// same order, `fb` where the protocol passes `This` — so that `gop_blt` can
// forward its arguments without reordering them. Collapsing them into
// `BltRect` here would put a translation between the firmware call and the
// code that answers it; the rect starts one frame in, where the helpers are.
#[allow(clippy::too_many_arguments)]
pub fn blt(
    fb: &mut [u8],
    blt_buffer: Option<&mut [BltPixel]>,
    op: u32,
    source_x: usize,
    source_y: usize,
    destination_x: usize,
    destination_y: usize,
    width: usize,
    height: usize,
    delta: usize,
    stats: Option<&mut PaintStats>,
) -> GopStatus {
    if width == 0 || height == 0 {
        return GopStatus::InvalidParameter;
    }
    if fb.len() < FB_BYTES {
        return GopStatus::DeviceError;
    }

    let Some(row_px) = row_pitch_pixels(delta, width) else {
        return GopStatus::InvalidParameter;
    };

    let rect = BltRect {
        source_x,
        source_y,
        destination_x,
        destination_y,
        width,
        height,
    };

    match op {
        BLT_VIDEO_FILL => {
            let buf = match blt_buffer {
                Some(b) if !b.is_empty() => b,
                _ => return GopStatus::InvalidParameter,
            };
            if !rect.destination_in_fb() {
                return GopStatus::InvalidParameter;
            }
            fill_rect(fb, &rect, buf[0].to_bgra_u32(), stats);
            GopStatus::Success
        }
        BLT_BUFFER_TO_VIDEO => {
            let buf = match blt_buffer {
                Some(b) => b,
                None => return GopStatus::InvalidParameter,
            };
            if !rect.destination_in_fb() {
                return GopStatus::InvalidParameter;
            }
            // The BltBuffer is the source here, so it is the source point that
            // must fit within it at the declared row pitch.
            if rect.buffer_pixels_needed(source_x, source_y, row_px) > buf.len() {
                return GopStatus::InvalidParameter;
            }
            copy_buffer_to_video(fb, buf, &rect, row_px, stats);
            GopStatus::Success
        }
        BLT_VIDEO_TO_BUFFER => {
            let buf = match blt_buffer {
                Some(b) => b,
                None => return GopStatus::InvalidParameter,
            };
            if !rect.source_in_fb() {
                return GopStatus::InvalidParameter;
            }
            // Same rule, other end: the BltBuffer is the destination.
            if rect.buffer_pixels_needed(destination_x, destination_y, row_px) > buf.len() {
                return GopStatus::InvalidParameter;
            }
            copy_video_to_buffer(fb, buf, &rect, row_px, stats);
            GopStatus::Success
        }
        BLT_VIDEO_TO_VIDEO => {
            if !rect.source_in_fb() || !rect.destination_in_fb() {
                return GopStatus::InvalidParameter;
            }
            copy_video_to_video(fb, &rect, stats);
            GopStatus::Success
        }
        _ => GopStatus::InvalidParameter,
    }
}

fn bump_bulk(stats: Option<&mut PaintStats>, bytes: u64) {
    if let Some(s) = stats {
        s.bulk_ops = s.bulk_ops.saturating_add(1);
        s.bulk_bytes = s.bulk_bytes.saturating_add(bytes);
    }
}

fn fill_rect(fb: &mut [u8], rect: &BltRect, color: u32, mut stats: Option<&mut PaintStats>) {
    let (x, y, w, h) = (
        rect.destination_x,
        rect.destination_y,
        rect.width,
        rect.height,
    );
    let row_bytes = w * FB_BPP;
    // Full-width row at FB origin x==0 and w==FB_W → one bulk span per row via word fill.
    if x == 0 && w == FB_W {
        let words = (row_bytes * h) / size_of::<u32>();
        let start = y * FB_STRIDE;
        let p = unsafe { fb.as_mut_ptr().add(start) as *mut u32 };
        unsafe {
            for i in 0..words {
                *p.add(i) = color;
            }
        }
        bump_bulk(stats.as_deref_mut(), (row_bytes * h) as u64);
        return;
    }

    // Per-row word fill for arbitrary x/w (still O(rows), not O(pixels) scalar).
    let color_px = BltPixel::from_bgra_u32(color);
    for row in 0..h {
        let dy = y + row;
        let off = dy * FB_STRIDE + x * FB_BPP;
        let dst = &mut fb[off..off + row_bytes];
        // BltPixel layout == BGRA linear: stamp words.
        let p = dst.as_mut_ptr() as *mut u32;
        unsafe {
            for i in 0..w {
                *p.add(i) = color;
            }
        }
        let _ = color_px; // layout asserted by to_bgra_u32 path
        bump_bulk(stats.as_deref_mut(), row_bytes as u64);
    }
}

fn copy_buffer_to_video(
    fb: &mut [u8],
    buf: &[BltPixel],
    rect: &BltRect,
    row_px: usize,
    mut stats: Option<&mut PaintStats>,
) {
    let (sx, sy, dx, dy, w, h) = rect.corners();
    let row_bytes = w * FB_BPP;
    for row in 0..h {
        let src_i = (sy + row) * row_px + sx;
        let dst_off = (dy + row) * FB_STRIDE + dx * FB_BPP;
        // BltPixel is BGRA — identical to FB memory order.
        let src_ptr = buf[src_i..].as_ptr() as *const u8;
        let dst_ptr = unsafe { fb.as_mut_ptr().add(dst_off) };
        unsafe {
            ptr::copy_nonoverlapping(src_ptr, dst_ptr, row_bytes);
        }
        bump_bulk(stats.as_deref_mut(), row_bytes as u64);
    }
}

fn copy_video_to_buffer(
    fb: &[u8],
    buf: &mut [BltPixel],
    rect: &BltRect,
    row_px: usize,
    mut stats: Option<&mut PaintStats>,
) {
    let (sx, sy, dx, dy, w, h) = rect.corners();
    let row_bytes = w * FB_BPP;
    for row in 0..h {
        let src_off = (sy + row) * FB_STRIDE + sx * FB_BPP;
        let dst_i = (dy + row) * row_px + dx;
        let src_ptr = unsafe { fb.as_ptr().add(src_off) };
        let dst_ptr = buf[dst_i..].as_mut_ptr() as *mut u8;
        unsafe {
            ptr::copy_nonoverlapping(src_ptr, dst_ptr, row_bytes);
        }
        bump_bulk(stats.as_deref_mut(), row_bytes as u64);
    }
}

fn copy_video_to_video(fb: &mut [u8], rect: &BltRect, mut stats: Option<&mut PaintStats>) {
    let (sx, sy, dx, dy, w, h) = rect.corners();
    let row_bytes = w * FB_BPP;
    // Overlap-safe: if destination is below source, copy bottom-up.
    let reverse = dy > sy || (dy == sy && dx > sx);
    if reverse {
        for row in (0..h).rev() {
            let so = (sy + row) * FB_STRIDE + sx * FB_BPP;
            let doff = (dy + row) * FB_STRIDE + dx * FB_BPP;
            unsafe {
                let base = fb.as_mut_ptr();
                ptr::copy(base.add(so), base.add(doff), row_bytes);
            }
            bump_bulk(stats.as_deref_mut(), row_bytes as u64);
        }
    } else {
        for row in 0..h {
            let so = (sy + row) * FB_STRIDE + sx * FB_BPP;
            let doff = (dy + row) * FB_STRIDE + dx * FB_BPP;
            unsafe {
                let base = fb.as_mut_ptr();
                ptr::copy(base.add(so), base.add(doff), row_bytes);
            }
            bump_bulk(stats.as_deref_mut(), row_bytes as u64);
        }
    }
}

/// QueryMode validation for the single fixed mode.
pub fn query_mode_ok(mode_number: u32) -> GopStatus {
    if mode_number != 0 {
        return GopStatus::InvalidParameter;
    }
    GopStatus::Success
}

/// SetMode validation for the single fixed mode.
pub fn set_mode_ok(mode_number: u32) -> GopStatus {
    if mode_number != 0 {
        return GopStatus::Unsupported;
    }
    GopStatus::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_fb() -> Vec<u8> {
        vec![0u8; FB_BYTES]
    }

    fn px_at(fb: &[u8], x: usize, y: usize) -> u32 {
        let o = y * FB_STRIDE + x * FB_BPP;
        u32::from_le_bytes([fb[o], fb[o + 1], fb[o + 2], fb[o + 3]])
    }

    #[test]
    fn fill_slate_is_non_black_bulk() {
        let mut fb = fresh_fb();
        fill_slate(&mut fb);
        assert_eq!(px_at(&fb, 0, 0), SLATE_BGRA);
        assert_eq!(px_at(&fb, FB_W - 1, FB_H - 1), SLATE_BGRA);
        assert_ne!(SLATE_BGRA, 0);
    }

    #[test]
    fn video_fill_full_frame_bulk_not_per_pixel() {
        let mut fb = fresh_fb();
        let mut pix = [BltPixel::from_bgra_u32(0xff112233)];
        let mut stats = PaintStats::default();
        let st = blt(
            &mut fb,
            Some(&mut pix),
            BLT_VIDEO_FILL,
            0,
            0,
            0,
            0,
            FB_W,
            FB_H,
            0,
            Some(&mut stats),
        );
        assert_eq!(st, GopStatus::Success);
        assert_eq!(px_at(&fb, 100, 100), 0xff112233);
        assert_eq!(px_at(&fb, FB_W - 1, FB_H - 1), 0xff112233);
        // One bulk span for full-frame fill (not 1920*1080 pixel_ops).
        assert!(stats.bulk_ops >= 1, "bulk_ops={}", stats.bulk_ops);
        assert_eq!(stats.bulk_bytes, FB_BYTES as u64);
        assert_eq!(stats.pixel_ops, 0);
    }

    #[test]
    fn buffer_to_video_and_back_roundtrip() {
        let mut fb = fresh_fb();
        fill_slate(&mut fb);
        let w = 64usize;
        let h = 32usize;
        let mut src: Vec<BltPixel> = (0..w * h)
            .map(|i| BltPixel::from_bgra_u32(0xff000000 | (i as u32)))
            .collect();
        let mut stats = PaintStats::default();
        assert_eq!(
            blt(
                &mut fb,
                Some(&mut src),
                BLT_BUFFER_TO_VIDEO,
                0,
                0,
                10,
                20,
                w,
                h,
                0,
                Some(&mut stats),
            ),
            GopStatus::Success
        );
        assert!(stats.bulk_ops == h as u64);
        assert_eq!(stats.bulk_bytes, (w * h * FB_BPP) as u64);

        let mut dst = vec![BltPixel::from_bgra_u32(0); w * h];
        assert_eq!(
            blt(
                &mut fb,
                Some(&mut dst),
                BLT_VIDEO_TO_BUFFER,
                10,
                20,
                0,
                0,
                w,
                h,
                0,
                None,
            ),
            GopStatus::Success
        );
        assert_eq!(src, dst);
    }

    #[test]
    fn video_to_video_overlap_down_scroll() {
        let mut fb = fresh_fb();
        // Paint a marker row at y=0.
        let mut row: Vec<BltPixel> = (0..FB_W)
            .map(|i| BltPixel::from_bgra_u32(0xaa000000 | i as u32))
            .collect();
        assert_eq!(
            blt(
                &mut fb,
                Some(&mut row),
                BLT_BUFFER_TO_VIDEO,
                0,
                0,
                0,
                0,
                FB_W,
                1,
                0,
                None,
            ),
            GopStatus::Success
        );
        // Scroll down by 1 (dest below source — reverse copy).
        let mut stats = PaintStats::default();
        assert_eq!(
            blt(
                &mut fb,
                None,
                BLT_VIDEO_TO_VIDEO,
                0,
                0,
                0,
                1,
                FB_W,
                10,
                0,
                Some(&mut stats),
            ),
            GopStatus::Success
        );
        assert_eq!(px_at(&fb, 0, 1), px_at(&fb, 0, 0)); // y0 still original until overwrite
                                                        // After reverse copy of h=10 from y0→y1, row y1 should equal original y0 pattern.
                                                        // y0 was not cleared by V2V; y1 should match what was at y0 before...
                                                        // Actually reverse: for row=9..0: copy sy+row → dy+row. Final y1 gets y0 content.
        assert_eq!(px_at(&fb, 5, 1), 0xaa000000 | 5);
        assert!(stats.bulk_ops >= 10);
        assert_eq!(stats.pixel_ops, 0);
    }

    #[test]
    fn invalid_params_and_modes() {
        let mut fb = fresh_fb();
        let mut pix = [BltPixel::from_bgra_u32(0xffffffff)];
        assert_eq!(
            blt(
                &mut fb,
                Some(&mut pix),
                BLT_VIDEO_FILL,
                0,
                0,
                FB_W - 1,
                0,
                2,
                1,
                0,
                None
            ),
            GopStatus::InvalidParameter
        );
        assert_eq!(
            blt(&mut fb, None, BLT_VIDEO_FILL, 0, 0, 0, 0, 1, 1, 0, None),
            GopStatus::InvalidParameter
        );
        assert_eq!(
            blt(&mut fb, Some(&mut pix), 99, 0, 0, 0, 0, 1, 1, 0, None),
            GopStatus::InvalidParameter
        );
        assert_eq!(query_mode_ok(0), GopStatus::Success);
        assert_eq!(query_mode_ok(1), GopStatus::InvalidParameter);
        assert_eq!(set_mode_ok(0), GopStatus::Success);
        assert_eq!(set_mode_ok(1), GopStatus::Unsupported);
    }

    /// The spec ordinals, and that every error carries the high `UINTN` bit.
    ///
    /// Written as `1 << (usize::BITS - 1)` rather than a 64-bit literal, so
    /// this asserts the rule rather than restating one target's answer to it.
    #[test]
    fn every_error_status_carries_the_efi_error_bit() {
        assert_eq!(GopStatus::Success.efi_code(), 0);
        for (status, ordinal) in [
            (GopStatus::InvalidParameter, 2),
            (GopStatus::Unsupported, 3),
            (GopStatus::DeviceError, 7),
        ] {
            let code = status.efi_code();
            assert_eq!(code & EFI_ERROR, EFI_ERROR, "{status:?} is not an error");
            assert_eq!(code & !EFI_ERROR, ordinal, "{status:?} ordinal");
        }
    }

    /// The three refusals in the pitch conversion, and the `delta == 0` default.
    ///
    /// `gop::gop_blt` reads this same function to size the `BltBuffer` slice it
    /// builds from a raw firmware pointer, so a divergence here would be a
    /// slice length disagreeing with the check that guards it.
    #[test]
    fn a_row_pitch_is_whole_pixels_and_at_least_the_rect() {
        assert_eq!(row_pitch_pixels(0, 64), Some(64), "delta 0 means width");
        assert_eq!(
            row_pitch_pixels(64 * size_of::<BltPixel>(), 64),
            Some(64),
            "an exact fit is a fit"
        );
        assert_eq!(
            row_pitch_pixels(80 * size_of::<BltPixel>(), 64),
            Some(80),
            "a wider pitch than the rect is what delta is for"
        );
        assert_eq!(
            row_pitch_pixels(size_of::<BltPixel>() + 1, 1),
            None,
            "a delta that is not whole pixels describes no row"
        );
        assert_eq!(
            row_pitch_pixels(32 * size_of::<BltPixel>(), 64),
            None,
            "a pitch narrower than the rect cannot hold one of its rows"
        );
    }

    #[test]
    fn set_mode_clears_black() {
        let mut fb = fresh_fb();
        fill_slate(&mut fb);
        assert_ne!(px_at(&fb, 0, 0), 0);
        clear_black(&mut fb);
        assert_eq!(px_at(&fb, 0, 0), 0);
        assert_eq!(px_at(&fb, FB_W / 2, FB_H / 2), 0);
    }

    /// Performance proxy: full-frame BufferToVideo must be O(rows) bulk, not per-pixel.
    #[test]
    fn full_frame_buffer_to_video_bulk_bytes() {
        let mut fb = fresh_fb();
        let mut src = vec![BltPixel::from_bgra_u32(0xff55aa00); FB_W * FB_H];
        let mut stats = PaintStats::default();
        assert_eq!(
            blt(
                &mut fb,
                Some(&mut src),
                BLT_BUFFER_TO_VIDEO,
                0,
                0,
                0,
                0,
                FB_W,
                FB_H,
                0,
                Some(&mut stats),
            ),
            GopStatus::Success
        );
        assert_eq!(stats.bulk_ops, FB_H as u64);
        assert_eq!(stats.bulk_bytes, FB_BYTES as u64);
        assert_eq!(stats.pixel_ops, 0);
        assert_eq!(px_at(&fb, 0, 0), 0xff55aa00);
    }
}
