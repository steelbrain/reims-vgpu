//! What the scaler's kernel is told for one transform: each dispatch's parameters, laid out as
//! `ScalerParams` in `scaler.metal` (reims-vgpu-metal's `src/gpu/scaler.metal`), and the taps,
//! all worked out from the plan as core's reference works its transform out (`reference::run`):
//! - the source's formats and planes, how its pixels and chroma samples turn, how its codes
//!   read, and the chroma samples the source window covers, which its chroma is interpolated
//!   from;
//! - the matrix, where RGB and Y'CbCr meet, and whether the scale is in linear light;
//! - where each destination pixel samples, and with what weights: the reference's own taps
//!   ([`reference::taps`]), each axis's weights divided by their sum, so that a change to where
//!   the reference samples reaches the kernel as it is;
//! - the destination window, its chroma's siting, how its codes are written, and one dispatch
//!   per plane of the destination, over the words the window covers: of `scaler_transform`
//!   (`scaler_transform_adaptive` where the adaptive filter moves a pass, and
//!   `scaler_transform_both_planes` for both planes of a 4:2:0 window where it moves none), or of
//!   `scaler_quads` for the RGB plane of a copy of a Y'CbCr source;
//! - where a destination pixel takes more than one pixel of the turned source each way, the lines:
//!   bands of the window's rows, each a dispatch of `scaler_lines` over the columns of the turned
//!   source the window's across taps take and the band's rows, then its planes' dispatches, which
//!   sum those lines across ([`LINES_BUDGET`]): of `scaler_across`, or of `scaler_across_shared`
//!   for a reduction across of a plane whose words hold more than one pixel.
//!
//! Each number the reference's functions give is taken from them (`reference::luma_of`,
//! `code_chroma`, `chroma_window` and the rest), not worked out again here.

use reims_vgpu_core::num::ToUsize as _;
use reims_vgpu_video_core::scaler::format::{Packing, PlaneSpan, ScalerFormat};
use reims_vgpu_video_core::scaler::plan::{Orientation, Plan, Site};
use reims_vgpu_video_core::scaler::reference::{self, Space};

/// What a dispatch's plane holds: `KIND_*` in `scaler.metal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum PlaneKind {
    Luma8 = 0,
    Luma16 = 1,
    Chroma8 = 2,
    Chroma16 = 3,
    Packed422 = 4,
    Rgb32 = 5,
    Biplanar8 = 6,
    Biplanar16 = 7,
}

/// One dispatch's parameters: `ScalerParams` in `scaler.metal`, field for field, each four
/// bytes, in its order ([`Params::bytes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Params {
    pub(super) source_packing: u32,
    pub(super) source_sample_bytes: u32,
    pub(super) source_order: [u32; 4],
    pub(super) source_luma_offset: u32,
    pub(super) source_luma_pitch: u32,
    pub(super) source_chroma_offset: u32,
    pub(super) source_chroma_pitch: u32,
    pub(super) source_luma_low: f32,
    pub(super) source_luma_scale: f32,
    pub(super) source_chroma_middle: f32,
    pub(super) source_chroma_scale: f32,
    pub(super) luma_map: [i32; 6],
    pub(super) turned_last_x: u32,
    pub(super) turned_last_y: u32,
    pub(super) source_subsample_x: u32,
    pub(super) source_subsample_y: u32,
    pub(super) chroma_step_x: u32,
    pub(super) chroma_step_y: u32,
    pub(super) chroma_origin_x: u32,
    pub(super) chroma_origin_y: u32,
    pub(super) chroma_last_x: u32,
    pub(super) chroma_last_y: u32,
    pub(super) rgb: u32,
    pub(super) linear: u32,
    pub(super) kr: f32,
    pub(super) kg: f32,
    pub(super) kb: f32,
    pub(super) cb_scale: f32,
    pub(super) cr_scale: f32,
    pub(super) b_of_cb: f32,
    pub(super) r_of_cr: f32,
    pub(super) g_scale: f32,
    pub(super) across_first: u32,
    pub(super) across_count: u32,
    pub(super) down_first: u32,
    pub(super) down_count: u32,
    pub(super) window_x: u32,
    pub(super) window_y: u32,
    pub(super) window_width: u32,
    pub(super) window_height: u32,
    pub(super) site_x: u32,
    pub(super) site_y: u32,
    pub(super) subsample_x: u32,
    pub(super) subsample_y: u32,
    pub(super) plane_kind: u32,
    pub(super) plane_order: [u32; 4],
    pub(super) plane_offset: u32,
    pub(super) plane_pitch: u32,
    pub(super) first_row: u32,
    pub(super) rows: u32,
    pub(super) first_byte: u32,
    pub(super) end_byte: u32,
    pub(super) words: u32,
    pub(super) luma_low: f32,
    pub(super) luma_span: f32,
    pub(super) chroma_middle: f32,
    pub(super) chroma_span: f32,
    pub(super) code_max: f32,
    pub(super) lines_first: u32,
    pub(super) lines_pitch: u32,
    pub(super) lines_row: u32,
    pub(super) lines_rows: u32,
    pub(super) chroma_plane_offset: u32,
    pub(super) chroma_plane_pitch: u32,
    pub(super) chroma_across_first: u32,
    pub(super) chroma_across_count: u32,
    pub(super) chroma_down_first: u32,
    pub(super) chroma_down_count: u32,
    pub(super) ycbcr_to_rgb: u32,
    pub(super) across_points_first: u32,
    pub(super) down_points_first: u32,
    pub(super) across_span_first: u32,
    pub(super) across_span_last: u32,
    pub(super) down_span_first: u32,
    pub(super) down_span_last: u32,
    pub(super) across_reduces: u32,
    pub(super) down_reduces: u32,
    pub(super) adapt_across: u32,
    pub(super) adapt_down: u32,
    pub(super) adaptive_channels: u32,
    pub(super) adaptive_scale: f32,
}

/// The words of `ScalerParams`.
pub(super) const PARAMS_WORDS: usize = 92;

impl Params {
    /// Its bytes, as the kernel reads them: each field in order, little-endian.
    pub(super) fn bytes(&self) -> Vec<u8> {
        let mut words: Vec<u32> = Vec::with_capacity(PARAMS_WORDS);
        let mut u = |v: u32| words.push(v);
        u(self.source_packing);
        u(self.source_sample_bytes);
        self.source_order.into_iter().for_each(&mut u);
        u(self.source_luma_offset);
        u(self.source_luma_pitch);
        u(self.source_chroma_offset);
        u(self.source_chroma_pitch);
        for f in [
            self.source_luma_low,
            self.source_luma_scale,
            self.source_chroma_middle,
            self.source_chroma_scale,
        ] {
            u(f.to_bits());
        }
        self.luma_map.into_iter().for_each(|v| u(v.cast_unsigned()));
        for v in [
            self.turned_last_x,
            self.turned_last_y,
            self.source_subsample_x,
            self.source_subsample_y,
            self.chroma_step_x,
            self.chroma_step_y,
            self.chroma_origin_x,
            self.chroma_origin_y,
            self.chroma_last_x,
            self.chroma_last_y,
            self.rgb,
            self.linear,
        ] {
            u(v);
        }
        for f in [
            self.kr,
            self.kg,
            self.kb,
            self.cb_scale,
            self.cr_scale,
            self.b_of_cb,
            self.r_of_cr,
            self.g_scale,
        ] {
            u(f.to_bits());
        }
        for v in [
            self.across_first,
            self.across_count,
            self.down_first,
            self.down_count,
            self.window_x,
            self.window_y,
            self.window_width,
            self.window_height,
            self.site_x,
            self.site_y,
            self.subsample_x,
            self.subsample_y,
            self.plane_kind,
        ] {
            u(v);
        }
        self.plane_order.into_iter().for_each(&mut u);
        for v in [
            self.plane_offset,
            self.plane_pitch,
            self.first_row,
            self.rows,
            self.first_byte,
            self.end_byte,
            self.words,
        ] {
            u(v);
        }
        for f in [
            self.luma_low,
            self.luma_span,
            self.chroma_middle,
            self.chroma_span,
            self.code_max,
        ] {
            u(f.to_bits());
        }
        for v in [
            self.lines_first,
            self.lines_pitch,
            self.lines_row,
            self.lines_rows,
            self.chroma_plane_offset,
            self.chroma_plane_pitch,
            self.chroma_across_first,
            self.chroma_across_count,
            self.chroma_down_first,
            self.chroma_down_count,
            self.ycbcr_to_rgb,
            self.across_points_first,
            self.down_points_first,
            self.across_span_first,
            self.across_span_last,
            self.down_span_first,
            self.down_span_last,
            self.across_reduces,
            self.down_reduces,
            self.adapt_across,
            self.adapt_down,
            self.adaptive_channels,
        ] {
            u(v);
        }
        u(self.adaptive_scale.to_bits());
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }
}

/// One plane of a surface as the kernel reaches it, in the buffer it is bound in: where it
/// starts and its row pitch, and the bytes of each row its samples fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BoundPlane {
    pub(super) offset: usize,
    pub(super) pitch: usize,
    pub(super) row_bytes: usize,
    pub(super) rows: usize,
}

impl BoundPlane {
    /// `span` of a surface whose first byte lies at `base` in its buffer.
    pub(super) fn at(span: &PlaneSpan, base: usize) -> Option<BoundPlane> {
        Some(BoundPlane {
            offset: span.offset.checked_add(base)?,
            pitch: span.bytes_per_row,
            row_bytes: span.row_bytes,
            rows: span.rows,
        })
    }

    /// Where its last row's samples end, from the buffer's first byte.
    pub(super) fn end(&self) -> Option<usize> {
        let rows = self.rows.checked_sub(1)?;
        rows.checked_mul(self.pitch)?
            .checked_add(self.row_bytes)?
            .checked_add(self.offset)
    }

    /// Whether the kernel can write it word by word: its rows start on 4-byte boundaries.
    pub(super) fn word_aligned(&self) -> bool {
        self.offset.is_multiple_of(4) && self.pitch.is_multiple_of(4)
    }
}

/// The kernel a dispatch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// `scaler_lines`: columns of the turned source down a band of the window's rows, into the
    /// lines buffer.
    Lines,
    /// `scaler_transform`: a plane of the destination, of a transform that is not separable.
    Transform,
    /// `scaler_across`: a plane of the destination, over a band of the window's rows, from the
    /// lines `scaler_lines` wrote, a thread a word.
    Across,
    /// `scaler_across_shared`: the same, a threadgroup's pixels worked out a thread a pixel into
    /// threadgroup memory first.
    Shared,
    /// `scaler_quads`: the RGB plane of a copy of a Y'CbCr source, a thread two by two pixels, in
    /// whole threadgroups.
    Quads,
    /// `scaler_transform_adaptive`: `scaler_transform` for a transform whose passes the adaptive
    /// filter moves, a kernel of its own so that the transforms it does not move run, and are
    /// made, without its code.
    TransformAdaptive,
    /// `scaler_transform_both_planes`: `scaler_transform` for both planes of a 4:2:0 window whose
    /// every edge is even, which the adaptive filter does not move, a kernel of its own so that the
    /// others are made without its unrolled code.
    TransformBothPlanes,
}

/// One dispatch: its kernel, its parameters, its threads across and down, and those of a
/// threadgroup.
pub(super) type Pass = (Kernel, Params, (u32, u32), (u32, u32));

/// The threads of a threadgroup of every kernel but `scaler_lines`: 64, which every GPU runs.
pub(super) const GROUP: (u32, u32) = (16, 4);

/// The columns of a threadgroup of `scaler_lines`, its `LINES_COLUMNS`.
const LINES_COLUMNS: u32 = 16;

/// The rows of a threadgroup of `scaler_lines` ([`lines_rows`]): 16, or 8 where the source's rows
/// that 16 rows of the window read do not fit the pixels it decodes once.
const LINES_ROWS: [u32; 2] = [16, 8];

/// Every kernel and threadgroup [`work`] dispatches, as a Vulkan driver makes a pipeline for
/// each at the first dispatch of its threadgroup's size: the dispatches that make them ahead of a
/// guest's first transform (`gpu.rs`, `make_ahead`), in their order, the cheapest to make first
/// (docs/video/scaler.md, "Made ahead").
pub(super) const DISPATCHED: [(Kernel, (u32, u32)); 8] = [
    (Kernel::Shared, GROUP),
    (Kernel::Quads, GROUP),
    (Kernel::Lines, (LINES_COLUMNS, LINES_ROWS[0])),
    (Kernel::Lines, (LINES_COLUMNS, LINES_ROWS[1])),
    (Kernel::Across, GROUP),
    (Kernel::TransformBothPlanes, GROUP),
    (Kernel::Transform, GROUP),
    (Kernel::TransformAdaptive, GROUP),
];

/// The pixels of the turned source a threadgroup of `scaler_lines` decodes once each, its
/// `SHARED_PIXELS`.
const SHARED_PIXELS: u32 = 768;

/// A transform as the kernel runs it: its dispatches, in order; the taps, as `(index, weight)`
/// pairs; and the bytes of lines its largest band takes, 0 where it is not separable.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Work {
    pub(super) passes: Vec<Pass>,
    pub(super) taps: Vec<(u32, f32)>,
    pub(super) lines: usize,
}

/// The bytes of lines a band takes at most, unless one pair of the window's rows takes more: 16
/// MiB.
pub(super) const LINES_BUDGET: usize = 16 << 20;

/// Why a transform cannot run on the kernel: an offset past what the kernel addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TooLarge;

fn small(v: usize) -> Result<u32, TooLarge> {
    u32::try_from(v).map_err(|_| TooLarge)
}

/// A count of the source's, which a `u32` holds, as an index.
fn index(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// `values`' four bytes' offsets, as the kernel takes them.
fn order(values: [usize; 4]) -> Result<[u32; 4], TooLarge> {
    let mut out = [0; 4];
    for (o, v) in out.iter_mut().zip(values) {
        *o = small(v)?;
    }
    Ok(out)
}

/// The turned picture's pixel `(x, y)` as the pixel of a `width` by `height` picture it is
/// ([`Orientation::source_point`] at its centre, as the reference turns a picture): the map's
/// six coefficients, `(m0 x + m1 y + m2, m3 x + m4 y + m5)`. A turn is a signed permutation, so
/// three points give it.
fn turn_map(orientation: Orientation, width: u32, height: u32) -> Result<[i32; 6], TooLarge> {
    let (w, h) = (f64::from(width), f64::from(height));
    let at = |x: f64, y: f64| {
        let (sx, sy) = orientation.source_point(x + 0.5, y + 0.5, w, h);
        (sx.floor(), sy.floor())
    };
    let (o, a, d) = (at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0));
    let int = |v: f64| {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "a whole number within a side, checked below"
        )]
        let i = v as i64;
        i32::try_from(i).map_err(|_| TooLarge)
    };
    Ok([
        int(a.0 - o.0)?,
        int(d.0 - o.0)?,
        int(o.0)?,
        int(a.1 - o.1)?,
        int(d.1 - o.1)?,
        int(o.1)?,
    ])
}

/// An affine map of `v` as `(f(0), f(1) - f(0))`: its value at 0, and its slope.
fn affine(f: impl Fn(f64) -> f64) -> (f64, f64) {
    let at_zero = f(0.0);
    (at_zero, f(1.0) - at_zero)
}

/// How the kernel reads a code of `format`'s in `space`, as the reference's `luma_of` and
/// `chroma_of` read it, each affine in the code: `(low, scale)` for a luma code's value,
/// `(code - low) * scale`, and `(middle, scale)` for a chroma code's.
fn decoding(format: ScalerFormat, space: Space) -> ((f64, f64), (f64, f64)) {
    let inverse = |(at_zero, slope): (f64, f64)| (-at_zero / slope, slope);
    (
        inverse(affine(|v| reference::luma_of(format, v, space))),
        inverse(affine(|v| reference::chroma_of(format, v, space))),
    )
}

/// How the kernel writes `format`'s codes, as the reference's `code_luma` and `code_chroma`
/// write them, each affine in the value: `(low, span)` for a luma code, `low + span * value`,
/// and `(middle, span)` for a chroma code.
fn encoding(format: ScalerFormat) -> ((f64, f64), (f64, f64)) {
    (
        affine(|y| reference::code_luma(format, y)),
        affine(|c| reference::code_chroma(format, c)),
    )
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "the kernel works in floats"
)]
fn float(v: f64) -> f32 {
    v as f32
}

/// The taps of one axis, as the kernel takes them: each pixel's merged by index, with no zero
/// weight, divided by their sum, and padded with zero weights to the most any pixel has; laid out
/// a tap at a time, that tap of every pixel in turn, so that neighbouring threads read
/// neighbouring taps. The count a pixel, and the table.
fn axis(taps: &[Vec<(usize, f64)>]) -> Result<(u32, Vec<(u32, f32)>), TooLarge> {
    let merged: Vec<Vec<(usize, f64)>> = taps
        .iter()
        .map(|pixel| {
            let total: f64 = pixel.iter().map(|(_, w)| w).sum();
            let mut out: Vec<(usize, f64)> = Vec::with_capacity(pixel.len());
            for &(i, w) in pixel {
                match out.iter_mut().find(|(j, _)| *j == i) {
                    Some((_, sum)) => *sum += w,
                    None => out.push((i, w)),
                }
            }
            out.retain(|(_, w)| *w != 0.0);
            if total != 0.0 {
                out.iter_mut().for_each(|(_, w)| *w /= total);
            }
            out
        })
        .collect();
    let count = merged.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut table = Vec::with_capacity(count * merged.len());
    for k in 0..count {
        for pixel in &merged {
            let pad = pixel.first().map_or(0, |(i, _)| *i);
            let (i, w) = pixel.get(k).copied().unwrap_or((pad, 0.0));
            table.push((small(i)?, float(w)));
        }
    }
    Ok((small(count)?, table))
}

const fn site(site: Site) -> u32 {
    match site {
        Site::First => 0,
        Site::Between => 1,
        Site::Second => 2,
    }
}

/// A destination plane's dispatch before its parameters: its kind, its index among the
/// destination's planes and that of the chroma plane it writes too, the rows and the bytes of a
/// row the window covers, each first and end, and its components' order.
type PlaneWork = (
    PlaneKind,
    usize,
    Option<usize>,
    (u32, u32),
    (u32, u32),
    [usize; 4],
);

/// The kernel's work for `plan`, its source's planes at `source` and its destination's at
/// `destination` in the buffers they are bound in. Every plane of the destination must be
/// [`BoundPlane::word_aligned`]; `Err` where an offset or a count is past what the kernel
/// addresses.
pub(super) fn work(
    plan: &Plan,
    source: &[BoundPlane],
    destination: &[BoundPlane],
) -> Result<Work, TooLarge> {
    let src = &plan.source_format;
    let dst = &plan.destination_format;
    let mut p = Params::default();
    // The source.
    let luma = source.first().ok_or(TooLarge)?;
    let chroma = source.get(1).unwrap_or(luma);
    p.source_packing = match src.format.packing() {
        Packing::Biplanar420 => 0,
        Packing::Packed422(order_of) => {
            p.source_order = order(order_of)?;
            1
        }
        Packing::Rgb32(order_of) => {
            p.source_order = order(order_of)?;
            2
        }
    };
    p.source_sample_bytes = small(src.format.bytes_per_sample())?;
    p.source_luma_offset = small(luma.offset)?;
    p.source_luma_pitch = small(luma.pitch)?;
    p.source_chroma_offset = small(chroma.offset)?;
    p.source_chroma_pitch = small(chroma.pitch)?;
    // Each plane's last byte the kernel reads lies inside the buffer's first 4 GiB.
    for plane in source {
        small(plane.end().ok_or(TooLarge)?)?;
    }
    let space = Space::between(src.format, dst.format);
    let ((luma_low, luma_scale), (chroma_middle, chroma_scale)) = decoding(src.format, space);
    p.source_luma_low = float(luma_low);
    p.source_luma_scale = float(luma_scale);
    p.source_chroma_middle = float(chroma_middle);
    p.source_chroma_scale = float(chroma_scale);
    let (sx, sy) = src.format.subsampling();
    (p.source_subsample_x, p.source_subsample_y) = (sx, sy);
    p.luma_map = turn_map(plan.orientation, src.width, src.height)?;
    let step = if plan.orientation.rotates() {
        (sy, sx)
    } else {
        (sx, sy)
    };
    (p.chroma_step_x, p.chroma_step_y) = step;
    // The taps, and the turned window's chroma samples: where the first sits, and the last.
    let taps = reference::taps(plan);
    let (origin_x, origin_y, last_x, last_y) =
        reference::chroma_window(taps.window, (index(step.0), index(step.1)));
    p.chroma_origin_x = small(origin_x)?;
    p.chroma_origin_y = small(origin_y)?;
    p.chroma_last_x = small(last_x)?;
    p.chroma_last_y = small(last_y)?;
    p.turned_last_x = small(taps.turned.0.saturating_sub(1))?;
    p.turned_last_y = small(taps.turned.1.saturating_sub(1))?;
    // The working space.
    p.rgb = u32::from(space == Space::Rgb);
    p.linear = u32::from(reference::in_linear_light(src.format, dst.format));
    let (kr, kb) = plan.matrix.weights();
    p.kr = float(kr);
    p.kg = float(1.0 - kr - kb);
    p.kb = float(kb);
    p.cb_scale = float(1.0 / (2.0 * (1.0 - kb)));
    p.cr_scale = float(1.0 / (2.0 * (1.0 - kr)));
    p.b_of_cb = float(2.0 * (1.0 - kb));
    p.r_of_cr = float(2.0 * (1.0 - kr));
    p.g_scale = float(1.0 / (1.0 - kr - kb));
    // A Y'CbCr source into RGB converts after the scale, as the reference converts it.
    p.ycbcr_to_rgb = u32::from(src.format.is_ycbcr() && space == Space::Rgb);
    let (across_count, across) = axis(&taps.across)?;
    let (down_count, down) = axis(&taps.down)?;
    // A Y'CbCr source's chroma's taps, along an axis where they are not its luma's.
    let chroma_across = taps.chroma_across.as_deref().map(axis).transpose()?;
    let chroma_down = taps.chroma_down.as_deref().map(axis).transpose()?;
    // A pixel of one source pixel each way is that pixel: into linear light and back out of it
    // is no change, so it goes neither way.
    if across_count == 1 && down_count == 1 {
        p.linear = 0;
    }
    p.across_first = 0;
    p.across_count = across_count;
    p.down_first = small(across.len())?;
    p.down_count = down_count;
    // The chroma's taps after the luma's, or the luma's where they are the same.
    let mut end = across.len() + down.len();
    let mut placed = |taps: &Option<(u32, Vec<(u32, f32)>)>, (first, count): (u32, u32)| {
        let Some((count, table)) = taps else {
            return Ok((first, count));
        };
        let at = small(end)?;
        end += table.len();
        Ok::<_, TooLarge>((at, *count))
    };
    (p.chroma_across_first, p.chroma_across_count) =
        placed(&chroma_across, (p.across_first, across_count))?;
    (p.chroma_down_first, p.chroma_down_count) = placed(&chroma_down, (p.down_first, down_count))?;
    // The adaptive filter: each column's and row's sample point after the taps, as the pixel it
    // lies in (an int's bits) and its fraction past it, and where each axis's pixels end. An
    // axis whose every pixel takes one source pixel moves none: its point lies on that pixel.
    let point = |&(pixel, fraction): &(i64, f64)| {
        let pixel = i32::try_from(pixel).map_err(|_| TooLarge)?;
        Ok::<_, TooLarge>((pixel.cast_unsigned(), float(fraction)))
    };
    let across_points = taps
        .across_points
        .iter()
        .map(point)
        .collect::<Result<Vec<_>, _>>()?;
    let down_points = taps
        .down_points
        .iter()
        .map(point)
        .collect::<Result<Vec<_>, _>>()?;
    p.across_points_first = small(end)?;
    p.down_points_first = small(end + across_points.len())?;
    (p.across_span_first, p.across_span_last) =
        (small(taps.across_span.0)?, small(taps.across_span.1)?);
    (p.down_span_first, p.down_span_last) = (small(taps.down_span.0)?, small(taps.down_span.1)?);
    p.across_reduces = u32::from(taps.reduces[0]);
    p.down_reduces = u32::from(taps.reduces[1]);
    let adaptive = reference::adaptive(src.format, space);
    p.adaptive_channels = small(adaptive.channels)?;
    p.adaptive_scale = float(adaptive.scale);
    p.adapt_across = u32::from(adaptive.channels > 0 && across_count > 1);
    p.adapt_down = u32::from(adaptive.channels > 0 && down_count > 1);
    let chroma_across_table = chroma_across.map(|(_, table)| table);
    let chroma_down_table = chroma_down.map(|(_, table)| table);
    // The window.
    let rect = plan.destination_rect;
    (p.window_x, p.window_y, p.window_width, p.window_height) =
        (rect.x, rect.y, rect.width, rect.height);
    p.site_x = site(plan.destination_siting.horizontal);
    p.site_y = site(plan.destination_siting.vertical);
    let (dx, dy) = dst.format.subsampling();
    (p.subsample_x, p.subsample_y) = (dx, dy);
    let ((luma_low, luma_span), (chroma_middle, chroma_span)) = encoding(dst.format);
    p.luma_low = float(luma_low);
    p.luma_span = float(luma_span);
    p.chroma_middle = float(chroma_middle);
    p.chroma_span = float(chroma_span);
    p.code_max = float(f64::from((1u32 << dst.format.bits()) - 1));
    // A dispatch a plane: its rows and bytes the window covers, as `coverage::covered` has them.
    let (x0, y0) = (rect.x, rect.y);
    let (x1, y1) = (rect.x + rect.width, rect.y + rect.height);
    let s = small(dst.format.bytes_per_sample())?;
    let even = [x0, x1, y0, y1].iter().all(|v| v % 2 == 0);
    let planes: Vec<PlaneWork> = match dst.format.packing() {
        // A window of even edges, as the plan takes 4:2:0 windows of two pixels or more: one
        // dispatch for both planes, each chroma word with the luma words of its pixels.
        Packing::Biplanar420 if even => {
            let kind = if s == 1 {
                PlaneKind::Biplanar8
            } else {
                PlaneKind::Biplanar16
            };
            vec![(kind, 0, Some(1), (y0 / 2, y1 / 2), (x0 * s, x1 * s), [0; 4])]
        }
        Packing::Biplanar420 => {
            let (luma_kind, chroma_kind) = if s == 1 {
                (PlaneKind::Luma8, PlaneKind::Chroma8)
            } else {
                (PlaneKind::Luma16, PlaneKind::Chroma16)
            };
            vec![
                (luma_kind, 0, None, (y0, y1), (x0 * s, x1 * s), [0; 4]),
                (
                    chroma_kind,
                    1,
                    None,
                    (y0 / 2, y1.div_ceil(2)),
                    (x0 / 2 * 2 * s, x1.div_ceil(2) * 2 * s),
                    [0; 4],
                ),
            ]
        }
        Packing::Packed422(order_of) => vec![(
            PlaneKind::Packed422,
            0,
            None,
            (y0, y1),
            (x0 / 2 * 4, x1.div_ceil(2) * 4),
            order_of,
        )],
        Packing::Rgb32(order_of) => {
            vec![(
                PlaneKind::Rgb32,
                0,
                None,
                (y0, y1),
                (x0 * 4, x1 * 4),
                order_of,
            )]
        }
    };
    let mut plane_passes = Vec::with_capacity(planes.len());
    for (kind, index, chroma, rows, (first_byte, end_byte), order_of) in planes {
        let plane = destination.get(index).ok_or(TooLarge)?;
        small(plane.end().ok_or(TooLarge)?)?;
        let mut pass = p;
        pass.plane_kind = kind as u32;
        pass.plane_order = order(order_of)?;
        pass.plane_offset = small(plane.offset)?;
        pass.plane_pitch = small(plane.pitch)?;
        if let Some(index) = chroma {
            let plane = destination.get(index).ok_or(TooLarge)?;
            small(plane.end().ok_or(TooLarge)?)?;
            pass.chroma_plane_offset = small(plane.offset)?;
            pass.chroma_plane_pitch = small(plane.pitch)?;
        }
        pass.first_byte = first_byte;
        pass.end_byte = end_byte;
        pass.words = (end_byte - (first_byte & !3)).div_ceil(4);
        plane_passes.push((pass, rows));
    }
    // A plane's rows `rows` of a band of the window's `[y0, y1)`: the window's for a luma or RGB
    // plane, the chroma rows a 4:2:0 plane's luma rows share. A band past the first starts on an
    // even row, so a chroma row lies in one band.
    let halved = [
        PlaneKind::Chroma8,
        PlaneKind::Chroma16,
        PlaneKind::Biplanar8,
        PlaneKind::Biplanar16,
    ]
    .map(|kind| kind as u32);
    let band_rows = |kind: u32, (first, end): (u32, u32), (b0, b1): (u32, u32)| {
        if halved.contains(&kind) {
            (first.max(b0 / 2), end.min(b1.div_ceil(2)))
        } else {
            (first.max(b0), end.min(b1))
        }
    };
    // Separable where a pixel takes more than one pixel each way: the lines span the columns the
    // across taps take, the chroma's among them, and the pixels the adaptive filter reads.
    let taken = (across.iter().chain(chroma_across_table.iter().flatten()))
        .filter(|(_, w)| *w != 0.0)
        .map(|(i, _)| *i);
    let read = (p.adapt_across != 0).then_some([p.across_span_first, p.across_span_last]);
    let taken = taken.chain(read.into_iter().flatten());
    let columns = taken.clone().min().zip(taken.max());
    let separable = (across_count.max(p.chroma_across_count) > 1
        && down_count.max(p.chroma_down_count) > 1)
        .then_some(columns)
        .flatten()
        .filter(|_| p.window_width > 0);
    // The bytes of one line: a float4 for an RGB source, three floats for a Y'CbCr source,
    // whose pixels are opaque.
    let line_bytes = if p.source_packing == 2 { 16 } else { 12 };
    let (bands, first, pitch) = match separable {
        Some((first, last)) => {
            let pitch = last - first + 1;
            let rows = LINES_BUDGET / line_bytes / pitch.to_usize();
            (
                bands((y0, y1), u32::try_from(rows).unwrap_or(u32::MAX), dy),
                first,
                pitch,
            )
        }
        None => (vec![(y0, y1)], 0, 0),
    };
    // The source rows each row of the window reads down: its taps' first and last, the chroma's
    // among them, and the six about its point where the pass down moves its sums.
    let row_runs: Vec<Option<(u32, u32)>> = (0..p.window_height)
        .map(|dy| {
            let mut rows: Vec<u32> = [&down, chroma_down_table.as_deref().unwrap_or(&[])]
                .iter()
                .flat_map(|table| {
                    let count = table.len() / p.window_height.max(1).to_usize();
                    (0..count).filter_map(move |k| {
                        table
                            .get(k * p.window_height.to_usize() + dy.to_usize())
                            .filter(|(_, w)| *w != 0.0)
                            .map(|(i, _)| *i)
                    })
                })
                .collect();
            if p.adapt_down != 0 {
                let b = i64::from(down_points[dy.to_usize()].0.cast_signed());
                let six = |k: i64| {
                    let row = (b - 2 + k)
                        .clamp(i64::from(p.down_span_first), i64::from(p.down_span_last));
                    u32::try_from(row).unwrap_or(0)
                };
                rows.extend([six(0), six(5)]);
            }
            rows.iter().copied().min().zip(rows.iter().copied().max())
        })
        .collect();
    let mut passes = Vec::new();
    let mut lines = 0usize;
    for band in bands {
        let mut lined = p;
        if separable.is_some() {
            lined.lines_first = first;
            lined.lines_pitch = pitch;
            lined.lines_row = band.0 - y0;
            lined.lines_rows = band.1 - band.0;
            let bytes = (lined.lines_rows.to_usize())
                .checked_mul(pitch.to_usize())
                .and_then(|n| n.checked_mul(line_bytes))
                .ok_or(TooLarge)?;
            small(bytes)?;
            lines = lines.max(bytes);
            let rows = lines_rows(&row_runs, (band.0 - y0, band.1 - y0));
            passes.push((
                Kernel::Lines,
                lined,
                (pitch, lined.lines_rows),
                (LINES_COLUMNS, rows),
            ));
        }
        for (plane, rows) in &plane_passes {
            let (first_row, end_row) = band_rows(plane.plane_kind, *rows, band);
            if first_row >= end_row {
                continue;
            }
            let mut pass = *plane;
            (
                pass.lines_first,
                pass.lines_pitch,
                pass.lines_row,
                pass.lines_rows,
            ) = (
                lined.lines_first,
                lined.lines_pitch,
                lined.lines_row,
                lined.lines_rows,
            );
            pass.first_row = first_row;
            pass.rows = end_row - first_row;
            let kernel = if separable.is_some() {
                // A reduction across of a plane whose words hold more than a pixel: each word's
                // pixels take lines far apart, which its threads read side by side a pixel each.
                if pitch > p.window_width && pass.plane_kind != PlaneKind::Rgb32 as u32 {
                    Kernel::Shared
                } else {
                    Kernel::Across
                }
            } else if across_count == 1
                && down_count == 1
                && pass.plane_kind == PlaneKind::Rgb32 as u32
                && p.source_packing != 2
            {
                // A copy of a Y'CbCr source into RGB: two by two pixels share chroma samples.
                Kernel::Quads
            } else if p.adapt_across != 0 || p.adapt_down != 0 {
                Kernel::TransformAdaptive
            } else if pass.plane_kind == PlaneKind::Biplanar8 as u32
                || pass.plane_kind == PlaneKind::Biplanar16 as u32
            {
                Kernel::TransformBothPlanes
            } else {
                Kernel::Transform
            };
            let threads = if kernel == Kernel::Quads {
                // Whole threadgroups: every thread of one reaches its barrier.
                (
                    pass.words.div_ceil(2).next_multiple_of(GROUP.0),
                    pass.rows.div_ceil(2).next_multiple_of(GROUP.1),
                )
            } else {
                (pass.words, pass.rows)
            };
            passes.push((kernel, pass, threads, GROUP));
        }
    }
    let mut table = across;
    table.extend(down);
    table.extend(chroma_across_table.into_iter().flatten());
    table.extend(chroma_down_table.into_iter().flatten());
    table.extend(across_points);
    table.extend(down_points);
    Ok(Work {
        passes,
        taps: table,
        lines,
    })
}

/// The rows of a threadgroup of `scaler_lines` for the band `[b0, b1)` of the window's rows,
/// each of which reads the source's rows `runs` gives, first and last: 16, where the run of the
/// source's rows each 16 rows of the band read fits the pixels it decodes once
/// ([`SHARED_PIXELS`]), as at a reduction of up to 2.5; else 8 ([`LINES_ROWS`]).
fn lines_rows(runs: &[Option<(u32, u32)>], (b0, b1): (u32, u32)) -> u32 {
    const ROWS: u32 = LINES_ROWS[0];
    let run = |dy: u32| runs.get(dy.to_usize()).copied().flatten();
    let fits = (b0..b1).step_by(ROWS.to_usize()).all(|top| {
        let bottom = (top + ROWS).min(b1) - 1;
        let first = run(top).map(|(first, _)| first);
        let last = run(bottom).map(|(_, last)| last);
        match first.zip(last) {
            Some((first, last)) if first <= last => {
                (last - first + 1) * LINES_COLUMNS <= SHARED_PIXELS
            }
            _ => true,
        }
    });
    if fits { ROWS } else { LINES_ROWS[1] }
}

/// The bands of the window's rows `[y0, y1)` of a separable transform: `rows` rows each at most,
/// or `step` where `rows` is fewer, each past the first starting on a multiple of `step` (the
/// destination's chroma rows), so that no chroma row spans two bands.
pub(super) fn bands((y0, y1): (u32, u32), rows: u32, step: u32) -> Vec<(u32, u32)> {
    let step = step.max(1);
    let rows = (rows / step * step).max(step);
    let mut out = Vec::new();
    let mut start = y0;
    while start < y1 {
        // To the multiple of `step` `rows` past the band's first group's start.
        let end = (start / step)
            .saturating_mul(step)
            .saturating_add(rows)
            .min(y1);
        out.push((start, end));
        start = end;
    }
    out
}

/// The taps' bytes, as the kernel reads them: each an index and a weight, little-endian.
pub(super) fn taps_bytes(taps: &[(u32, f32)]) -> impl Iterator<Item = u8> + '_ {
    taps.iter()
        .flat_map(|(i, w)| i.to_le_bytes().into_iter().chain(w.to_bits().to_le_bytes()))
}

#[cfg(test)]
mod tests;
