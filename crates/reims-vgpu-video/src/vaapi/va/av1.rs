//! AV1 decoding on VA-API: [`crate::codec::av1::Decoder`] over an accelerator that fills
//! libva's AV1 buffers (va_dec_av1.h) and decodes into a pool of NV12 surfaces for 8 bits and
//! P010 ones for 10, on the driver's `AV1Profile0`.
//!
//! A frame that applies film grain is decoded into two surfaces, as va_dec_av1.h has it:
//! `current_frame`, without grain, which later frames predict from, and
//! `current_display_picture`, with grain applied by the driver, which is what is output. A
//! frame without grain is one surface, both of them.

use std::sync::Arc;

use super::buffer::{Buffer, render_decode};
use super::caps::{Context, Entrypoint, Profile};
use super::display::{Display, Error};
use super::ffi::{self, BitField, av1 as va};
use super::frame::Frame;
use super::surface::{Rect, Surface, SurfaceFormat, SurfacePool, end_context};
use super::wait::{self, Deadline};
use crate::codec::av1::frame::{FrameHeader, FrameType};
use crate::codec::av1::sequence::SequenceHeader;
use crate::codec::av1::tile_group::TileGroup;
use crate::codec::av1::{Accelerator, Decoder, DecoderConfiguration, FrameParams, Limits, Output};

/// `VAProfileAV1Profile0`: Main, 4:2:0 at 8 or 10 bits.
pub const AV1_PROFILE0: Profile = Profile(va::VAProfileAV1Profile0);

/// An AV1 decoder on a VA display.
///
/// Samples are temporal units in the low overhead bitstream format, as VideoToolbox's and
/// MP4's samples are, with or without their temporal delimiters. Frames come out as they
/// are shown. 8-bit streams decode into NV12 frames, 10-bit ones into P010.
pub struct Av1Decoder {
    inner: Decoder<VaAv1>,
}

impl Av1Decoder {
    /// The frames a caller is to hold at once, beyond those the decoder holds itself.
    pub const SPARE_FRAMES: usize = 2;

    /// Has the driver apply grain of the parameters `alter` makes of each frame's (the codec
    /// decoder's `alter_film_grain`), for the card tests.
    #[cfg(all(test, target_os = "linux"))]
    pub(crate) fn alter_film_grain(
        &mut self,
        alter: fn(&mut crate::codec::av1::frame::FilmGrainParams),
    ) {
        self.inner.alter_film_grain(alter);
    }

    /// A decoder for samples described by an `av1C` record, whose sequence header, where the
    /// record has one, is checked against what the decoder and the driver take, so that a
    /// stream they cannot decode is refused here. `limits` is the largest frame the driver
    /// lists for the profile.
    pub fn new(display: &Display, av1c: &[u8], limits: Limits) -> Result<Self, Error> {
        let config = DecoderConfiguration::parse(av1c)?;
        let mut inner = Decoder::new(VaAv1::new(display), limits);
        inner.configure(&config)?;
        if let Some(sequence) = inner.sequence() {
            select_profile(display, sequence)?;
        }
        Ok(Self { inner })
    }

    /// A decoder whose samples carry their sequence headers.
    pub fn without_record(display: &Display, limits: Limits) -> Self {
        Self {
            inner: Decoder::new(VaAv1::new(display), limits),
        }
    }

    /// Decodes one temporal unit, tagged `id`, and returns the frames it shows. The decode
    /// may still be running on the GPU: a frame's readback waits for it.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame>, Error> {
        let deadline = Deadline::default();
        let outputs = wait::by(&deadline, || self.inner.decode(sample, id));
        Ok(outputs?
            .into_iter()
            .map(|output| frame_from_output(output, &deadline))
            .collect())
    }

    /// Forgets every reference, as for a seek.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The frames decoded so far, shown or not.
    pub fn frames_decoded(&self) -> u64 {
        self.inner.frames_decoded()
    }

    /// The VA profile the decoder runs, once it has started a frame.
    pub fn profile(&self) -> Option<Profile> {
        self.inner
            .accelerator()
            .session
            .as_ref()
            .map(|_| AV1_PROFILE0)
    }
}

// Decoders move between threads: libva is thread-safe (va.h, "Multithreading Guide"), and
// each decoder is used by one thread at a time through `&mut`.
const _: fn() = || {
    fn send<T: Send>() {}
    send::<Av1Decoder>();
};

/// A decoded frame's surfaces.
pub(crate) struct Picture {
    /// `current_frame`: the frame later frames predict from.
    reference: Arc<Surface>,
    /// `current_display_picture`, where the frame applies film grain: the frame with grain.
    display: Option<Arc<Surface>>,
}

/// The frame VA-API decoded for one of the decoder's outputs.
fn frame_from_output(output: Output<Picture>, deadline: &Deadline) -> Frame {
    let surface = output
        .picture
        .display
        .as_ref()
        .unwrap_or(&output.picture.reference);
    let (width, height) = output.size;
    Frame {
        surface: Arc::clone(surface),
        id: output.id,
        poc: 0,
        idr: output.frame_type == FrameType::Key && !output.shown_existing,
        reference: true,
        rect: Rect {
            x: 0,
            y: 0,
            width,
            height,
        },
        coded_size: output.size,
        full_range: output.full_range,
        deadline: deadline.clone(),
    }
}

/// The VA-API half of the decoder.
struct VaAv1 {
    display: Display,
    session: Option<Session>,
}

/// What one sequence header's frames are decoded with.
struct Session {
    /// Destroyed first, before the surfaces it renders into: it holds its config.
    context: Option<Context>,
    pool: SurfacePool,
}

impl Drop for Session {
    fn drop(&mut self) {
        // The context goes first. The pool may outlive it: frames the caller holds keep their
        // surfaces.
        end_context(self.context.take(), [&self.pool]);
    }
}

impl VaAv1 {
    fn new(display: &Display) -> Self {
        Self {
            display: display.clone_ref(),
            session: None,
        }
    }

    fn session(&self) -> Result<&Session, Error> {
        self.session
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding before a sequence header".into()))
    }
}

/// The surface format of `sequence`'s frames: NV12 at 8 bits, P010 at 10.
/// `check_supported` has refused other depths and formats.
fn format_for(sequence: &SequenceHeader) -> SurfaceFormat {
    if sequence.color.bit_depth > 8 {
        SurfaceFormat::P010
    } else {
        SurfaceFormat::Nv12
    }
}

/// Checks that the driver decodes `sequence`'s stream at its largest frame.
fn select_profile(display: &Display, sequence: &SequenceHeader) -> Result<(), Error> {
    let format = format_for(sequence);
    if !display.supports(AV1_PROFILE0, Entrypoint::VLD) {
        return Err(Error::Unsupported(
            "AV1Profile0: the driver has no decoder".into(),
        ));
    }
    let limits = display.config_attributes(
        AV1_PROFILE0,
        Entrypoint::VLD,
        &[
            ffi::VAConfigAttribRTFormat,
            ffi::VAConfigAttribMaxPictureWidth,
            ffi::VAConfigAttribMaxPictureHeight,
        ],
    )?;
    if limits[0].is_some_and(|f| f & format.rt_format() == 0) {
        return Err(Error::Unsupported(format!(
            "AV1Profile0: no {} render targets",
            format.name()
        )));
    }
    let (width, height) = (sequence.max_frame_width, sequence.max_frame_height);
    if limits[1].is_some_and(|max| width > max) || limits[2].is_some_and(|max| height > max) {
        return Err(Error::Unsupported(format!(
            "AV1Profile0 at {width}x{height}: the driver's maximum is {}x{}",
            limits[1].unwrap_or(0),
            limits[2].unwrap_or(0)
        )));
    }
    Ok(())
}

impl Accelerator for VaAv1 {
    type Picture = Picture;
    type Error = Error;

    fn configure(&mut self, sequence: &SequenceHeader) -> Result<(), Error> {
        self.session = None;
        select_profile(&self.display, sequence)?;
        let format = format_for(sequence);
        let mut attribs = [ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: format.rt_format(),
        }];
        // Eight reference slots.
        let (context, pool) = Context::new_decoder(
            &self.display,
            AV1_PROFILE0,
            &mut attribs,
            format,
            (sequence.max_frame_width, sequence.max_frame_height),
            8,
        )?;
        self.session = Some(Session {
            context: Some(context),
            pool,
        });
        Ok(())
    }

    fn new_picture(&mut self, header: &FrameHeader) -> Result<Picture, Error> {
        let pool = &self.session()?.pool;
        let reference = Arc::new(pool.get()?);
        let display = if header.film_grain.apply_grain {
            Some(Arc::new(pool.get()?))
        } else {
            None
        };
        Ok(Picture { reference, display })
    }

    fn decode_frame(
        &mut self,
        picture: &Picture,
        params: &FrameParams<'_, Picture>,
        tile_groups: &[TileGroup<'_>],
    ) -> Result<(), Error> {
        let session = self.session()?;
        let decoding = session
            .context
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding without a context".into()))?;
        let context = decoding.id;
        let display = &self.display;
        let parameters = picture_parameters(picture, params);
        let picture_buffer = Buffer::with_value(
            display,
            context,
            ffi::VAPictureParameterBufferType,
            &parameters,
        )?;
        let mut tile_buffers = Vec::with_capacity(tile_groups.len());
        for group in tile_groups {
            let slices = slice_parameters(group);
            tile_buffers.push((
                Buffer::with_values(display, context, ffi::VASliceParameterBufferType, &slices)?,
                Buffer::with_bytes(display, context, ffi::VASliceDataBufferType, group.data)?,
            ));
        }
        let mut groups: Vec<Vec<&Buffer<'_>>> = vec![vec![&picture_buffer]];
        groups.extend(
            tile_buffers
                .iter()
                .map(|(parameters, data)| vec![parameters, data]),
        );
        let groups: Vec<&[&Buffer<'_>]> = groups.iter().map(Vec::as_slice).collect();
        // The decode renders into current_frame, the display picture beside it.
        picture.reference.stalled_if(render_decode(
            display,
            decoding,
            picture.reference.id(),
            &groups,
        ))
    }
}

fn flag(field: BitField, value: bool) -> u32 {
    field.put(u32::from(value))
}

fn narrow_u16(value: u32) -> u16 {
    u16::try_from(value).unwrap_or(u16::MAX)
}

fn narrow_u8(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// `VADecPictureParameterBufferAV1` for a frame, its fields as ffmpeg's VA-API AV1 decode
/// fills them.
#[expect(
    clippy::too_many_lines,
    reason = "one assignment of every field va_dec_av1.h has"
)]
fn picture_parameters(
    picture: &Picture,
    p: &FrameParams<'_, Picture>,
) -> va::VADecPictureParameterBufferAV1 {
    let (seq, h) = (p.sequence, p.header);
    let color = &seq.color;
    let mut pp = va::VADecPictureParameterBufferAV1::zeroed();
    pp.profile = seq.seq_profile;
    pp.order_hint_bits_minus_1 = narrow_u8(seq.order_hint_bits.saturating_sub(1));
    pp.bit_depth_idx = match color.bit_depth {
        8 => 0,
        10 => 1,
        _ => 2,
    };
    pp.matrix_coefficients = color.matrix_coefficients;
    {
        use va::seq_info_fields as f;
        pp.seq_info_fields = flag(f::STILL_PICTURE, seq.still_picture)
            | flag(f::USE_128X128_SUPERBLOCK, seq.use_128x128_superblock)
            | flag(f::ENABLE_FILTER_INTRA, seq.enable_filter_intra)
            | flag(f::ENABLE_INTRA_EDGE_FILTER, seq.enable_intra_edge_filter)
            | flag(
                f::ENABLE_INTERINTRA_COMPOUND,
                seq.enable_interintra_compound,
            )
            | flag(f::ENABLE_MASKED_COMPOUND, seq.enable_masked_compound)
            | flag(f::ENABLE_DUAL_FILTER, seq.enable_dual_filter)
            | flag(f::ENABLE_ORDER_HINT, seq.enable_order_hint)
            | flag(f::ENABLE_JNT_COMP, seq.enable_jnt_comp)
            | flag(f::ENABLE_CDEF, seq.enable_cdef)
            | flag(f::MONO_CHROME, color.mono_chrome)
            | flag(f::COLOR_RANGE, color.color_range)
            | flag(f::SUBSAMPLING_X, color.subsampling_x)
            | flag(f::SUBSAMPLING_Y, color.subsampling_y)
            | f::CHROMA_SAMPLE_POSITION.put(u32::from(color.chroma_sample_position))
            | flag(f::FILM_GRAIN_PARAMS_PRESENT, seq.film_grain_params_present);
    }
    pp.current_frame = picture.reference.id();
    pp.current_display_picture = picture
        .display
        .as_ref()
        .map_or(picture.reference.id(), |display| display.id());
    pp.frame_width_minus1 = narrow_u16(h.size.upscaled_width.saturating_sub(1));
    pp.frame_height_minus1 = narrow_u16(h.size.frame_height.saturating_sub(1));
    // ref_frame_map: no slot for a key frame shown at once, which predicts from none.
    let shown_key = h.frame_type == FrameType::Key && h.show_frame;
    for (map, slot) in pp.ref_frame_map.iter_mut().zip(&p.slots) {
        *map = match slot {
            Some(reference) if !shown_key => reference.picture.reference.id(),
            _ => ffi::VA_INVALID_SURFACE,
        };
    }
    pp.ref_frame_idx = h.ref_frame_idx;
    pp.primary_ref_frame = h.primary_ref_frame;
    pp.order_hint = h.order_hint;
    let segmentation = &h.segmentation;
    {
        use va::segment_info_fields as f;
        pp.seg_info.segment_info_fields = flag(f::ENABLED, segmentation.enabled)
            | flag(f::UPDATE_MAP, segmentation.update_map)
            | flag(f::TEMPORAL_UPDATE, segmentation.temporal_update)
            | flag(f::UPDATE_DATA, segmentation.update_data);
    }
    if segmentation.enabled {
        pp.seg_info.feature_data = segmentation.features.data;
        for (mask, enabled) in pp
            .seg_info
            .feature_mask
            .iter_mut()
            .zip(&segmentation.features.enabled)
        {
            *mask = enabled
                .iter()
                .rev()
                .fold(0u8, |mask, &enabled| (mask << 1) | u8::from(enabled));
        }
    }
    pp.film_grain_info = film_grain(h);
    let tiles = &h.tile_info;
    pp.tile_cols = narrow_u8(tiles.cols());
    pp.tile_rows = narrow_u8(tiles.rows());
    for (to, &from) in pp.width_in_sbs_minus_1.iter_mut().zip(&tiles.col_widths_sb) {
        *to = narrow_u16(from.saturating_sub(1));
    }
    for (to, &from) in pp
        .height_in_sbs_minus_1
        .iter_mut()
        .zip(&tiles.row_heights_sb)
    {
        *to = narrow_u16(from.saturating_sub(1));
    }
    pp.context_update_tile_id = narrow_u16(tiles.context_update_tile_id);
    {
        use va::pic_info_fields as f;
        pp.pic_info_fields = f::FRAME_TYPE.put(u32::from(h.frame_type.code()))
            | flag(f::SHOW_FRAME, h.show_frame)
            | flag(f::SHOWABLE_FRAME, h.showable_frame)
            | flag(f::ERROR_RESILIENT_MODE, h.error_resilient_mode)
            | flag(f::DISABLE_CDF_UPDATE, h.disable_cdf_update)
            | flag(f::ALLOW_SCREEN_CONTENT_TOOLS, h.allow_screen_content_tools)
            | flag(f::FORCE_INTEGER_MV, h.force_integer_mv)
            | flag(f::ALLOW_INTRABC, h.allow_intrabc)
            | flag(f::USE_SUPERRES, h.size.use_superres)
            | flag(f::ALLOW_HIGH_PRECISION_MV, h.allow_high_precision_mv)
            | flag(f::IS_MOTION_MODE_SWITCHABLE, h.is_motion_mode_switchable)
            | flag(f::USE_REF_FRAME_MVS, h.use_ref_frame_mvs)
            | flag(
                f::DISABLE_FRAME_END_UPDATE_CDF,
                h.disable_frame_end_update_cdf,
            )
            | flag(f::UNIFORM_TILE_SPACING_FLAG, tiles.uniform)
            | flag(f::ALLOW_WARPED_MOTION, h.allow_warped_motion);
    }
    pp.superres_scale_denominator = narrow_u8(h.size.superres_denom);
    pp.interp_filter = h.interpolation_filter;
    let lf = &h.loop_filter;
    pp.filter_level = [lf.level[0], lf.level[1]];
    pp.filter_level_u = lf.level[2];
    pp.filter_level_v = lf.level[3];
    {
        use va::loop_filter_info_fields as f;
        pp.loop_filter_info_fields = narrow_u8(
            f::SHARPNESS_LEVEL.put(u32::from(lf.sharpness))
                | flag(f::MODE_REF_DELTA_ENABLED, lf.delta_enabled)
                | flag(f::MODE_REF_DELTA_UPDATE, lf.delta_update),
        );
    }
    pp.ref_deltas = lf.ref_deltas;
    pp.mode_deltas = lf.mode_deltas;
    let q = &h.quantization;
    pp.base_qindex = q.base_q_idx;
    pp.y_dc_delta_q = q.delta_q_y_dc;
    pp.u_dc_delta_q = q.delta_q_u_dc;
    pp.u_ac_delta_q = q.delta_q_u_ac;
    pp.v_dc_delta_q = q.delta_q_v_dc;
    pp.v_ac_delta_q = q.delta_q_v_ac;
    {
        use va::qmatrix_fields as f;
        pp.qmatrix_fields = narrow_u16(
            flag(f::USING_QMATRIX, q.using_qmatrix)
                | f::QM_Y.put(u32::from(q.qm_y))
                | f::QM_U.put(u32::from(q.qm_u))
                | f::QM_V.put(u32::from(q.qm_v)),
        );
    }
    {
        use va::mode_control_fields as f;
        pp.mode_control_fields = flag(f::DELTA_Q_PRESENT_FLAG, h.delta_q_present)
            | f::LOG2_DELTA_Q_RES.put(u32::from(h.delta_q_res))
            | flag(f::DELTA_LF_PRESENT_FLAG, h.delta_lf_present)
            | f::LOG2_DELTA_LF_RES.put(u32::from(h.delta_lf_res))
            | flag(f::DELTA_LF_MULTI, h.delta_lf_multi)
            | f::TX_MODE.put(u32::from(h.tx_mode))
            | flag(f::REFERENCE_SELECT, h.reference_select)
            | flag(f::REDUCED_TX_SET_USED, h.reduced_tx_set)
            | flag(f::SKIP_MODE_PRESENT, h.skip_mode_present);
    }
    let cdef = &h.cdef;
    pp.cdef_damping_minus_3 = cdef.damping_minus_3;
    pp.cdef_bits = cdef.bits;
    // (primary << 2) | secondary, the secondary as coded.
    for (i, (y, uv)) in pp
        .cdef_y_strengths
        .iter_mut()
        .zip(pp.cdef_uv_strengths.iter_mut())
        .enumerate()
    {
        let strength = |pri: &[u8; 8], sec: &[u8; 8]| {
            (pri.get(i).copied().unwrap_or(0) << 2) | (sec.get(i).copied().unwrap_or(0) & 3)
        };
        *y = strength(&cdef.y_pri_strength, &cdef.y_sec_strength);
        *uv = strength(&cdef.uv_pri_strength, &cdef.uv_sec_strength);
    }
    let lr = &h.loop_restoration;
    {
        use va::loop_restoration_fields as f;
        let [y, cb, cr] = lr.frame_restoration_type;
        pp.loop_restoration_fields = narrow_u16(
            f::YFRAME_RESTORATION_TYPE.put(u32::from(y))
                | f::CBFRAME_RESTORATION_TYPE.put(u32::from(cb))
                | f::CRFRAME_RESTORATION_TYPE.put(u32::from(cr))
                | f::LR_UNIT_SHIFT.put(u32::from(lr.unit_shift))
                | f::LR_UV_SHIFT.put(u32::from(lr.uv_shift)),
        );
    }
    for (wm, motion) in pp.wm.iter_mut().zip(h.global_motion.iter().skip(1)) {
        wm.wmtype = i32::from(motion.gm_type);
        for (to, &from) in wm.wmmat.iter_mut().zip(&motion.params) {
            *to = from;
        }
        wm.invalid = u8::from(!motion.warp_valid);
    }
    pp
}

/// `VAFilmGrainStructAV1` of the parameters in force.
fn film_grain(h: &FrameHeader) -> va::VAFilmGrainStructAV1 {
    use va::film_grain_info_fields as f;
    let g = &h.film_grain;
    let mut out = va::VAFilmGrainStructAV1::default();
    if !g.apply_grain {
        return out;
    }
    out.film_grain_info_fields = flag(f::APPLY_GRAIN, g.apply_grain)
        | flag(f::CHROMA_SCALING_FROM_LUMA, g.chroma_scaling_from_luma)
        | f::GRAIN_SCALING_MINUS_8.put(u32::from(g.grain_scaling_minus_8))
        | f::AR_COEFF_LAG.put(u32::from(g.ar_coeff_lag))
        | f::AR_COEFF_SHIFT_MINUS_6.put(u32::from(g.ar_coeff_shift_minus_6))
        | f::GRAIN_SCALE_SHIFT.put(u32::from(g.grain_scale_shift))
        | flag(f::OVERLAP_FLAG, g.overlap_flag)
        | flag(f::CLIP_TO_RESTRICTED_RANGE, g.clip_to_restricted_range);
    out.grain_seed = g.grain_seed;
    out.num_y_points = g.num_y_points;
    out.point_y_value = g.point_y_value;
    out.point_y_scaling = g.point_y_scaling;
    out.num_cb_points = g.num_cb_points;
    out.point_cb_value = g.point_cb_value;
    out.point_cb_scaling = g.point_cb_scaling;
    out.num_cr_points = g.num_cr_points;
    out.point_cr_value = g.point_cr_value;
    out.point_cr_scaling = g.point_cr_scaling;
    out.ar_coeffs_y = g.ar_coeffs_y;
    out.ar_coeffs_cb = g.ar_coeffs_cb;
    out.ar_coeffs_cr = g.ar_coeffs_cr;
    out.cb_mult = g.cb_mult;
    out.cb_luma_mult = g.cb_luma_mult;
    out.cb_offset = g.cb_offset;
    out.cr_mult = g.cr_mult;
    out.cr_luma_mult = g.cr_luma_mult;
    out.cr_offset = g.cr_offset;
    out
}

/// A tile group's `VASliceParameterBufferAV1`s: one a tile, its bytes within the group's
/// tile data, which goes in the data buffer after them.
fn slice_parameters(group: &TileGroup<'_>) -> Vec<va::VASliceParameterBufferAV1> {
    group
        .tiles
        .iter()
        .map(|tile| va::VASliceParameterBufferAV1 {
            slice_data_size: tile.size,
            slice_data_offset: tile.offset,
            slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
            tile_row: narrow_u16(tile.row),
            tile_column: narrow_u16(tile.column),
            tg_start: narrow_u16(group.start),
            tg_end: narrow_u16(group.end),
            ..va::VASliceParameterBufferAV1::default()
        })
        .collect()
}

/// The largest frame the driver lists for `AV1Profile0`'s decoder: the decoder's [`Limits`].
/// `None` where it lists no decoder, or no size.
pub fn limits(display: &Display) -> Option<Limits> {
    if !display.supports(AV1_PROFILE0, Entrypoint::VLD) {
        return None;
    }
    let listed = display
        .config_attributes(
            AV1_PROFILE0,
            Entrypoint::VLD,
            &[
                ffi::VAConfigAttribMaxPictureWidth,
                ffi::VAConfigAttribMaxPictureHeight,
            ],
        )
        .ok()?;
    Some(Limits::up_to(listed[0]?, listed[1]?))
}

/// Whether the driver's `AV1Profile0` decoder takes P010 render targets, which a 10-bit
/// stream's frames are decoded into: its `VAConfigAttribRTFormat` lists them.
pub fn decodes_ten_bit(display: &Display) -> bool {
    display.supports(AV1_PROFILE0, Entrypoint::VLD)
        && display
            .config_attributes(
                AV1_PROFILE0,
                Entrypoint::VLD,
                &[ffi::VAConfigAttribRTFormat],
            )
            .ok()
            .and_then(|listed| listed[0])
            .is_some_and(|formats| formats & SurfaceFormat::P010.rt_format() != 0)
}

#[cfg(test)]
#[cfg(unix)]
mod tests;
