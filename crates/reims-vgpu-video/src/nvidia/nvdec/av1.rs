//! AV1 on NVDEC: the codec crate's decoder over an accelerator that fills `CUVIDAV1PICPARAMS`
//! from the parsed sequence and frame headers, as ffmpeg's NVDEC AV1 decode fills it, and
//! hands NVDEC the frame's tile groups. NVIDIA's own parser is not used.
//!
//! The tile groups' tile data go as one bitstream, with `pSliceDataOffsets` giving each
//! tile's start and end, two offsets a tile: a frame of one tile group is sent in place, its
//! group's bytes as they lie in the sample, and one of more is gathered into one buffer.
//!
//! A frame that applies film grain is decoded into two surfaces: `decodePicIdx`, without
//! grain, which later frames predict from (`ref_frame_map`), and `CurrPicIdx`, with the grain
//! NVDEC applies, which is what is output.

use std::ffi::{c_char, c_int, c_short};
use std::sync::Arc;

use crate::codec::av1::frame::{FrameHeader, PRIMARY_REF_NONE};
use crate::codec::av1::sequence::SequenceHeader;
use crate::codec::av1::tile_group::TileGroup;
use crate::codec::av1::{Accelerator, FrameParams};

use super::decoder::{
    Decoder, DecoderSpec, InPlace, Picture, SPARE_SURFACES, Slices, active, surface_count,
};
use super::ffi::av1::{CUVIDAV1PICPARAMS, bits};
use super::ffi::{self, Zeroed};
use super::{ChromaFormat, NvdecCodec};
use crate::nvidia::cuda::Gpu;
use crate::nvidia::error::Error;

/// `ref_frame_map`'s entry for a slot that holds no frame, as ffmpeg sends it.
const NO_FRAME: u8 = 0xff;

/// A decoded frame's surfaces.
pub(crate) struct Av1Picture {
    /// `decodePicIdx`: the frame later frames predict from.
    pub(crate) reference: Arc<Picture>,
    /// `CurrPicIdx`, where the frame applies film grain: the frame with grain.
    pub(crate) display: Option<Arc<Picture>>,
}

impl Av1Picture {
    /// The surface that is output: the frame with grain where it applies grain.
    pub(crate) fn output(&self) -> &Arc<Picture> {
        self.display.as_ref().unwrap_or(&self.reference)
    }
}

/// The NVDEC half of the AV1 decoder.
pub(crate) struct NvdecAv1 {
    gpu: Arc<Gpu>,
    decoder: Option<Decoder>,
    /// The frames sent in place, and gathered, as [`InPlace`] counts H.264's and HEVC's.
    pub(crate) in_place: InPlace,
}

impl NvdecAv1 {
    pub(crate) fn new(gpu: Arc<Gpu>) -> Self {
        Self {
            gpu,
            decoder: None,
            in_place: InPlace::default(),
        }
    }
}

/// The decoder `sequence`'s frames need: 4:2:0 at its bit depth, at its largest frame, with a
/// surface for each slot and each slot's frame with grain, the frame being decoded and its
/// frame with grain, and spares.
pub(crate) fn spec(sequence: &SequenceHeader) -> DecoderSpec {
    DecoderSpec {
        codec: NvdecCodec::Av1,
        chroma: ChromaFormat::Yuv420,
        bit_depth: u32::from(sequence.color.bit_depth),
        width: sequence.max_frame_width,
        height: sequence.max_frame_height,
        surfaces: surface_count(2 * (8 + 1) + SPARE_SURFACES),
    }
}

impl Accelerator for NvdecAv1 {
    type Picture = Av1Picture;
    type Error = Error;

    fn configure(&mut self, sequence: &SequenceHeader) -> Result<(), Error> {
        self.decoder = None;
        self.decoder = Some(Decoder::new(Arc::clone(&self.gpu), &spec(sequence))?);
        Ok(())
    }

    fn new_picture(&mut self, header: &FrameHeader) -> Result<Av1Picture, Error> {
        let decoder = active(self.decoder.as_ref())?;
        // Mapped at the frame's size, rounded up to even sides.
        let (max_width, max_height) = decoder.size();
        let size = (
            header
                .size
                .upscaled_width
                .next_multiple_of(2)
                .min(max_width),
            header.size.frame_height.next_multiple_of(2).min(max_height),
        );
        let reference = Arc::new(decoder.picture_of_size(size)?);
        let display = if header.film_grain.apply_grain {
            Some(Arc::new(decoder.picture_of_size(size)?))
        } else {
            None
        };
        Ok(Av1Picture { reference, display })
    }

    fn decode_frame(
        &mut self,
        picture: &Av1Picture,
        params: &FrameParams<'_, Av1Picture>,
        tile_groups: &[TileGroup<'_>],
    ) -> Result<(), Error> {
        let mut pic = picture_parameters(
            (picture.output().index(), picture.reference.index()),
            params,
            |slot: &Av1Picture| slot.reference.index(),
        )?;
        let decoder = active(self.decoder.as_ref())?;
        // Two offsets a tile, each from the bitstream's start.
        let mut offsets = Vec::new();
        let mut base = 0u32;
        for group in tile_groups {
            for tile in &group.tiles {
                let start = base.saturating_add(tile.offset);
                offsets.extend([start, start.saturating_add(tile.size)]);
            }
            base = base.saturating_add(u32::try_from(group.data.len()).unwrap_or(u32::MAX));
        }
        if let [group] = tile_groups {
            self.in_place.in_place += 1;
            return decoder.decode(&mut pic, group.data, Slices::Tiles(&offsets));
        }
        self.in_place.gathered += 1;
        let bitstream: Vec<u8> = tile_groups
            .iter()
            .flat_map(|g| g.data.iter().copied())
            .collect();
        decoder.decode(&mut pic, &bitstream, Slices::Tiles(&offsets))
    }
}

fn narrow(value: impl TryInto<u8>) -> u8 {
    value.try_into().unwrap_or(u8::MAX)
}

fn int(value: impl TryInto<c_int>) -> c_int {
    value.try_into().unwrap_or(c_int::MAX)
}

/// An `i8` as the header's `char`.
fn char_of(value: i8) -> c_char {
    c_char::from_ne_bytes(value.to_ne_bytes())
}

fn put<T: Into<u32>>(word: u32, field: super::ffi::bits::Bits, value: T) -> u32 {
    field.put(word, value.into())
}

/// The frame's `CUVIDPICPARAMS`, all but its bitstream: `current` its surfaces (`CurrPicIdx`,
/// the frame output, and `decodePicIdx`, the frame later frames predict from), `index_of` the
/// surface a slot's frame lies in.
///
/// NVDEC takes surfaces, not slots, wherever the header names a frame: `ref_frame_map[]`, each
/// reference's `index`, and `primary_ref_frame`, which is the surface of the frame whose
/// contexts the frame starts from, or `0xff` where it starts from none (`PRIMARY_REF_NONE`),
/// as ffmpeg's NVDEC decode sends it: the slot's number there sends NVDEC to another surface's
/// state, and libnvcuvid faulted on the first key frame so (**measured** on the RTX 3090,
/// driver 620.70.06, 2026-11-27). A frame that names a surface where the slots hold none is
/// refused before NVDEC sees it, as the codec module refuses it first.
#[expect(
    clippy::too_many_lines,
    reason = "one assignment of every field CUVIDAV1PICPARAMS has"
)]
fn picture_parameters<P>(
    current: (c_int, c_int),
    p: &FrameParams<'_, P>,
    index_of: impl Fn(&P) -> c_int,
) -> Result<ffi::CUVIDPICPARAMS, Error> {
    let (seq, h) = (p.sequence, p.header);
    let color = &seq.color;
    let mut out = ffi::CUVIDPICPARAMS::zeroed();
    out.PicWidthInMbs = int(h.size.upscaled_width.div_ceil(16));
    out.FrameHeightInMbs = int(h.size.frame_height.div_ceil(16));
    out.CurrPicIdx = current.0;
    out.ref_pic_flag = c_int::from(h.refresh_frame_flags != 0);
    out.intra_pic_flag = c_int::from(h.frame_type.is_intra());

    let mut a = CUVIDAV1PICPARAMS::zeroed();
    a.width = h.size.upscaled_width;
    a.height = h.size.frame_height;
    a.frame_offset = u32::from(h.order_hint);
    a.decodePicIdx = current.1;
    {
        use bits::sequence as f;
        let mut w = 0;
        w = put(w, f::profile, seq.seq_profile);
        w = put(w, f::use_128x128_superblock, seq.use_128x128_superblock);
        w = put(w, f::subsampling_x, color.subsampling_x);
        w = put(w, f::subsampling_y, color.subsampling_y);
        w = put(w, f::mono_chrome, color.mono_chrome);
        w = put(w, f::bit_depth_minus8, color.bit_depth.saturating_sub(8));
        w = put(w, f::enable_filter_intra, seq.enable_filter_intra);
        w = put(w, f::enable_intra_edge_filter, seq.enable_intra_edge_filter);
        w = put(
            w,
            f::enable_interintra_compound,
            seq.enable_interintra_compound,
        );
        w = put(w, f::enable_masked_compound, seq.enable_masked_compound);
        w = put(w, f::enable_dual_filter, seq.enable_dual_filter);
        w = put(w, f::enable_order_hint, seq.enable_order_hint);
        w = put(
            w,
            f::order_hint_bits_minus1,
            seq.order_hint_bits.saturating_sub(1),
        );
        w = put(w, f::enable_jnt_comp, seq.enable_jnt_comp);
        w = put(w, f::enable_superres, seq.enable_superres);
        w = put(w, f::enable_cdef, seq.enable_cdef);
        w = put(w, f::enable_restoration, seq.enable_restoration);
        w = put(w, f::enable_fgs, seq.film_grain_params_present);
        a.sequence_word = w;
    }
    {
        use bits::frame as f;
        let mut w = 0;
        w = put(w, f::frame_type, h.frame_type.code());
        w = put(w, f::show_frame, h.show_frame);
        w = put(w, f::disable_cdf_update, h.disable_cdf_update);
        w = put(
            w,
            f::allow_screen_content_tools,
            h.allow_screen_content_tools,
        );
        w = put(w, f::force_integer_mv, h.force_integer_mv);
        w = put(w, f::coded_denom, h.size.coded_denom);
        w = put(w, f::allow_intrabc, h.allow_intrabc);
        w = put(w, f::allow_high_precision_mv, h.allow_high_precision_mv);
        w = put(w, f::interp_filter, h.interpolation_filter);
        w = put(w, f::switchable_motion_mode, h.is_motion_mode_switchable);
        w = put(w, f::use_ref_frame_mvs, h.use_ref_frame_mvs);
        w = put(
            w,
            f::disable_frame_end_update_cdf,
            h.disable_frame_end_update_cdf,
        );
        w = put(w, f::delta_q_present, h.delta_q_present);
        w = put(w, f::delta_q_res, h.delta_q_res);
        w = put(w, f::using_qmatrix, h.quantization.using_qmatrix);
        w = put(w, f::coded_lossless, h.coded_lossless);
        w = put(w, f::use_superres, h.size.use_superres);
        w = put(w, f::tx_mode, h.tx_mode);
        w = put(w, f::reference_mode, h.reference_select);
        w = put(w, f::allow_warped_motion, h.allow_warped_motion);
        w = put(w, f::reduced_tx_set, h.reduced_tx_set);
        w = put(w, f::skip_mode, h.skip_mode_present);
        a.frame_word = w;
    }
    let tiles = &h.tile_info;
    {
        use bits::tiling as f;
        let mut w = 0;
        w = put(w, f::num_tile_cols, tiles.cols());
        w = put(w, f::num_tile_rows, tiles.rows());
        w = put(w, f::context_update_tile_id, tiles.context_update_tile_id);
        a.tiling_word = w;
    }
    for (to, &from) in a.tile_widths.iter_mut().zip(&tiles.col_widths_sb) {
        *to = u16::try_from(from).unwrap_or(u16::MAX);
    }
    for (to, &from) in a.tile_heights.iter_mut().zip(&tiles.row_heights_sb) {
        *to = u16::try_from(from).unwrap_or(u16::MAX);
    }
    let cdef = &h.cdef;
    a.cdef_word = narrow(put(
        put(0, bits::cdef::cdef_damping_minus_3, cdef.damping_minus_3),
        bits::cdef::cdef_bits,
        cdef.bits,
    ));
    // Primary in bits 0 to 3, secondary as coded in 4 to 7.
    for (i, (y, uv)) in a
        .cdef_y_strength
        .iter_mut()
        .zip(a.cdef_uv_strength.iter_mut())
        .enumerate()
    {
        let strength = |pri: &[u8; 8], sec: &[u8; 8]| {
            (pri.get(i).copied().unwrap_or(0) & 0xf) | (sec.get(i).copied().unwrap_or(0) << 4)
        };
        *y = strength(&cdef.y_pri_strength, &cdef.y_sec_strength);
        *uv = strength(&cdef.uv_pri_strength, &cdef.uv_sec_strength);
    }
    if h.skip_mode_present {
        a.skip_mode_word = narrow(put(
            put(0, bits::skip_mode::SkipModeFrame0, h.skip_mode_frame[0]),
            bits::skip_mode::SkipModeFrame1,
            h.skip_mode_frame[1],
        ));
    }
    let q = &h.quantization;
    a.base_qindex = q.base_q_idx;
    a.qp_y_dc_delta_q = char_of(q.delta_q_y_dc);
    a.qp_u_dc_delta_q = char_of(q.delta_q_u_dc);
    a.qp_v_dc_delta_q = char_of(q.delta_q_v_dc);
    a.qp_u_ac_delta_q = char_of(q.delta_q_u_ac);
    a.qp_v_ac_delta_q = char_of(q.delta_q_v_ac);
    a.qm_y = q.qm_y;
    a.qm_u = q.qm_u;
    a.qm_v = q.qm_v;
    let segmentation = &h.segmentation;
    {
        use bits::segmentation as f;
        let mut w = 0;
        w = put(w, f::segmentation_enabled, segmentation.enabled);
        w = put(w, f::segmentation_update_map, segmentation.update_map);
        w = put(w, f::segmentation_update_data, segmentation.update_data);
        w = put(
            w,
            f::segmentation_temporal_update,
            segmentation.temporal_update,
        );
        a.segmentation_word = narrow(w);
    }
    if segmentation.enabled {
        for ((mask, data), (enabled, values)) in a
            .segmentation_feature_mask
            .iter_mut()
            .zip(a.segmentation_feature_data.iter_mut())
            .zip(
                segmentation
                    .features
                    .enabled
                    .iter()
                    .zip(&segmentation.features.data),
            )
        {
            *mask = enabled
                .iter()
                .rev()
                .fold(0u8, |mask, &enabled| (mask << 1) | u8::from(enabled));
            for (to, &from) in data.iter_mut().zip(values) {
                *to = c_short::from(from);
            }
        }
    }
    let lf = &h.loop_filter;
    a.loop_filter_level = [lf.level[0], lf.level[1]];
    a.loop_filter_level_u = lf.level[2];
    a.loop_filter_level_v = lf.level[3];
    a.loop_filter_sharpness = lf.sharpness;
    for (to, &from) in a.loop_filter_ref_deltas.iter_mut().zip(&lf.ref_deltas) {
        *to = char_of(from);
    }
    for (to, &from) in a.loop_filter_mode_deltas.iter_mut().zip(&lf.mode_deltas) {
        *to = char_of(from);
    }
    {
        use bits::loop_filter as f;
        let mut w = 0;
        w = put(w, f::loop_filter_delta_enabled, lf.delta_enabled);
        w = put(w, f::loop_filter_delta_update, lf.delta_update);
        w = put(w, f::delta_lf_present, h.delta_lf_present);
        w = put(w, f::delta_lf_res, h.delta_lf_res);
        w = put(w, f::delta_lf_multi, h.delta_lf_multi);
        a.loop_filter_word = narrow(w);
    }
    // 1 + lr_unit_shift for luma, less lr_uv_shift for chroma, as ffmpeg gives them.
    let lr = &h.loop_restoration;
    let luma = lr.unit_shift.saturating_add(1);
    let chroma = luma.saturating_sub(lr.uv_shift);
    a.lr_unit_size = [luma, chroma, chroma];
    a.lr_type = lr.frame_restoration_type;
    for (map, slot) in a.ref_frame_map.iter_mut().zip(&p.slots) {
        *map = slot
            .as_ref()
            .map_or(NO_FRAME, |slot| narrow(index_of(slot.picture)));
    }
    let surface_of = |index: u8| {
        a.ref_frame_map
            .get(usize::from(index))
            .copied()
            .unwrap_or(NO_FRAME)
    };
    a.primary_ref_frame = if h.primary_ref_frame == PRIMARY_REF_NONE {
        NO_FRAME
    } else {
        let index = h
            .ref_frame_idx
            .get(usize::from(h.primary_ref_frame))
            .copied()
            .unwrap_or(u8::MAX);
        match surface_of(index) {
            NO_FRAME => return Err(Error::Bitstream("a primary reference no slot holds".into())),
            surface => surface,
        }
    };
    {
        use bits::layer as f;
        a.layer_word = narrow(put(
            put(0, f::temporal_layer_id, h.temporal_id),
            f::spatial_layer_id,
            h.spatial_id,
        ));
    }
    // Every frame's, as ffmpeg's NVDEC decode sends them: an intra frame's name slot 0.
    for (reference, &index) in a.ref_frame.iter_mut().zip(&h.ref_frame_idx) {
        reference.index = surface_of(index);
        if let Some(slot) = p.slots.get(usize::from(index)).and_then(Option::as_ref) {
            reference.width = slot.state.size.upscaled_width;
            reference.height = slot.state.size.frame_height;
        } else if !h.frame_type.is_intra() {
            return Err(Error::Bitstream("a reference no slot holds".into()));
        }
    }
    // `invalid` where the reference has no global motion, as ffmpeg's NVDEC decode sends it.
    for (motion, gm) in a
        .global_motion
        .iter_mut()
        .zip(h.global_motion.iter().skip(1))
    {
        use bits::motion as f;
        motion.motion_word = narrow(put(
            put(0, f::invalid, !gm.is_global),
            f::wmtype,
            gm.gm_type,
        ));
        motion.wmmat = gm.params;
    }
    film_grain(h, &mut a);
    out.CodecSpecific.av1 = a;
    Ok(out)
}

/// The film grain parameters in force, where they apply grain.
fn film_grain(h: &FrameHeader, a: &mut CUVIDAV1PICPARAMS) {
    use bits::film_grain as f;
    let g = &h.film_grain;
    if !g.apply_grain {
        return;
    }
    let mut w = 0;
    w = put(w, f::apply_grain, g.apply_grain);
    w = put(w, f::overlap_flag, g.overlap_flag);
    w = put(w, f::scaling_shift_minus8, g.grain_scaling_minus_8);
    w = put(w, f::chroma_scaling_from_luma, g.chroma_scaling_from_luma);
    w = put(w, f::ar_coeff_lag, g.ar_coeff_lag);
    w = put(w, f::ar_coeff_shift_minus6, g.ar_coeff_shift_minus_6);
    w = put(w, f::grain_scale_shift, g.grain_scale_shift);
    w = put(w, f::clip_to_restricted_range, g.clip_to_restricted_range);
    a.film_grain_word = u16::try_from(w).unwrap_or(0);
    a.num_y_points = g.num_y_points;
    a.num_cb_points = g.num_cb_points;
    a.num_cr_points = g.num_cr_points;
    let points = |to: &mut [[u8; 2]], values: &[u8], scalings: &[u8]| {
        for (point, (&value, &scaling)) in to.iter_mut().zip(values.iter().zip(scalings)) {
            *point = [value, scaling];
        }
    };
    points(
        &mut a.scaling_points_y,
        &g.point_y_value,
        &g.point_y_scaling,
    );
    points(
        &mut a.scaling_points_cb,
        &g.point_cb_value,
        &g.point_cb_scaling,
    );
    points(
        &mut a.scaling_points_cr,
        &g.point_cr_value,
        &g.point_cr_scaling,
    );
    a.random_seed = g.grain_seed;
    let coefficients = |to: &mut [c_short], from: &[i8]| {
        for (to, &from) in to.iter_mut().zip(from) {
            *to = c_short::from(from);
        }
    };
    coefficients(&mut a.ar_coeffs_y, &g.ar_coeffs_y);
    coefficients(&mut a.ar_coeffs_cb, &g.ar_coeffs_cb);
    coefficients(&mut a.ar_coeffs_cr, &g.ar_coeffs_cr);
    a.cb_mult = g.cb_mult;
    a.cb_luma_mult = g.cb_luma_mult;
    a.cb_offset = c_short::try_from(g.cb_offset).unwrap_or(0);
    a.cr_mult = g.cr_mult;
    a.cr_luma_mult = g.cr_luma_mult;
    a.cr_offset = c_short::try_from(g.cr_offset).unwrap_or(0);
}

#[cfg(test)]
mod tests {
    //! What [`picture_parameters`] names of each frame of our streams, decoded by the codec
    //! module over an accelerator whose pictures are numbers: the surfaces NVDEC is sent
    //! wherever the header names a frame, and the refusal of a frame naming an empty slot.

    use super::*;
    use crate::codec::av1::frame::NUM_REF_FRAMES;
    use crate::codec::av1::sequence::SequenceHeader;
    use crate::codec::av1::tests::{LIMITS, STREAMS, samples, without_temporal_delimiters};
    use crate::codec::av1::{Decoder as Av1Decoder, Error as Av1Error};

    /// A frame as sent: its header, the surface each slot held, and the parameters.
    struct Sent {
        header: FrameHeader,
        held: [Option<u32>; NUM_REF_FRAMES],
        params: CUVIDAV1PICPARAMS,
        ref_pic_flag: c_int,
        current: (c_int, c_int),
    }

    /// Fills each frame's parameters, its pictures numbered from 1.
    #[derive(Default)]
    struct Filler {
        next: u32,
        sent: Vec<Sent>,
    }

    impl Accelerator for Filler {
        type Picture = u32;
        type Error = Av1Error;

        fn configure(&mut self, _sequence: &SequenceHeader) -> Result<(), Av1Error> {
            Ok(())
        }

        fn new_picture(&mut self, _header: &FrameHeader) -> Result<u32, Av1Error> {
            self.next += 1;
            Ok(self.next)
        }

        fn decode_frame(
            &mut self,
            picture: &u32,
            params: &FrameParams<'_, u32>,
            _tile_groups: &[TileGroup<'_>],
        ) -> Result<(), Av1Error> {
            let current = (
                c_int::try_from(*picture).unwrap(),
                c_int::try_from(*picture).unwrap(),
            );
            let out = picture_parameters(current, params, |p: &u32| c_int::try_from(*p).unwrap())
                .expect("a frame of our streams");
            // SAFETY: the union's AV1 member, which picture_parameters wrote.
            let av1 = unsafe { out.CodecSpecific.av1 };
            self.sent.push(Sent {
                header: params.header.clone(),
                held: params
                    .slots
                    .each_ref()
                    .map(|s| s.as_ref().map(|s| *s.picture)),
                params: av1,
                ref_pic_flag: out.ref_pic_flag,
                current,
            });
            Ok(())
        }
    }

    #[test]
    fn each_frame_names_surfaces_where_its_header_names_frames() {
        for stream in STREAMS.iter().filter(|s| s.decodes) {
            let mut decoder = Av1Decoder::new(Filler::default(), LIMITS);
            for sample in samples(stream.ivf) {
                decoder
                    .decode(&without_temporal_delimiters(sample), 0)
                    .unwrap();
            }
            let sent = &decoder.accelerator().sent;
            assert!(!sent.is_empty(), "{}", stream.name);
            let surface = |held: &[Option<u32>; NUM_REF_FRAMES], slot: u8| {
                held[usize::from(slot)].map_or(NO_FRAME, |p| u8::try_from(p).unwrap())
            };
            for (n, frame) in sent.iter().enumerate() {
                let (h, a) = (&frame.header, &frame.params);
                let what = format!("{} frame {n}", stream.name);
                // The primary reference is a surface, or none: never the slot's number.
                let primary = if h.primary_ref_frame == PRIMARY_REF_NONE {
                    NO_FRAME
                } else {
                    surface(
                        &frame.held,
                        h.ref_frame_idx[usize::from(h.primary_ref_frame)],
                    )
                };
                assert_eq!(a.primary_ref_frame, primary, "{what}");
                if h.primary_ref_frame != PRIMARY_REF_NONE {
                    assert_ne!(a.primary_ref_frame, NO_FRAME, "{what}");
                }
                for (slot, &map) in a.ref_frame_map.iter().enumerate() {
                    assert_eq!(
                        map,
                        surface(&frame.held, u8::try_from(slot).unwrap()),
                        "{what}"
                    );
                }
                for (reference, &slot) in a.ref_frame.iter().zip(&h.ref_frame_idx) {
                    assert_eq!(reference.index, surface(&frame.held, slot), "{what}");
                    if !h.frame_type.is_intra() {
                        assert_ne!(reference.index, NO_FRAME, "{what}");
                        assert_ne!(reference.width, 0, "{what}");
                    }
                }
                assert_eq!(
                    frame.ref_pic_flag,
                    c_int::from(h.refresh_frame_flags != 0),
                    "{what}"
                );
                assert_eq!(a.decodePicIdx, frame.current.1, "{what}");
            }
            // The first frame, a key frame shown at once, starts from no frame.
            assert_eq!(
                sent[0].params.primary_ref_frame, NO_FRAME,
                "{}",
                stream.name
            );
            assert!(
                sent[0].params.ref_frame_map.iter().all(|&m| m == NO_FRAME),
                "{}",
                stream.name
            );
        }
    }

    #[test]
    fn a_frame_naming_a_slot_that_holds_no_frame_is_refused_before_nvdec() {
        // An inter frame of our streams, with a primary reference, handed slots that hold
        // nothing: what a stream whose references were emptied would ask, had the codec module
        // not refused it first.
        let stream = STREAMS.iter().find(|s| s.name == "svt-main8").unwrap();
        let mut decoder = Av1Decoder::new(Filler::default(), LIMITS);
        for sample in samples(stream.ivf) {
            decoder
                .decode(&without_temporal_delimiters(sample), 0)
                .unwrap();
        }
        let sequence = decoder.sequence().unwrap().clone();
        let sent = &decoder.accelerator().sent;
        let inter = sent
            .iter()
            .find(|f| {
                !f.header.frame_type.is_intra() && f.header.primary_ref_frame != PRIMARY_REF_NONE
            })
            .expect("an inter frame with a primary reference");
        let empty = FrameParams::<u32> {
            sequence: &sequence,
            header: &inter.header,
            slots: Default::default(),
        };
        assert!(matches!(
            picture_parameters((1, 1), &empty, |p: &u32| c_int::try_from(*p).unwrap()),
            Err(Error::Bitstream(_))
        ));
        let mut no_primary = inter.header.clone();
        no_primary.primary_ref_frame = PRIMARY_REF_NONE;
        let empty = FrameParams::<u32> {
            sequence: &sequence,
            header: &no_primary,
            slots: Default::default(),
        };
        assert!(matches!(
            picture_parameters((1, 1), &empty, |p: &u32| c_int::try_from(*p).unwrap()),
            Err(Error::Bitstream(_))
        ));
    }
}
