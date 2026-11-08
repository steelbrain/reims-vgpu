//! libva's encode buffers, filled from the encoder core's stream and
//! picture parameters: the same values the packed headers carry, so that a
//! driver reading either sees one stream (va_enc_h264.h, va_enc_hevc.h,
//! va.h's misc parameters).

use super::super::ffi::{
    self, VAPictureH264,
    encode::{
        self as enc, VAEncMiscParameterBuffer, VAEncMiscParameterBufferQualityLevel,
        VAEncMiscParameterFrameRate, VAEncMiscParameterHRD, VAEncMiscParameterRateControl,
        VAEncPictureParameterBufferH264, VAEncPictureParameterBufferHEVC,
        VAEncSequenceParameterBufferH264, VAEncSequenceParameterBufferHEVC,
        VAEncSliceParameterBufferH264, VAEncSliceParameterBufferHEVC, VAPictureHEVC,
        h264_pic_fields as hp, h264_seq_fields as hs, h264_vui_fields as hv, hevc_pic_fields as ep,
        hevc_seq_fields as es, hevc_slice_fields as el, hevc_vui_fields as ev, rc_flags,
    },
};
use crate::codec::encode::encoder::{PictureParams, ReferencePicture, StreamParams};
use crate::codec::encode::{EncoderSettings, FrameType, RateControl, h264, hevc};

/// Plain C data, for which all zero bytes is a valid value: integers,
/// arrays of them, and null pointers.
pub(crate) trait Plain: Copy {
    fn zeroed() -> Self {
        // SAFETY: implemented only for libva's repr(C) structs of integers
        // and arrays of them, where zero is every field's valid value.
        unsafe { core::mem::zeroed() }
    }
}

impl Plain for VAEncSequenceParameterBufferH264 {}
impl Plain for VAEncPictureParameterBufferH264 {}
impl Plain for VAEncSliceParameterBufferH264 {}
impl Plain for VAEncSequenceParameterBufferHEVC {}
impl Plain for VAEncPictureParameterBufferHEVC {}
impl Plain for VAEncSliceParameterBufferHEVC {}

/// Frames from one IDR picture to the next, for the driver's rate control
/// to budget a GOP by: the interval asked for, or ten seconds of frames.
fn intra_period(stream: &StreamParams<'_>) -> u32 {
    if stream.intra_period > 0 {
        return stream.intra_period;
    }
    let fps = stream.settings.frame_rate.as_f64();
    let frames = (fps * 10.0).round();
    if frames >= 1.0 && frames < f64::from(u32::MAX) {
        frames as u32
    } else {
        300
    }
}

/// Ours: the least bitrate the driver is given, and so the least HRD
/// buffer, a second of it. Below it VCN's output no longer falls: H.264
/// High and HEVC Main at 640x480 and 1280x720, of `testsrc2` and
/// `mandelbrot`, made the same streams at 32,000 bit/s as at the lowest
/// average swept, and 1920x1080 and 3840x2160 encoded at it, on the RX
/// 7900 XTX and the Radeon 780M (2026-10-24). So a guest's
/// `AverageBitRate` of 1 gets the stream the encoder can make, as from
/// Apple's AVE, and the driver never gets a buffer of a bit.
pub(crate) const MIN_BITS_PER_SECOND: u32 = 32_000;

/// The rate control as the driver gets it: no rate under
/// [`MIN_BITS_PER_SECOND`], and a peak at least the average.
fn floored(rate_control: RateControl) -> RateControl {
    let floor = |rate: u32| rate.max(MIN_BITS_PER_SECOND);
    match rate_control {
        RateControl::ConstantQp { .. } => rate_control,
        RateControl::ConstantBitRate { bits_per_second } => RateControl::ConstantBitRate {
            bits_per_second: floor(bits_per_second),
        },
        RateControl::VariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
        } => {
            let average = floor(average_bits_per_second);
            RateControl::VariableBitRate {
                average_bits_per_second: average,
                peak_bits_per_second: peak_bits_per_second.max(average),
            }
        }
        RateControl::QualityVariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
            quality,
        } => {
            let average = floor(average_bits_per_second);
            RateControl::QualityVariableBitRate {
                average_bits_per_second: average,
                peak_bits_per_second: peak_bits_per_second.max(average),
                quality,
            }
        }
    }
}

/// The bitrate the sequence parameters carry: the target.
fn target_bits_per_second(settings: &EncoderSettings) -> u32 {
    match floored(settings.rate_control) {
        RateControl::ConstantQp { .. } => 0,
        RateControl::ConstantBitRate { bits_per_second } => bits_per_second,
        RateControl::VariableBitRate {
            average_bits_per_second,
            ..
        }
        | RateControl::QualityVariableBitRate {
            average_bits_per_second,
            ..
        } => average_bits_per_second,
    }
}

fn vui_aspect(sar: Option<(u16, u16)>) -> (bool, u8, u32, u32) {
    match sar {
        Some((h, v)) if h == v => (true, 1, 1, 1),
        Some((h, v)) => (true, 255, u32::from(h), u32::from(v)),
        None => (false, 0, 0, 0),
    }
}

pub(crate) fn h264_sequence(
    stream: &StreamParams<'_>,
    settings: &EncoderSettings,
    sps: &h264::Sps,
) -> VAEncSequenceParameterBufferH264 {
    let mut p = VAEncSequenceParameterBufferH264::zeroed();
    p.seq_parameter_set_id = 0;
    p.level_idc = sps.level_idc;
    p.intra_period = intra_period(stream);
    p.intra_idr_period = p.intra_period;
    p.ip_period = stream.b_frames + 1;
    p.bits_per_second = target_bits_per_second(settings);
    p.max_num_ref_frames = u32::from(sps.max_num_ref_frames);
    p.picture_width_in_mbs = u16::try_from(sps.width_in_mbs).unwrap_or(u16::MAX);
    p.picture_height_in_mbs = u16::try_from(sps.height_in_mbs).unwrap_or(u16::MAX);
    p.seq_fields = hs::CHROMA_FORMAT_IDC.put(1)
        | hs::FRAME_MBS_ONLY_FLAG.put(1)
        | hs::DIRECT_8X8_INFERENCE_FLAG.put(1)
        | hs::LOG2_MAX_FRAME_NUM_MINUS4.put(u32::from(h264::LOG2_MAX_FRAME_NUM_MINUS4))
        | hs::PIC_ORDER_CNT_TYPE.put(0)
        | hs::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4
            .put(u32::from(h264::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4));
    let cropped = sps.crop_right > 0 || sps.crop_bottom > 0;
    p.frame_cropping_flag = u8::from(cropped);
    p.frame_crop_right_offset = sps.crop_right / 2;
    p.frame_crop_bottom_offset = sps.crop_bottom / 2;
    p.vui_parameters_present_flag = 1;
    let (aspect, idc, sar_width, sar_height) = vui_aspect(sps.vui.sample_aspect_ratio);
    p.vui_fields = hv::ASPECT_RATIO_INFO_PRESENT_FLAG.put(u32::from(aspect))
        | hv::TIMING_INFO_PRESENT_FLAG.put(u32::from(sps.vui.timing.is_some()))
        | hv::BITSTREAM_RESTRICTION_FLAG.put(1)
        | hv::LOG2_MAX_MV_LENGTH_HORIZONTAL.put(15)
        | hv::LOG2_MAX_MV_LENGTH_VERTICAL.put(15)
        | hv::MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG.put(1);
    p.aspect_ratio_idc = idc;
    p.sar_width = sar_width;
    p.sar_height = sar_height;
    if let Some((num_units_in_tick, time_scale)) = sps.vui.timing {
        p.num_units_in_tick = num_units_in_tick;
        p.time_scale = time_scale;
    }
    p
}

fn h264_reference(surface: ffi::VASurfaceID, reference: &ReferencePicture) -> VAPictureH264 {
    VAPictureH264 {
        picture_id: surface,
        frame_idx: reference.frame_num,
        flags: ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE,
        top_field_order_cnt: reference.poc,
        bottom_field_order_cnt: reference.poc,
        va_reserved: [0; 4],
    }
}

/// Looks a reference's reconstructed surface up by its number.
pub(crate) type SurfaceOf<'a> = &'a dyn Fn(u64) -> Option<ffi::VASurfaceID>;

pub(crate) fn h264_picture(
    p: &PictureParams<'_>,
    pps: &h264::Pps,
    reconstructed: ffi::VASurfaceID,
    coded_buffer: ffi::VABufferID,
    surface_of: SurfaceOf<'_>,
) -> VAEncPictureParameterBufferH264 {
    let mut b = VAEncPictureParameterBufferH264::zeroed();
    b.curr_pic = VAPictureH264 {
        picture_id: reconstructed,
        frame_idx: p.frame_num,
        flags: if p.reference.is_some() {
            ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
        } else {
            0
        },
        top_field_order_cnt: p.poc,
        bottom_field_order_cnt: p.poc,
        va_reserved: [0; 4],
    };
    b.reference_frames = [VAPictureH264::invalid(); 16];
    for (slot, reference) in b.reference_frames.iter_mut().zip(p.dpb) {
        if let Some(surface) = surface_of(reference.number) {
            *slot = h264_reference(surface, reference);
        }
    }
    b.coded_buf = coded_buffer;
    b.frame_num = u16::try_from(p.frame_num).unwrap_or(0);
    b.pic_init_qp = pps.pic_init_qp;
    b.pic_fields = hp::IDR_PIC_FLAG.put(u32::from(p.frame_type == FrameType::Idr))
        | hp::REFERENCE_PIC_FLAG.put(u32::from(p.reference.is_some()))
        | hp::ENTROPY_CODING_MODE_FLAG.put(u32::from(
            pps.entropy == crate::codec::encode::Entropy::Cabac,
        ))
        | hp::TRANSFORM_8X8_MODE_FLAG.put(u32::from(pps.transform_8x8_mode_flag))
        | hp::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG.put(1);
    b
}

pub(crate) fn h264_slice(
    p: &PictureParams<'_>,
    sps: &h264::Sps,
    slice: &h264::SliceHeader,
    surface_of: SurfaceOf<'_>,
) -> VAEncSliceParameterBufferH264 {
    let mut b = VAEncSliceParameterBufferH264::zeroed();
    b.macroblock_address = 0;
    b.num_macroblocks = sps.width_in_mbs * sps.height_in_mbs;
    b.macroblock_info = ffi::VA_INVALID_ID;
    b.slice_type = slice.slice_type.raw();
    b.idr_pic_id = u16::try_from(slice.idr_pic_id).unwrap_or(0);
    b.pic_order_cnt_lsb = u16::try_from(slice.pic_order_cnt_lsb).unwrap_or(0);
    b.direct_spatial_mv_pred_flag = u8::from(slice.slice_type == h264::SliceType::B);
    b.ref_pic_list0 = [VAPictureH264::invalid(); 32];
    b.ref_pic_list1 = [VAPictureH264::invalid(); 32];
    for (list, reference) in [
        (&mut b.ref_pic_list0, p.list0),
        (&mut b.ref_pic_list1, p.list1),
    ] {
        if let Some(reference) = reference
            && let Some(surface) = surface_of(reference.number)
        {
            list[0] = h264_reference(surface, &reference);
        }
    }
    b.slice_qp_delta = i8::try_from(slice.slice_qp_delta).unwrap_or(0);
    b
}

pub(crate) fn hevc_sequence(
    stream: &StreamParams<'_>,
    settings: &EncoderSettings,
    sps: &hevc::Sps,
) -> VAEncSequenceParameterBufferHEVC {
    let t = &sps.tools;
    let depth = u32::from(sps.profile.bit_depth()) - 8;
    let mut p = VAEncSequenceParameterBufferHEVC::zeroed();
    p.general_profile_idc = sps.profile.profile_idc();
    p.general_level_idc = sps.level_idc;
    p.general_tier_flag = 0;
    p.intra_period = intra_period(stream);
    p.intra_idr_period = p.intra_period;
    p.ip_period = stream.b_frames + 1;
    p.bits_per_second = target_bits_per_second(settings);
    p.pic_width_in_luma_samples = u16::try_from(sps.width).unwrap_or(u16::MAX);
    p.pic_height_in_luma_samples = u16::try_from(sps.height).unwrap_or(u16::MAX);
    p.seq_fields = es::CHROMA_FORMAT_IDC.put(1)
        | es::BIT_DEPTH_LUMA_MINUS8.put(depth)
        | es::BIT_DEPTH_CHROMA_MINUS8.put(depth)
        | es::STRONG_INTRA_SMOOTHING_ENABLED_FLAG.put(u32::from(t.strong_intra_smoothing))
        | es::AMP_ENABLED_FLAG.put(u32::from(t.amp))
        | es::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG.put(u32::from(t.sao))
        | es::LOW_DELAY_SEQ.put(u32::from(stream.b_frames == 0));
    p.log2_min_luma_coding_block_size_minus3 = t.log2_min_cb_size - 3;
    p.log2_diff_max_min_luma_coding_block_size = t.log2_ctb_size - t.log2_min_cb_size;
    p.log2_min_transform_block_size_minus2 = t.log2_min_tb_size - 2;
    p.log2_diff_max_min_transform_block_size = t.log2_max_tb_size - t.log2_min_tb_size;
    p.max_transform_hierarchy_depth_inter = t.max_transform_hierarchy_depth_inter;
    p.max_transform_hierarchy_depth_intra = t.max_transform_hierarchy_depth_intra;
    p.vui_parameters_present_flag = 1;
    let (aspect, idc, sar_width, sar_height) = vui_aspect(sps.vui.sample_aspect_ratio);
    p.vui_fields = ev::ASPECT_RATIO_INFO_PRESENT_FLAG.put(u32::from(aspect))
        | ev::VUI_TIMING_INFO_PRESENT_FLAG.put(u32::from(sps.vui.timing.is_some()))
        | ev::BITSTREAM_RESTRICTION_FLAG.put(1)
        | ev::MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG.put(1)
        | ev::RESTRICTED_REF_PIC_LISTS_FLAG.put(1)
        | ev::LOG2_MAX_MV_LENGTH_HORIZONTAL.put(15)
        | ev::LOG2_MAX_MV_LENGTH_VERTICAL.put(15);
    p.aspect_ratio_idc = idc;
    p.sar_width = sar_width;
    p.sar_height = sar_height;
    if let Some((num_units_in_tick, time_scale)) = sps.vui.timing {
        p.vui_num_units_in_tick = num_units_in_tick;
        p.vui_time_scale = time_scale;
    }
    p.max_bytes_per_pic_denom = 2;
    p.max_bits_per_min_cu_denom = 1;
    p
}

fn hevc_reference(surface: ffi::VASurfaceID, reference: &ReferencePicture) -> VAPictureHEVC {
    VAPictureHEVC {
        picture_id: surface,
        pic_order_cnt: reference.poc,
        flags: 0,
        va_reserved: [0; 4],
    }
}

pub(crate) fn hevc_picture(
    p: &PictureParams<'_>,
    pps: &hevc::Pps,
    slice: &hevc::SliceHeader,
    reconstructed: ffi::VASurfaceID,
    coded_buffer: ffi::VABufferID,
    surface_of: SurfaceOf<'_>,
) -> VAEncPictureParameterBufferHEVC {
    let t = &pps.tools;
    let mut b = VAEncPictureParameterBufferHEVC::zeroed();
    b.decoded_curr_pic = VAPictureHEVC {
        picture_id: reconstructed,
        pic_order_cnt: p.poc,
        flags: 0,
        va_reserved: [0; 4],
    };
    b.reference_frames = [VAPictureHEVC::invalid(); 15];
    for (slot, reference) in b.reference_frames.iter_mut().zip(p.dpb) {
        if let Some(surface) = surface_of(reference.number) {
            *slot = hevc_reference(surface, reference);
        }
    }
    b.coded_buf = coded_buffer;
    // No temporal motion vector prediction: no collocated picture.
    b.collocated_ref_pic_index = 0xff;
    b.pic_init_qp = pps.init_qp;
    b.nal_unit_type = slice.nal_unit_type;
    // coding_type: 1 I, 2 P, 3 B (va_enc_hevc.h).
    let coding_type = match p.frame_type {
        FrameType::Idr => 1,
        FrameType::P => 2,
        FrameType::B => 3,
    };
    b.pic_fields = ep::IDR_PIC_FLAG.put(u32::from(p.frame_type == FrameType::Idr))
        | ep::CODING_TYPE.put(coding_type)
        | ep::REFERENCE_PIC_FLAG.put(u32::from(p.reference.is_some()))
        | ep::CONSTRAINED_INTRA_PRED_FLAG.put(u32::from(t.constrained_intra_pred))
        | ep::TRANSFORM_SKIP_ENABLED_FLAG.put(u32::from(t.transform_skip))
        | ep::CU_QP_DELTA_ENABLED_FLAG.put(u32::from(t.cu_qp_delta))
        | ep::PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG.put(1);
    b
}

pub(crate) fn hevc_slice(
    p: &PictureParams<'_>,
    sps: &hevc::Sps,
    slice: &hevc::SliceHeader,
    surface_of: SurfaceOf<'_>,
) -> VAEncSliceParameterBufferHEVC {
    let ctb = 1u32 << sps.tools.log2_ctb_size;
    let mut b = VAEncSliceParameterBufferHEVC::zeroed();
    b.slice_segment_address = 0;
    b.num_ctu_in_slice = sps.width.div_ceil(ctb) * sps.height.div_ceil(ctb);
    b.slice_type = slice.slice_type.raw();
    b.ref_pic_list0 = [VAPictureHEVC::invalid(); 15];
    b.ref_pic_list1 = [VAPictureHEVC::invalid(); 15];
    for (list, reference) in [
        (&mut b.ref_pic_list0, p.list0),
        (&mut b.ref_pic_list1, p.list1),
    ] {
        if let Some(reference) = reference
            && let Some(surface) = surface_of(reference.number)
        {
            list[0] = hevc_reference(surface, &reference);
        }
    }
    b.max_num_merge_cand = slice.max_num_merge_cand;
    b.slice_qp_delta = i8::try_from(slice.slice_qp_delta).unwrap_or(0);
    let sao = u32::from(sps.tools.sao);
    b.slice_fields = el::LAST_SLICE_OF_PIC_FLAG.put(1)
        | el::SLICE_SAO_LUMA_FLAG.put(sao)
        | el::SLICE_SAO_CHROMA_FLAG.put(sao)
        | el::SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG.put(1)
        | el::COLLOCATED_FROM_L0_FLAG.put(1);
    b
}

/// The rate control misc parameter, `reset` when it changes mid-stream;
/// none for a constant QP.
pub(crate) fn rate_control(
    settings: &EncoderSettings,
    reset: bool,
) -> Option<VAEncMiscParameterBuffer<VAEncMiscParameterRateControl>> {
    let (bits_per_second, target_percentage, quality_factor, stuffing) =
        match floored(settings.rate_control) {
            RateControl::ConstantQp { .. } => return None,
            RateControl::ConstantBitRate { bits_per_second } => (bits_per_second, 100, 0, true),
            RateControl::VariableBitRate {
                average_bits_per_second,
                peak_bits_per_second,
            } => (
                peak_bits_per_second,
                percentage(average_bits_per_second, peak_bits_per_second),
                0,
                false,
            ),
            RateControl::QualityVariableBitRate {
                average_bits_per_second,
                peak_bits_per_second,
                quality,
            } => (
                peak_bits_per_second,
                percentage(average_bits_per_second, peak_bits_per_second),
                u32::from(quality),
                false,
            ),
        };
    Some(VAEncMiscParameterBuffer {
        type_: enc::VAEncMiscParameterTypeRateControl,
        data: VAEncMiscParameterRateControl {
            bits_per_second,
            target_percentage,
            window_size: 1000,
            rc_flags: rc_flags::RESET.put(u32::from(reset))
                | rc_flags::DISABLE_FRAME_SKIP.put(1)
                | rc_flags::DISABLE_BIT_STUFFING.put(u32::from(!stuffing)),
            quality_factor,
            ..VAEncMiscParameterRateControl::default()
        },
    })
}

fn percentage(average: u32, peak: u32) -> u32 {
    let percent = u64::from(average) * 100 / u64::from(peak.max(1));
    u32::try_from(percent.clamp(1, 100)).unwrap_or(100)
}

/// The HRD buffer rate control works in: as long at the peak rate as the
/// settings say, a second unless a hard limit needs less, three quarters
/// full at the start, as libva's users set it.
pub(crate) fn hrd(
    settings: &EncoderSettings,
) -> Option<VAEncMiscParameterBuffer<VAEncMiscParameterHRD>> {
    let peak = floored(settings.rate_control).peak_bits_per_second()?;
    let (buffer_size, initial_buffer_fullness) = settings.buffer_bits(peak);
    Some(VAEncMiscParameterBuffer {
        type_: enc::VAEncMiscParameterTypeHRD,
        data: VAEncMiscParameterHRD {
            buffer_size,
            initial_buffer_fullness,
            ..VAEncMiscParameterHRD::default()
        },
    })
}

/// The frame rate: numerator in the low 16 bits, denominator in the high.
pub(crate) fn frame_rate(
    settings: &EncoderSettings,
) -> VAEncMiscParameterBuffer<VAEncMiscParameterFrameRate> {
    let rate = settings.frame_rate.reduced();
    VAEncMiscParameterBuffer {
        type_: enc::VAEncMiscParameterTypeFrameRate,
        data: VAEncMiscParameterFrameRate {
            framerate: rate.numerator | (rate.denominator << 16),
            ..VAEncMiscParameterFrameRate::default()
        },
    }
}

/// The quality level, when the settings ask for one.
pub(crate) fn quality_level(
    settings: &EncoderSettings,
) -> Option<VAEncMiscParameterBuffer<VAEncMiscParameterBufferQualityLevel>> {
    let quality_level = settings.quality_level?;
    Some(VAEncMiscParameterBuffer {
        type_: enc::VAEncMiscParameterTypeQualityLevel,
        data: VAEncMiscParameterBufferQualityLevel {
            quality_level,
            ..VAEncMiscParameterBufferQualityLevel::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode::Profile;

    fn settings(rate_control: RateControl) -> EncoderSettings {
        let mut settings = EncoderSettings::new(Profile::H264High, 640, 480);
        settings.rate_control = rate_control;
        settings
    }

    /// What reaches the driver: the sequence's target, the rate control's
    /// rate and percentage, and the HRD buffer and its fullness.
    fn sent(rate_control: RateControl) -> (u32, u32, u32, u32, u32) {
        let settings = settings(rate_control);
        let rc = rate_control_of(&settings);
        let hrd = hrd(&settings).map_or((0, 0), |hrd| {
            (hrd.data.buffer_size, hrd.data.initial_buffer_fullness)
        });
        (target_bits_per_second(&settings), rc.0, rc.1, hrd.0, hrd.1)
    }

    fn rate_control_of(settings: &EncoderSettings) -> (u32, u32) {
        rate_control(settings, false).map_or((0, 0), |rc| {
            (rc.data.bits_per_second, rc.data.target_percentage)
        })
    }

    #[test]
    fn rates_under_the_floor_reach_the_driver_at_it() {
        // An `AverageBitRate` of 1: the session's VBR of 1 peaking at 1.
        let one = RateControl::VariableBitRate {
            average_bits_per_second: 1,
            peak_bits_per_second: 1,
        };
        assert_eq!(sent(one), (32_000, 32_000, 100, 32_000, 24_000));
        // One under the floor, peaking at 1.5 times it.
        let under = RateControl::VariableBitRate {
            average_bits_per_second: MIN_BITS_PER_SECOND - 1,
            peak_bits_per_second: (MIN_BITS_PER_SECOND - 1) * 3 / 2,
        };
        assert_eq!(sent(under), (32_000, 47_998, 66, 47_998, 35_997));
        let cbr = RateControl::ConstantBitRate { bits_per_second: 0 };
        assert_eq!(sent(cbr), (32_000, 32_000, 100, 32_000, 24_000));
        let qvbr = RateControl::QualityVariableBitRate {
            average_bits_per_second: 8,
            peak_bits_per_second: 12,
            quality: 25,
        };
        assert_eq!(sent(qvbr), (32_000, 32_000, 100, 32_000, 24_000));
    }

    #[test]
    fn rates_at_the_floor_and_over_it_reach_the_driver_as_asked() {
        let at = RateControl::VariableBitRate {
            average_bits_per_second: MIN_BITS_PER_SECOND,
            peak_bits_per_second: MIN_BITS_PER_SECOND * 3 / 2,
        };
        assert_eq!(sent(at), (32_000, 48_000, 66, 48_000, 36_000));
        let over = RateControl::VariableBitRate {
            average_bits_per_second: 2_000_000,
            peak_bits_per_second: 3_000_000,
        };
        assert_eq!(sent(over), (2_000_000, 3_000_000, 66, 3_000_000, 2_250_000));
        let cqp = RateControl::ConstantQp {
            intra: 24,
            predicted: 26,
            bidirectional: 28,
        };
        assert_eq!(sent(cqp), (0, 0, 0, 0, 0));
    }

    #[test]
    fn a_limited_stream_gets_a_half_second_buffer() {
        // The session's settings for a DataRateLimits of 3 Mbit/s alone.
        let mut settings = settings(RateControl::VariableBitRate {
            average_bits_per_second: 2_000_000,
            peak_bits_per_second: 2_000_000,
        });
        settings.buffer_milliseconds = 500;
        let hrd =
            hrd(&settings).map(|hrd| (hrd.data.buffer_size, hrd.data.initial_buffer_fullness));
        assert_eq!(hrd, Some((1_000_000, 750_000)));
    }
}
