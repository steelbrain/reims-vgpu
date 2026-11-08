//! H.264 decoding on VA-API: [`crate::codec::h264::Decoder`] over an accelerator
//! that fills libva's H.264 buffers (va.h, "H.264/AVC data structures")
//! and decodes into a pool of NV12 surfaces.

use super::buffer::{Buffer, render_decode};
use super::caps::{Config, Context, Entrypoint, Profile};
use super::display::{Display, Error};
use super::ffi;
use super::frame::Frame;
use super::surface::{Rect, Surface, SurfaceFormat, SurfacePool, end_context};
use super::wait::{self, Deadline};
use crate::codec::h264::{
    self, Accelerator, Decoder, NalFormat, Output, OutputOrder, PictureParams, ReferenceFrame,
    SliceParams, slice::SliceType, sps::Sps,
};

/// Surfaces a pool makes beyond the DPB and the current picture, so that
/// a frame the caller still holds does not stall the next decode.
const SPARE_SURFACES: usize = 2;

/// The most macroblocks of an H.264 picture the decoder takes: iHD and
/// radeonsi answer 4096x4096 as their largest H.264 decode, yet decode
/// nothing of a picture of 65,536 (256 by 256), every call answering
/// success (docs/video/bugs.md, U28).
pub const H264_MAX_MACROBLOCKS: u32 = 65_535;

/// An H.264 decoder on a VA display.
///
/// Samples are access units: length-prefixed NAL units with the `avcC`
/// record's length size, as VideoToolbox's samples are, or Annex B for
/// [`H264Decoder::new_annex_b`]. Frames come out in decode order unless
/// [`H264Decoder::set_output_order`] asks for display order.
pub struct H264Decoder {
    inner: Decoder<VaH264>,
}

impl H264Decoder {
    /// The frames a caller may hold at once, beyond those the decoder holds
    /// itself, before a decode finds no free surface (`Error::OutOfSurfaces`):
    /// the pool's spare surfaces.
    pub const SPARE_FRAMES: usize = SPARE_SURFACES;

    /// A decoder for samples described by an `avcC` record (ISO/IEC
    /// 14496-15, 5.3.3.1), whose parameter sets it takes. The record's
    /// first SPS is checked against what the decoder and the driver take,
    /// so that a stream they cannot decode is refused here.
    pub fn new(display: &Display, avcc: &[u8]) -> Result<Self, Error> {
        let inner = Decoder::from_avcc(VaH264::new(display), avcc)?;
        let config = h264::avcc::DecoderConfiguration::parse(avcc)?;
        if let Some(&sps) = config.sequence_parameter_sets.first() {
            let sps = Sps::parse(sps)?;
            h264::check_supported(&sps)?;
            select_profile(display, &sps)?;
        }
        Ok(Self { inner })
    }

    /// A decoder for Annex B samples, which carry their parameter sets.
    pub fn new_annex_b(display: &Display) -> Self {
        Self {
            inner: Decoder::new(VaH264::new(display), NalFormat::AnnexB),
        }
    }

    /// Chooses decode order (the default) or display order.
    pub fn set_output_order(&mut self, order: OutputOrder) {
        self.inner.set_output_order(order);
    }

    /// Decodes one access unit, tagged `id`, and returns the frames that
    /// come out (see [`crate::codec::h264::Decoder::decode`]). The decode may
    /// still be running on the GPU: a frame's readback waits for it.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame>, Error> {
        // One deadline for the call's driver calls and its frames' read
        // backs (`super::wait`).
        let deadline = Deadline::default();
        let outputs = wait::by(&deadline, || self.inner.decode(sample, id))?;
        Ok(outputs
            .into_iter()
            .map(|output| frame_from_output(output, &deadline))
            .collect())
    }

    /// Emits the frames still waiting for display order. References are
    /// kept, so decoding can go on.
    pub fn flush(&mut self) -> Result<Vec<Frame>, Error> {
        let deadline = Deadline::default();
        let outputs = wait::by(&deadline, || self.inner.flush());
        Ok(outputs
            .into_iter()
            .map(|output| frame_from_output(output, &deadline))
            .collect())
    }

    /// Forgets every reference, as for a seek.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The VA profile the decoder runs, once it has started a picture.
    pub fn profile(&self) -> Option<Profile> {
        self.inner.accelerator().session.as_ref().map(|s| s.profile)
    }
}

// Decoders and frames move between threads, and frames are read back from
// any: libva is thread-safe (va.h, "Multithreading Guide"), and each
// decoder is used by one thread at a time through `&mut`.
const _: fn() = || {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<H264Decoder>();
    send::<Frame>();
    sync::<Frame>();
    send::<Display>();
    sync::<Display>();
};

/// The frame VA-API decoded for one of the decoder's outputs.
fn frame_from_output(output: Output<Surface>, deadline: &Deadline) -> Frame {
    let (x, y, width, height) = output.crop;
    Frame {
        surface: output.picture,
        id: output.id,
        poc: output.poc,
        idr: output.idr,
        reference: output.reference,
        rect: Rect {
            x,
            y,
            width,
            height,
        },
        coded_size: output.coded_size,
        full_range: output.full_range,
        deadline: deadline.clone(),
    }
}

/// The VA-API half of the decoder.
struct VaH264 {
    display: Display,
    session: Option<Session>,
}

/// What one SPS's frames are decoded with.
struct Session {
    profile: Profile,
    /// Destroyed first, before the surfaces it renders into: it holds
    /// its config.
    context: Option<Context>,
    pool: SurfacePool,
}

impl Drop for Session {
    fn drop(&mut self) {
        // The context goes first. The pool may outlive it: frames the
        // caller holds keep their surfaces.
        end_context(self.context.take(), [&self.pool]);
    }
}

impl VaH264 {
    fn new(display: &Display) -> Self {
        Self {
            display: display.clone_ref(),
            session: None,
        }
    }

    fn session(&self) -> Result<&Session, Error> {
        self.session
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))
    }
}

/// The VA profiles that decode `sps`'s stream, most constrained first:
/// Baseline streams (with FMO refused and redundant slices dropped) are
/// Constrained Baseline's subset of Main.
fn profile_candidates(sps: &Sps) -> &'static [Profile] {
    match sps.profile_idc {
        66 => &[
            Profile::H264_CONSTRAINED_BASELINE,
            Profile::H264_MAIN,
            Profile::H264_HIGH,
        ],
        // Extended's pictures that reach a driver are Main's: its own tools
        // are refused at their slices (h264::check_supported).
        77 | 88 => &[Profile::H264_MAIN, Profile::H264_HIGH],
        _ => &[Profile::H264_HIGH],
    }
}

/// The first candidate profile the driver decodes at `sps`'s size.
fn select_profile(display: &Display, sps: &Sps) -> Result<Profile, Error> {
    let (width, height) = (sps.coded_width(), sps.coded_height());
    let macroblocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
    if macroblocks > u64::from(H264_MAX_MACROBLOCKS) {
        return Err(Error::Unsupported(format!(
            "H.264 at {width}x{height}: {macroblocks} macroblocks, past the \
             {H264_MAX_MACROBLOCKS} the drivers decode"
        )));
    }
    for &profile in profile_candidates(sps) {
        if !display.supports(profile, Entrypoint::VLD) {
            continue;
        }
        let limits = display.config_attributes(
            profile,
            Entrypoint::VLD,
            &[
                ffi::VAConfigAttribRTFormat,
                ffi::VAConfigAttribMaxPictureWidth,
                ffi::VAConfigAttribMaxPictureHeight,
            ],
        )?;
        if limits[0].is_some_and(|f| f & ffi::VA_RT_FORMAT_YUV420 == 0) {
            continue;
        }
        if limits[1].is_some_and(|max| width > max) || limits[2].is_some_and(|max| height > max) {
            return Err(Error::Unsupported(format!(
                "H.264 at {width}x{height}: the driver's maximum is {}x{}",
                limits[1].unwrap_or(0),
                limits[2].unwrap_or(0)
            )));
        }
        return Ok(profile);
    }
    Err(Error::Unsupported(format!(
        "H.264 profile_idc {} (no VA profile of {:?})",
        sps.profile_idc,
        profile_candidates(sps)
            .iter()
            .filter_map(|p| p.name())
            .collect::<Vec<_>>()
    )))
}

impl Accelerator for VaH264 {
    type Picture = Surface;
    type Error = Error;

    fn configure(&mut self, sps: &Sps, dpb_frames: usize) -> Result<(), Error> {
        self.session = None;
        let profile = select_profile(&self.display, sps)?;
        let (width, height) = (sps.coded_width(), sps.coded_height());
        let mut attribs = [ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: ffi::VA_RT_FORMAT_YUV420,
        }];
        let config = Config::new(&self.display, profile, Entrypoint::VLD, &mut attribs)?;
        let pool = SurfacePool::new(
            &self.display,
            SurfaceFormat::Nv12,
            width,
            height,
            dpb_frames + 1 + SPARE_SURFACES,
        )?;
        let mut targets = pool.ids();
        let context = Context::new(&self.display, config, width, height, &mut targets)?;
        self.session = Some(Session {
            profile,
            context: Some(context),
            pool,
        });
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Surface, Error> {
        self.session()?.pool.get()
    }

    /// A copy of `from` in a new surface of the pool: a stand-in for a
    /// lost frame (ours: Apple's AVC decoder conceals one some way of its
    /// own).
    fn conceal(&mut self, from: &Surface) -> Result<Option<Surface>, Error> {
        let to = self.session()?.pool.get()?;
        Ok(from.copy_to(&to)?.then_some(to))
    }

    fn decode_picture(
        &mut self,
        picture: &Surface,
        params: &PictureParams<'_, Surface>,
        slices: &[SliceParams<'_, Surface>],
    ) -> Result<(), Error> {
        let session = self.session()?;
        let decoding = session
            .context
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding without a context".into()))?;
        let context = decoding.id;
        let display = &self.display;
        let picture_buffer = Buffer::with_value(
            display,
            context,
            ffi::VAPictureParameterBufferType,
            &picture_parameters(picture, params),
        )?;
        let matrix_buffer = Buffer::with_value(
            display,
            context,
            ffi::VAIQMatrixBufferType,
            &iq_matrix(params),
        )?;
        let mut slice_buffers = Vec::with_capacity(slices.len());
        for slice in slices {
            let parameters = slice_parameters(slice)?;
            slice_buffers.push((
                Buffer::with_value(
                    display,
                    context,
                    ffi::VASliceParameterBufferType,
                    &parameters,
                )?,
                Buffer::with_bytes(display, context, ffi::VASliceDataBufferType, slice.nal)?,
            ));
        }
        let first = [&picture_buffer, &matrix_buffer];
        let slices: Vec<[&Buffer<'_>; 2]> = (slice_buffers.iter())
            .map(|(parameters, data)| [parameters, data])
            .collect();
        let groups: Vec<&[&Buffer<'_>]> = core::iter::once(first.as_slice())
            .chain(slices.iter().map(<[_; 2]>::as_slice))
            .collect();
        // A first picture past its deadline may still be submitted: the
        // pool is kept until the GPU is done with it.
        picture.stalled_if(render_decode(display, decoding, picture.id(), &groups))
    }
}

fn va_picture(reference: &ReferenceFrame<'_, Surface>) -> ffi::VAPictureH264 {
    ffi::VAPictureH264 {
        picture_id: reference.picture.id(),
        frame_idx: reference.frame_idx,
        flags: if reference.long_term {
            ffi::VA_PICTURE_H264_LONG_TERM_REFERENCE
        } else {
            ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
        },
        top_field_order_cnt: reference.top_poc,
        bottom_field_order_cnt: reference.bottom_poc,
        va_reserved: [0; 4],
    }
}

fn picture_parameters(
    picture: &Surface,
    p: &PictureParams<'_, Surface>,
) -> ffi::VAPictureParameterBufferH264 {
    use ffi::{pic_fields as pf, seq_fields as sf};
    let (sps, pps, header) = (p.sps, p.pps, p.header);
    let mut reference_frames = [ffi::VAPictureH264::invalid(); 16];
    for (slot, reference) in reference_frames.iter_mut().zip(&p.references) {
        *slot = va_picture(reference);
    }
    let reference = header.nal_ref_idc != 0;
    let flag = u32::from;
    let seq_fields = sf::CHROMA_FORMAT_IDC.put(u32::from(sps.chroma_format_idc))
        | sf::RESIDUAL_COLOUR_TRANSFORM_FLAG.put(flag(sps.separate_colour_plane_flag))
        | sf::GAPS_IN_FRAME_NUM_VALUE_ALLOWED_FLAG.put(flag(sps.gaps_in_frame_num_value_allowed_flag))
        | sf::FRAME_MBS_ONLY_FLAG.put(flag(sps.frame_mbs_only_flag))
        | sf::MB_ADAPTIVE_FRAME_FIELD_FLAG.put(flag(sps.mb_adaptive_frame_field_flag))
        | sf::DIRECT_8X8_INFERENCE_FLAG.put(flag(sps.direct_8x8_inference_flag))
        // Table A-1: MinLumaBiPredSize is 8x8 from level 3.1 up.
        | sf::MIN_LUMA_BI_PRED_SIZE8X8.put(flag(sps.level_idc >= 31))
        | sf::LOG2_MAX_FRAME_NUM_MINUS4.put(u32::from(sps.log2_max_frame_num_minus4))
        | sf::PIC_ORDER_CNT_TYPE.put(u32::from(sps.pic_order_cnt_type))
        | sf::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4.put(u32::from(sps.log2_max_pic_order_cnt_lsb_minus4))
        | sf::DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG.put(flag(sps.delta_pic_order_always_zero_flag));
    let pic_fields = pf::ENTROPY_CODING_MODE_FLAG.put(flag(pps.entropy_coding_mode_flag))
        | pf::WEIGHTED_PRED_FLAG.put(flag(pps.weighted_pred_flag))
        | pf::WEIGHTED_BIPRED_IDC.put(u32::from(pps.weighted_bipred_idc))
        | pf::TRANSFORM_8X8_MODE_FLAG.put(flag(pps.transform_8x8_mode_flag))
        | pf::FIELD_PIC_FLAG.put(flag(header.field_pic_flag))
        | pf::CONSTRAINED_INTRA_PRED_FLAG.put(flag(pps.constrained_intra_pred_flag))
        | pf::PIC_ORDER_PRESENT_FLAG.put(flag(pps.bottom_field_pic_order_in_frame_present_flag))
        | pf::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG
            .put(flag(pps.deblocking_filter_control_present_flag))
        | pf::REDUNDANT_PIC_CNT_PRESENT_FLAG.put(flag(pps.redundant_pic_cnt_present_flag))
        | pf::REFERENCE_PIC_FLAG.put(flag(reference));
    let narrow = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
    ffi::VAPictureParameterBufferH264 {
        curr_pic: ffi::VAPictureH264 {
            picture_id: picture.id(),
            frame_idx: header.frame_num,
            // As libva's users mark the current picture: a reference is
            // flagged short-term until its marking is known.
            flags: if reference {
                ffi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
            } else {
                0
            },
            top_field_order_cnt: p.top_poc,
            bottom_field_order_cnt: p.bottom_poc,
            va_reserved: [0; 4],
        },
        reference_frames,
        picture_width_in_mbs_minus1: narrow(sps.width_in_mbs() - 1),
        picture_height_in_mbs_minus1: narrow(sps.frame_height_in_mbs() - 1),
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        num_ref_frames: sps.max_num_ref_frames,
        seq_fields,
        num_slice_groups_minus1: 0,
        slice_group_map_type: 0,
        slice_group_change_rate_minus1: 0,
        pic_init_qp_minus26: pps.pic_init_qp_minus26,
        pic_init_qs_minus26: pps.pic_init_qs_minus26,
        chroma_qp_index_offset: pps.chroma_qp_index_offset,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
        pic_fields,
        frame_num: narrow(header.frame_num),
        va_reserved: [0; 8],
    }
}

/// The scaling lists in raster order (va.h: "4x4 scaling list, in raster
/// scan order"), the 8x8 lists being Intra Y and Inter Y.
fn iq_matrix(p: &PictureParams<'_, Surface>) -> ffi::VAIQMatrixBufferH264 {
    let lists = &p.scaling_lists;
    ffi::VAIQMatrixBufferH264 {
        scaling_list_4x4: core::array::from_fn(|i| lists.raster_4x4(i)),
        scaling_list_8x8: [lists.raster_8x8(0), lists.raster_8x8(1)],
        va_reserved: [0; 4],
    }
}

fn slice_parameters(
    slice: &SliceParams<'_, Surface>,
) -> Result<ffi::VASliceParameterBufferH264, Error> {
    let h = slice.header;
    let mut lists = [[ffi::VAPictureH264::invalid(); 32]; 2];
    for (va_list, list) in lists.iter_mut().zip(&slice.ref_lists) {
        for (slot, reference) in va_list.iter_mut().zip(list) {
            if let Some(reference) = reference {
                *slot = va_picture(reference);
            }
        }
    }
    let [list0, list1] = lists;
    let inter = h.slice_type.is_inter();
    let b = h.slice_type == SliceType::B;
    let mut out = ffi::VASliceParameterBufferH264 {
        slice_data_size: u32::try_from(slice.nal.len())
            .map_err(|_| Error::Unsupported("a slice over 4 GiB".into()))?,
        slice_data_offset: 0,
        slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
        // va.h: "relative to and includes the NAL unit byte", counted
        // "after the removal of any emulation prevention bytes".
        slice_data_bit_offset: u16::try_from(h.header_bits)
            .map_err(|_| Error::Unsupported("a slice header over 64 Kibit".into()))?,
        first_mb_in_slice: u16::try_from(h.first_mb_in_slice)
            .map_err(|_| Error::Unsupported("frames over 65536 macroblocks".into()))?,
        slice_type: h.slice_type.raw(),
        direct_spatial_mv_pred_flag: u8::from(h.direct_spatial_mv_pred_flag),
        num_ref_idx_l0_active_minus1: if inter {
            h.num_ref_idx_l0_active_minus1
        } else {
            0
        },
        num_ref_idx_l1_active_minus1: if b { h.num_ref_idx_l1_active_minus1 } else { 0 },
        cabac_init_idc: h.cabac_init_idc,
        slice_qp_delta: h.slice_qp_delta,
        disable_deblocking_filter_idc: h.disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2: h.slice_alpha_c0_offset_div2,
        slice_beta_offset_div2: h.slice_beta_offset_div2,
        ref_pic_list0: list0,
        ref_pic_list1: list1,
        luma_log2_weight_denom: 0,
        chroma_log2_weight_denom: 0,
        luma_weight_l0_flag: 0,
        luma_weight_l0: [0; 32],
        luma_offset_l0: [0; 32],
        chroma_weight_l0_flag: 0,
        chroma_weight_l0: [[0; 2]; 32],
        chroma_offset_l0: [[0; 2]; 32],
        luma_weight_l1_flag: 0,
        luma_weight_l1: [0; 32],
        luma_offset_l1: [0; 32],
        chroma_weight_l1_flag: 0,
        chroma_weight_l1: [[0; 2]; 32],
        chroma_offset_l1: [[0; 2]; 32],
        va_reserved: [0; 4],
    };
    // VA-API wants the weights 7.4.3.2 infers where the stream codes none,
    // as libva's users fill them: 1 << denominator, offset 0.
    let counts = [
        if inter {
            usize::from(h.num_ref_idx_l0_active_minus1) + 1
        } else {
            0
        },
        if b {
            usize::from(h.num_ref_idx_l1_active_minus1) + 1
        } else {
            0
        },
    ];
    let table = h.pred_weight_table.as_ref();
    if let Some(t) = table {
        out.luma_log2_weight_denom = t.luma_log2_weight_denom;
        out.chroma_log2_weight_denom = t.chroma_log2_weight_denom;
    }
    let luma_default = 1i16 << out.luma_log2_weight_denom;
    let chroma_default = 1i16 << out.chroma_log2_weight_denom;
    for (list, &count) in counts.iter().enumerate() {
        let weights = table.map(|t| &t.lists[list]);
        let (luma_flag, luma_weight, luma_offset, chroma_flag, chroma_weight, chroma_offset) =
            if list == 0 {
                (
                    &mut out.luma_weight_l0_flag,
                    &mut out.luma_weight_l0,
                    &mut out.luma_offset_l0,
                    &mut out.chroma_weight_l0_flag,
                    &mut out.chroma_weight_l0,
                    &mut out.chroma_offset_l0,
                )
            } else {
                (
                    &mut out.luma_weight_l1_flag,
                    &mut out.luma_weight_l1,
                    &mut out.luma_offset_l1,
                    &mut out.chroma_weight_l1_flag,
                    &mut out.chroma_weight_l1,
                    &mut out.chroma_offset_l1,
                )
            };
        *luma_flag = u8::from(weights.is_some_and(|w| w.iter().any(|e| e.luma_present)));
        *chroma_flag = u8::from(weights.is_some_and(|w| w.iter().any(|e| e.chroma_present)));
        for i in 0..count.min(32) {
            match weights.and_then(|w| w.get(i)) {
                Some(e) => {
                    (luma_weight[i], luma_offset[i]) = e.luma;
                    for c in 0..2 {
                        (chroma_weight[i][c], chroma_offset[i][c]) = e.chroma[c];
                    }
                }
                None => {
                    luma_weight[i] = luma_default;
                    chroma_weight[i] = [chroma_default; 2];
                }
            }
        }
    }
    Ok(out)
}
