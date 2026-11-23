//! HEVC decoding on VA-API: [`crate::codec::hevc::Decoder`] over an accelerator
//! that fills libva's HEVC buffers (va_dec_hevc.h) and decodes into a pool
//! of NV12 surfaces for Main, P010 ones for Main 10, and, for a caller that
//! lands 4:4:4 pictures ([`HevcDecoder::lands_4_4_4`]), XYUV ones for the
//! range extensions' 4:4:4 at 8 bits, on the driver's `HEVCMain444`. Those
//! pictures' parameters go as va_dec_hevc.h's `…Extension` structures, the
//! base one with the range extensions' after it, in the picture's and each
//! slice's parameter buffer, as ffmpeg sends them for that profile.

use super::buffer::{Buffer, render_decode};
use super::caps::{Context, Entrypoint, Profile};
use super::display::{Display, Error};
use super::ffi;
use super::frame::Frame;
use super::surface::{Rect, Surface, SurfaceFormat, SurfacePool, end_context};
use super::target::TargetFor;
use super::wait::{self, Deadline};
use crate::codec::PictureRole;
use crate::codec::hevc::{
    self, Accelerator, Decoder, NalFormat, Output, OutputOrder, PictureParams, RpsMembership,
    SliceParams,
    hvcc::DecoderConfiguration,
    nal::{NalUnit, is_idr, is_irap, nal_type},
    pps::Pps,
    slice::SliceType,
    sps::Sps,
};

/// An entry of `RefPicList[][]` that names no picture (va_dec_hevc.h).
const NO_REFERENCE: u8 = 0xff;

/// The most pictures `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter` and
/// `RefPicSetLtCurr` hold together: va.h says their sum, `NumPocTotalCurr`,
/// "should be equal to or smaller than 8". radeonsi decoded a picture with
/// 9 wrongly, and every one after it, as ffmpeg's VA-API decode did
/// (kvazaar's `--ref 15`, radeonsi, Mesa 26.2.4, 2026-10-24).
const MAX_CURRENT_REFERENCES: usize = 8;

/// An HEVC decoder on a VA display.
///
/// Samples are access units: length-prefixed NAL units with the `hvcC`
/// record's length size, as VideoToolbox's samples are, or Annex B for
/// [`HevcDecoder::new_annex_b`]. Frames come out in decode order unless
/// [`HevcDecoder::set_output_order`] asks for display order. Main streams
/// decode into NV12 frames, Main 10 ones into P010, and 4:4:4 ones at 8 bits
/// into XYUV where the caller lands them ([`HevcDecoder::lands_4_4_4`]).
pub struct HevcDecoder {
    inner: Decoder<VaHevc>,
}

impl HevcDecoder {
    /// The frames a caller is to hold at once, beyond those the decoder holds itself. The
    /// decoder's pool makes a surface for a picture as it needs one, whether the DPB, the
    /// caller or the picture being decoded holds it, up to a bound past which a decode finds no
    /// free surface (`Error::OutOfSurfaces`).
    pub const SPARE_FRAMES: usize = 2;

    /// A decoder for samples described by an `hvcC` record (ISO/IEC
    /// 14496-15, 8.3.3.1), whose parameter sets it takes. The record's
    /// first SPS is checked against what the decoder and the driver take,
    /// so that a stream they cannot decode is refused here. A 4:4:4 stream
    /// the driver decodes is taken here, and refused at its first picture
    /// unless [`Self::lands_4_4_4`] was called before it.
    pub fn new(display: &Display, hvcc: &[u8]) -> Result<Self, Error> {
        let inner = Decoder::from_hvcc(VaHevc::new(display), hvcc)?;
        let config = DecoderConfiguration::parse(hvcc)?;
        let first_sps = config
            .nal_units
            .iter()
            .find(|unit| NalUnit::new(unit).is_ok_and(|nal| nal.unit_type() == nal_type::SPS));
        if let Some(sps) = first_sps {
            let sps = Sps::parse(sps)?;
            hevc::check_supported(&sps)?;
            select_profile(display, &sps, true)?;
        }
        Ok(Self { inner })
    }

    /// A decoder for Annex B samples, which carry their parameter sets.
    pub fn new_annex_b(display: &Display) -> Self {
        Self {
            inner: Decoder::new(VaHevc::new(display), NalFormat::AnnexB),
        }
    }

    /// Has a 4:4:4 stream at 8 bits decoded, into XYUV surfaces, whose
    /// frames the caller reads back with [`super::ReadBack::copy_4_4_4_into`]
    /// ([`Frame::is_4_4_4`]); without it, a 4:4:4 stream is refused when its
    /// SPS is taken.
    pub fn lands_4_4_4(&mut self) {
        self.inner.accelerator_mut().four_four_four = true;
    }

    /// Chooses decode order (the default) or display order.
    pub fn set_output_order(&mut self, order: OutputOrder) {
        self.inner.set_output_order(order);
    }

    /// Decodes one access unit, tagged `id`, and returns the frames that
    /// come out (see [`crate::codec::hevc::Decoder::decode`]). The decode may
    /// still be running on the GPU: a frame's readback waits for it.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame>, Error> {
        self.decode_into(sample, id, None)
    }

    /// [`Self::decode`], the access unit's picture decoded into the target `target_for` gives
    /// where it gives one of the picture's shape (`super::target`): that picture's frame is then
    /// in the target ([`Frame::is_in_target`]). A frame of another access unit, or one whose
    /// target was not given or not taken, is in a surface of the decoder's own.
    pub fn decode_into(
        &mut self,
        sample: &[u8],
        id: u64,
        target_for: Option<TargetFor>,
    ) -> Result<Vec<Frame>, Error> {
        // One deadline for the call's driver calls and its frames' read
        // backs (`super::wait`).
        let deadline = Deadline::default();
        self.inner.accelerator_mut().target_for = target_for;
        let outputs = wait::by(&deadline, || self.inner.decode(sample, id));
        // A target not asked for (an access unit of no picture) is let go of here.
        self.inner.accelerator_mut().target_for = None;
        Ok(outputs?
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

    /// How many surfaces of its own the decoder's session made, once it has started a picture.
    #[cfg(test)]
    pub(super) fn surfaces_made(&self) -> Option<usize> {
        let session = self.inner.accelerator().session.as_ref()?;
        Some(session.pool.made())
    }
}

// Decoders move between threads: libva is thread-safe (va.h,
// "Multithreading Guide"), and each decoder is used by one thread at a
// time through `&mut`.
const _: fn() = || {
    fn send<T: Send>() {}
    send::<HevcDecoder>();
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
struct VaHevc {
    display: Display,
    session: Option<Session>,
    /// What gives the decode in hand its picture's target, taken as the picture is made.
    target_for: Option<TargetFor>,
    /// Whether the caller lands 4:4:4 pictures ([`HevcDecoder::lands_4_4_4`]).
    four_four_four: bool,
}

/// What one SPS's pictures are decoded with.
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

impl VaHevc {
    fn new(display: &Display) -> Self {
        Self {
            display: display.clone_ref(),
            session: None,
            target_for: None,
            four_four_four: false,
        }
    }

    fn session(&self) -> Result<&Session, Error> {
        self.session
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))
    }
}

/// The VA profile and surface format for `sps`'s stream: Main at 8 bits,
/// Main 10 at 10, of 4:2:0, and Main 4:4:4 of the range extensions' 4:4:4
/// at 8 bits where the caller lands its pictures (`four_four_four`).
/// `check_supported` has refused other depths and formats.
fn profile_for(sps: &Sps, four_four_four: bool) -> Result<(Profile, SurfaceFormat), Error> {
    match (sps.chroma_format_idc, sps.bit_depth_luma_minus8) {
        (1, 0) => Ok((Profile::HEVC_MAIN, SurfaceFormat::Nv12)),
        (1, _) => Ok((Profile::HEVC_MAIN10, SurfaceFormat::P010)),
        (3, 0) if four_four_four => Ok((Profile::HEVC_MAIN444, SurfaceFormat::Xyuv)),
        (3, 0) => Err(Error::Unsupported(
            "HEVC 4:4:4: its caller lands 4:2:0 pictures alone".into(),
        )),
        _ => Err(Error::Unsupported(
            "HEVC: VA-API decodes 4:2:0 here, and 4:4:4 at 8 bits".into(),
        )),
    }
}

/// The VA profile, when the driver decodes `sps`'s stream at its size.
fn select_profile(display: &Display, sps: &Sps, four_four_four: bool) -> Result<Profile, Error> {
    let (profile, format) = profile_for(sps, four_four_four)?;
    let name = profile.name().unwrap_or("HEVC");
    if !display.supports(profile, Entrypoint::VLD) {
        return Err(Error::Unsupported(format!(
            "{name}: the driver has no decoder"
        )));
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
    if limits[0].is_some_and(|f| f & format.rt_format() == 0) {
        return Err(Error::Unsupported(format!(
            "{name}: no {} render targets",
            format.name()
        )));
    }
    let (width, height) = (
        sps.pic_width_in_luma_samples,
        sps.pic_height_in_luma_samples,
    );
    if limits[1].is_some_and(|max| width > max) || limits[2].is_some_and(|max| height > max) {
        return Err(Error::Unsupported(format!(
            "{name} at {width}x{height}: the driver's maximum is {}x{}",
            limits[1].unwrap_or(0),
            limits[2].unwrap_or(0)
        )));
    }
    Ok(profile)
}

impl Accelerator for VaHevc {
    type Picture = Surface;
    type Error = Error;

    fn configure(&mut self, sps: &Sps, _dpb_size: usize) -> Result<(), Error> {
        self.session = None;
        let profile = select_profile(&self.display, sps, self.four_four_four)?;
        let (_, format) = profile_for(sps, self.four_four_four)?;
        let (width, height) = (
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
        );
        let mut attribs = [ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: format.rt_format(),
        }];
        // The DPB the driver sizes the decoder by: `sps_max_dec_pic_buffering_minus1`.
        let references = u32::from(sps.highest_ordering().max_dec_pic_buffering_minus1);
        let (context, pool) = Context::new_decoder(
            &self.display,
            profile,
            &mut attribs,
            format,
            (width, height),
            references,
        )?;
        self.session = Some(Session {
            profile,
            context: Some(context),
            pool,
        });
        Ok(())
    }

    /// The decode's target where it was handed one of the picture's shape for a picture of
    /// `role`, else a surface of the pool.
    fn new_picture(&mut self, role: PictureRole) -> Result<Surface, Error> {
        let target_for = self.target_for.take();
        let pool = &self.session()?.pool;
        let lent = target_for
            .zip(pool.shape())
            .and_then(|(target_for, shape)| target_for(shape, role))
            .and_then(|target| pool.lend_target(target));
        match lent {
            Some(surface) => Ok(surface),
            None => pool.get(),
        }
    }

    fn decode_picture(
        &mut self,
        picture: &Surface,
        params: &PictureParams<'_, Surface>,
        slices: &[SliceParams<'_>],
    ) -> Result<(), Error> {
        let session = self.session()?;
        let decoding = session
            .context
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding without a context".into()))?;
        let context = decoding.id;
        let display = &self.display;
        // The range extensions' profile takes the extension structures
        // whole (va_dec_hevc.h), as ffmpeg sends them.
        let extensions = session.profile == Profile::HEVC_MAIN444;
        let base = picture_parameters(picture, params)?;
        let picture_buffer = if extensions {
            let parameters = ffi::VAPictureParameterBufferHEVCExtension {
                base,
                rext: range_extension_parameters(params.sps, params.pps),
                scc: ffi::VAPictureParameterBufferHEVCScc::none(),
            };
            Buffer::with_value(
                display,
                context,
                ffi::VAPictureParameterBufferType,
                &parameters,
            )?
        } else {
            Buffer::with_value(display, context, ffi::VAPictureParameterBufferType, &base)?
        };
        // va_dec_hevc.h: the matrix is sent "only when
        // scaling_list_enabled_flag = 1".
        let matrix_buffer = match &params.scaling_lists {
            Some(lists) => Some(Buffer::with_value(
                display,
                context,
                ffi::VAIQMatrixBufferType,
                &iq_matrix(lists),
            )?),
            None => None,
        };
        let mut slice_buffers = Vec::with_capacity(slices.len());
        for slice in slices {
            let base = slice_parameters(slice)?;
            let parameters = if extensions {
                let parameters = ffi::VASliceParameterBufferHEVCExtension {
                    base,
                    rext: slice_range_extension(slice),
                };
                Buffer::with_value(
                    display,
                    context,
                    ffi::VASliceParameterBufferType,
                    &parameters,
                )?
            } else {
                Buffer::with_value(display, context, ffi::VASliceParameterBufferType, &base)?
            };
            slice_buffers.push((
                parameters,
                Buffer::with_bytes(display, context, ffi::VASliceDataBufferType, slice.nal)?,
            ));
        }
        let mut first: Vec<&Buffer<'_>> = vec![&picture_buffer];
        first.extend(matrix_buffer.as_ref());
        let mut groups: Vec<Vec<&Buffer<'_>>> = vec![first];
        groups.extend(
            slice_buffers
                .iter()
                .map(|(parameters, data)| vec![parameters, data]),
        );
        let groups: Vec<&[&Buffer<'_>]> = groups.iter().map(Vec::as_slice).collect();
        // A first picture past its deadline may still be submitted: the
        // pool is kept until the GPU is done with it.
        picture.stalled_if(render_decode(display, decoding, picture.id(), &groups))
    }
}

/// Refuses a picture whose current sets hold more pictures than VA-API
/// takes, before anything of it reaches the driver: an error, not a wrong
/// picture, as the NVIDIA backend refuses a set past its 8.
fn current_references(sets: &[Vec<usize>; 3]) -> Result<(), Error> {
    let count: usize = sets.iter().map(Vec::len).sum();
    if count > MAX_CURRENT_REFERENCES {
        return Err(Error::Unsupported(format!(
            "{count} pictures in the current reference sets: VA-API takes \
             {MAX_CURRENT_REFERENCES}"
        )));
    }
    Ok(())
}

fn narrow_u16(value: u32) -> u16 {
    u16::try_from(value).unwrap_or(u16::MAX)
}

fn picture_parameters(
    picture: &Surface,
    p: &PictureParams<'_, Surface>,
) -> Result<ffi::VAPictureParameterBufferHEVC, Error> {
    use ffi::{hevc_pic_fields as pf, hevc_slice_parsing_fields as sf};
    let (sps, pps, header) = (p.sps, p.pps, p.header);
    current_references(&p.current_sets)?;
    let mut reference_frames = [ffi::VAPictureHEVC::invalid(); 15];
    for (slot, reference) in reference_frames.iter_mut().zip(&p.references) {
        let set = match reference.membership {
            RpsMembership::StCurrBefore => ffi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
            RpsMembership::StCurrAfter => ffi::VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
            RpsMembership::LtCurr => ffi::VA_PICTURE_HEVC_RPS_LT_CURR,
            RpsMembership::Foll => 0,
        };
        let long_term = if reference.long_term {
            ffi::VA_PICTURE_HEVC_LONG_TERM_REFERENCE
        } else {
            0
        };
        *slot = ffi::VAPictureHEVC {
            picture_id: reference.picture.id(),
            pic_order_cnt: reference.poc,
            flags: set | long_term,
            va_reserved: [0; 4],
        };
    }
    let flag = u32::from;
    let pcm = sps.pcm.unwrap_or_default();
    let ordering = sps.highest_ordering();
    let pic_fields = pf::CHROMA_FORMAT_IDC.put(u32::from(sps.chroma_format_idc))
        | pf::SEPARATE_COLOUR_PLANE_FLAG.put(flag(sps.separate_colour_plane_flag))
        | pf::PCM_ENABLED_FLAG.put(flag(sps.pcm.is_some()))
        | pf::SCALING_LIST_ENABLED_FLAG.put(flag(sps.scaling_list_enabled_flag))
        | pf::TRANSFORM_SKIP_ENABLED_FLAG.put(flag(pps.transform_skip_enabled_flag))
        | pf::AMP_ENABLED_FLAG.put(flag(sps.amp_enabled_flag))
        | pf::STRONG_INTRA_SMOOTHING_ENABLED_FLAG
            .put(flag(sps.strong_intra_smoothing_enabled_flag))
        | pf::SIGN_DATA_HIDING_ENABLED_FLAG.put(flag(pps.sign_data_hiding_enabled_flag))
        | pf::CONSTRAINED_INTRA_PRED_FLAG.put(flag(pps.constrained_intra_pred_flag))
        | pf::CU_QP_DELTA_ENABLED_FLAG.put(flag(pps.cu_qp_delta_enabled_flag))
        | pf::WEIGHTED_PRED_FLAG.put(flag(pps.weighted_pred_flag))
        | pf::WEIGHTED_BIPRED_FLAG.put(flag(pps.weighted_bipred_flag))
        | pf::TRANSQUANT_BYPASS_ENABLED_FLAG.put(flag(pps.transquant_bypass_enabled_flag))
        | pf::TILES_ENABLED_FLAG.put(flag(pps.tiles_enabled_flag))
        | pf::ENTROPY_CODING_SYNC_ENABLED_FLAG.put(flag(pps.entropy_coding_sync_enabled_flag))
        | pf::PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG
            .put(flag(pps.pps_loop_filter_across_slices_enabled_flag))
        | pf::LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG
            .put(flag(pps.loop_filter_across_tiles_enabled_flag))
        | pf::PCM_LOOP_FILTER_DISABLED_FLAG.put(flag(pcm.loop_filter_disabled_flag))
        | pf::NO_PIC_REORDERING_FLAG.put(flag(ordering.max_num_reorder_pics == 0))
        | pf::NO_BI_PRED_FLAG.put(flag(!p.has_b_slices));
    let unit_type = header.nal_unit_type;
    let slice_parsing_fields = sf::LISTS_MODIFICATION_PRESENT_FLAG
        .put(flag(pps.lists_modification_present_flag))
        | sf::LONG_TERM_REF_PICS_PRESENT_FLAG.put(flag(sps.long_term_ref_pics_present_flag))
        | sf::SPS_TEMPORAL_MVP_ENABLED_FLAG.put(flag(sps.sps_temporal_mvp_enabled_flag))
        | sf::CABAC_INIT_PRESENT_FLAG.put(flag(pps.cabac_init_present_flag))
        | sf::OUTPUT_FLAG_PRESENT_FLAG.put(flag(pps.output_flag_present_flag))
        | sf::DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG
            .put(flag(pps.dependent_slice_segments_enabled_flag))
        | sf::PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG
            .put(flag(pps.pps_slice_chroma_qp_offsets_present_flag))
        | sf::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG
            .put(flag(sps.sample_adaptive_offset_enabled_flag))
        | sf::DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG
            .put(flag(pps.deblocking_filter_override_enabled_flag))
        | sf::PPS_DISABLE_DEBLOCKING_FILTER_FLAG.put(flag(pps.pps_deblocking_filter_disabled_flag))
        | sf::SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG
            .put(flag(pps.slice_segment_header_extension_present_flag))
        // va_dec_hevc.h: "NUT between 16 and 21", and "19 and 20".
        | sf::RAP_PIC_FLAG.put(flag(is_irap(unit_type) && unit_type <= nal_type::CRA_NUT))
        | sf::IDR_PIC_FLAG.put(flag(is_idr(unit_type)))
        | sf::INTRA_PIC_FLAG.put(flag(p.intra_only));
    // va_dec_hevc.h: "when uniform_spacing_flag equals 1, application
    // should populate column_width_minus[], and row_height_minus1[]".
    let columns = pps.column_widths(sps)?;
    let rows = pps.row_heights(sps)?;
    let narrow = |v: u32| u8::try_from(v).unwrap_or(u8::MAX);
    Ok(ffi::VAPictureParameterBufferHEVC {
        curr_pic: ffi::VAPictureHEVC {
            picture_id: picture.id(),
            pic_order_cnt: p.poc,
            flags: 0,
            va_reserved: [0; 4],
        },
        reference_frames,
        pic_width_in_luma_samples: narrow_u16(sps.pic_width_in_luma_samples),
        pic_height_in_luma_samples: narrow_u16(sps.pic_height_in_luma_samples),
        pic_fields,
        sps_max_dec_pic_buffering_minus1: ordering.max_dec_pic_buffering_minus1,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        pcm_sample_bit_depth_luma_minus1: pcm.sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1: pcm.sample_bit_depth_chroma_minus1,
        log2_min_luma_coding_block_size_minus3: sps.log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_luma_coding_block_size,
        log2_min_transform_block_size_minus2: sps.log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_transform_block_size: sps.log2_diff_max_min_luma_transform_block_size,
        log2_min_pcm_luma_coding_block_size_minus3: pcm.log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size: pcm.log2_diff_max_min_luma_coding_block_size,
        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
        init_qp_minus26: pps.init_qp_minus26,
        diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth,
        pps_cb_qp_offset: pps.pps_cb_qp_offset,
        pps_cr_qp_offset: pps.pps_cr_qp_offset,
        log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level_minus2,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        column_width_minus1: core::array::from_fn(|i| {
            columns
                .get(i)
                .map_or(0, |&w| narrow_u16(w.saturating_sub(1)))
        }),
        row_height_minus1: core::array::from_fn(|i| {
            rows.get(i).map_or(0, |&h| narrow_u16(h.saturating_sub(1)))
        }),
        slice_parsing_fields,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        num_short_term_ref_pic_sets: narrow(
            u32::try_from(sps.short_term_ref_pic_sets.len()).unwrap_or(0),
        ),
        num_long_term_ref_pic_sps: narrow(u32::try_from(sps.long_term_ref_pics.len()).unwrap_or(0)),
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        pps_beta_offset_div2: pps.pps_beta_offset_div2,
        pps_tc_offset_div2: pps.pps_tc_offset_div2,
        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
        st_rps_bits: header.st_rps_bits,
        va_reserved: [0; 8],
    })
}

/// The scaling lists as `VAIQMatrixBufferHEVC` takes them: each list in
/// raster order, as va_dec_hevc.h says ("Matrix entries are in raster scan
/// order"), not in the coding order the standard indexes `ScalingList` by,
/// and 32x32's two lists as `matrixId` 0 and 3. Mesa reads them so: lists
/// sent in coding order decoded every picture of a stream with the default
/// lists wrongly on radeonsi, and in raster order matched ffmpeg.
fn iq_matrix(lists: &hevc::scaling::ScalingLists) -> ffi::VAIQMatrixBufferHEVC {
    ffi::VAIQMatrixBufferHEVC {
        scaling_list_4x4: lists.list_4x4.map(|list| raster(&list, &DIAGONAL_4X4)),
        scaling_list_8x8: lists.list_8x8.map(|list| raster(&list, &DIAGONAL_8X8)),
        scaling_list_16x16: lists.list_16x16.map(|list| raster(&list, &DIAGONAL_8X8)),
        scaling_list_32x32: [0, 3].map(|matrix| raster(&lists.list_32x32[matrix], &DIAGONAL_8X8)),
        scaling_list_dc_16x16: lists.dc_16x16,
        scaling_list_dc_32x32: [lists.dc_32x32[0], lists.dc_32x32[3]],
        va_reserved: [0; 4],
    }
}

/// The up-right diagonal scan of an `N`x`N` block (H.265 6.5.3): the
/// raster position, `y * N + x`, of each coefficient in coding order.
const fn diagonal_scan<const N: usize, const LEN: usize>() -> [usize; LEN] {
    let mut scan = [0; LEN];
    let (mut i, mut x, mut y) = (0usize, 0usize, 0isize);
    while i < LEN {
        while y >= 0 {
            if x < N && (y as usize) < N {
                scan[i] = y as usize * N + x;
                i += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x as isize;
        x = 0;
    }
    scan
}

const DIAGONAL_4X4: [usize; 16] = diagonal_scan::<4, 16>();
const DIAGONAL_8X8: [usize; 64] = diagonal_scan::<8, 64>();

/// A list in coding order laid out in raster order by `scan`.
fn raster<const LEN: usize>(coded: &[u8; LEN], scan: &[usize; LEN]) -> [u8; LEN] {
    let mut out = [0; LEN];
    for (&value, &position) in coded.iter().zip(scan) {
        out[position] = value;
    }
    out
}

fn slice_parameters(slice: &SliceParams<'_>) -> Result<ffi::VASliceParameterBufferHEVC, Error> {
    use ffi::hevc_long_slice_flags as lf;
    let h = slice.header;
    let mut ref_pic_list = [[NO_REFERENCE; 15]; 2];
    for (list, va_list) in slice.ref_lists.iter().zip(&mut ref_pic_list) {
        for (slot, &index) in va_list.iter_mut().zip(list) {
            *slot = u8::try_from(index)
                .map_err(|_| Error::Unsupported("a reference past ReferenceFrames".into()))?;
        }
    }
    let flag = u32::from;
    let long_slice_flags = lf::LAST_SLICE_OF_PIC.put(flag(slice.last))
        | lf::DEPENDENT_SLICE_SEGMENT_FLAG.put(flag(h.dependent_slice_segment_flag))
        | lf::SLICE_TYPE.put(u32::from(h.slice_type.raw()))
        | lf::COLOR_PLANE_ID.put(u32::from(h.colour_plane_id))
        | lf::SLICE_SAO_LUMA_FLAG.put(flag(h.slice_sao_luma_flag))
        | lf::SLICE_SAO_CHROMA_FLAG.put(flag(h.slice_sao_chroma_flag))
        | lf::MVD_L1_ZERO_FLAG.put(flag(h.mvd_l1_zero_flag))
        | lf::CABAC_INIT_FLAG.put(flag(h.cabac_init_flag))
        | lf::SLICE_TEMPORAL_MVP_ENABLED_FLAG.put(flag(h.slice_temporal_mvp_enabled_flag))
        | lf::SLICE_DEBLOCKING_FILTER_DISABLED_FLAG
            .put(flag(h.slice_deblocking_filter_disabled_flag))
        | lf::COLLOCATED_FROM_L0_FLAG.put(flag(h.collocated_from_l0_flag))
        | lf::SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG
            .put(flag(h.slice_loop_filter_across_slices_enabled_flag));
    let inter = h.slice_type.is_inter();
    let b = h.slice_type == SliceType::B;
    let mut out = ffi::VASliceParameterBufferHEVC {
        slice_data_size: u32::try_from(slice.nal.len())
            .map_err(|_| Error::Unsupported("a slice over 4 GiB".into()))?,
        slice_data_offset: 0,
        slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
        // va_dec_hevc.h: "the number of bytes parsed in the slice_header()
        // after the removal of any emulation prevention bytes".
        slice_data_byte_offset: h.header_bytes,
        slice_segment_address: h.slice_segment_address,
        ref_pic_list,
        long_slice_flags,
        // va_dec_hevc.h: "when slice_temporal_mvp_enabled_flag equals 0, it
        // should take value 0xFF".
        collocated_ref_idx: if h.slice_temporal_mvp_enabled_flag {
            h.collocated_ref_idx
        } else {
            NO_REFERENCE
        },
        num_ref_idx_l0_active_minus1: if inter {
            h.num_ref_idx_l0_active_minus1
        } else {
            0
        },
        num_ref_idx_l1_active_minus1: if b { h.num_ref_idx_l1_active_minus1 } else { 0 },
        slice_qp_delta: h.slice_qp_delta,
        slice_cb_qp_offset: h.slice_cb_qp_offset,
        slice_cr_qp_offset: h.slice_cr_qp_offset,
        slice_beta_offset_div2: h.slice_beta_offset_div2,
        slice_tc_offset_div2: h.slice_tc_offset_div2,
        luma_log2_weight_denom: 0,
        delta_chroma_log2_weight_denom: 0,
        delta_luma_weight_l0: [0; 15],
        luma_offset_l0: [0; 15],
        delta_chroma_weight_l0: [[0; 2]; 15],
        chroma_offset_l0: [[0; 2]; 15],
        delta_luma_weight_l1: [0; 15],
        luma_offset_l1: [0; 15],
        delta_chroma_weight_l1: [[0; 2]; 15],
        chroma_offset_l1: [[0; 2]; 15],
        five_minus_max_num_merge_cand: h.five_minus_max_num_merge_cand,
        num_entry_point_offsets: narrow_u16(h.num_entry_point_offsets),
        entry_offset_to_subset_array: 0,
        slice_data_num_emu_prevn_bytes: narrow_u16(h.emulation_prevention_bytes),
        va_reserved: [0; 2],
    };
    if let Some(table) = &h.pred_weight_table {
        // An entry without explicit weights keeps delta 0 and offset 0:
        // the inferred 2^denominator weight (7.4.7.3).
        out.luma_log2_weight_denom = table.luma_log2_weight_denom;
        out.delta_chroma_log2_weight_denom = table.delta_chroma_log2_weight_denom;
        for list in 0..2 {
            let (luma_weight, luma_offset, chroma_weight, chroma_offset) = if list == 0 {
                (
                    &mut out.delta_luma_weight_l0,
                    &mut out.luma_offset_l0,
                    &mut out.delta_chroma_weight_l0,
                    &mut out.chroma_offset_l0,
                )
            } else {
                (
                    &mut out.delta_luma_weight_l1,
                    &mut out.luma_offset_l1,
                    &mut out.delta_chroma_weight_l1,
                    &mut out.chroma_offset_l1,
                )
            };
            for (i, entry) in table.entries[list].iter().enumerate().take(15) {
                if let Some((weight, offset)) = entry.luma {
                    luma_weight[i] = weight;
                    luma_offset[i] = offset;
                }
                if let Some(pairs) = entry.chroma {
                    for (c, &(weight, _)) in pairs.iter().enumerate() {
                        chroma_weight[i][c] = weight;
                        chroma_offset[i][c] =
                            i8::try_from(table.chroma_offset(list, i, c)).unwrap_or(0);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `VAPictureParameterBufferHEVCRext` of an SPS's and a PPS's range
/// extensions: their flags, and the PPS's chroma QP offset lists (7.3.2.3.2).
fn range_extension_parameters(sps: &Sps, pps: &Pps) -> ffi::VAPictureParameterBufferHEVCRext {
    use ffi::hevc_range_extension_pic_fields as rf;
    let flag = u32::from;
    let (s, p) = (&sps.range_extension, &pps.range_extension);
    let range_extension_pic_fields = rf::TRANSFORM_SKIP_ROTATION_ENABLED_FLAG
        .put(flag(s.transform_skip_rotation_enabled_flag))
        | rf::TRANSFORM_SKIP_CONTEXT_ENABLED_FLAG.put(flag(s.transform_skip_context_enabled_flag))
        | rf::IMPLICIT_RDPCM_ENABLED_FLAG.put(flag(s.implicit_rdpcm_enabled_flag))
        | rf::EXPLICIT_RDPCM_ENABLED_FLAG.put(flag(s.explicit_rdpcm_enabled_flag))
        | rf::EXTENDED_PRECISION_PROCESSING_FLAG.put(flag(s.extended_precision_processing_flag))
        | rf::INTRA_SMOOTHING_DISABLED_FLAG.put(flag(s.intra_smoothing_disabled_flag))
        | rf::HIGH_PRECISION_OFFSETS_ENABLED_FLAG.put(flag(s.high_precision_offsets_enabled_flag))
        | rf::PERSISTENT_RICE_ADAPTATION_ENABLED_FLAG
            .put(flag(s.persistent_rice_adaptation_enabled_flag))
        | rf::CABAC_BYPASS_ALIGNMENT_ENABLED_FLAG.put(flag(s.cabac_bypass_alignment_enabled_flag))
        | rf::CROSS_COMPONENT_PREDICTION_ENABLED_FLAG
            .put(flag(p.cross_component_prediction_enabled_flag))
        | rf::CHROMA_QP_OFFSET_LIST_ENABLED_FLAG.put(flag(p.chroma_qp_offset_list_enabled_flag));
    let mut out = ffi::VAPictureParameterBufferHEVCRext {
        range_extension_pic_fields,
        diff_cu_chroma_qp_offset_depth: p.diff_cu_chroma_qp_offset_depth,
        chroma_qp_offset_list_len_minus1: u8::try_from(
            p.chroma_qp_offset_lists.len().saturating_sub(1),
        )
        .unwrap_or(u8::MAX),
        log2_sao_offset_scale_luma: p.log2_sao_offset_scale_luma,
        log2_sao_offset_scale_chroma: p.log2_sao_offset_scale_chroma,
        log2_max_transform_skip_block_size_minus2: p.log2_max_transform_skip_block_size_minus2,
        ..Default::default()
    };
    for ((cb, cr), &(cb_offset, cr_offset)) in (out.cb_qp_offset_list.iter_mut())
        .zip(out.cr_qp_offset_list.iter_mut())
        .zip(&p.chroma_qp_offset_lists)
    {
        (*cb, *cr) = (cb_offset, cr_offset);
    }
    out
}

/// `VASliceParameterBufferHEVCRext` of a slice: its prediction weights'
/// offsets at 16 bits, as the base structure has them at 8 (the same
/// values, at the 8 bits this decodes), and `cu_chroma_qp_offset_enabled_flag`.
fn slice_range_extension(slice: &SliceParams<'_>) -> ffi::VASliceParameterBufferHEVCRext {
    let h = slice.header;
    let mut out = ffi::VASliceParameterBufferHEVCRext {
        slice_ext_flags: ffi::hevc_slice_ext_flags::CU_CHROMA_QP_OFFSET_ENABLED_FLAG
            .put(u32::from(h.cu_chroma_qp_offset_enabled_flag)),
        ..Default::default()
    };
    if let Some(table) = &h.pred_weight_table {
        for list in 0..2 {
            let (luma_offset, chroma_offset) = if list == 0 {
                (&mut out.luma_offset_l0, &mut out.chroma_offset_l0)
            } else {
                (&mut out.luma_offset_l1, &mut out.chroma_offset_l1)
            };
            for (i, entry) in table.entries[list].iter().enumerate().take(15) {
                if let Some((_, offset)) = entry.luma {
                    luma_offset[i] = i16::from(offset);
                }
                if entry.chroma.is_some() {
                    for (c, slot) in chroma_offset[i].iter_mut().enumerate() {
                        *slot = table.chroma_offset(list, i, c);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_diagonal_scan_runs_up_and_right() {
        // H.265 6.5.3 for a 4x4 block: (0,0), (0,1), (1,0), (0,2), ...
        assert_eq!(
            DIAGONAL_4X4,
            [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15]
        );
        assert_eq!(DIAGONAL_8X8[..6], [0, 8, 1, 16, 9, 2]);
        assert_eq!(DIAGONAL_8X8[63], 63);
        let coded: [u8; 16] = core::array::from_fn(|i| i as u8);
        assert_eq!(raster(&coded, &DIAGONAL_4X4)[..4], [0, 2, 5, 9]);
    }

    #[test]
    fn a_4_4_4_stream_takes_main_4_4_4_into_xyuv_where_its_caller_lands_it() -> Result<(), Error> {
        use crate::codec::hevc::tests::{sps_4_4_4_with, sps_of};
        let four = Sps::parse(&sps_4_4_4_with(0))?;
        assert_eq!(
            profile_for(&four, true)?,
            (Profile::HEVC_MAIN444, SurfaceFormat::Xyuv)
        );
        assert!(matches!(
            profile_for(&four, false),
            Err(Error::Unsupported(why)) if why == "HEVC 4:4:4: its caller lands 4:2:0 pictures alone"
        ));
        let main = Sps::parse(&sps_of(1, 1, 0, false, None))?;
        let ten = Sps::parse(&sps_of(2, 1, 2, false, None))?;
        for four_four_four in [false, true] {
            assert_eq!(
                profile_for(&main, four_four_four)?,
                (Profile::HEVC_MAIN, SurfaceFormat::Nv12)
            );
            assert_eq!(
                profile_for(&ten, four_four_four)?,
                (Profile::HEVC_MAIN10, SurfaceFormat::P010)
            );
        }
        Ok(())
    }

    #[test]
    fn range_extension_parameters_carry_each_flag_and_the_chroma_qp_offset_lists()
    -> Result<(), Error> {
        use crate::codec::hevc::pps::PpsRangeExtension;
        use crate::codec::hevc::tests::{pps_with_range_extension, sps_4_4_4_with};
        // sps_range_extension()'s flags in syntax order: rotation, implicit RDPCM,
        // persistent Rice adaptation and CABAC bypass alignment on.
        let sps = Sps::parse(&sps_4_4_4_with(0b1_0100_0011))?;
        let pps = Pps::parse(&pps_with_range_extension(PpsRangeExtension {
            log2_max_transform_skip_block_size_minus2: 2,
            cross_component_prediction_enabled_flag: true,
            chroma_qp_offset_list_enabled_flag: true,
            diff_cu_chroma_qp_offset_depth: 1,
            chroma_qp_offset_lists: vec![(-3, 4), (5, -6)],
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
        }))?;
        let rext = range_extension_parameters(&sps, &pps);
        // va_dec_hevc.h's bits: 0, 2, 7 and 8 of the SPS's, 9 and 10 of the PPS's.
        assert_eq!(rext.range_extension_pic_fields, 0x785);
        assert_eq!(rext.diff_cu_chroma_qp_offset_depth, 1);
        assert_eq!(rext.chroma_qp_offset_list_len_minus1, 1);
        assert_eq!(rext.log2_max_transform_skip_block_size_minus2, 2);
        assert_eq!(rext.cb_qp_offset_list, [-3, 5, 0, 0, 0, 0]);
        assert_eq!(rext.cr_qp_offset_list, [4, -6, 0, 0, 0, 0]);
        let plain = range_extension_parameters(&Sps::parse(&sps_4_4_4_with(0))?, &pps);
        assert_eq!(plain.range_extension_pic_fields, 0x600);
        Ok(())
    }

    #[test]
    fn more_than_eight_current_references_are_refused() {
        let sets = |before: usize, after: usize, long_term: usize| {
            [
                (0..before).collect::<Vec<_>>(),
                (before..before + after).collect(),
                (0..long_term).collect(),
            ]
        };
        assert!(current_references(&sets(8, 0, 0)).is_ok());
        assert!(current_references(&sets(4, 3, 1)).is_ok());
        for refused in [sets(9, 0, 0), sets(5, 4, 0), sets(4, 4, 1)] {
            assert!(matches!(
                current_references(&refused),
                Err(Error::Unsupported(why)) if why.contains("VA-API takes 8")
            ));
        }
    }
}
