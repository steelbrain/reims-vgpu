//! VP9 decoding on VA-API: [`crate::codec::vp9::Decoder`] over an accelerator
//! that fills libva's VP9 buffers (va_dec_vp9.h) and decodes each frame into
//! a pool of NV12 surfaces for profile 0, P010 ones for profile 2 at 10 bits.
//!
//! A frame goes to the driver whole, its uncompressed header first, as one
//! slice (`VA_SLICE_DATA_FLAG_ALL`), with the header's fields in the picture
//! parameters and each segment's derived quantizer scales and loop filter
//! levels in the slice's, as ffmpeg sends them. The eight reference slots go
//! as their surfaces, `VA_INVALID_SURFACE` for an empty one. A frame of
//! another size or format than the last makes a new context and pool; the
//! slots keep the surfaces of the frames before it, which the driver scales
//! from as references.

use super::buffer::{Buffer, render_decode};
use super::caps::{Context, Entrypoint, Profile};
use super::display::{Display, Error};
use super::ffi;
use super::ffi::vp9::{
    VADecPictureParameterBufferVP9, VASegmentParameterVP9, VASliceParameterBufferVP9,
    vp9_pic_fields as pf, vp9_segment_flags as sf,
};
use super::frame::Frame;
use super::surface::{Rect, Surface, SurfaceFormat, SurfacePool, end_context};
use super::wait::{self, Deadline};
use crate::codec::vp9::header::{FrameHeader, MAX_SEGMENTS};
use crate::codec::vp9::quant::SegmentParams;
use crate::codec::vp9::{Accelerator, Decoder, Format, FrameParams, Output};

/// The segment feature that names a reference frame (`SEG_LVL_REF_FRAME`).
const SEG_LVL_REF_FRAME: usize = 2;
/// The segment feature that skips the residual (`SEG_LVL_SKIP`).
const SEG_LVL_SKIP: usize = 3;

/// A VP9 decoder on a VA display.
///
/// Samples are a container's: a frame, or a superframe of frames, whole.
/// Frames come out as the stream shows them, which is decode order, at most
/// one a sample a container holds.
pub struct Vp9Decoder {
    inner: Decoder<VaVp9>,
}

impl Vp9Decoder {
    /// The frames a caller is to hold at once, beyond those the decoder's
    /// eight slots hold: the decoder's pool makes a surface for a frame as it
    /// needs one, up to a bound past which a decode finds no free surface
    /// (`Error::OutOfSurfaces`).
    pub const SPARE_FRAMES: usize = 2;

    /// A decoder on `display`. Each frame's profile, bit depth and size are
    /// checked against what the driver decodes as it comes.
    pub fn new(display: &Display) -> Self {
        Self {
            inner: Decoder::new(VaVp9 {
                display: display.clone_ref(),
                session: None,
            }),
        }
    }

    /// Why the driver decodes none of a stream of VP9 `profile` at
    /// `bit_depth` and `width` by `height`, or `Ok` where its VLD config of
    /// the profile takes it.
    pub fn supports(
        display: &Display,
        profile: u8,
        bit_depth: u8,
        (width, height): (u32, u32),
    ) -> Result<(), Error> {
        let format = Format {
            profile,
            bit_depth,
            width,
            height,
        };
        select_profile(display, &format).map(|_| ())
    }

    /// Decodes one sample, tagged `id`, and returns the frames it shows. The
    /// decode may still be running on the GPU: a frame's readback waits for
    /// it.
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

    /// Forgets every slot, as for a seek: the next frame decodes only if it
    /// is a key frame.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The VA profile the decoder runs, once it has decoded a frame.
    pub fn profile(&self) -> Option<Profile> {
        self.inner.accelerator().session.as_ref().map(|s| s.profile)
    }
}

// Decoders move between threads: libva is thread-safe (va.h,
// "Multithreading Guide"), and each decoder is used by one thread at a
// time through `&mut`.
const _: fn() = || {
    fn send<T: Send>() {}
    send::<Vp9Decoder>();
};

/// The frame VA-API decoded for one of the decoder's outputs.
fn frame_from_output(output: Output<Surface>, deadline: &Deadline) -> Frame {
    let (width, height) = output.size;
    Frame {
        surface: output.picture,
        id: output.id,
        poc: 0,
        idr: output.key_frame,
        reference: output.reference,
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
struct VaVp9 {
    display: Display,
    session: Option<Session>,
}

/// What one format's frames are decoded with.
struct Session {
    profile: Profile,
    /// Destroyed first, before the surfaces it renders into: it holds its
    /// config.
    context: Option<Context>,
    pool: SurfacePool,
}

impl Drop for Session {
    fn drop(&mut self) {
        // The context goes first. The pool may outlive it: the slots and
        // the frames the caller holds keep their surfaces.
        end_context(self.context.take(), [&self.pool]);
    }
}

impl VaVp9 {
    fn session(&self) -> Result<&Session, Error> {
        self.session
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding before a format is configured".into()))
    }
}

/// The VA profile and surface format of `format`'s frames: profile 0 in
/// NV12, profile 2 at 10 bits in P010. The codec's decoder has refused the
/// rest.
fn profile_for(format: &Format) -> Result<(Profile, SurfaceFormat), Error> {
    match (format.profile, format.bit_depth) {
        (0, 8) => Ok((Profile::VP9_PROFILE0, SurfaceFormat::Nv12)),
        (2, 10) => Ok((Profile::VP9_PROFILE2, SurfaceFormat::P010)),
        (profile, depth) => Err(Error::Unsupported(format!(
            "VP9 profile {profile} at {depth} bits: VA-API decodes profile 0, and 2 at 10 bits, \
             here"
        ))),
    }
}

/// The VA profile, where the driver decodes `format`'s frames at their size.
fn select_profile(display: &Display, format: &Format) -> Result<Profile, Error> {
    let (profile, surfaces) = profile_for(format)?;
    let name = profile.name().unwrap_or("VP9");
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
    if limits[0].is_some_and(|f| f & surfaces.rt_format() == 0) {
        return Err(Error::Unsupported(format!(
            "{name}: no {} render targets",
            surfaces.name()
        )));
    }
    let (width, height) = (format.width, format.height);
    if limits[1].is_some_and(|max| width > max) || limits[2].is_some_and(|max| height > max) {
        return Err(Error::Unsupported(format!(
            "{name} at {width}x{height}: the driver's maximum is {}x{}",
            limits[1].unwrap_or(0),
            limits[2].unwrap_or(0)
        )));
    }
    Ok(profile)
}

impl Accelerator for VaVp9 {
    type Picture = Surface;
    type Error = Error;

    fn configure(&mut self, format: &Format) -> Result<(), Error> {
        self.session = None;
        let profile = select_profile(&self.display, format)?;
        let (_, surfaces) = profile_for(format)?;
        let mut attribs = [ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: surfaces.rt_format(),
        }];
        // Eight slots and the frame being decoded.
        let references = 8;
        let (context, pool) = Context::new_decoder(
            &self.display,
            profile,
            &mut attribs,
            surfaces,
            (format.width, format.height),
            references,
        )?;
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

    fn decode_frame(
        &mut self,
        picture: &Surface,
        params: &FrameParams<'_, Surface>,
    ) -> Result<(), Error> {
        let session = self.session()?;
        let decoding = session
            .context
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding without a context".into()))?;
        let context = decoding.id;
        let display = &self.display;
        let picture_parameters = picture_parameters(params)?;
        let slice_parameters = slice_parameters(params)?;
        let picture_buffer = Buffer::with_value(
            display,
            context,
            ffi::VAPictureParameterBufferType,
            &picture_parameters,
        )?;
        let slice_buffer = Buffer::with_value(
            display,
            context,
            ffi::VASliceParameterBufferType,
            &slice_parameters,
        )?;
        let data = Buffer::with_bytes(display, context, ffi::VASliceDataBufferType, params.frame)?;
        let groups: [&[&Buffer<'_>]; 2] = [&[&picture_buffer], &[&slice_buffer, &data]];
        // A first frame past its deadline may still be submitted: the pool
        // is kept until the GPU is done with it.
        picture.stalled_if(render_decode(display, decoding, picture.id(), &groups))
    }
}

/// A value the uncompressed header bounds, narrowed.
fn narrow_u8(value: u32, what: &'static str) -> Result<u8, Error> {
    u8::try_from(value).map_err(|_| Error::Unsupported(format!("VP9 {what} of {value}")))
}

/// `VADecPictureParameterBufferVP9` of the frame, as ffmpeg's VA-API VP9
/// decoder fills it.
fn picture_parameters(
    params: &FrameParams<'_, Surface>,
) -> Result<VADecPictureParameterBufferVP9, Error> {
    let h: &FrameHeader = params.header;
    let flag = u32::from;
    let seg = &h.segmentation;
    let [last, golden, alt] = h.ref_frame_idx;
    let [last_bias, golden_bias, alt_bias] = h.ref_frame_sign_bias;
    let pic_fields = pf::SUBSAMPLING_X.put(flag(h.color.subsampling_x))
        | pf::SUBSAMPLING_Y.put(flag(h.color.subsampling_y))
        // `frame_type`: 0 a key frame, 1 another.
        | pf::FRAME_TYPE.put(flag(!h.key_frame))
        | pf::SHOW_FRAME.put(flag(h.show_frame))
        | pf::ERROR_RESILIENT_MODE.put(flag(h.error_resilient_mode))
        | pf::INTRA_ONLY.put(flag(h.intra_only))
        | pf::ALLOW_HIGH_PRECISION_MV.put(flag(h.allow_high_precision_mv))
        | pf::MCOMP_FILTER_TYPE.put(u32::from(h.interp_filter))
        | pf::FRAME_PARALLEL_DECODING_MODE.put(flag(h.frame_parallel_decoding_mode))
        | pf::RESET_FRAME_CONTEXT.put(u32::from(h.reset_frame_context))
        | pf::REFRESH_FRAME_CONTEXT.put(flag(h.refresh_frame_context))
        | pf::FRAME_CONTEXT_IDX.put(u32::from(h.frame_context_idx))
        | pf::SEGMENTATION_ENABLED.put(flag(seg.enabled))
        | pf::SEGMENTATION_TEMPORAL_UPDATE.put(flag(seg.temporal_update))
        | pf::SEGMENTATION_UPDATE_MAP.put(flag(seg.update_map))
        | pf::LAST_REF_FRAME.put(u32::from(last))
        | pf::LAST_REF_FRAME_SIGN_BIAS.put(flag(last_bias))
        | pf::GOLDEN_REF_FRAME.put(u32::from(golden))
        | pf::GOLDEN_REF_FRAME_SIGN_BIAS.put(flag(golden_bias))
        | pf::ALT_REF_FRAME.put(u32::from(alt))
        | pf::ALT_REF_FRAME_SIGN_BIAS.put(flag(alt_bias))
        | pf::LOSSLESS_FLAG.put(flag(h.quantization.lossless()));
    let reference_frames = params.references.each_ref().map(|reference| {
        reference
            .as_ref()
            .map_or(ffi::VA_INVALID_SURFACE, |r| r.picture.id())
    });
    let narrow_u16 = |value: u32, what: &'static str| {
        u16::try_from(value).map_err(|_| Error::Unsupported(format!("VP9 {what} of {value}")))
    };
    Ok(VADecPictureParameterBufferVP9 {
        frame_width: narrow_u16(h.width, "a frame width")?,
        frame_height: narrow_u16(h.height, "a frame height")?,
        reference_frames,
        pic_fields,
        filter_level: h.loop_filter.level,
        sharpness_level: h.loop_filter.sharpness,
        log2_tile_rows: h.tile_rows_log2,
        log2_tile_columns: h.tile_cols_log2,
        frame_header_length_in_bytes: h.uncompressed_header_size,
        first_partition_size: h.header_size_in_bytes,
        mb_segment_tree_probs: seg.tree_probs,
        // ffmpeg sends 255 where the map's prediction is not temporal.
        segment_pred_probs: if seg.temporal_update {
            seg.pred_probs
        } else {
            [255; 3]
        },
        profile: h.profile,
        bit_depth: narrow_u8(u32::from(h.color.bit_depth), "bit depth")?,
        va_reserved: [0; ffi::VA_PADDING_MEDIUM],
    })
}

/// One segment's `VASegmentParameterVP9`: its reference and skip features,
/// where segmentation is on, and its derived values.
fn segment_parameters(
    header: &FrameHeader,
    segment: usize,
    derived: &SegmentParams,
) -> VASegmentParameterVP9 {
    let seg = &header.segmentation;
    let enabled = |level: usize| {
        seg.enabled
            && seg
                .feature_enabled
                .get(segment)
                .and_then(|features| features.get(level))
                .is_some_and(|&on| on)
    };
    let reference = seg
        .feature_data
        .get(segment)
        .and_then(|data| data.get(SEG_LVL_REF_FRAME))
        .and_then(|&value| u32::try_from(value).ok())
        .unwrap_or(0);
    let flags = sf::SEGMENT_REFERENCE_ENABLED.put(u32::from(enabled(SEG_LVL_REF_FRAME)))
        | sf::SEGMENT_REFERENCE.put(reference)
        | sf::SEGMENT_REFERENCE_SKIPPED.put(u32::from(enabled(SEG_LVL_SKIP)));
    VASegmentParameterVP9 {
        // The flags' four bits.
        segment_flags: u16::try_from(flags).unwrap_or(0),
        filter_level: derived.filter_level,
        luma_ac_quant_scale: derived.luma_ac,
        luma_dc_quant_scale: derived.luma_dc,
        chroma_ac_quant_scale: derived.chroma_ac,
        chroma_dc_quant_scale: derived.chroma_dc,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    }
}

/// `VASliceParameterBufferVP9` of the frame: all of it one slice, and each
/// segment's parameters.
fn slice_parameters(params: &FrameParams<'_, Surface>) -> Result<VASliceParameterBufferVP9, Error> {
    let size = u32::try_from(params.frame.len())
        .map_err(|_| Error::Unsupported("a VP9 frame past 4 GiB".into()))?;
    let mut seg_param = [VASegmentParameterVP9::default(); MAX_SEGMENTS];
    for (segment, (out, derived)) in seg_param.iter_mut().zip(params.segments).enumerate() {
        *out = segment_parameters(params.header, segment, derived);
    }
    Ok(VASliceParameterBufferVP9 {
        slice_data_size: size,
        slice_data_offset: 0,
        slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
        seg_param,
        va_reserved: [0; ffi::VA_PADDING_LOW],
    })
}

#[cfg(test)]
mod tests;
