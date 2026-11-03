//! H.265 decoding on Vulkan Video: the codec crate's HEVC decoder over an
//! accelerator that maps its parameter sets onto a session parameters
//! object and its DPB onto DPB slots, each a DPB picture of the session's,
//! and lands each picture in an output its caller supplies, as the H.264
//! decoder does ([`crate::H264Decoder`]).
//!
//! The codec crate runs the decoding process above the slice data: order
//! counts (8.3.1), the reference picture set's search and marking (8.3.2),
//! random access (IDR, CRA and BLA pictures, `NoRaslOutputFlag` and the
//! RASL pictures it skips, 8.1.3) and output order (C.5.2). The hardware
//! reads each slice segment header itself: what goes with the slices is
//! the picture's std information, its order count and the slots of its
//! three current sets (`RefPicSetStCurrBefore`, `RefPicSetStCurrAfter`,
//! `RefPicSetLtCurr`), from which it builds the reference lists, and each
//! reference's order count and marking, as NVDEC takes them.
//!
//! **Slots.** Every picture of the reference picture set the DPB holds, a
//! current one or one kept for later pictures (`RefPicSetStFoll`,
//! `RefPicSetLtFoll`), goes to the decode as a reference with the slot its
//! decode set up, so that it keeps its slot while it is a reference; the
//! picture takes a slot none of them holds, the session's choice
//! (`Session::free_slot`: the least, or, where decodes overlap copies, the
//! one copied out of longest ago). The session has the DPB's
//! `sps_max_dec_pic_buffering_minus1 + 1` slots, which the codec crate
//! holds the set's pictures beside the current one to (C.5.2.2), and one
//! more where decodes overlap copies and the device has it.
//! A picture held only for output is no reference and holds no slot: its
//! output was written when it was decoded.

pub(crate) mod params;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoDecodeH265PictureInfo, StdVideoDecodeH265PictureInfoFlags,
    StdVideoDecodeH265ReferenceInfo, StdVideoDecodeH265ReferenceInfoFlags,
};
use reims_vgpu_video_codec::hevc::nal::{is_idr, is_irap};
use reims_vgpu_video_codec::hevc::{
    self, Accelerator, Decoder, NalFormat, Output, OutputOrder, PictureParams, SliceParams,
    hvcc::DecoderConfiguration, nal::NalUnit, nal::nal_type, sps::Sps,
};

use self::params::ParameterSets;
use crate::device::VideoDevice;
use crate::error::Error;
use crate::h264::new_picture;
use crate::import::SampleMemory;
use crate::picture::Picture;
use crate::profile::{H265Capabilities, H265Profile, VideoProfile, h265_capabilities};
use crate::session::{BitstreamCounts, CodecInfo, Decode, InMemory, Session, Target};
use crate::std_types::zeroed;
use crate::surface::{Landing, Surface, SurfaceSource, SurfaceSpec};
use crate::timeline::Timeline;

/// A decoded H.265 frame: its picture, which holds the output it landed
/// in, and what the decoder knows of it.
pub type H265Frame<T> = Output<Picture<T>>;

/// The pictures a current set of `StdVideoDecodeH265PictureInfo` holds
/// (`STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE`).
const CURRENT_SET_SIZE: usize = 8;

/// A slot index no picture holds, in a current set of
/// `StdVideoDecodeH265PictureInfo`.
const NO_SLOT: u8 = 0xff;

/// An H.265 decoder on a [`VideoDevice`], Main and Main 10, landing its
/// pictures in the outputs `S` gives: NV12 at 8 bits, P010 at 10.
///
/// Samples are access units: length-prefixed NAL units with the `hvcC`
/// record's length size, as VideoToolbox's samples are, or Annex B for
/// [`H265Decoder::new_annex_b`]. Frames come out in decode order unless
/// [`H265Decoder::set_output_order`] asks for display order. Each decode
/// is submitted and not waited for: a frame's picture names the timeline
/// value after which its output holds it.
///
/// A sample whose one picture is a RASL picture the decoder skips (one
/// after the CRA picture it started or restarted at, or after a BLA
/// picture, 8.1.3) is answered with [`Error::H265Stream`] of
/// [`hevc::Error::SkippedLeadingPicture`]: no picture, and no failure.
/// After a sample that fails, pictures are refused with
/// [`hevc::Error::Unrecovered`] until an IRAP picture decodes, as Apple's
/// HEVC decoder does.
pub struct H265Decoder<S: SurfaceSource> {
    inner: Decoder<VulkanH265<S>>,
}

impl<S: SurfaceSource> H265Decoder<S> {
    /// A decoder for samples described by an `hvcC` record (ISO/IEC
    /// 14496-15, 8.3.3.1), whose parameter sets it takes. The record's
    /// first SPS is checked against what the decoder and the device take,
    /// so that a stream they cannot decode is refused here.
    pub fn new(device: &Arc<VideoDevice>, source: S, hvcc: &[u8]) -> Result<Self, Error> {
        let accelerator = VulkanH265::new(device, source)?;
        let config = DecoderConfiguration::parse(hvcc)?;
        let first_sps = config
            .nal_units
            .iter()
            .find(|unit| NalUnit::new(unit).is_ok_and(|nal| nal.unit_type() == nal_type::SPS));
        if let Some(sps) = first_sps {
            let sps = Sps::parse(sps)?;
            hevc::check_supported(&sps)?;
            select(&accelerator.capabilities, &sps)?;
        }
        Ok(Self {
            inner: Decoder::from_hvcc(accelerator, hvcc)?,
        })
    }

    /// A decoder for Annex B samples, which carry their parameter sets.
    pub fn new_annex_b(device: &Arc<VideoDevice>, source: S) -> Result<Self, Error> {
        Ok(Self {
            inner: Decoder::new(VulkanH265::new(device, source)?, NalFormat::AnnexB),
        })
    }

    /// Chooses decode order (the default) or display order.
    pub fn set_output_order(&mut self, order: OutputOrder) {
        self.inner.set_output_order(order);
    }

    /// Decodes one access unit, tagged `id`, and returns the frames that
    /// come out ([`reims_vgpu_video_codec::hevc::Decoder::decode`]). Their
    /// decodes may still be running: each frame's picture says the value
    /// its decode signals.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<H265Frame<S::Surface>>, Error> {
        self.inner.decode(sample, id)
    }

    /// [`Self::decode`] of a sample whose bytes lie in `memory` too, which
    /// each picture's decode reads in place, with no copy, where the
    /// device takes them there: the picture's first slice segment's start
    /// code at a multiple of `minBitstreamBufferOffsetAlignment` in the
    /// memory, and the sample holding the slice segments padded to a
    /// multiple of `minBitstreamBufferSizeAlignment` from there, the
    /// padding zeros as the caller wrote them. Elsewhere the slice segments
    /// are copied as `decode` copies them.
    ///
    /// # Safety
    ///
    /// `memory` is memory of the decoder's device that holds `sample`'s
    /// bytes at its offset, valid while its `keep` lives; nothing writes
    /// those bytes until the decodes reading them are done, which they are
    /// once [`Self::semaphore`] reaches [`Self::submitted`] after the call.
    pub unsafe fn decode_in_place(
        &mut self,
        sample: &[u8],
        id: u64,
        memory: SampleMemory,
    ) -> Result<Vec<H265Frame<S::Surface>>, Error> {
        self.inner.accelerator_mut().in_memory = Some(InMemory {
            start: std::ptr::NonNull::from(sample).cast(),
            len: sample.len(),
            memory,
        });
        let decoded = self.inner.decode(sample, id);
        self.inner.accelerator_mut().in_memory = None;
        decoded
    }

    /// The value [`Self::semaphore`] reaches once every decode and copy
    /// submitted so far is done.
    pub fn submitted(&self) -> u64 {
        self.inner.accelerator().timeline.last_submitted()
    }

    /// Emits the frames still waiting for display order. References are
    /// kept, so decoding can go on.
    pub fn flush(&mut self) -> Vec<H265Frame<S::Surface>> {
        self.inner.flush()
    }

    /// Forgets every reference, as for a seek: the next picture must be an
    /// IRAP picture, whose RASL pictures are skipped.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The timeline semaphore each decode, and each copy out of the DPB,
    /// signals.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.inner.accelerator().timeline.semaphore()
    }

    /// The profile the stream's session decodes with, once a picture has
    /// started.
    pub fn profile(&self) -> Option<H265Profile> {
        match self.inner.accelerator().session.as_ref()?.caps.profile {
            VideoProfile::H265(profile) => Some(profile),
            VideoProfile::H264(_) => None,
        }
    }

    /// How the decodes so far read their slice segments: in place, or
    /// copied.
    pub fn bitstreams(&self) -> BitstreamCounts {
        self.inner.accelerator().bitstreams
    }
}

/// The Vulkan half of the decoder.
struct VulkanH265<S: SurfaceSource> {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    capabilities: Vec<H265Capabilities>,
    source: S,
    /// The stream's session, which holds the timeline itself, and waits
    /// for its decodes as it goes.
    session: Option<Session<ParameterSets>>,
    /// Where the sample being decoded lies in device memory, while
    /// [`H265Decoder::decode_in_place`] decodes it.
    in_memory: Option<InMemory>,
    /// How the decodes so far read their slice segments.
    bitstreams: BitstreamCounts,
}

impl<S: SurfaceSource> VulkanH265<S> {
    fn new(device: &Arc<VideoDevice>, source: S) -> Result<Self, Error> {
        let capabilities = h265_capabilities(device);
        if capabilities.is_empty() {
            return Err(Error::NoVideoDecode(format!(
                "{} decodes no H.265 profile into NV12 or P010 images it can copy or sample",
                device.name()
            )));
        }
        Ok(Self {
            device: Arc::clone(device),
            timeline: Timeline::new(device)?,
            capabilities,
            source,
            session: None,
            in_memory: None,
            bitstreams: BitstreamCounts::default(),
        })
    }
}

/// The capabilities of the profile that decodes `sps`'s samples
/// ([`H265Profile::for_bit_depth`]), where the device decodes it at `sps`'s
/// coded size.
fn select(capabilities: &[H265Capabilities], sps: &Sps) -> Result<H265Capabilities, Error> {
    let depth = sps.bit_depth_luma_minus8.saturating_add(8);
    let profile = H265Profile::for_bit_depth(depth)
        .ok_or_else(|| Error::Unsupported(format!("H.265 at {depth} bits")))?;
    let caps = capabilities
        .iter()
        .find(|c| c.profile == profile)
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "H.265 at {depth} bits: the device decodes profiles {:?}",
                capabilities
                    .iter()
                    .map(|c| c.profile.profile_idc)
                    .collect::<Vec<_>>()
            ))
        })?;
    caps.check_extent(
        "H.265",
        sps.pic_width_in_luma_samples,
        sps.pic_height_in_luma_samples,
    )?;
    Ok(*caps)
}

impl<S: SurfaceSource> Accelerator for VulkanH265<S> {
    type Picture = Picture<S::Surface>;
    type Error = Error;

    /// A session whose DPB holds `dpb_size` pictures: the references the
    /// codec crate holds beside the current picture, and the current one.
    fn configure(&mut self, sps: &Sps, dpb_size: usize) -> Result<(), Error> {
        // The old session goes first: it waits for its decodes.
        self.session = None;
        let caps = select(&self.capabilities, sps)?;
        let references = u32::try_from(dpb_size.saturating_sub(1)).unwrap_or(u32::MAX);
        if references >= caps.max_dpb_slots || references > caps.max_active_reference_pictures {
            return Err(Error::Unsupported(format!(
                "a DPB of {dpb_size} pictures: the device has {} DPB slots and {} active \
                 references",
                caps.max_dpb_slots, caps.max_active_reference_pictures
            )));
        }
        let extent = vk::Extent2D {
            width: sps.pic_width_in_luma_samples,
            height: sps.pic_height_in_luma_samples,
        };
        self.session = Some(Session::new(
            &self.device,
            &self.timeline,
            caps.video(),
            extent,
            references,
        )?);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Self::Picture, Error> {
        let (spec, extent) = {
            let session = self
                .session
                .as_ref()
                .ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))?;
            let spec = SurfaceSpec {
                extent: session.image_extent,
                profile: session.caps.profile,
                decode_target: session.caps.decodes_into_output,
            };
            (spec, session.extent)
        };
        new_picture(
            &self.device,
            &self.timeline,
            &mut self.source,
            &spec,
            extent,
        )
    }

    fn decode_picture(
        &mut self,
        picture: &Self::Picture,
        params: &PictureParams<'_, Self::Picture>,
        slices: &[SliceParams<'_>],
    ) -> Result<(), Error> {
        self.timeline.collect();
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))?;
        let VideoProfile::H265(profile) = session.caps.profile else {
            return Err(Error::Unsupported(
                "an H.265 picture in another session".into(),
            ));
        };
        let mut plan = plan(params, |p: &Self::Picture| p.slot(), session.slots())?;
        // The session's choice among the slots no reference holds, which is
        // the plan's own where its decodes wait for every copy.
        plan.setup_slot = session
            .free_slot(&plan.reference_slots)
            .ok_or_else(|| Error::Unsupported("no DPB slot is free".into()))?;
        let (sps, pps) = (params.sps, params.pps);
        let parameters = session.parameters(
            |sets| sets.holds(sps, pps),
            |device, handle, old| ParameterSets::new(device, handle, profile, old, sps, pps),
        )?;
        let nals: Vec<&[u8]> = slices.iter().map(|s| s.nal).collect();
        let decode = Decode {
            slices: &nals,
            in_memory: self.in_memory.as_ref(),
            parameters,
            output: match picture.surface().landing() {
                Landing::Image(image) => Target::Image(image, picture.output_view()),
                Landing::Buffer(planes) => Target::Buffer(planes, sps.crop_rect()),
            },
            setup_slot: plan.setup_slot,
            reference_slots: &plan.reference_slots,
            codec: CodecInfo::H265 {
                picture: &plan.picture_info,
                setup: &plan.setup_info,
                references: &plan.reference_infos,
            },
        };
        let submitted = session.decode(&decode)?;
        self.bitstreams.count(submitted.in_place);
        picture.decoded(plan.setup_slot, submitted.landed);
        Ok(())
    }
}

/// What one picture's decode names: the slot it is set up in, each
/// reference's slot and std information, in the order of the picture's
/// references, and the picture's own std information as a reference and
/// as the picture decoded.
#[derive(Debug)]
pub(crate) struct Plan {
    pub(crate) setup_slot: i32,
    pub(crate) reference_slots: Vec<i32>,
    pub(crate) reference_infos: Vec<StdVideoDecodeH265ReferenceInfo>,
    pub(crate) setup_info: StdVideoDecodeH265ReferenceInfo,
    pub(crate) picture_info: StdVideoDecodeH265PictureInfo,
}

/// The decode of the picture `params` describes, each reference's slot as
/// `slot_of` answers for its picture, on a session of `slots` slots: every
/// reference the DPB holds named with its slot, the current sets as
/// slots, and the picture set up in the least slot no reference holds.
///
/// Refused where a reference has no slot (it was never decoded) or shares
/// one, where no slot is free, and where a current set holds more than the
/// std structure's 8 pictures.
pub(crate) fn plan<P>(
    params: &PictureParams<'_, P>,
    slot_of: impl Fn(&P) -> Option<i32>,
    slots: u32,
) -> Result<Plan, Error> {
    let mut reference_slots = Vec::with_capacity(params.references.len());
    let mut reference_infos = Vec::with_capacity(params.references.len());
    for reference in &params.references {
        let slot = slot_of(reference.picture)
            .ok_or_else(|| Error::Unsupported("a reference that was never decoded".into()))?;
        if reference_slots.contains(&slot) {
            return Err(Error::Unsupported(format!(
                "two references in DPB slot {slot}"
            )));
        }
        reference_slots.push(slot);
        let mut flags: StdVideoDecodeH265ReferenceInfoFlags = zeroed();
        flags.set_used_for_long_term_reference(u32::from(reference.long_term));
        flags.set_unused_for_reference(0);
        let mut info: StdVideoDecodeH265ReferenceInfo = zeroed();
        info.flags = flags;
        info.PicOrderCntVal = reference.poc;
        reference_infos.push(info);
    }
    let max = i32::try_from(slots).unwrap_or(i32::MAX);
    let setup_slot = (0..max)
        .find(|slot| !reference_slots.contains(slot))
        .ok_or_else(|| Error::Unsupported("no DPB slot is free".into()))?;

    let header = params.header;
    let mut flags: StdVideoDecodeH265PictureInfoFlags = zeroed();
    flags.set_IrapPicFlag(u32::from(is_irap(header.nal_unit_type)));
    flags.set_IdrPicFlag(u32::from(is_idr(header.nal_unit_type)));
    // Every decoded picture is marked "used for short-term reference"
    // (8.1.3, C.5.2.3): a sub-layer non-reference picture may still be
    // predicted from by a higher sub-layer.
    flags.set_IsReference(1);
    flags.set_short_term_ref_pic_set_sps_flag(u32::from(header.short_term_ref_pic_set_sps_flag));
    let mut picture_info: StdVideoDecodeH265PictureInfo = zeroed();
    picture_info.flags = flags;
    picture_info.sps_video_parameter_set_id = params.sps.sps_video_parameter_set_id;
    picture_info.pps_seq_parameter_set_id = params.pps.pps_seq_parameter_set_id;
    picture_info.pps_pic_parameter_set_id = params.pps.pps_pic_parameter_set_id;
    // NumDeltaPocs is 16 at most, and a set's bits fit 16 bits for any set
    // the codec crate reads (16 pictures of 2^15 apart at most, 7.4.8).
    picture_info.NumDeltaPocsOfRefRpsIdx =
        u8::try_from(header.st_rps_reference_delta_pocs).unwrap_or(u8::MAX);
    picture_info.NumBitsForSTRefPicSetInSlice =
        u16::try_from(header.st_rps_bits).unwrap_or(u16::MAX);
    picture_info.PicOrderCntVal = params.poc;
    let [before, after, long_term] = &params.current_sets;
    picture_info.RefPicSetStCurrBefore = current_set(before, &reference_slots)?;
    picture_info.RefPicSetStCurrAfter = current_set(after, &reference_slots)?;
    picture_info.RefPicSetLtCurr = current_set(long_term, &reference_slots)?;

    let mut setup_flags: StdVideoDecodeH265ReferenceInfoFlags = zeroed();
    setup_flags.set_used_for_long_term_reference(0);
    let mut setup_info: StdVideoDecodeH265ReferenceInfo = zeroed();
    setup_info.flags = setup_flags;
    setup_info.PicOrderCntVal = params.poc;
    Ok(Plan {
        setup_slot,
        reference_slots,
        reference_infos,
        setup_info,
        picture_info,
    })
}

/// A current set, indices into the references, as the slots they hold,
/// in its order, the rest of the eight [`NO_SLOT`].
fn current_set(set: &[usize], slots: &[i32]) -> Result<[u8; CURRENT_SET_SIZE], Error> {
    if set.len() > CURRENT_SET_SIZE {
        return Err(Error::Unsupported(format!(
            "{} pictures in one of the current reference sets: Vulkan Video takes {CURRENT_SET_SIZE}",
            set.len()
        )));
    }
    let mut out = [NO_SLOT; CURRENT_SET_SIZE];
    for (entry, &index) in out.iter_mut().zip(set) {
        let slot = slots
            .get(index)
            .and_then(|&slot| u8::try_from(slot).ok())
            .ok_or_else(|| Error::Unsupported("a current set naming no reference".into()))?;
        *entry = slot;
    }
    Ok(out)
}
