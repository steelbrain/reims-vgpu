//! H.264 decoding on Vulkan Video: the codec crate's decoder over an
//! accelerator that maps its parameter sets onto a session parameters
//! object and its DPB onto DPB slots, each a DPB picture of the session's,
//! and lands each picture in an output its caller supplies.
//!
//! The hardware reads the slice headers itself: what goes with the slices
//! is the picture's std information and each reference's (its frame
//! number or long-term index, its order counts), from which it builds the
//! reference lists, as NVDEC does.

mod params;

use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags,
    StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH264ReferenceInfoFlags,
};
use reims_vgpu_video_codec::h264::{
    self, Accelerator, Decoder, NalFormat, Output, OutputOrder, PictureParams, ReferenceFrame,
    SliceParams, avcc::DecoderConfiguration, slice::MemoryManagement, slice::SliceType, sps::Sps,
};

use self::params::ParameterSets;
use crate::device::VideoDevice;
use crate::error::Error;
use crate::import::SampleMemory;
use crate::picture::{ImageView, Picture};
use crate::profile::{H264Capabilities, H264Profile, VideoProfile, h264_capabilities};
use crate::session::{BitstreamCounts, CodecInfo, Decode, InMemory, Session, Target};
use crate::std_types::zeroed;
use crate::surface::{Landing, Surface, SurfaceSource, SurfaceSpec};
use crate::timeline::Timeline;

/// A decoded frame: its picture, which holds the output it landed in, and
/// what the decoder knows of it.
pub type Frame<T> = Output<Picture<T>>;

/// An H.264 decoder on a [`VideoDevice`], landing its pictures in the
/// outputs `S` gives.
///
/// Samples are access units: length-prefixed NAL units with the `avcC`
/// record's length size, as VideoToolbox's samples are, or Annex B for
/// [`H264Decoder::new_annex_b`]. Frames come out in decode order unless
/// [`H264Decoder::set_output_order`] asks for display order. Each decode
/// is submitted and not waited for: a frame's picture names the timeline
/// value after which its output holds it.
pub struct H264Decoder<S: SurfaceSource> {
    inner: Decoder<VulkanH264<S>>,
}

impl<S: SurfaceSource> H264Decoder<S> {
    /// A decoder for samples described by an `avcC` record (ISO/IEC
    /// 14496-15, 5.3.3.1), whose parameter sets it takes. The record's
    /// first SPS is checked against what the decoder and the device take,
    /// so that a stream they cannot decode is refused here.
    pub fn new(device: &Arc<VideoDevice>, source: S, avcc: &[u8]) -> Result<Self, Error> {
        let accelerator = VulkanH264::new(device, source)?;
        let config = DecoderConfiguration::parse(avcc)?;
        if let Some(&sps) = config.sequence_parameter_sets.first() {
            let sps = Sps::parse(sps)?;
            h264::check_supported(&sps)?;
            select(&accelerator.capabilities, &sps)?;
        }
        Ok(Self {
            inner: Decoder::from_avcc(accelerator, avcc)?,
        })
    }

    /// A decoder for Annex B samples, which carry their parameter sets.
    pub fn new_annex_b(device: &Arc<VideoDevice>, source: S) -> Result<Self, Error> {
        Ok(Self {
            inner: Decoder::new(VulkanH264::new(device, source)?, NalFormat::AnnexB),
        })
    }

    /// Chooses decode order (the default) or display order.
    pub fn set_output_order(&mut self, order: OutputOrder) {
        self.inner.set_output_order(order);
    }

    /// Decodes one access unit, tagged `id`, and returns the frames that
    /// come out ([`reims_vgpu_video_codec::h264::Decoder::decode`]). Their
    /// decodes may still be running: each frame's picture says the value
    /// its decode signals.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame<S::Surface>>, Error> {
        self.inner.decode(sample, id)
    }

    /// [`Self::decode`] of a sample whose bytes lie in `memory` too, which
    /// each picture's decode reads in place, with no copy, where the
    /// device takes them there: the first slice's start code at a multiple
    /// of `minBitstreamBufferOffsetAlignment` in the memory, and the
    /// sample holding the slices padded to a multiple of
    /// `minBitstreamBufferSizeAlignment` from there, the padding zeros as
    /// the caller wrote them. Elsewhere the slices are copied as `decode`
    /// copies them.
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
    ) -> Result<Vec<Frame<S::Surface>>, Error> {
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
    pub fn flush(&mut self) -> Vec<Frame<S::Surface>> {
        self.inner.flush()
    }

    /// Forgets every reference, as for a seek.
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
    pub fn profile(&self) -> Option<H264Profile> {
        match self.inner.accelerator().session.as_ref()?.caps.profile {
            VideoProfile::H264(profile) => Some(profile),
            VideoProfile::H265(_) => None,
        }
    }

    /// How the decodes so far read their slices: in place, or copied.
    pub fn bitstreams(&self) -> BitstreamCounts {
        self.inner.accelerator().bitstreams
    }
}

/// The Vulkan half of the decoder.
struct VulkanH264<S: SurfaceSource> {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    capabilities: Vec<H264Capabilities>,
    source: S,
    /// The stream's session, which holds the timeline itself, and waits
    /// for its decodes as it goes.
    session: Option<Session<ParameterSets>>,
    /// Where the sample being decoded lies in device memory, while
    /// [`H264Decoder::decode_in_place`] decodes it.
    in_memory: Option<InMemory>,
    /// How the decodes so far read their slices.
    bitstreams: BitstreamCounts,
}

impl<S: SurfaceSource> VulkanH264<S> {
    fn new(device: &Arc<VideoDevice>, source: S) -> Result<Self, Error> {
        let capabilities = h264_capabilities(device);
        if capabilities.is_empty() {
            return Err(Error::NoVideoDecode(format!(
                "{} decodes no H.264 profile into NV12 images it can copy or sample",
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

    fn session(&self) -> Result<&Session<ParameterSets>, Error> {
        self.session
            .as_ref()
            .ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))
    }
}

/// The capabilities of the first profile of `sps`'s candidates the device
/// decodes, where it decodes `sps`'s coded size.
fn select(capabilities: &[H264Capabilities], sps: &Sps) -> Result<H264Capabilities, Error> {
    for profile in H264Profile::candidates(sps.profile_idc) {
        let Some(caps) = capabilities.iter().find(|c| c.profile == *profile) else {
            continue;
        };
        caps.check_extent("H.264", sps.coded_width(), sps.coded_height())?;
        return Ok(*caps);
    }
    Err(Error::Unsupported(format!(
        "H.264 profile_idc {}: the device decodes profiles {:?}",
        sps.profile_idc,
        capabilities
            .iter()
            .map(|c| c.profile.profile_idc)
            .collect::<Vec<_>>()
    )))
}

impl<S: SurfaceSource> Accelerator for VulkanH264<S> {
    type Picture = Picture<S::Surface>;
    type Error = Error;

    /// A session whose DPB holds `sps`'s `max_num_ref_frames` references
    /// and the picture being decoded: the codec crate's `dpb_frames` also
    /// counts the frames held for display order, which are no DPB slot's.
    fn configure(&mut self, sps: &Sps, _dpb_frames: usize) -> Result<(), Error> {
        // The old session goes first: it waits for its decodes.
        self.session = None;
        let caps = select(&self.capabilities, sps)?;
        let extent = vk::Extent2D {
            width: sps.coded_width(),
            height: sps.coded_height(),
        };
        self.session = Some(Session::new(
            &self.device,
            &self.timeline,
            caps.video(),
            extent,
            u32::from(sps.max_num_ref_frames),
        )?);
        Ok(())
    }

    fn new_picture(&mut self) -> Result<Self::Picture, Error> {
        let (spec, extent) = {
            let session = self.session()?;
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

    /// A stand-in for a lost frame: `from`'s own output and DPB slot,
    /// under a picture of its own ([`Picture`]), since only a decode
    /// sets a slot up (ours: Apple's AVC decoder conceals some way of its
    /// own). The DPB names it in place of its source where its source is
    /// no reference ([`kept_references`]).
    fn conceal(&mut self, from: &Self::Picture) -> Result<Option<Self::Picture>, Error> {
        Ok(Some(from.stand_in()))
    }

    fn decode_picture(
        &mut self,
        picture: &Self::Picture,
        params: &PictureParams<'_, Self::Picture>,
        slices: &[SliceParams<'_, Self::Picture>],
    ) -> Result<(), Error> {
        self.timeline.collect();
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))?;
        let (sps, pps) = (params.sps, params.pps);
        let parameters = session.parameters(
            |sets| sets.holds(sps, pps),
            |device, handle, old| ParameterSets::new(device, handle, old, sps, pps),
        )?;

        let mut slotted = Vec::with_capacity(params.references.len());
        for reference in &params.references {
            let slot = reference
                .picture
                .slot()
                .ok_or_else(|| Error::Unsupported("a reference that was never decoded".into()))?;
            slotted.push((slot, reference.picture.is_stand_in()));
        }
        let kept = kept_references(&slotted);
        let mut references: Vec<(i32, StdVideoDecodeH264ReferenceInfo)> = kept
            .iter()
            .filter_map(|&i| {
                let reference = params.references.get(i)?;
                let &(slot, _) = slotted.get(i)?;
                Some((slot, reference_info(reference)))
            })
            .collect();
        let used: Vec<i32> = references.iter().map(|&(slot, _)| slot).collect();
        let setup_slot = session
            .free_slot(&used)
            .ok_or_else(|| Error::Unsupported("no DPB slot is free".into()))?;

        let mut picture_info = picture_info(params, slices);
        let mut setup_info = setup_info(params);
        let frame_num = renumber(
            picture_info.frame_num,
            params.sps.log2_max_frame_num_minus4 + 4,
            &mut references,
        );
        picture_info.frame_num = frame_num;
        if setup_info.flags.used_for_long_term_reference() == 0 {
            setup_info.FrameNum = frame_num;
        }

        let nals: Vec<&[u8]> = slices.iter().map(|s| s.nal).collect();
        let (reference_slots, reference_infos): (Vec<i32>, Vec<_>) = references.into_iter().unzip();
        let decode = Decode {
            slices: &nals,
            in_memory: self.in_memory.as_ref(),
            parameters,
            output: match picture.surface().landing() {
                Landing::Image(image) => Target::Image(image, picture.output_view()),
                Landing::Buffer(planes) => Target::Buffer(planes, params.sps.crop_rect()),
            },
            setup_slot,
            reference_slots: &reference_slots,
            codec: CodecInfo::H264 {
                picture: &picture_info,
                setup: &setup_info,
                references: &reference_infos,
            },
        };
        let submitted = session.decode(&decode)?;
        self.bitstreams.count(submitted.in_place);
        picture.decoded(setup_slot, submitted.landed);
        Ok(())
    }
}

/// A picture of `spec`, decoded at `extent`, landing in an output of
/// `source`'s, with the decode view a decode that writes an output image
/// takes.
pub(crate) fn new_picture<S: SurfaceSource>(
    device: &Arc<VideoDevice>,
    timeline: &Arc<Timeline>,
    source: &mut S,
    spec: &SurfaceSpec,
    extent: vk::Extent2D,
) -> Result<Picture<S::Surface>, Error> {
    let surface = source.surface(spec)?;
    let view = match surface.landing() {
        Landing::Image(image) if spec.decode_target => Some(ImageView::for_decode(
            device,
            image,
            spec.format(),
            vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
        )?),
        _ => None,
    };
    Ok(Picture::new(timeline, surface, view, extent, spec.format()))
}

/// Which references a decode names, by their index in `references`
/// (each its slot, and whether it stands in for a lost frame), in their
/// order: one per slot, since a slot is one reference's. A stand-in
/// shares its source's slot, and is named only where no picture decoded
/// into that slot is a reference, as after a lost frame the sliding
/// window leaves the stand-in alone; where both are, the source stands for
/// both, its samples being the stand-in's.
fn kept_references(references: &[(i32, bool)]) -> Vec<usize> {
    (0..references.len())
        .filter(|&i| {
            let Some(&(slot, stand_in)) = references.get(i) else {
                return false;
            };
            let same_slot = |&(s, _): &(i32, bool)| s == slot;
            let first_decoded = references.iter().position(|r| same_slot(r) && !r.1);
            match first_decoded {
                Some(first) => first == i,
                None => stand_in && references.iter().position(same_slot) == Some(i),
            }
        })
        .collect()
}

/// The frame numbers a decode hands the driver: the picture's `frame_num`,
/// returned, and each short-term reference's `FrameNum`, in place. Where a
/// reference's `FrameNum` exceeds `frame_num` (it was decoded before
/// `frame_num` wrapped), all of them are shifted alike so that the picture
/// is `MaxFrameNum - 1` and each reference its `FrameNumWrap` (8.2.4.1)
/// shifted the same: no reference wraps then, and every `PicNum` and
/// `CurrPicNum` moves by the same amount, so that the initial reference
/// lists, which order short-term frames by `PicNum`, and each
/// modification, which counts from `CurrPicNum`, name the same frames.
/// Elsewhere the numbers are the stream's.
///
/// NVIDIA's Vulkan driver (615.71.09 on the RTX 5080, 11-02) builds a P
/// slice's initial list from `FrameNum` without wrapping it: from the
/// first P picture after a wrap whose list holds frames from both sides of
/// it, every such picture predicts from the wrong frames, ffmpeg's Vulkan
/// decoder's too, while NVDEC through CUDA decodes them right. Its list
/// modifications do wrap. Shifted, the H.264 check's streams decode right.
/// Long-term references keep their `LongTermFrameIdx`.
fn renumber(
    frame_num: u16,
    log2_max_frame_num: u8,
    references: &mut [(i32, StdVideoDecodeH264ReferenceInfo)],
) -> u16 {
    let short_term =
        |info: &StdVideoDecodeH264ReferenceInfo| info.flags.used_for_long_term_reference() == 0;
    if !references
        .iter()
        .any(|(_, info)| short_term(info) && info.FrameNum > frame_num)
    {
        return frame_num;
    }
    // MaxFrameNum is 2^16 at most (log2_max_frame_num is 16 at most).
    let max = 1u32 << log2_max_frame_num.min(16);
    let last = max - 1;
    let shift = last - u32::from(frame_num).min(last);
    for (_, info) in references.iter_mut().filter(|(_, info)| short_term(info)) {
        let number = u32::from(info.FrameNum);
        // FrameNumWrap + shift: at least 0, as FrameNumWrap is above
        // frame_num - MaxFrameNum, and at most MaxFrameNum - 1.
        let shifted = if number > u32::from(frame_num) {
            (number + shift).saturating_sub(max)
        } else {
            number + shift
        };
        info.FrameNum = u16::try_from(shifted).unwrap_or(u16::MAX);
    }
    u16::try_from(last).unwrap_or(u16::MAX)
}

/// A reference's std information: a frame, short-term with its
/// `FrameNum` or long-term with its `LongTermFrameIdx`, and its order
/// counts.
fn reference_info(reference: &ReferenceFrame<'_, impl Sized>) -> StdVideoDecodeH264ReferenceInfo {
    let mut flags: StdVideoDecodeH264ReferenceInfoFlags = zeroed();
    flags.set_used_for_long_term_reference(u32::from(reference.long_term));
    let mut info: StdVideoDecodeH264ReferenceInfo = zeroed();
    info.flags = flags;
    // 16 bits hold both: MaxFrameNum is 2^16 at most, and a long-term
    // index below 16.
    info.FrameNum = u16::try_from(reference.frame_idx).unwrap_or(u16::MAX);
    info.PicOrderCnt = [reference.top_poc, reference.bottom_poc];
    info
}

/// The picture's own information as a reference, for the slot its decode
/// sets up: long-term where its marking makes it so (an IDR picture's
/// `long_term_reference_flag`, or operation 6).
fn setup_info(params: &PictureParams<'_, impl Sized>) -> StdVideoDecodeH264ReferenceInfo {
    let header = params.header;
    let marking = &header.ref_pic_marking;
    let long_term = if header.idr && marking.long_term_reference_flag {
        Some(0)
    } else {
        marking.adaptive.iter().flatten().find_map(|op| match op {
            MemoryManagement::CurrentToLongTerm(idx) => Some(*idx),
            _ => None,
        })
    };
    let mut flags: StdVideoDecodeH264ReferenceInfoFlags = zeroed();
    flags.set_used_for_long_term_reference(u32::from(long_term.is_some()));
    let mut info: StdVideoDecodeH264ReferenceInfo = zeroed();
    info.flags = flags;
    info.FrameNum = u16::try_from(long_term.unwrap_or(header.frame_num)).unwrap_or(u16::MAX);
    info.PicOrderCnt = [params.top_poc, params.bottom_poc];
    info
}

/// The picture's std information: a frame, its parameter sets' ids,
/// `frame_num`, `idr_pic_id` and order counts.
fn picture_info(
    params: &PictureParams<'_, impl Sized>,
    slices: &[SliceParams<'_, impl Sized>],
) -> StdVideoDecodeH264PictureInfo {
    let header = params.header;
    let mut flags: StdVideoDecodeH264PictureInfoFlags = zeroed();
    flags.set_field_pic_flag(0);
    flags.set_is_intra(u32::from(
        slices.iter().all(|s| s.header.slice_type == SliceType::I),
    ));
    flags.set_IdrPicFlag(u32::from(header.idr));
    flags.set_bottom_field_flag(0);
    flags.set_is_reference(u32::from(header.nal_ref_idc != 0));
    flags.set_complementary_field_pair(0);
    let mut info: StdVideoDecodeH264PictureInfo = zeroed();
    info.flags = flags;
    info.seq_parameter_set_id = params.sps.seq_parameter_set_id;
    info.pic_parameter_set_id = params.pps.pic_parameter_set_id;
    // frame_num is below MaxFrameNum, 2^16 at most; idr_pic_id is 65535 at
    // most (7.4.3).
    info.frame_num = u16::try_from(header.frame_num).unwrap_or(u16::MAX);
    info.idr_pic_id = u16::try_from(header.idr_pic_id).unwrap_or(u16::MAX);
    info.PicOrderCnt = [params.top_poc, params.bottom_poc];
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    fn short_term(frame_num: u16) -> (i32, StdVideoDecodeH264ReferenceInfo) {
        let mut info: StdVideoDecodeH264ReferenceInfo = zeroed();
        info.FrameNum = frame_num;
        (0, info)
    }

    fn long_term(index: u16) -> (i32, StdVideoDecodeH264ReferenceInfo) {
        let (slot, mut info) = short_term(index);
        info.flags.set_used_for_long_term_reference(1);
        (slot, info)
    }

    fn numbers(references: &[(i32, StdVideoDecodeH264ReferenceInfo)]) -> Vec<u16> {
        references.iter().map(|(_, info)| info.FrameNum).collect()
    }

    #[test]
    fn frame_numbers_are_the_streams_where_none_wraps() {
        let mut references = [short_term(4), short_term(3), long_term(0)];
        assert_eq!(renumber(5, 4, &mut references), 5);
        assert_eq!(numbers(&references), [4, 3, 0]);
    }

    #[test]
    fn frame_numbers_shift_past_a_wrap_keeping_their_order() {
        // MaxFrameNum 16, frame_num 1: FrameNumWrap 0, -1 and -2, shifted
        // by 14 so that the picture is 15. The long-term index stays.
        let mut references = [short_term(15), short_term(0), short_term(14), long_term(1)];
        assert_eq!(renumber(1, 4, &mut references), 15);
        assert_eq!(numbers(&references), [13, 14, 12, 1]);
        // A frame_num of 0, every reference before the wrap.
        let mut references = [short_term(15), short_term(13)];
        assert_eq!(renumber(0, 4, &mut references), 15);
        assert_eq!(numbers(&references), [14, 12]);
        // MaxFrameNum 2^16.
        let mut references = [short_term(65535), short_term(2)];
        assert_eq!(renumber(3, 16, &mut references), 65535);
        assert_eq!(numbers(&references), [65531, 65534]);
    }

    #[test]
    fn a_stand_in_is_named_only_where_its_source_is_no_reference() {
        // Its source gone: the stand-in alone holds the slot.
        assert_eq!(kept_references(&[(4, true)]), [0]);
        // Its source a reference too: the source stands for both.
        assert_eq!(
            kept_references(&[(4, false), (2, false), (4, true)]),
            [0, 1]
        );
        assert_eq!(kept_references(&[(4, true), (4, false)]), [1]);
        // Two stand-ins of one source: the first.
        assert_eq!(kept_references(&[(4, true), (4, true), (1, false)]), [0, 2]);
    }
}
