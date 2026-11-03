//! One stream's encode session, of either codec: the `VkVideoSessionKHR`
//! and its memory, its DPB pictures, its parameters object, its rate
//! control state, and the command buffers its encodes are submitted with,
//! each reused once the timeline has passed the work that used it. What is
//! the codec's goes in with each encode ([`CodecInfo`]).
//!
//! **A picture's work**, one command buffer on the encode queue:
//! - its feedback query, and its timestamps where timed, reset;
//! - a source in a buffer copied into a picture of the session's, the last
//!   row and column repeated into the padding the device reads; a source
//!   image read where it is;
//! - in-band parameter sets written in front of where the encode writes,
//!   so that they end where it starts;
//! - the encode, in a coding scope that resets the session first and sets
//!   its rate control and quality level, and sets the rate control again
//!   where it changed;
//! - its writes made available to the host.
//!
//! It waits for the caller's semaphores, and signals the next value of the
//! encoder's timeline. Encodes on one queue run in order, so each picture's
//! references, written by encodes before it, are there.
//!
//! **Layouts.** With unified layouts every image stays in `GENERAL`. Else a
//! source image is taken from `GENERAL` to `VIDEO_ENCODE_SRC` and back, and
//! DPB pictures are kept in `VIDEO_ENCODE_DPB`.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoEncodeH264PictureInfo, StdVideoEncodeH264ReferenceInfo, StdVideoEncodeH264SliceHeader,
    StdVideoEncodeH265PictureInfo, StdVideoEncodeH265ReferenceInfo,
    StdVideoEncodeH265SliceSegmentHeader, StdVideoH264LevelIdc, StdVideoH265LevelIdc,
};

use crate::device::VideoDevice;
use crate::encode::Settings;
use crate::encode::caps::{EncodeCapabilities, EncodeCodec};
use crate::encode::frame::QuerySlot;
use crate::encode::io::{BitstreamBuffer, SourcePicture};
use crate::encode::rate::RatePlan;
use crate::error::{Error, Named};
use crate::params::Parameters;
use crate::picture::{ImageView, color_range, plane};
use crate::session::{allocate_command, begin, bind_session_memory, command_pool, granular};
use crate::surface::{BufferPlanes, bind_image_memory};
use crate::timeline::Timeline;

/// What an encode names of its codec: the std structures Vulkan Video's
/// codec extension chains to the encode, to each slice and to its slots,
/// the references' in the order of [`EncodeOp::reference_slots`].
#[derive(Clone, Copy)]
pub(crate) enum CodecInfo<'a> {
    H264 {
        picture: &'a StdVideoEncodeH264PictureInfo,
        slices: &'a [StdVideoEncodeH264SliceHeader],
        setup: &'a StdVideoEncodeH264ReferenceInfo,
        references: &'a [StdVideoEncodeH264ReferenceInfo],
    },
    H265 {
        picture: &'a StdVideoEncodeH265PictureInfo,
        slices: &'a [StdVideoEncodeH265SliceSegmentHeader],
        setup: &'a StdVideoEncodeH265ReferenceInfo,
        references: &'a [StdVideoEncodeH265ReferenceInfo],
    },
}

impl CodecInfo<'_> {
    fn reference_count(&self) -> usize {
        match self {
            Self::H264 { references, .. } => references.len(),
            Self::H265 { references, .. } => references.len(),
        }
    }
}

/// The codec's part of the rate control state: the GOP it is told of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CodecRate {
    /// `VkVideoEncodeH264RateControlFlagsKHR` or H.265's, raw.
    pub(crate) flags: u32,
    /// Frames from one I picture to the next; 0 for no limit.
    pub(crate) gop_frame_count: u32,
    /// Frames from one IDR picture to the next; 0 for no limit.
    pub(crate) idr_period: u32,
    /// Consecutive B pictures.
    pub(crate) b_frames: u32,
}

/// The codec's part of the session's create info.
#[derive(Clone, Copy)]
pub(crate) enum CodecSession {
    H264 { max_level: StdVideoH264LevelIdc },
    H265 { max_level: StdVideoH265LevelIdc },
}

/// The parameter sets `get` names, Annex B, as the device writes them
/// (`vkGetEncodedVideoSessionParametersKHR`), with whether it overrode
/// any in `feedback`.
pub(crate) fn encoded_parameters(
    device: &VideoDevice,
    get: &vk::VideoEncodeSessionParametersGetInfoKHR<'_>,
    feedback: &mut vk::VideoEncodeSessionParametersFeedbackInfoKHR<'_>,
) -> Result<Vec<u8>, Error> {
    let fp = device.encoding()?.encode.fp();
    let mut size = 0usize;
    // SAFETY: the parameters object `get` names is the device's; `get`,
    // `feedback` and their chains live through the call; a null output
    // asks for the size.
    unsafe {
        (fp.get_encoded_video_session_parameters_khr)(
            device.device().handle(),
            get,
            feedback,
            &mut size,
            std::ptr::null_mut(),
        )
    }
    .named("vkGetEncodedVideoSessionParametersKHR")?;
    let mut data = vec![0u8; size];
    // SAFETY: as above, with an output of `size` bytes.
    unsafe {
        (fp.get_encoded_video_session_parameters_khr)(
            device.device().handle(),
            get,
            feedback,
            &mut size,
            data.as_mut_ptr().cast(),
        )
    }
    .named("vkGetEncodedVideoSessionParametersKHR")?;
    data.truncate(size);
    Ok(data)
}

/// One picture's encode, as [`EncodeSession::encode`] records it.
pub(crate) struct EncodeOp<'a> {
    /// The frame.
    pub(crate) source: SourcePicture,
    /// Where its bitstream goes.
    pub(crate) target: BitstreamBuffer,
    /// In-band parameter sets to write before it, Annex B.
    pub(crate) header: Option<&'a [u8]>,
    /// The DPB slot the picture is set up in.
    pub(crate) setup_slot: i32,
    /// Each reference's slot, in the order of the codec's references.
    pub(crate) reference_slots: &'a [i32],
    /// The codec's std information.
    pub(crate) codec: CodecInfo<'a>,
    /// The QP of its slices: for a constant QP, else 0.
    pub(crate) qp: i32,
    /// The caller's semaphores and values the encode waits for.
    pub(crate) waits: &'a [(vk::Semaphore, u64)],
    /// Its feedback query and timestamps.
    pub(crate) query: &'a QuerySlot,
}

/// What [`EncodeSession::encode`] submitted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Submitted {
    /// The value after which the target holds the picture.
    pub(crate) value: u64,
    /// Where the encode writes from, in the target's buffer.
    pub(crate) encode_offset: u64,
    /// Where the in-band parameter sets went, offset and bytes.
    pub(crate) header: Option<(u64, u64)>,
    /// Whether the source was copied into a picture of the session's.
    pub(crate) copied: bool,
}

/// An image of the session's: a DPB picture, or a picture a source in a
/// buffer is copied into. Destroyed as it goes.
struct SessionImage {
    device: Arc<VideoDevice>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: Option<ImageView>,
}

impl Drop for SessionImage {
    fn drop(&mut self) {
        drop(self.view.take());
        // SAFETY: the image is the device's, and its session's work is
        // done; a null handle is nothing.
        unsafe { self.device.device().destroy_image(self.image, None) };
        // SAFETY: its memory, now bound to nothing.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}

impl SessionImage {
    /// An image of `extent` of `caps`'s profile's format, of `usage`,
    /// exclusive to the encode family, with a view of the video usage
    /// `view_usage`.
    fn new(
        device: &Arc<VideoDevice>,
        caps: &EncodeCapabilities,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        view_usage: vk::ImageUsageFlags,
    ) -> Result<Self, Error> {
        let format = caps.profile.format();
        let image = caps.profile.with_list(|list| {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(list);
            // SAFETY: `info` and its chain live through the call.
            unsafe { device.device().create_image(&info, None) }.named("vkCreateImage")
        })?;
        let mut made = Self {
            device: Arc::clone(device),
            image,
            memory: vk::DeviceMemory::null(),
            view: None,
        };
        made.memory = bind_image_memory(device, image)?;
        made.view = Some(ImageView::for_decode(device, image, format, view_usage)?);
        Ok(made)
    }

    fn view(&self) -> vk::ImageView {
        self.view
            .as_ref()
            .map_or(vk::ImageView::null(), ImageView::handle)
    }
}

/// A picture sources in buffers are copied into, free once the timeline
/// passes `free_after`.
struct Staging {
    image: SessionImage,
    free_after: u64,
}

/// The command buffer one encode at a time uses.
struct Submission {
    command: vk::CommandBuffer,
    /// The value the last work submitted with it signals.
    value: u64,
}

/// One stream's encode session.
pub(crate) struct EncodeSession {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    /// What the device answers for the session's profile.
    pub(crate) caps: EncodeCapabilities,
    /// The coded extent each picture is encoded at.
    pub(crate) extent: vk::Extent2D,
    /// The frame's own size, which a source in a buffer has.
    frame_size: (u32, u32),
    /// The extent of the session's pictures, and the least of a source
    /// image's: the coded extent, rounded up to the picture access and
    /// input granularities.
    pub(crate) image_extent: vk::Extent2D,
    handle: vk::VideoSessionKHR,
    memory: Vec<vk::DeviceMemory>,
    /// One a slot, by its index.
    dpb: Vec<SessionImage>,
    parameters: Option<Parameters>,
    quality_level: u32,
    /// The rate control state wanted, and the codec's part of it.
    rate: (RatePlan, CodecRate),
    /// The state the session has, once an encode set it.
    state: Option<(RatePlan, CodecRate)>,
    pool: vk::CommandPool,
    busy: VecDeque<Submission>,
    free: Vec<Submission>,
    staging: Vec<Staging>,
}

/// A rate control state as Vulkan takes it, its parts linked by
/// [`RateState::info`].
struct RateState {
    plan: RatePlan,
    layers: [vk::VideoEncodeRateControlLayerInfoKHR<'static>; 1],
    h264_layer: vk::VideoEncodeH264RateControlLayerInfoKHR<'static>,
    h265_layer: vk::VideoEncodeH265RateControlLayerInfoKHR<'static>,
    h264: vk::VideoEncodeH264RateControlInfoKHR<'static>,
    h265: vk::VideoEncodeH265RateControlInfoKHR<'static>,
}

impl RateState {
    fn new((plan, codec): &(RatePlan, CodecRate)) -> Self {
        Self {
            plan: *plan,
            layers: [vk::VideoEncodeRateControlLayerInfoKHR::default()
                .average_bitrate(plan.average)
                .max_bitrate(plan.peak)
                .frame_rate_numerator(plan.frame_rate.numerator)
                .frame_rate_denominator(plan.frame_rate.denominator)],
            h264_layer: vk::VideoEncodeH264RateControlLayerInfoKHR::default(),
            h265_layer: vk::VideoEncodeH265RateControlLayerInfoKHR::default(),
            h264: vk::VideoEncodeH264RateControlInfoKHR::default()
                .flags(vk::VideoEncodeH264RateControlFlagsKHR::from_raw(
                    codec.flags,
                ))
                .gop_frame_count(codec.gop_frame_count)
                .idr_period(codec.idr_period)
                .consecutive_b_frame_count(codec.b_frames)
                .temporal_layer_count(plan.layers()),
            h265: vk::VideoEncodeH265RateControlInfoKHR::default()
                .flags(vk::VideoEncodeH265RateControlFlagsKHR::from_raw(
                    codec.flags,
                ))
                .gop_frame_count(codec.gop_frame_count)
                .idr_period(codec.idr_period)
                .consecutive_b_frame_count(codec.b_frames)
                .sub_layer_count(plan.layers()),
        }
    }

    /// `VkVideoEncodeRateControlInfoKHR` of the state, its layer and the
    /// codec's parts linked, valid while this is.
    fn info(&mut self, codec: EncodeCodec) -> vk::VideoEncodeRateControlInfoKHR<'_> {
        let (layer, rate): (*const c_void, *const c_void) = match codec {
            EncodeCodec::H264 => (
                std::ptr::from_ref(&self.h264_layer).cast(),
                std::ptr::from_ref(&self.h264).cast(),
            ),
            EncodeCodec::H265 => (
                std::ptr::from_ref(&self.h265_layer).cast(),
                std::ptr::from_ref(&self.h265).cast(),
            ),
        };
        for each in &mut self.layers {
            each.p_next = layer;
        }
        let count = usize::try_from(self.plan.layers()).unwrap_or(0);
        let mut info = vk::VideoEncodeRateControlInfoKHR::default()
            .rate_control_mode(self.plan.mode)
            .virtual_buffer_size_in_ms(self.plan.buffer_ms)
            .initial_virtual_buffer_size_in_ms(self.plan.initial_ms)
            .layers(self.layers.get(..count).unwrap_or(&[]));
        info.p_next = rate;
        info
    }
}

impl EncodeSession {
    /// A session for `settings`' pictures (its coded size, its profile),
    /// with as many DPB slots and active references as its limits keep,
    /// its rate control and its quality level, of `codec`.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        timeline: &Arc<Timeline>,
        settings: &Settings,
        codec: CodecSession,
    ) -> Result<Self, Error> {
        let caps = settings.caps.clone();
        let extent = settings.coded;
        let frame_size = (settings.config.width, settings.config.height);
        let (slots, references) = (settings.limits.slots(), settings.limits.max_references);
        let rate = (settings.rate, settings.codec_rate());
        let quality_level = settings.quality_level;
        let family = device.encoding()?.family;
        if slots > caps.max_dpb_slots || references > caps.max_active_reference_pictures {
            return Err(Error::Unsupported(format!(
                "{references} references in {slots} slots: the device has {} DPB slots and {} \
                 active references",
                caps.max_dpb_slots, caps.max_active_reference_pictures
            )));
        }
        if caps.requires_gop_remaining_frames {
            return Err(Error::Unsupported(
                "a device that needs each encode's GOP remaining frames".into(),
            ));
        }
        let image_extent = granular(
            granular(extent, caps.picture_access_granularity),
            caps.input_granularity,
        );
        let mut session = Self {
            device: Arc::clone(device),
            timeline: Arc::clone(timeline),
            caps,
            extent,
            frame_size,
            image_extent,
            handle: vk::VideoSessionKHR::null(),
            memory: Vec::new(),
            dpb: Vec::new(),
            parameters: None,
            quality_level,
            rate,
            state: None,
            pool: vk::CommandPool::null(),
            busy: VecDeque::new(),
            free: Vec::new(),
            staging: Vec::new(),
        };
        let fp = device.video().fp();
        let vk_device = device.device().handle();
        let caps = &session.caps;
        let format = caps.profile.format();
        let mut h264 = vk::VideoEncodeH264SessionCreateInfoKHR::default();
        let mut h265 = vk::VideoEncodeH265SessionCreateInfoKHR::default();
        let handle = caps.profile.with_info(|profile| {
            let info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(family)
                .video_profile(profile)
                .picture_format(format)
                .max_coded_extent(extent)
                .reference_picture_format(format)
                .max_dpb_slots(slots)
                .max_active_reference_pictures(references)
                .std_header_version(&caps.std_header_version);
            let info = match codec {
                CodecSession::H264 { max_level } => {
                    h264 = h264.use_max_level_idc(true).max_level_idc(max_level);
                    info.push_next(&mut h264)
                }
                CodecSession::H265 { max_level } => {
                    h265 = h265.use_max_level_idc(true).max_level_idc(max_level);
                    info.push_next(&mut h265)
                }
            };
            let mut handle = vk::VideoSessionKHR::null();
            // SAFETY: `info` and what it points at live through the call.
            unsafe {
                (fp.create_video_session_khr)(vk_device, &info, std::ptr::null(), &mut handle)
            }
            .named("vkCreateVideoSessionKHR")?;
            Ok::<_, Error>(handle)
        })?;
        session.handle = handle;
        bind_session_memory(device, session.handle, &mut session.memory)?;
        let dpb = vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR;
        for _ in 0..slots {
            let image = SessionImage::new(device, &session.caps, image_extent, dpb, dpb)?;
            session.dpb.push(image);
        }
        session.pool = command_pool(device, family)?;
        Ok(session)
    }

    /// The session's handle, for its parameters object.
    pub(crate) fn handle(&self) -> vk::VideoSessionKHR {
        self.handle
    }

    /// The quality level the session is set to, which its parameters
    /// object is made with.
    pub(crate) fn quality_level(&self) -> u32 {
        self.quality_level
    }

    /// Takes `parameters` as the session's parameters object, the old one
    /// retired once the encodes submitted with it are done.
    pub(crate) fn set_parameters(&mut self, parameters: Parameters) {
        if let Some(old) = self.parameters.replace(parameters) {
            self.timeline
                .retire(self.timeline.last_submitted(), Box::new(old));
        }
    }

    /// Asks for `rate` from the next encode on.
    pub(crate) fn set_rate(&mut self, rate: (RatePlan, CodecRate)) {
        self.rate = rate;
    }

    /// The rate control state wanted.
    pub(crate) fn rate(&self) -> &(RatePlan, CodecRate) {
        &self.rate
    }

    fn layout(&self, unified: vk::ImageLayout) -> vk::ImageLayout {
        if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            unified
        }
    }

    fn dpb_picture(&self, slot: i32) -> Result<&SessionImage, Error> {
        usize::try_from(slot)
            .ok()
            .and_then(|slot| self.dpb.get(slot))
            .ok_or_else(|| Error::Unsupported(format!("DPB slot {slot} past the session's")))
    }

    /// Records and submits `op`, and returns what it submitted.
    pub(crate) fn encode(&mut self, op: &EncodeOp<'_>) -> Result<Submitted, Error> {
        if op.reference_slots.len() != op.codec.reference_count() {
            return Err(Error::Unsupported(
                "references with no std information of their own".into(),
            ));
        }
        let parameters = self
            .parameters
            .as_ref()
            .map(Parameters::handle)
            .ok_or_else(|| Error::Unsupported("an encode before the parameter sets".into()))?;
        let mut submission = self.submission()?;
        let recorded = self.record_and_submit(&mut submission, op, parameters);
        submission.value = match &recorded {
            Ok((submitted, _)) => submitted.value,
            Err(_) => self.timeline.last_submitted(),
        };
        self.busy.push_back(submission);
        let (submitted, retire) = recorded?;
        for item in retire {
            self.timeline.retire(submitted.value, item);
        }
        Ok(submitted)
    }

    /// Where the encode writes in `target`, and where in-band parameter
    /// sets of `header` bytes go before it: the encode from the least
    /// offset the device takes with room for them before it, the header
    /// ending there, padded at its front with zeros to whole words (zeros
    /// between NAL units are trailing zeros in Annex B, 7.4.1.2.4), as
    /// `vkCmdUpdateBuffer` writes. Returns the encode's offset and range,
    /// and the header's offset with its padding.
    pub(crate) fn placement(
        target: &BitstreamBuffer,
        header: usize,
        offset_alignment: u64,
        size_alignment: u64,
    ) -> Result<(u64, u64, Option<u64>), Error> {
        let padded = u64::try_from(header.next_multiple_of(4)).unwrap_or(u64::MAX);
        let alignment = offset_alignment.max(1).next_multiple_of(4);
        let from = if header == 0 {
            target.offset.next_multiple_of(offset_alignment.max(1))
        } else {
            target
                .offset
                .saturating_add(padded)
                .next_multiple_of(alignment)
        };
        let end = target.offset.saturating_add(target.size);
        let range = end
            .saturating_sub(from)
            .checked_div(size_alignment.max(1))
            .unwrap_or(0)
            * size_alignment.max(1);
        if range == 0 {
            return Err(Error::Unsupported(format!(
                "a bitstream of {} bytes from {}: none left at the device's alignment",
                target.size, target.offset
            )));
        }
        let header_at = (header > 0).then(|| from - padded);
        Ok((from, range, header_at))
    }

    fn record_and_submit(
        &mut self,
        submission: &mut Submission,
        op: &EncodeOp<'_>,
        parameters: vk::VideoSessionParametersKHR,
    ) -> Result<(Submitted, Vec<Box<dyn Send>>), Error> {
        // Its own handle on the device, so that the session is free to
        // change below.
        let video = Arc::clone(&self.device);
        let copies = video.encoding()?.copies;
        let device = video.device();
        let command = submission.command;
        let header_len = op.header.map_or(0, <[u8]>::len);
        if header_len > 0 && !copies {
            return Err(Error::Unsupported(
                "in-band parameter sets on an encode family that copies nothing".into(),
            ));
        }
        let (encode_offset, range, header_at) = Self::placement(
            &op.target,
            header_len,
            self.caps.min_bitstream_buffer_offset_alignment,
            self.caps.min_bitstream_buffer_size_alignment,
        )?;
        let mut retire: Vec<Box<dyn Send>> = Vec::new();
        let mut staging_at = None;
        begin(device, command)?;
        let query = op.query;
        // SAFETY: recording into the session's command buffer, whose last
        // work is done, outside a coding scope; so for each command below,
        // each with what it reads alive through the call.
        unsafe { device.cmd_reset_query_pool(command, query.feedback, query.index(), 1) };
        let timed = query.stamps != vk::QueryPool::null();
        if timed {
            // SAFETY: as above.
            unsafe {
                device.cmd_reset_query_pool(
                    command,
                    query.stamps,
                    query.stamp(0),
                    QuerySlot::stamps_count(),
                );
            }
            // SAFETY: as above.
            unsafe {
                device.cmd_write_timestamp2(
                    command,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    query.stamps,
                    query.stamp(0),
                );
            }
        }

        // The source: read in place, or copied into a picture of ours.
        let src_layout = self.layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR);
        let (source_view, copied, caller_image) = match op.source {
            SourcePicture::Image(image) => {
                let view = ImageView::for_decode(
                    &self.device,
                    image,
                    self.caps.profile.format(),
                    vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
                )?;
                let handle = view.handle();
                retire.push(Box::new(view));
                if src_layout != vk::ImageLayout::GENERAL {
                    let to_src = [vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
                        .dst_access_mask(vk::AccessFlags2::VIDEO_ENCODE_READ_KHR)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(src_layout)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image)
                        .subresource_range(color_range())];
                    // SAFETY: as above.
                    unsafe {
                        self.device.cmd_pipeline_barrier2(
                            command,
                            &vk::DependencyInfo::default().image_memory_barriers(&to_src),
                        );
                    }
                }
                (handle, false, Some(image))
            }
            SourcePicture::Buffer(planes) => {
                if !copies {
                    return Err(Error::Unsupported(
                        "a source in a buffer on an encode family that copies nothing".into(),
                    ));
                }
                let mut taken = self.staging()?;
                taken.free_after = self.timeline.last_submitted();
                let image = taken.image.image;
                let view = taken.image.view();
                // Back in the pool: free once this encode, the next
                // submission, is done, or at once where it is never
                // submitted.
                self.staging.push(taken);
                let at = self.staging.len() - 1;
                self.record_upload(command, &planes, image, src_layout)?;
                staging_at = Some(at);
                (view, true, None)
            }
        };
        if timed {
            // SAFETY: as above.
            unsafe {
                device.cmd_write_timestamp2(
                    command,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    query.stamps,
                    query.stamp(1),
                );
            }
        }

        // In-band parameter sets, ending where the encode starts.
        let header = match (op.header, header_at) {
            (Some(bytes), Some(at)) => {
                let padded = bytes.len().next_multiple_of(4);
                let mut words = vec![0u8; padded];
                words[padded - bytes.len()..].copy_from_slice(bytes);
                // SAFETY: as above; the buffer has `TRANSFER_DST` usage
                // (`BitstreamSpec`), and `at` and the size are multiples of
                // 4 within the target (`placement`), at most 65536 bytes.
                unsafe { device.cmd_update_buffer(command, op.target.buffer, at, &words) };
                let skipped = u64::try_from(padded - bytes.len()).unwrap_or(0);
                Some((at + skipped, u64::try_from(bytes.len()).unwrap_or(0)))
            }
            _ => None,
        };

        let (setup_image, setup_view) = {
            let setup = self.dpb_picture(op.setup_slot)?;
            (setup.image, setup.view())
        };
        let dpb_layout = self.layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR);
        let encode_access =
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR;
        // The DPB picture set up drops what it held; the references were
        // written by encodes before this one, on this queue.
        let fresh = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            .dst_access_mask(encode_access)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(dpb_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(setup_image)
            .subresource_range(color_range())];
        let references_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            .src_access_mask(vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            .dst_access_mask(vk::AccessFlags2::VIDEO_ENCODE_READ_KHR)];
        // SAFETY: as above.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&references_barrier)
                    .image_memory_barriers(&fresh),
            );
        }

        let resource = |view| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(self.extent)
                .base_array_layer(0)
                .image_view_binding(view)
        };
        let mut reference_resources = Vec::with_capacity(op.reference_slots.len());
        for &slot in op.reference_slots {
            reference_resources.push(resource(self.dpb_picture(slot)?.view()));
        }
        let setup_resource = resource(setup_view);
        let source_resource = resource(source_view);

        // The slots bound for the scope: every reference's, and the
        // picture's DPB picture, not yet a slot's (-1), as the encode sets
        // it up.
        let mut bound: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = op
            .reference_slots
            .iter()
            .zip(&reference_resources)
            .map(|(&slot, resource)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
            })
            .collect();
        bound.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource),
        );
        let fp = self.device.video().fp();
        let codec = self.caps.profile.codec;
        let wanted = self.rate;
        let current = self.state;
        {
            let mut held = current.as_ref().map(RateState::new);
            let state = held.as_mut().map(|state| state.info(codec));
            let mut begin_coding = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.handle)
                .video_session_parameters(parameters)
                .reference_slots(&bound);
            if let Some(state) = &state {
                begin_coding.p_next = std::ptr::from_ref(state).cast();
            }
            // SAFETY: as above, with the rate control state the session
            // has chained, alive through the call.
            unsafe { (fp.cmd_begin_video_coding_khr)(command, &begin_coding) };
        }
        if current != Some(wanted) {
            let mut state = RateState::new(&wanted);
            let rate = state.info(codec);
            let mut quality =
                vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(self.quality_level);
            quality.p_next = std::ptr::from_ref(&rate).cast();
            let mut control = vk::VideoCodingControlInfoKHR::default();
            if current.is_none() {
                control = control.flags(
                    vk::VideoCodingControlFlagsKHR::RESET
                        | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                        | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
                );
                control.p_next = std::ptr::from_ref(&quality).cast();
            } else {
                control = control.flags(vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL);
                control.p_next = std::ptr::from_ref(&rate).cast();
            }
            // SAFETY: as above, inside the coding scope, with the chain
            // alive through the call.
            unsafe { (fp.cmd_control_video_coding_khr)(command, &control) };
        }
        // SAFETY: as above: the query was reset, and is used by this
        // encode alone.
        unsafe {
            device.cmd_begin_query(
                command,
                query.feedback,
                query.index(),
                vk::QueryControlFlags::empty(),
            );
        }
        self.record_encode(
            command,
            op,
            &reference_resources,
            &setup_resource,
            &source_resource,
            (encode_offset, range),
        );
        // SAFETY: as above.
        unsafe { device.cmd_end_query(command, query.feedback, query.index()) };
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: as above, closing the scope.
        unsafe { (fp.cmd_end_video_coding_khr)(command, &end) };

        let mut images = Vec::new();
        if let Some(image) = caller_image
            && src_layout != vk::ImageLayout::GENERAL
        {
            images.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(src_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color_range()),
            );
        }
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::VIDEO_ENCODE_KHR | vk::PipelineStageFlags2::ALL_TRANSFER,
            )
            .src_access_mask(
                vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR | vk::AccessFlags2::TRANSFER_WRITE,
            )
            .dst_stage_mask(vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::HOST_READ | vk::AccessFlags2::MEMORY_READ)];
        // SAFETY: as above.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&to_host)
                    .image_memory_barriers(&images),
            );
        }
        if timed {
            // SAFETY: as above.
            unsafe {
                device.cmd_write_timestamp2(
                    command,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    query.stamps,
                    query.stamp(2),
                );
            }
        }
        // SAFETY: as above.
        unsafe { device.end_command_buffer(command) }.named("vkEndCommandBuffer")?;

        let value = self.timeline.next();
        let waits: Vec<vk::SemaphoreSubmitInfo<'_>> = op
            .waits
            .iter()
            .map(|&(semaphore, value)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submits = [vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(&signals)];
        self.device.submit_encode(&submits)?;
        self.timeline.submitted(value);
        if let Some(staging) = staging_at.and_then(|at| self.staging.get_mut(at)) {
            staging.free_after = value;
        }
        self.state = Some(wanted);
        Ok((
            Submitted {
                value,
                encode_offset,
                header,
                copied,
            },
            retire,
        ))
    }

    /// Records `vkCmdEncodeVideoKHR` of `op`, its bitstream at `dst`
    /// (offset and range in the target's buffer).
    fn record_encode(
        &self,
        command: vk::CommandBuffer,
        op: &EncodeOp<'_>,
        reference_resources: &[vk::VideoPictureResourceInfoKHR<'_>],
        setup_resource: &vk::VideoPictureResourceInfoKHR<'_>,
        source_resource: &vk::VideoPictureResourceInfoKHR<'_>,
        (offset, range): (u64, u64),
    ) {
        let encode_fp = match self.device.encoding() {
            Ok(encode) => encode.encode.fp(),
            Err(_) => return,
        };
        let info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(op.target.buffer)
            .dst_buffer_offset(offset)
            .dst_buffer_range(range)
            .src_picture_resource(*source_resource);
        match op.codec {
            CodecInfo::H264 {
                picture,
                slices,
                setup,
                references,
            } => {
                let mut setup_codec =
                    vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(setup);
                let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(op.setup_slot)
                    .picture_resource(setup_resource)
                    .push_next(&mut setup_codec);
                let mut codecs: Vec<vk::VideoEncodeH264DpbSlotInfoKHR<'_>> = references
                    .iter()
                    .map(|r| vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(r))
                    .collect();
                let slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = op
                    .reference_slots
                    .iter()
                    .zip(reference_resources)
                    .zip(codecs.iter_mut())
                    .map(|((&slot, resource), codec)| {
                        vk::VideoReferenceSlotInfoKHR::default()
                            .slot_index(slot)
                            .picture_resource(resource)
                            .push_next(codec)
                    })
                    .collect();
                let nalus: Vec<vk::VideoEncodeH264NaluSliceInfoKHR<'_>> = slices
                    .iter()
                    .map(|header| {
                        vk::VideoEncodeH264NaluSliceInfoKHR::default()
                            .constant_qp(op.qp)
                            .std_slice_header(header)
                    })
                    .collect();
                let mut h264 = vk::VideoEncodeH264PictureInfoKHR::default()
                    .nalu_slice_entries(&nalus)
                    .std_picture_info(picture)
                    .generate_prefix_nalu(false);
                let info = info
                    .setup_reference_slot(&setup_slot)
                    .reference_slots(&slots)
                    .push_next(&mut h264);
                // SAFETY: recording into the session's command buffer
                // inside its coding scope, with every structure the encode
                // reads alive through the call.
                unsafe { (encode_fp.cmd_encode_video_khr)(command, &info) };
            }
            CodecInfo::H265 {
                picture,
                slices,
                setup,
                references,
            } => {
                let mut setup_codec =
                    vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(setup);
                let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(op.setup_slot)
                    .picture_resource(setup_resource)
                    .push_next(&mut setup_codec);
                let mut codecs: Vec<vk::VideoEncodeH265DpbSlotInfoKHR<'_>> = references
                    .iter()
                    .map(|r| vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(r))
                    .collect();
                let slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = op
                    .reference_slots
                    .iter()
                    .zip(reference_resources)
                    .zip(codecs.iter_mut())
                    .map(|((&slot, resource), codec)| {
                        vk::VideoReferenceSlotInfoKHR::default()
                            .slot_index(slot)
                            .picture_resource(resource)
                            .push_next(codec)
                    })
                    .collect();
                let nalus: Vec<vk::VideoEncodeH265NaluSliceSegmentInfoKHR<'_>> = slices
                    .iter()
                    .map(|header| {
                        vk::VideoEncodeH265NaluSliceSegmentInfoKHR::default()
                            .constant_qp(op.qp)
                            .std_slice_segment_header(header)
                    })
                    .collect();
                let mut h265 = vk::VideoEncodeH265PictureInfoKHR::default()
                    .nalu_slice_segment_entries(&nalus)
                    .std_picture_info(picture);
                let info = info
                    .setup_reference_slot(&setup_slot)
                    .reference_slots(&slots)
                    .push_next(&mut h265);
                // SAFETY: as above.
                unsafe { (encode_fp.cmd_encode_video_khr)(command, &info) };
            }
        }
    }

    /// Records the copy of `planes`, the frame at its own size, into
    /// `image` at the coded size, the last column and row repeated into the
    /// padding, leaving it in `layout` for the encode.
    fn record_upload(
        &self,
        command: vk::CommandBuffer,
        planes: &BufferPlanes,
        image: vk::Image,
        layout: vk::ImageLayout,
    ) -> Result<(), Error> {
        let device = self.device.device();
        let sample = self.caps.profile.sample_bytes();
        let (width, height) = self.frame_size;
        planes.check(width, height, sample)?;
        let barrier = |src_stage, src_access, dst_stage, dst_access, old, new| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
        };
        let copy_write = (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let copy_any = (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );
        let to_copy = [barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::NONE,
            copy_write.0,
            copy_write.1,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        )];
        // SAFETY: recording into the session's command buffer outside a
        // coding scope, with what each command reads alive through the
        // call; the staging picture is the session's, its last reader done.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy),
            );
        }
        let regions = crate::session::buffer_regions(planes, (0, 0, width, height), sample);
        // SAFETY: as above; the planes fit the buffer and the copy's rules
        // (`BufferPlanes::check`), and the picture holds the frame.
        unsafe {
            device.cmd_copy_buffer_to_image(
                command,
                planes.buffer,
                image,
                vk::ImageLayout::GENERAL,
                &regions,
            );
        }
        // The picture's whole extent, which the device reads: past the
        // coded size to its input granularity (the RTX 5080 codes H.265 to
        // whole 32x32 CTBs, 2176 rows for 2160).
        let (coded_width, coded_height) = (self.image_extent.width, self.image_extent.height);
        // Each plane's padding: the last column copied rightwards, then
        // the last row, padding included, downwards.
        let plane_sizes = [
            (
                vk::ImageAspectFlags::PLANE_0,
                width,
                height,
                coded_width,
                coded_height,
            ),
            (
                vk::ImageAspectFlags::PLANE_1,
                width.div_ceil(2),
                height.div_ceil(2),
                coded_width / 2,
                coded_height / 2,
            ),
        ];
        let at = |x: u32, y: u32| vk::Offset3D {
            x: i32::try_from(x).unwrap_or(i32::MAX),
            y: i32::try_from(y).unwrap_or(i32::MAX),
            z: 0,
        };
        let columns: Vec<vk::ImageCopy> = plane_sizes
            .iter()
            .flat_map(|&(aspect, w, h, cw, _)| {
                (w..cw).map(move |x| {
                    vk::ImageCopy::default()
                        .src_subresource(plane(aspect))
                        .src_offset(at(w - 1, 0))
                        .dst_subresource(plane(aspect))
                        .dst_offset(at(x, 0))
                        .extent(vk::Extent3D {
                            width: 1,
                            height: h,
                            depth: 1,
                        })
                })
            })
            .collect();
        let rows: Vec<vk::ImageCopy> = plane_sizes
            .iter()
            .flat_map(|&(aspect, _, h, cw, ch)| {
                (h..ch).map(move |y| {
                    vk::ImageCopy::default()
                        .src_subresource(plane(aspect))
                        .src_offset(at(0, h - 1))
                        .dst_subresource(plane(aspect))
                        .dst_offset(at(0, y))
                        .extent(vk::Extent3D {
                            width: cw,
                            height: 1,
                            depth: 1,
                        })
                })
            })
            .collect();
        for regions in [columns, rows] {
            if regions.is_empty() {
                continue;
            }
            let between = [barrier(
                copy_write.0,
                copy_write.1,
                copy_any.0,
                copy_any.1,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
            )];
            // SAFETY: as above.
            unsafe {
                self.device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&between),
                );
            }
            // SAFETY: as above; each region reads a column or row the copies
            // before wrote, and writes another, within the picture.
            unsafe {
                device.cmd_copy_image(
                    command,
                    image,
                    vk::ImageLayout::GENERAL,
                    image,
                    vk::ImageLayout::GENERAL,
                    &regions,
                );
            }
        }
        let to_encode = [barrier(
            copy_write.0,
            copy_write.1,
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
            vk::ImageLayout::GENERAL,
            layout,
        )];
        // SAFETY: as above.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&to_encode),
            );
        }
        Ok(())
    }

    /// A picture sources in buffers are copied into, whose last encode is
    /// done.
    fn staging(&mut self) -> Result<Staging, Error> {
        let completed = self.timeline.completed();
        // Each is taken out and put back last, so the free one is found
        // first.
        if let Some(at) = self.staging.iter().position(|s| s.free_after <= completed) {
            return Ok(self.staging.swap_remove(at));
        }
        let usage = vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR;
        let image = SessionImage::new(
            &self.device,
            &self.caps,
            self.image_extent,
            usage,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
        )?;
        Ok(Staging {
            image,
            free_after: 0,
        })
    }

    /// A submission whose last work is done.
    fn submission(&mut self) -> Result<Submission, Error> {
        let completed = self.timeline.completed();
        while self.busy.front().is_some_and(|s| s.value <= completed) {
            if let Some(done) = self.busy.pop_front() {
                self.free.push(done);
            }
        }
        if let Some(free) = self.free.pop() {
            return Ok(free);
        }
        Ok(Submission {
            command: allocate_command(&self.device, self.pool)?,
            value: 0,
        })
    }
}

impl Drop for EncodeSession {
    fn drop(&mut self) {
        // Every work of the session was submitted before the timeline's
        // last value.
        if let Err(error) = self.timeline.wait(
            self.timeline.last_submitted(),
            std::time::Duration::from_secs(10),
        ) {
            log::error!("an encode session's work did not finish ({error}): leaking it");
            return;
        }
        self.busy.clear();
        self.free.clear();
        self.staging.clear();
        drop(self.parameters.take());
        self.dpb.clear();
        let device = self.device.device();
        // SAFETY: the pool is the session's, its command buffers idle; they
        // go with it. A null pool is nothing.
        unsafe { device.destroy_command_pool(self.pool, None) };
        if self.handle != vk::VideoSessionKHR::null() {
            // SAFETY: the session is the device's and no work uses it.
            unsafe {
                (self.device.video().fp().destroy_video_session_khr)(
                    device.handle(),
                    self.handle,
                    std::ptr::null(),
                );
            }
        }
        for &memory in &self.memory {
            // SAFETY: the session bound to it is destroyed.
            unsafe { device.free_memory(memory, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(offset: u64, size: u64) -> BitstreamBuffer {
        BitstreamBuffer {
            buffer: vk::Buffer::null(),
            offset,
            size,
        }
    }

    #[test]
    fn a_bitstream_starts_at_the_devices_alignment() {
        let (from, range, header) =
            EncodeSession::placement(&target(512, 100_000), 0, 256, 256).unwrap();
        assert_eq!((from, header), (512, None));
        assert_eq!(range, 99_840);
        let (from, _, _) = EncodeSession::placement(&target(600, 100_000), 0, 256, 256).unwrap();
        assert_eq!(from, 768);
    }

    #[test]
    fn in_band_parameter_sets_end_where_the_picture_starts() {
        // 27 bytes of parameter sets, padded to 28 in front: the encode
        // from 768, the header from 740, its first byte at 741.
        let (from, range, header) =
            EncodeSession::placement(&target(512, 100_000), 27, 256, 256).unwrap();
        assert_eq!((from, header), (768, Some(740)));
        assert_eq!(range, 99_584);
        // Room before the alignment: the encode still starts past it.
        let (from, _, header) =
            EncodeSession::placement(&target(500, 100_000), 8, 256, 256).unwrap();
        assert_eq!((from, header), (512, Some(504)));
        // An alignment of 1 still writes whole words.
        let (from, _, header) = EncodeSession::placement(&target(3, 1000), 5, 1, 1).unwrap();
        assert_eq!((from, header), (12, Some(4)));
    }

    #[test]
    fn a_bitstream_with_no_room_at_the_alignment_is_refused() {
        assert!(EncodeSession::placement(&target(512, 200), 0, 256, 256).is_err());
        assert!(EncodeSession::placement(&target(0, 300), 64, 256, 256).is_err());
    }
}
