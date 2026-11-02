//! One stream's video session: the `VkVideoSessionKHR` and its memory,
//! its DPB pictures, its parameters object, and the command buffers and
//! bitstream buffers its decodes are submitted with, each reused once the
//! timeline has passed the work that used it.
//!
//! **DPB pictures.** The session owns one NV12 image a DPB slot, so that a
//! reference is never an output: a guest recycles an output once it has
//! shown it, whether or not the picture is still a reference. Each decode
//! sets up the least slot no reference holds, and takes its picture from
//! `UNDEFINED`. With unified layouts every DPB picture stays in `GENERAL`;
//! without, in `VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR`.
//!
//! **Outputs.** Where the device decodes into an output beside the DPB
//! picture (`DPB_AND_OUTPUT_DISTINCT`), the decode writes the output. Where
//! the DPB picture is the decode's output (`DPB_AND_OUTPUT_COINCIDE`, all
//! the 5080 offers), one GPU copy then takes the picture from it into the
//! output: in the decode's own command buffer where the decode family
//! copies, else on the copy queue behind the decode's value, signalling
//! the next, which the following decode waits for before it sets a slot up
//! again. Every output is left in `GENERAL`.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use ash::vk::native::{StdVideoDecodeH264PictureInfo, StdVideoDecodeH264ReferenceInfo};
use reims_vgpu_video_codec::h264::{pps::Pps, sps::Sps};

use super::params::ParameterSets;
use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::picture::{ImageView, color_range, plane, planes};
use crate::profile::{H264Capabilities, PICTURE_FORMAT};
use crate::surface::bind_image_memory;
use crate::timeline::Timeline;

/// The least a bitstream buffer holds, so that most streams make one
/// buffer a submission and never grow it.
const MIN_BITSTREAM: u64 = 2 << 20;

/// The start code put before each slice: the Vulkan specification has
/// `pSliceOffsets` locate each slice in the buffer and says nothing of
/// start codes; ffmpeg's Vulkan decoder (`vulkan_h264.c`) puts a
/// three-byte start code at each offset, and the 5080 decodes every
/// stream of the H.264 check so.
const START_CODE: [u8; 3] = [0, 0, 1];

/// Where each slice goes in a picture's bitstream, after its start code,
/// and the bytes the decode reads: the slices' total, padded to
/// `size_alignment`.
pub(crate) fn slice_layout(
    slices: &[&[u8]],
    size_alignment: u64,
) -> Result<(Vec<u32>, u64), Error> {
    let mut offsets = Vec::with_capacity(slices.len());
    let mut at = 0u64;
    for slice in slices {
        offsets.push(
            u32::try_from(at).map_err(|_| Error::Unsupported("a picture over 4 GiB".into()))?,
        );
        at += u64::try_from(START_CODE.len() + slice.len()).unwrap_or(u64::MAX);
    }
    Ok((offsets, at.next_multiple_of(size_alignment.max(1))))
}

/// A decode as [`Session::decode`] records it.
pub(crate) struct Decode<'a> {
    /// The picture's slices, each its NAL unit from the header byte on.
    pub(crate) slices: &'a [&'a [u8]],
    /// The parameters object holding the picture's SPS and PPS.
    pub(crate) parameters: vk::VideoSessionParametersKHR,
    /// The output the picture lands in, and its decode view where the
    /// decode writes it.
    pub(crate) output: (vk::Image, Option<vk::ImageView>),
    /// The DPB slot the picture is set up in.
    pub(crate) setup_slot: i32,
    /// The picture as a reference, for its slot.
    pub(crate) setup_info: StdVideoDecodeH264ReferenceInfo,
    /// The picture.
    pub(crate) picture_info: StdVideoDecodeH264PictureInfo,
    /// Each reference: its slot and its std information.
    pub(crate) references: &'a [(i32, StdVideoDecodeH264ReferenceInfo)],
}

/// One stream's session.
pub(crate) struct Session {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    /// What the device answers for the session's profile.
    pub(crate) caps: H264Capabilities,
    /// The coded extent each picture is decoded at.
    pub(crate) extent: vk::Extent2D,
    handle: vk::VideoSessionKHR,
    memory: Vec<vk::DeviceMemory>,
    /// One a slot, by its index.
    dpb: Vec<DpbPicture>,
    parameters: Option<ParameterSets>,
    /// Whether a decode has reset the session, which its first does.
    reset: bool,
    pool: vk::CommandPool,
    /// The copy queue's, where copies run on it rather than the decode
    /// queue.
    copy_pool: vk::CommandPool,
    /// The value the last copy on the copy queue signals, which the next
    /// decode waits for; 0 before one.
    last_copy: u64,
    /// Submissions whose work may be in flight, oldest first.
    busy: VecDeque<Submission>,
    /// Submissions whose work is done.
    free: Vec<Submission>,
}

/// A DPB picture of the session's, destroyed as it goes.
struct DpbPicture {
    device: Arc<VideoDevice>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: Option<ImageView>,
}

impl Drop for DpbPicture {
    fn drop(&mut self) {
        // The view goes before its image.
        drop(self.view.take());
        // SAFETY: the image is the device's, and its session's work is
        // done; a null handle is nothing.
        unsafe { self.device.device().destroy_image(self.image, None) };
        // SAFETY: its memory, now bound to nothing.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}

impl DpbPicture {
    /// A DPB picture of `extent` for `caps`'s profile: also the decode's
    /// output, copied out of, where the two coincide.
    fn new(
        device: &Arc<VideoDevice>,
        caps: &H264Capabilities,
        extent: vk::Extent2D,
    ) -> Result<Self, Error> {
        let view_usage = if caps.decodes_into_output {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
        } else {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
        };
        let usage = if caps.decodes_into_output {
            view_usage
        } else {
            view_usage | vk::ImageUsageFlags::TRANSFER_SRC
        };
        let families = device.dpb_families();
        let image = caps.profile.with_list(|list| {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(PICTURE_FORMAT)
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
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(list);
            let info = if families.len() > 1 {
                info.sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families)
            } else {
                info.sharing_mode(vk::SharingMode::EXCLUSIVE)
            };
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
        made.view = Some(ImageView::for_decode(device, image, view_usage)?);
        Ok(made)
    }

    fn view(&self) -> vk::ImageView {
        self.view
            .as_ref()
            .map_or(vk::ImageView::null(), ImageView::handle)
    }
}

/// The command buffers and the bitstream buffer one decode at a time uses.
struct Submission {
    command: vk::CommandBuffer,
    /// The copy queue's, where copies run on it.
    copy_command: vk::CommandBuffer,
    bitstream: Bitstream,
    /// The value the last work submitted with it signals.
    value: u64,
}

/// A host-visible buffer a picture's slices are written into, mapped for
/// as long as it lives.
struct Bitstream {
    device: Arc<VideoDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: NonNull<u8>,
    size: u64,
}

// SAFETY: the mapping is the buffer's memory alone, written by the one
// session that holds it, through `&mut`.
unsafe impl Send for Bitstream {}

/// Where a pipeline barrier's first scope starts: after a decode on this
/// queue, or after a semaphore wait that already holds it.
#[derive(Clone, Copy)]
enum After {
    DecodeOnThisQueue,
    SemaphoreWait,
}

impl Session {
    /// A session for frames of `extent` whose stream holds
    /// `max_num_ref_frames` references, of `caps`'s profile: a DPB slot
    /// for each reference and one for the picture being decoded.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        timeline: &Arc<Timeline>,
        caps: H264Capabilities,
        extent: vk::Extent2D,
        max_num_ref_frames: u8,
    ) -> Result<Self, Error> {
        let references = u32::from(max_num_ref_frames);
        let slots = references + 1;
        if slots > caps.max_dpb_slots || references > caps.max_active_reference_pictures {
            return Err(Error::Unsupported(format!(
                "{references} references: the device has {} DPB slots and {} active references",
                caps.max_dpb_slots, caps.max_active_reference_pictures
            )));
        }
        let mut session = Self {
            device: Arc::clone(device),
            timeline: Arc::clone(timeline),
            caps,
            extent,
            handle: vk::VideoSessionKHR::null(),
            memory: Vec::new(),
            dpb: Vec::new(),
            parameters: None,
            reset: false,
            pool: vk::CommandPool::null(),
            copy_pool: vk::CommandPool::null(),
            last_copy: 0,
            busy: VecDeque::new(),
            free: Vec::new(),
        };
        let fp = device.video().fp();
        let vk_device = device.device().handle();
        caps.profile.with_info(|profile| {
            let info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(device.decode_family())
                .video_profile(profile)
                .picture_format(PICTURE_FORMAT)
                .max_coded_extent(extent)
                .reference_picture_format(PICTURE_FORMAT)
                .max_dpb_slots(slots)
                .max_active_reference_pictures(references)
                .std_header_version(&caps.std_header_version);
            // SAFETY: `info` and what it points at live through the call.
            unsafe {
                (fp.create_video_session_khr)(
                    vk_device,
                    &info,
                    std::ptr::null(),
                    &mut session.handle,
                )
            }
            .named("vkCreateVideoSessionKHR")
        })?;
        session.bind_memory()?;
        for _ in 0..slots {
            session.dpb.push(DpbPicture::new(device, &caps, extent)?);
        }
        session.pool = command_pool(device, device.decode_family())?;
        if !caps.decodes_into_output && !device.copies_on_decode_queue() {
            session.copy_pool = command_pool(device, device.copy_family())?;
        }
        Ok(session)
    }

    /// Binds memory of its own to each of the session's bindings.
    fn bind_memory(&mut self) -> Result<(), Error> {
        let fp = self.device.video().fp();
        let vk_device = self.device.device().handle();
        let mut count = 0;
        // SAFETY: the session is the device's; a null output asks for the
        // count alone.
        unsafe {
            (fp.get_video_session_memory_requirements_khr)(
                vk_device,
                self.handle,
                &mut count,
                std::ptr::null_mut(),
            )
        }
        .named("vkGetVideoSessionMemoryRequirementsKHR")?;
        let mut requirements = vec![
            vk::VideoSessionMemoryRequirementsKHR::default();
            usize::try_from(count).unwrap_or(0)
        ];
        // SAFETY: as above, with an output of `count` elements.
        unsafe {
            (fp.get_video_session_memory_requirements_khr)(
                vk_device,
                self.handle,
                &mut count,
                requirements.as_mut_ptr(),
            )
        }
        .named("vkGetVideoSessionMemoryRequirementsKHR")?;
        requirements.truncate(usize::try_from(count).unwrap_or(0));
        let mut binds = Vec::with_capacity(requirements.len());
        for requirement in &requirements {
            let memory = self.device.allocate(
                &requirement.memory_requirements,
                &[
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                ],
                None,
            )?;
            self.memory.push(memory);
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(requirement.memory_bind_index)
                    .memory(memory)
                    .memory_offset(0)
                    .memory_size(requirement.memory_requirements.size),
            );
        }
        if binds.is_empty() {
            return Ok(());
        }
        // SAFETY: each memory was allocated for its binding, which is bound
        // to none yet; the array lives through the call.
        unsafe {
            (fp.bind_video_session_memory_khr)(
                vk_device,
                self.handle,
                u32::try_from(binds.len()).unwrap_or(u32::MAX),
                binds.as_ptr(),
            )
        }
        .named("vkBindVideoSessionMemoryKHR")
    }

    /// The DPB slots the session has.
    pub(crate) fn slots(&self) -> u32 {
        u32::try_from(self.dpb.len()).unwrap_or(u32::MAX)
    }

    /// The parameters object that holds `sps` and `pps`: the one it has,
    /// or a new one in its place, the old retired once the decodes
    /// submitted with it are done.
    pub(crate) fn parameters(
        &mut self,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<vk::VideoSessionParametersKHR, Error> {
        if let Some(sets) = &self.parameters
            && sets.holds(sps, pps)
        {
            return Ok(sets.handle());
        }
        let new = ParameterSets::new(
            &self.device,
            self.handle,
            self.parameters.as_ref(),
            sps,
            pps,
        )?;
        let handle = new.handle();
        if let Some(old) = self.parameters.replace(new) {
            self.timeline.retire(
                self.timeline.last_submitted(),
                Box::new(old.into_parameters()),
            );
        }
        Ok(handle)
    }

    /// Records and submits `decode`, and the copy of its picture into its
    /// output where the device decodes into DPB pictures alone. Returns
    /// the value after which the output holds the picture.
    pub(crate) fn decode(&mut self, decode: &Decode<'_>) -> Result<u64, Error> {
        let (offsets, range) =
            slice_layout(decode.slices, self.caps.min_bitstream_buffer_size_alignment)?;
        let mut submission = self.submission(range)?;
        match self.record_and_submit(&mut submission, decode, &offsets, range) {
            Ok(value) => {
                submission.value = value;
                self.busy.push_back(submission);
                Ok(value)
            }
            Err(error) => {
                // Its work was not submitted, or it signals no later than
                // the timeline's last value: done once that is.
                submission.value = self.timeline.last_submitted();
                self.busy.push_back(submission);
                Err(error)
            }
        }
    }

    /// The layout DPB pictures are kept in.
    fn dpb_layout(&self) -> vk::ImageLayout {
        if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        }
    }

    fn dpb_picture(&self, slot: i32) -> Result<&DpbPicture, Error> {
        usize::try_from(slot)
            .ok()
            .and_then(|slot| self.dpb.get(slot))
            .ok_or_else(|| Error::Unsupported(format!("DPB slot {slot} past the session's")))
    }

    fn record_and_submit(
        &mut self,
        submission: &mut Submission,
        decode: &Decode<'_>,
        offsets: &[u32],
        range: u64,
    ) -> Result<u64, Error> {
        submission.bitstream.write(decode.slices, offsets, range);
        let device = self.device.device();
        let command = submission.command;
        begin(device, command)?;

        let (setup_image, setup_view) = {
            let setup = self.dpb_picture(decode.setup_slot)?;
            (setup.image, setup.view())
        };
        let dpb_layout = self.dpb_layout();
        let decode_access =
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR;
        let fresh = |image, layout| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(decode_access)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
        };
        // The DPB picture set up drops what it held; an output the decode
        // writes, likewise. The references were written by decodes before
        // this one, on this queue.
        let output_view = if self.caps.decodes_into_output {
            Some(decode.output.1.ok_or_else(|| {
                Error::Unsupported("an output a decode writes, with no view".into())
            })?)
        } else {
            None
        };
        let decode_output_layout = if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DST_KHR
        };
        let mut images = vec![fresh(setup_image, dpb_layout)];
        if output_view.is_some() {
            images.push(fresh(decode.output.0, decode_output_layout));
        }
        let references_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_READ_KHR)];
        let dependency = vk::DependencyInfo::default()
            .memory_barriers(&references_barrier)
            .image_memory_barriers(&images);
        // SAFETY: recording into the session's command buffer, whose last
        // work is done, with the arrays alive through the call; so for each
        // command recorded below.
        unsafe { device.cmd_pipeline_barrier2(command, &dependency) };

        let resource = |view| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(self.extent)
                .base_array_layer(0)
                .image_view_binding(view)
        };
        let mut reference_resources = Vec::with_capacity(decode.references.len());
        for &(slot, _) in decode.references {
            reference_resources.push(resource(self.dpb_picture(slot)?.view()));
        }
        let mut reference_dpb: Vec<vk::VideoDecodeH264DpbSlotInfoKHR<'_>> = decode
            .references
            .iter()
            .map(|(_, info)| vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(info))
            .collect();
        let setup_resource = resource(setup_view);
        let output_resource = output_view.map_or(setup_resource, resource);
        let mut setup_dpb =
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&decode.setup_info);

        // The slots bound for the scope: every reference's, and the
        // picture's DPB picture, not yet a slot's (-1), as the decode sets
        // it up.
        let mut bound: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = decode
            .references
            .iter()
            .zip(&reference_resources)
            .map(|(&(slot, _), resource)| {
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
        let begin_coding = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.handle)
            .video_session_parameters(decode.parameters)
            .reference_slots(&bound);
        let fp = self.device.video().fp();
        // SAFETY: as above.
        unsafe { (fp.cmd_begin_video_coding_khr)(command, &begin_coding) };
        if !self.reset {
            let control = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            // SAFETY: as above, inside the coding scope.
            unsafe { (fp.cmd_control_video_coding_khr)(command, &control) };
        }
        let references: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = decode
            .references
            .iter()
            .zip(&reference_resources)
            .zip(reference_dpb.iter_mut())
            .map(|((&(slot, _), resource), dpb)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
                    .push_next(dpb)
            })
            .collect();
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(decode.setup_slot)
            .picture_resource(&setup_resource)
            .push_next(&mut setup_dpb);
        let mut h264 = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&decode.picture_info)
            .slice_offsets(offsets);
        let info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(submission.bitstream.buffer)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(output_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&references)
            .push_next(&mut h264);
        // SAFETY: as above, inside the coding scope, with every structure
        // the decode reads alive through the call.
        unsafe { (self.device.decode().fp().cmd_decode_video_khr)(command, &info) };
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: as above, closing the scope.
        unsafe { (fp.cmd_end_video_coding_khr)(command, &end) };

        let copy_here = !self.caps.decodes_into_output && self.device.copies_on_decode_queue();
        if self.caps.decodes_into_output && decode_output_layout != vk::ImageLayout::GENERAL {
            let to_general = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(decode_output_layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(decode.output.0)
                .subresource_range(color_range())];
            // SAFETY: as above.
            unsafe {
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_general),
                );
            }
        } else if copy_here {
            self.record_copy(
                command,
                setup_image,
                decode.output.0,
                After::DecodeOnThisQueue,
            );
        }
        // SAFETY: as above.
        unsafe { device.end_command_buffer(command) }.named("vkEndCommandBuffer")?;

        // A copy on the copy queue may still read a DPB picture this decode
        // sets up again.
        let value = self.timeline.next();
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore())
            .value(self.last_copy)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let waits = if self.last_copy > 0 {
            &waits[..]
        } else {
            &[][..]
        };
        submit(
            &self.device,
            &self.timeline,
            submission.command,
            waits,
            value,
            Queue::Decode,
        )?;
        self.reset = true;
        if self.caps.decodes_into_output || copy_here {
            return Ok(value);
        }

        // The copy out, on the copy queue once the decode is done.
        let copy = submission.copy_command;
        begin(device, copy)?;
        self.record_copy(copy, setup_image, decode.output.0, After::SemaphoreWait);
        // SAFETY: as above, for the copy queue's command buffer.
        unsafe { device.end_command_buffer(copy) }.named("vkEndCommandBuffer")?;
        let landed = self.timeline.next();
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        submit(
            &self.device,
            &self.timeline,
            copy,
            &waits,
            landed,
            Queue::Copy,
        )?;
        self.last_copy = landed;
        Ok(landed)
    }

    /// Records the copy of DPB picture `dpb`'s two planes into `output`,
    /// which it takes from `UNDEFINED` and leaves in `GENERAL`, the DPB
    /// picture left in the DPB layout.
    fn record_copy(
        &self,
        command: vk::CommandBuffer,
        dpb: vk::Image,
        output: vk::Image,
        after: After,
    ) {
        let device = self.device.device();
        let dpb_layout = self.dpb_layout();
        let copy_layout = if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        // After the decode on this queue, or after the semaphore wait,
        // which makes its writes visible to the copy queue.
        let (src_stage, src_access) = match after {
            After::DecodeOnThisQueue => (
                vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            ),
            After::SemaphoreWait => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
        };
        let barrier = |image, old, new| {
            vk::ImageMemoryBarrier2::default()
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .old_layout(old)
                .new_layout(new)
                .image(image)
                .subresource_range(color_range())
        };
        let before = [
            barrier(dpb, dpb_layout, copy_layout)
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ),
            barrier(output, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL)
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE),
        ];
        // SAFETY: recording into a command buffer of the session's, with
        // the array alive through the call; so for each command below.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
        }
        let regions = planes(self.extent).map(|(aspect, extent)| {
            vk::ImageCopy::default()
                .src_subresource(plane(aspect))
                .dst_subresource(plane(aspect))
                .extent(extent)
        });
        // SAFETY: as above; both images hold the extent, in the layouts
        // the barrier left them in.
        unsafe {
            device.cmd_copy_image(
                command,
                dpb,
                copy_layout,
                output,
                vk::ImageLayout::GENERAL,
                &regions,
            );
        }
        if copy_layout != dpb_layout {
            let back = [barrier(dpb, copy_layout, dpb_layout)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::NONE)];
            // SAFETY: as above.
            unsafe {
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&back),
                );
            }
        }
    }

    /// A submission whose last work is done, with a bitstream buffer of
    /// `size` bytes at least.
    fn submission(&mut self, size: u64) -> Result<Submission, Error> {
        let completed = self.timeline.completed();
        while self.busy.front().is_some_and(|s| s.value <= completed) {
            if let Some(done) = self.busy.pop_front() {
                self.free.push(done);
            }
        }
        if let Some(at) = self.free.iter().position(|s| s.bitstream.size >= size) {
            return Ok(self.free.swap_remove(at));
        }
        let (command, copy_command) = match self.free.pop() {
            // Its buffer is let go: too small, and its work done.
            Some(small) => (small.command, small.copy_command),
            None => {
                let command = allocate_command(&self.device, self.pool)?;
                let copy_command = if self.copy_pool == vk::CommandPool::null() {
                    vk::CommandBuffer::null()
                } else {
                    allocate_command(&self.device, self.copy_pool)?
                };
                (command, copy_command)
            }
        };
        let size = size
            .max(MIN_BITSTREAM)
            .next_power_of_two()
            .next_multiple_of(self.caps.min_bitstream_buffer_size_alignment.max(1));
        // The command buffers go with their pools where this fails.
        let bitstream = Bitstream::new(&self.device, &self.caps, size)?;
        Ok(Submission {
            command,
            copy_command,
            bitstream,
            value: 0,
        })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Every work of the session was submitted before the timeline's
        // last value.
        if let Err(error) = self.timeline.wait(
            self.timeline.last_submitted(),
            std::time::Duration::from_secs(10),
        ) {
            log::error!("a video session's work did not finish ({error}): leaking it");
            return;
        }
        self.busy.clear();
        self.free.clear();
        drop(self.parameters.take());
        self.dpb.clear();
        let device = self.device.device();
        // SAFETY: the pools are the session's, their command buffers idle;
        // they go with them. A null pool is nothing.
        unsafe { device.destroy_command_pool(self.pool, None) };
        // SAFETY: as above.
        unsafe { device.destroy_command_pool(self.copy_pool, None) };
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

/// A command pool of `family` whose command buffers are reset one by one.
fn command_pool(device: &VideoDevice, family: u32) -> Result<vk::CommandPool, Error> {
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(family);
    // SAFETY: `info` lives through the call.
    unsafe { device.device().create_command_pool(&info, None) }.named("vkCreateCommandPool")
}

/// A primary command buffer of `pool`, which the session alone uses.
fn allocate_command(
    device: &VideoDevice,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, Error> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool is the session's, used by it alone.
    let commands = unsafe { device.device().allocate_command_buffers(&info) }
        .named("vkAllocateCommandBuffers")?;
    commands.first().copied().ok_or(Error::Vulkan(
        "vkAllocateCommandBuffers",
        vk::Result::ERROR_UNKNOWN,
    ))
}

/// Begins `command` anew, its last work done.
fn begin(device: &ash::Device, command: vk::CommandBuffer) -> Result<(), Error> {
    // SAFETY: the command buffer is a session's, from a pool that resets
    // them one by one, and its last work is done.
    unsafe { device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()) }
        .named("vkResetCommandBuffer")?;
    let info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: as above.
    unsafe { device.begin_command_buffer(command, &info) }.named("vkBeginCommandBuffer")
}

/// The queue a submission goes to.
#[derive(Clone, Copy)]
enum Queue {
    Decode,
    Copy,
}

/// Submits `command` to `queue` after `waits`, signalling the timeline's
/// `value`, which it records as submitted.
fn submit(
    device: &VideoDevice,
    timeline: &Timeline,
    command: vk::CommandBuffer,
    waits: &[vk::SemaphoreSubmitInfo<'_>],
    value: u64,
    queue: Queue,
) -> Result<(), Error> {
    let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
    let signals = [vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline.semaphore())
        .value(value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let submits = [vk::SubmitInfo2::default()
        .command_buffer_infos(&commands)
        .wait_semaphore_infos(waits)
        .signal_semaphore_infos(&signals)];
    match queue {
        Queue::Decode => device.submit_decode(&submits)?,
        Queue::Copy => device.submit_copy(&submits, vk::Fence::null())?,
    }
    timeline.submitted(value);
    Ok(())
}

impl Bitstream {
    /// A buffer of `size` bytes for `caps`'s profile, mapped.
    fn new(device: &Arc<VideoDevice>, caps: &H264Capabilities, size: u64) -> Result<Self, Error> {
        let vk_device = device.device();
        let buffer = caps.profile.with_list(|list| {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(list);
            // SAFETY: `info` and its chain live through the call.
            unsafe { vk_device.create_buffer(&info, None) }.named("vkCreateBuffer")
        })?;
        let mut made = Self {
            device: Arc::clone(device),
            buffer,
            memory: vk::DeviceMemory::null(),
            mapped: NonNull::dangling(),
            size,
        };
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        made.memory = device.allocate(&requirements, &[host], None)?;
        // SAFETY: the memory is unbound and of the buffer's requirements.
        unsafe { vk_device.bind_buffer_memory(buffer, made.memory, 0) }
            .named("vkBindBufferMemory")?;
        // SAFETY: the memory is host-visible and unmapped.
        let mapped = unsafe {
            vk_device.map_memory(made.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .named("vkMapMemory")?;
        made.mapped = NonNull::new(mapped.cast::<u8>()).ok_or(Error::Vulkan(
            "vkMapMemory",
            vk::Result::ERROR_MEMORY_MAP_FAILED,
        ))?;
        Ok(made)
    }

    /// Writes each slice after its start code at its offset, and zeros to
    /// `range`.
    fn write(&mut self, slices: &[&[u8]], offsets: &[u32], range: u64) {
        let len = usize::try_from(range.min(self.size)).unwrap_or(0);
        // SAFETY: the mapping is `size` bytes of host-coherent memory that
        // no GPU work reads now (its last decode is done), and `len` is no
        // more than that.
        let out = unsafe { std::slice::from_raw_parts_mut(self.mapped.as_ptr(), len) };
        out.fill(0);
        for (slice, &offset) in slices.iter().zip(offsets) {
            let at = usize::try_from(offset).unwrap_or(usize::MAX);
            let end = at.saturating_add(START_CODE.len() + slice.len());
            if let Some(to) = out.get_mut(at..end) {
                let (code, data) = to.split_at_mut(START_CODE.len());
                code.copy_from_slice(&START_CODE);
                data.copy_from_slice(slice);
            }
        }
    }
}

impl Drop for Bitstream {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        // SAFETY: the buffer is the device's, and no work uses it: its
        // holder lets it go once its last decode is done. A null handle is
        // nothing.
        unsafe { vk_device.destroy_buffer(self.buffer, None) };
        // SAFETY: its memory, unmapped as it is freed.
        unsafe { vk_device.free_memory(self.memory, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_slice_offset_is_its_start_code() {
        let slices: [&[u8]; 3] = [&[0x65, 1, 2], &[0x41; 10], &[0x01]];
        let (offsets, range) = slice_layout(&slices, 256).unwrap();
        assert_eq!(offsets, [0, 6, 19]);
        assert_eq!(range, 256);
        let (_, range) = slice_layout(&slices, 1).unwrap();
        assert_eq!(range, 23);
        // An alignment of 0 is taken as 1.
        assert_eq!(slice_layout(&slices, 0).unwrap().1, 23);
    }
}
