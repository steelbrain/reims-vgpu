//! One stream's video session: the `VkVideoSessionKHR` and its memory,
//! its parameters object, and the command buffers and bitstream buffers
//! its decodes are submitted with, each reused once the timeline has
//! passed the decode that used it.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use ash::vk::native::{StdVideoDecodeH264PictureInfo, StdVideoDecodeH264ReferenceInfo};
use reims_vgpu_video_codec::h264::{pps::Pps, sps::Sps};

use super::params::ParameterSets;
use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::picture::color_range;
use crate::profile::{H264Capabilities, PICTURE_FORMAT};
use crate::timeline::Timeline;

/// The least a bitstream buffer holds, so that most streams make one
/// buffer a submission and never grow it.
const MIN_BITSTREAM: u64 = 2 << 20;

/// The start code put before each slice: the Vulkan specification has
/// `pSliceOffsets` locate each slice in the buffer and says nothing of
/// start codes, and the decoders in use read each slice from a three-byte
/// start code at its offset, as ffmpeg's Vulkan decoder writes them.
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
    /// The target's image and its decode view.
    pub(crate) target: (vk::Image, vk::ImageView),
    /// The DPB slot the target is set up in.
    pub(crate) setup_slot: i32,
    /// The target as a reference, for its slot.
    pub(crate) setup_info: StdVideoDecodeH264ReferenceInfo,
    /// The picture.
    pub(crate) picture_info: StdVideoDecodeH264PictureInfo,
    /// Each reference: its slot, its decode view, and its std information.
    pub(crate) references: &'a [(i32, vk::ImageView, StdVideoDecodeH264ReferenceInfo)],
}

/// One stream's session.
pub(crate) struct Session {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    /// What the device answers for the session's profile.
    pub(crate) caps: H264Capabilities,
    /// The coded extent each picture is decoded at.
    pub(crate) extent: vk::Extent2D,
    /// The DPB slots it has.
    pub(crate) max_dpb_slots: u32,
    handle: vk::VideoSessionKHR,
    memory: Vec<vk::DeviceMemory>,
    parameters: Option<ParameterSets>,
    /// Whether a decode has reset the session, which its first does.
    reset: bool,
    pool: vk::CommandPool,
    /// Submissions whose work may be in flight, oldest first.
    busy: VecDeque<Submission>,
    /// Submissions whose work is done.
    free: Vec<Submission>,
}

/// A command buffer and a bitstream buffer, which one decode at a time
/// uses.
struct Submission {
    command: vk::CommandBuffer,
    bitstream: Bitstream,
    /// The value of the last decode submitted with it.
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

impl Session {
    /// A session for frames of `extent` with the DPB `dpb_frames` frames
    /// need, of `caps`'s profile.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        timeline: &Arc<Timeline>,
        caps: H264Capabilities,
        extent: vk::Extent2D,
        dpb_frames: usize,
    ) -> Result<Self, Error> {
        let frames = u32::try_from(dpb_frames).unwrap_or(u32::MAX);
        // A slot for each frame the DPB holds, and one for the picture
        // being decoded.
        let max_dpb_slots = frames.saturating_add(1);
        if max_dpb_slots > caps.max_dpb_slots || frames > caps.max_active_reference_pictures {
            return Err(Error::Unsupported(format!(
                "a DPB of {frames} frames: the device has {} slots and {} active references",
                caps.max_dpb_slots, caps.max_active_reference_pictures
            )));
        }
        let mut session = Self {
            device: Arc::clone(device),
            timeline: Arc::clone(timeline),
            caps,
            extent,
            max_dpb_slots,
            handle: vk::VideoSessionKHR::null(),
            memory: Vec::new(),
            parameters: None,
            reset: false,
            pool: vk::CommandPool::null(),
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
                .max_dpb_slots(max_dpb_slots)
                .max_active_reference_pictures(frames)
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
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.decode_family());
        // SAFETY: `info` lives through the call.
        session.pool = unsafe { device.device().create_command_pool(&info, None) }
            .named("vkCreateCommandPool")?;
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

    /// Records and submits `decode`, signalling the timeline's next value,
    /// which it returns.
    pub(crate) fn decode(&mut self, decode: &Decode<'_>) -> Result<u64, Error> {
        let (offsets, range) =
            slice_layout(decode.slices, self.caps.min_bitstream_buffer_size_alignment)?;
        let mut submission = self.submission(range)?;
        let result = self.record_and_submit(&mut submission, decode, &offsets, range);
        match result {
            Ok(value) => {
                submission.value = value;
                self.busy.push_back(submission);
                Ok(value)
            }
            Err(error) => {
                // Nothing was submitted with it.
                self.free.push(submission);
                Err(error)
            }
        }
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
        // SAFETY: the command buffer is the session's pool's, and its last
        // submission is done; so for each command recorded below.
        unsafe { device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()) }
            .named("vkResetCommandBuffer")?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: as above.
        unsafe { device.begin_command_buffer(command, &begin) }.named("vkBeginCommandBuffer")?;

        // The target drops what it held; the references were written by
        // decodes before this one on this queue.
        let layout = if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        };
        let decode_access =
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR;
        let target_barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .dst_access_mask(decode_access)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(decode.target.0)
            .subresource_range(color_range())];
        let references_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_READ_KHR)];
        let dependency = vk::DependencyInfo::default()
            .memory_barriers(&references_barrier)
            .image_memory_barriers(&target_barrier);
        // SAFETY: as above, with the arrays alive through the call.
        unsafe { device.cmd_pipeline_barrier2(command, &dependency) };

        let resource = |view| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(self.extent)
                .base_array_layer(0)
                .image_view_binding(view)
        };
        let reference_resources: Vec<vk::VideoPictureResourceInfoKHR<'_>> = decode
            .references
            .iter()
            .map(|&(_, view, _)| resource(view))
            .collect();
        let mut reference_dpb: Vec<vk::VideoDecodeH264DpbSlotInfoKHR<'_>> = decode
            .references
            .iter()
            .map(|(_, _, info)| {
                vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(info)
            })
            .collect();
        let target_resource = resource(decode.target.1);
        let mut setup_dpb =
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&decode.setup_info);

        // The slots bound for the scope: every reference, and the target,
        // not yet a DPB slot's (-1), as the decode sets it up.
        let mut bound: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = decode
            .references
            .iter()
            .zip(&reference_resources)
            .map(|(&(slot, _, _), resource)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
            })
            .collect();
        bound.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&target_resource),
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
            .map(|((&(slot, _, _), resource), dpb)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
                    .push_next(dpb)
            })
            .collect();
        let setup = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(decode.setup_slot)
            .picture_resource(&target_resource)
            .push_next(&mut setup_dpb);
        let mut h264 = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&decode.picture_info)
            .slice_offsets(offsets);
        let info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(submission.bitstream.buffer)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(target_resource)
            .setup_reference_slot(&setup)
            .reference_slots(&references)
            .push_next(&mut h264);
        // SAFETY: as above, inside the coding scope, with every structure
        // the decode reads alive through the call.
        unsafe { (self.device.decode().fp().cmd_decode_video_khr)(command, &info) };
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: as above, closing the scope.
        unsafe { (fp.cmd_end_video_coding_khr)(command, &end) };
        // SAFETY: as above.
        unsafe { device.end_command_buffer(command) }.named("vkEndCommandBuffer")?;

        let value = self.timeline.next();
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals)];
        self.device.submit_decode(&submit)?;
        self.timeline.submitted(value);
        self.reset = true;
        Ok(value)
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
        let command = match self.free.pop() {
            Some(small) => {
                // Its buffer is let go: too small, and its work done.
                small.command
            }
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                // SAFETY: the pool is the session's, used by it alone.
                let commands = unsafe { self.device.device().allocate_command_buffers(&info) }
                    .named("vkAllocateCommandBuffers")?;
                commands.first().copied().unwrap_or_default()
            }
        };
        let size = size
            .max(MIN_BITSTREAM)
            .next_power_of_two()
            .next_multiple_of(self.caps.min_bitstream_buffer_size_alignment.max(1));
        match Bitstream::new(&self.device, &self.caps, size) {
            Ok(bitstream) => Ok(Submission {
                command,
                bitstream,
                value: 0,
            }),
            Err(error) => {
                // SAFETY: the command buffer is the pool's, and idle.
                unsafe {
                    self.device
                        .device()
                        .free_command_buffers(self.pool, &[command]);
                }
                Err(error)
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Every decode of the session was submitted before the timeline's
        // last value.
        if let Err(error) = self.timeline.wait(
            self.timeline.last_submitted(),
            std::time::Duration::from_secs(10),
        ) {
            log::error!("a video session's decodes did not finish ({error}): leaking it");
            return;
        }
        self.busy.clear();
        self.free.clear();
        drop(self.parameters.take());
        let device = self.device.device();
        // SAFETY: the pool is the session's, its command buffers idle;
        // they go with it. A null pool is nothing.
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
