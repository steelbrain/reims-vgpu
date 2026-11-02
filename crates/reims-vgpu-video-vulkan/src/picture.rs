//! A decoded picture: its surface, the view decodes bind it through, the
//! DPB slot it was set up in, and the timeline value of its decode.

use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::time::Duration;

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::profile::PICTURE_FORMAT;
use crate::surface::SurfaceImage;
use crate::timeline::Timeline;

/// How long a read back waits for the decodes before it.
const READ_BACK_WAIT: Duration = Duration::from_secs(10);

/// A picture a decoder made: the surface it decoded into, and the
/// timeline value its decode signals.
///
/// A stand-in for a lost frame shares its source's surface and DPB slot
/// ([`reims_vgpu_video_codec::h264::Accelerator::conceal`]): a surface is
/// decoded into only when it is given, so the stand-in holds what its
/// source holds for as long as it lives.
pub struct Picture<T: SurfaceImage> {
    inner: Arc<Inner<T>>,
    stand_in: bool,
}

struct Inner<T: SurfaceImage> {
    timeline: Arc<Timeline>,
    /// Taken as it goes, to wait for the GPU in the timeline.
    held: ManuallyDrop<Held<T>>,
    extent: vk::Extent2D,
    /// The DPB slot its decode set up, -1 before it.
    slot: AtomicI32,
    /// The value of its decode's submission, 0 before it.
    decoded: AtomicU64,
    /// The value of the last submission that reads or writes it.
    last_use: AtomicU64,
}

/// What a picture holds that GPU work may still use as it goes.
struct Held<T> {
    surface: T,
    view: ImageView,
}

/// An image view, destroyed as it goes.
pub(crate) struct ImageView {
    device: Arc<VideoDevice>,
    view: vk::ImageView,
}

impl ImageView {
    /// A view of the whole of `image` for decoding into and predicting
    /// from: its usage is the decode usages alone, since a sampled view of
    /// NV12 would need a sampler conversion.
    pub(crate) fn for_decode(device: &Arc<VideoDevice>, image: vk::Image) -> Result<Self, Error> {
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(
            vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
        );
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(PICTURE_FORMAT)
            .subresource_range(color_range())
            .push_next(&mut usage);
        // SAFETY: `info` and its chain live through the call, and the image
        // is the device's (the surface's contract).
        let view =
            unsafe { device.device().create_image_view(&info, None) }.named("vkCreateImageView")?;
        Ok(Self {
            device: Arc::clone(device),
            view,
        })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        // SAFETY: the view is the device's, and the work that used it is
        // done (its holder waits for it in the timeline).
        unsafe { self.device.device().destroy_image_view(self.view, None) };
    }
}

/// The whole of a one-level, one-layer image, all its planes.
pub(crate) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

impl<T: SurfaceImage> Picture<T> {
    pub(crate) fn new(
        timeline: &Arc<Timeline>,
        surface: T,
        view: ImageView,
        extent: vk::Extent2D,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                timeline: Arc::clone(timeline),
                held: ManuallyDrop::new(Held { surface, view }),
                extent,
                slot: AtomicI32::new(-1),
                decoded: AtomicU64::new(0),
                last_use: AtomicU64::new(0),
            }),
            stand_in: false,
        }
    }

    /// Another picture of this one's surface and slot, standing in for a
    /// lost frame.
    pub(crate) fn stand_in(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            stand_in: true,
        }
    }

    /// Whether it stands in for a lost frame.
    pub(crate) fn is_stand_in(&self) -> bool {
        self.stand_in
    }

    /// The surface it was decoded into.
    pub fn surface(&self) -> &T {
        &self.inner.held.surface
    }

    /// The coded extent decoded: the frame before cropping, from the
    /// surface's origin.
    pub fn coded_extent(&self) -> vk::Extent2D {
        self.inner.extent
    }

    /// The decoder's timeline semaphore.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.inner.timeline.semaphore()
    }

    /// The value [`Self::semaphore`] reaches once the decode into the
    /// surface is done: graphics work that samples it waits for it.
    pub fn value(&self) -> u64 {
        self.inner.decoded.load(Ordering::Acquire)
    }

    /// Blocks until the decode is done, for `timeout` at most.
    pub fn wait(&self, timeout: Duration) -> Result<(), Error> {
        self.inner.timeline.wait(self.value(), timeout)
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.inner.held.view.view
    }

    /// The DPB slot its decode set up, `None` before it.
    pub(crate) fn slot(&self) -> Option<i32> {
        let slot = self.inner.slot.load(Ordering::Acquire);
        (slot >= 0).then_some(slot)
    }

    /// Records its decode: the slot it set up and its submission's value.
    pub(crate) fn decoded(&self, slot: i32, value: u64) {
        self.inner.slot.store(slot, Ordering::Release);
        self.inner.decoded.store(value, Ordering::Release);
        self.used(value);
    }

    /// Records a submission of `value` that reads it.
    pub(crate) fn used(&self, value: u64) {
        self.inner.last_use.fetch_max(value, Ordering::AcqRel);
    }

    /// The decoded planes, copied to host memory (tests and tools): the
    /// luma plane then the interleaved chroma plane, each of the coded
    /// extent, tightly packed.
    ///
    /// It waits for every decode the decoder has submitted, so that no
    /// decode reads the image while the copy moves its layout, and blocks
    /// until the copy is done.
    pub fn read_back(&self, device: &Arc<VideoDevice>) -> Result<Nv12, Error> {
        let family = device
            .transfer_family()
            .ok_or_else(|| Error::Unsupported("no queue of ours copies".into()))?;
        let timeline = &self.inner.timeline;
        let extent = self.inner.extent;
        let (width, height) = (u64::from(extent.width), u64::from(extent.height));
        let luma = width * height;
        let size = luma + luma / 2;
        let mut copy = ReadBack::new(device, family, size)?;
        copy.record(self.surface().image(), extent, luma)?;
        copy.submit(timeline, timeline.last_submitted())?;
        copy.wait()?;
        let bytes = copy.bytes(size)?;
        let luma = usize::try_from(luma).map_err(|_| Error::Unsupported("a huge frame".into()))?;
        let (y, uv) = bytes.split_at(luma.min(bytes.len()));
        Ok(Nv12 {
            width: extent.width,
            height: extent.height,
            y: y.to_vec(),
            uv: uv.to_vec(),
        })
    }
}

impl<T: SurfaceImage> Drop for Inner<T> {
    fn drop(&mut self) {
        // SAFETY: the drop runs once, and nothing reads `held` after it.
        let held = unsafe { ManuallyDrop::take(&mut self.held) };
        self.timeline
            .retire(self.last_use.load(Ordering::Acquire), Box::new(held));
    }
}

/// A frame's planes in host memory, of the coded extent.
pub struct Nv12 {
    /// The width in luma samples.
    pub width: u32,
    /// The height in luma samples.
    pub height: u32,
    /// The luma plane, `width` bytes a row.
    pub y: Vec<u8>,
    /// The chroma plane, Cb and Cr interleaved, `width` bytes a row and
    /// half as many rows.
    pub uv: Vec<u8>,
}

/// One read back's objects, destroyed as it goes.
struct ReadBack<'a> {
    device: &'a Arc<VideoDevice>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    fence: vk::Fence,
    family: u32,
    /// Whether the copy was submitted, and so signals the fence.
    submitted: bool,
}

impl<'a> ReadBack<'a> {
    fn new(device: &'a Arc<VideoDevice>, family: u32, size: u64) -> Result<Self, Error> {
        let vk_device = device.device();
        let mut copy = Self {
            device,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            fence: vk::Fence::null(),
            family,
            submitted: false,
        };
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(family);
        // SAFETY: `info` lives through the call; so for each call below.
        copy.pool =
            unsafe { vk_device.create_command_pool(&info, None) }.named("vkCreateCommandPool")?;
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(copy.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: as above.
        let commands = unsafe { vk_device.allocate_command_buffers(&info) }
            .named("vkAllocateCommandBuffers")?;
        copy.command = commands.first().copied().unwrap_or_default();
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: as above.
        copy.buffer = unsafe { vk_device.create_buffer(&info, None) }.named("vkCreateBuffer")?;
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(copy.buffer) };
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        copy.memory = device.allocate(
            &requirements,
            &[host | vk::MemoryPropertyFlags::HOST_CACHED, host],
            None,
        )?;
        // SAFETY: the memory is unbound and of the buffer's requirements.
        unsafe { vk_device.bind_buffer_memory(copy.buffer, copy.memory, 0) }
            .named("vkBindBufferMemory")?;
        // SAFETY: as above.
        copy.fence = unsafe { vk_device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .named("vkCreateFence")?;
        Ok(copy)
    }

    /// Records the copy of `image`'s two planes into the buffer, the
    /// chroma plane at `luma`, moving the image out of the DPB layout and
    /// back where the layouts are not unified.
    fn record(&self, image: vk::Image, extent: vk::Extent2D, luma: u64) -> Result<(), Error> {
        let vk_device = self.device.device();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is ours and not recording.
        unsafe { vk_device.begin_command_buffer(self.command, &begin) }
            .named("vkBeginCommandBuffer")?;
        let unified = self.device.unified_image_layouts();
        let (held, copied) = if unified {
            (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
        } else {
            (
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )
        };
        let barrier = |old, new, src_stage, src_access, dst_stage, dst_access| {
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
        let before = [barrier(
            held,
            copied,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        )];
        // SAFETY: recording into our command buffer; the barrier's arrays
        // live through the call.
        unsafe {
            vk_device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
        }
        let plane = |aspect, offset, width, height| {
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
        };
        let regions = [
            plane(
                vk::ImageAspectFlags::PLANE_0,
                0,
                extent.width,
                extent.height,
            ),
            plane(
                vk::ImageAspectFlags::PLANE_1,
                luma,
                extent.width / 2,
                extent.height / 2,
            ),
        ];
        // SAFETY: as above; the image is in `copied`, and the buffer holds
        // both planes.
        unsafe {
            vk_device.cmd_copy_image_to_buffer(self.command, image, copied, self.buffer, &regions);
        }
        let after = [barrier(
            copied,
            held,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::NONE,
        )];
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        let dependency = vk::DependencyInfo::default().memory_barriers(&to_host);
        let dependency = if unified {
            dependency
        } else {
            dependency.image_memory_barriers(&after)
        };
        // SAFETY: as above.
        unsafe { vk_device.cmd_pipeline_barrier2(self.command, &dependency) };
        // SAFETY: as above.
        unsafe { vk_device.end_command_buffer(self.command) }.named("vkEndCommandBuffer")
    }

    /// Submits the copy, after the decoder's submission of `value`.
    fn submit(&mut self, timeline: &Timeline, value: u64) -> Result<(), Error> {
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(self.command)];
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .wait_semaphore_infos(&waits)];
        self.device
            .submit_transfer(self.family, &submit, self.fence)?;
        self.submitted = true;
        Ok(())
    }

    fn wait(&self) -> Result<(), Error> {
        let nanos = u64::try_from(READ_BACK_WAIT.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the fence is ours, and the submission signals it.
        match unsafe {
            self.device
                .device()
                .wait_for_fences(&[self.fence], true, nanos)
        } {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => Err(Error::Timeout),
            Err(result) => Err(Error::Vulkan("vkWaitForFences", result)),
        }
    }

    /// The buffer's first `size` bytes.
    fn bytes(&self, size: u64) -> Result<Vec<u8>, Error> {
        let vk_device = self.device.device();
        // SAFETY: the memory is host-visible, unmapped, and the copy into
        // it is done.
        let mapped =
            unsafe { vk_device.map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty()) }
                .named("vkMapMemory")?;
        let len = usize::try_from(size).map_err(|_| Error::Unsupported("a huge frame".into()))?;
        // SAFETY: the mapping is `size` bytes, and nothing writes them.
        let bytes = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), len) }.to_vec();
        // SAFETY: the memory is mapped, and the slice of it is gone.
        unsafe { vk_device.unmap_memory(self.memory) };
        Ok(bytes)
    }
}

impl Drop for ReadBack<'_> {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        // A copy that did not finish keeps what it uses.
        // SAFETY: the fence is ours.
        let done =
            !self.submitted || unsafe { vk_device.get_fence_status(self.fence) }.unwrap_or(false);
        if !done {
            log::error!("a read back did not finish: leaking its buffer");
            return;
        }
        // SAFETY: each object is ours and no GPU work uses it; destroying a
        // null handle does nothing.
        unsafe { vk_device.destroy_fence(self.fence, None) };
        // SAFETY: as above.
        unsafe { vk_device.destroy_buffer(self.buffer, None) };
        // SAFETY: as above.
        unsafe { vk_device.free_memory(self.memory, None) };
        // SAFETY: as above; its command buffers go with it.
        unsafe { vk_device.destroy_command_pool(self.pool, None) };
    }
}
