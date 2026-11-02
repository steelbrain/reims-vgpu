//! A decoded picture: the output it landed in, the DPB slot its decode set
//! up, and the timeline value after which the output holds it.

use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::time::Duration;

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::profile::P010_FORMAT;
use crate::surface::{Landing, Surface};
use crate::timeline::Timeline;

/// How long a read back waits for its picture.
const READ_BACK_WAIT: Duration = Duration::from_secs(10);

/// A picture a decoder made: the output it landed in, and the timeline
/// value after which the output holds it.
///
/// Its samples as a reference are in the decoder's DPB picture of its
/// slot. A stand-in for a lost H.264 frame shares its source's slot
/// ([`reims_vgpu_video_codec::h264::Accelerator::conceal`]): a DPB picture
/// is written only by the decode that sets its slot up, which no slot a
/// reference holds is, so the stand-in holds what its source holds for as
/// long as it is a reference.
pub struct Picture<T: Surface> {
    inner: Arc<Inner<T>>,
    stand_in: bool,
}

struct Inner<T: Surface> {
    timeline: Arc<Timeline>,
    /// Taken as it goes, to wait for the GPU in the timeline.
    held: ManuallyDrop<Held<T>>,
    extent: vk::Extent2D,
    /// The format it was decoded in: NV12 or P010.
    format: vk::Format,
    /// The DPB slot its decode set up, -1 before it.
    slot: AtomicI32,
    /// The value after which the output holds the picture, 0 before its
    /// decode is submitted.
    landed: AtomicU64,
}

/// What a picture holds that GPU work may still use as it goes.
struct Held<T> {
    surface: T,
    /// The output's decode view, where a decode writes it.
    view: Option<ImageView>,
}

/// An image view, destroyed as it goes.
pub(crate) struct ImageView {
    device: Arc<VideoDevice>,
    view: vk::ImageView,
}

impl ImageView {
    /// A view of the whole of `image`, of `format`, for decoding into and
    /// predicting from, of the decode usages `usage` alone: a sampled view
    /// of a two-plane format would need a sampler conversion.
    pub(crate) fn for_decode(
        device: &Arc<VideoDevice>,
        image: vk::Image,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, Error> {
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(usage);
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range())
            .push_next(&mut usage);
        // SAFETY: `info` and its chain live through the call, and the image
        // is the device's.
        let view =
            unsafe { device.device().create_image_view(&info, None) }.named("vkCreateImageView")?;
        Ok(Self {
            device: Arc::clone(device),
            view,
        })
    }

    pub(crate) fn handle(&self) -> vk::ImageView {
        self.view
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

impl<T: Surface> Picture<T> {
    /// A picture of `extent` in `format`, landing in `surface`.
    pub(crate) fn new(
        timeline: &Arc<Timeline>,
        surface: T,
        view: Option<ImageView>,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                timeline: Arc::clone(timeline),
                held: ManuallyDrop::new(Held { surface, view }),
                extent,
                format,
                slot: AtomicI32::new(-1),
                landed: AtomicU64::new(0),
            }),
            stand_in: false,
        }
    }

    /// Another picture of this one's slot, standing in for a lost frame.
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

    /// The output it landed in.
    pub fn surface(&self) -> &T {
        &self.inner.held.surface
    }

    /// The coded extent decoded: the frame before cropping, from the
    /// output's origin.
    pub fn coded_extent(&self) -> vk::Extent2D {
        self.inner.extent
    }

    /// The format it was decoded in, and its output has:
    /// [`crate::PICTURE_FORMAT`] (NV12) or [`crate::P010_FORMAT`].
    pub fn format(&self) -> vk::Format {
        self.inner.format
    }

    /// The bytes of one of its samples: 1 in NV12, 2 in P010.
    pub fn sample_bytes(&self) -> u32 {
        if self.inner.format == P010_FORMAT {
            2
        } else {
            1
        }
    }

    /// The decoder's timeline semaphore.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.inner.timeline.semaphore()
    }

    /// The value [`Self::semaphore`] reaches once the output holds the
    /// picture: graphics work that samples it waits for it.
    pub fn value(&self) -> u64 {
        self.inner.landed.load(Ordering::Acquire)
    }

    /// Blocks until the output holds the picture, for `timeout` at most.
    pub fn wait(&self, timeout: Duration) -> Result<(), Error> {
        self.inner.timeline.wait(self.value(), timeout)
    }

    /// The output's decode view, where a decode writes it.
    pub(crate) fn output_view(&self) -> Option<vk::ImageView> {
        self.inner.held.view.as_ref().map(ImageView::handle)
    }

    /// The DPB slot its decode set up, `None` before it.
    pub(crate) fn slot(&self) -> Option<i32> {
        let slot = self.inner.slot.load(Ordering::Acquire);
        (slot >= 0).then_some(slot)
    }

    /// Records its decode: the slot it set up, and the value after which
    /// its output holds it.
    pub(crate) fn decoded(&self, slot: i32, landed: u64) {
        self.inner.slot.store(slot, Ordering::Release);
        self.inner.landed.store(landed, Ordering::Release);
    }

    /// The output image's planes, copied to host memory (tests and
    /// tools): the luma plane then the interleaved chroma plane, each of
    /// the coded extent, tightly packed. It blocks until the copy is done,
    /// which runs on the copy queue once the output holds the picture. A
    /// buffer landing is refused: the caller reads its own buffer.
    pub fn read_back(&self, device: &Arc<VideoDevice>) -> Result<Nv12, Error> {
        let extent = self.inner.extent;
        let sample_bytes = self.sample_bytes();
        let (width, height) = (u64::from(extent.width), u64::from(extent.height));
        let luma = width * height * u64::from(sample_bytes);
        let size = luma + luma / 2;
        let Landing::Image(image) = self.surface().landing() else {
            return Err(Error::Unsupported(
                "a read back of a picture that landed in a buffer, which holds it".into(),
            ));
        };
        let mut copy = ReadBack::new(device, size)?;
        copy.record(image, extent, luma)?;
        copy.submit(&self.inner.timeline, self.value())?;
        copy.wait()?;
        let bytes = copy.bytes(size)?;
        let luma = usize::try_from(luma).map_err(|_| Error::Unsupported("a huge frame".into()))?;
        let (y, uv) = bytes.split_at(luma.min(bytes.len()));
        Ok(Nv12 {
            width: extent.width,
            height: extent.height,
            sample_bytes,
            y: y.to_vec(),
            uv: uv.to_vec(),
        })
    }
}

impl<T: Surface> Drop for Inner<T> {
    fn drop(&mut self) {
        // SAFETY: the drop runs once, and nothing reads `held` after it.
        let held = unsafe { ManuallyDrop::take(&mut self.held) };
        // The output is used by its picture's own work alone.
        self.timeline
            .retire(self.landed.load(Ordering::Acquire), Box::new(held));
    }
}

/// A frame's planes in host memory, of the coded extent: NV12, or P010
/// where its samples are of 2 bytes, each a little-endian 16-bit value
/// whose high 10 bits are the sample.
pub struct Nv12 {
    /// The width in luma samples.
    pub width: u32,
    /// The height in luma samples.
    pub height: u32,
    /// The bytes of a sample: 1 (NV12) or 2 (P010).
    pub sample_bytes: u32,
    /// The luma plane, `width * sample_bytes` bytes a row.
    pub y: Vec<u8>,
    /// The chroma plane, Cb and Cr interleaved, `width * sample_bytes`
    /// bytes a row and half as many rows.
    pub uv: Vec<u8>,
}

/// A two-plane 4:2:0 image's planes, NV12's or P010's: each one's aspect
/// and extent in its own texels.
pub(crate) fn planes(extent: vk::Extent2D) -> [(vk::ImageAspectFlags, vk::Extent3D); 2] {
    [
        (
            vk::ImageAspectFlags::PLANE_0,
            vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
        ),
        (
            vk::ImageAspectFlags::PLANE_1,
            vk::Extent3D {
                width: extent.width / 2,
                height: extent.height / 2,
                depth: 1,
            },
        ),
    ]
}

/// One plane's subresource.
pub(crate) fn plane(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: aspect,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// One read back's objects, destroyed as it goes.
struct ReadBack<'a> {
    device: &'a Arc<VideoDevice>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    fence: vk::Fence,
    /// Whether the copy was submitted, and so signals the fence.
    submitted: bool,
}

impl<'a> ReadBack<'a> {
    fn new(device: &'a Arc<VideoDevice>, size: u64) -> Result<Self, Error> {
        let vk_device = device.device();
        let mut copy = Self {
            device,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            fence: vk::Fence::null(),
            submitted: false,
        };
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.copy_family());
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

    /// Records the copy of `image`'s two planes, in `GENERAL` as every
    /// output is left, into the buffer, tightly packed, the chroma plane
    /// at `luma`.
    fn record(&self, image: vk::Image, extent: vk::Extent2D, luma: u64) -> Result<(), Error> {
        let vk_device = self.device.device();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is ours and not recording.
        unsafe { vk_device.begin_command_buffer(self.command, &begin) }
            .named("vkBeginCommandBuffer")?;
        let regions = planes(extent).map(|(aspect, size)| {
            let offset = if aspect == vk::ImageAspectFlags::PLANE_0 {
                0
            } else {
                luma
            };
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(plane(aspect))
                .image_extent(size)
        });
        // SAFETY: recording into our command buffer; the image is in
        // `GENERAL`, the semaphore wait orders the copy after the work
        // that wrote it, and the buffer holds both planes.
        unsafe {
            vk_device.cmd_copy_image_to_buffer(
                self.command,
                image,
                vk::ImageLayout::GENERAL,
                self.buffer,
                &regions,
            );
        }
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: as above, with the array alive through the call.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
        }
        // SAFETY: as above.
        unsafe { vk_device.end_command_buffer(self.command) }.named("vkEndCommandBuffer")
    }

    /// Submits the copy, after the decoder's timeline reaches `value`.
    fn submit(&mut self, timeline: &Timeline, value: u64) -> Result<(), Error> {
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(self.command)];
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .wait_semaphore_infos(&waits)];
        self.device.submit_copy(&submit, self.fence)?;
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
