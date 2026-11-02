//! Where a decoder's pictures land: outputs its caller supplies, one a
//! picture, each an image or two planes in a buffer ([`Landing`]), NV12
//! at 8 bits and P010 at 10. In the product an image is the GPU copy of a
//! guest IOSurface, which the guest's Metal then samples per plane,
//! through R8 and RG8 views (R16 and RG16 for P010), and a buffer the
//! IOSurface's own bytes in host memory. The decoder's
//! references are DPB pictures of its own, never an output: a guest
//! recycles an output once it has shown it, whether or not the picture is
//! still a reference.

use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::profile::{OUTPUT_FLAGS, OUTPUT_USAGE, VideoProfile};

/// How a decoder's output images are made, which it gives its
/// [`SurfaceSource`] with each request.
///
/// An output image is 2D, one mip level and one layer, of the profile's
/// format ([`Self::format`]: `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` at 8
/// bits, `VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` at 10),
/// optimally tiled, at least `extent`, with [`OUTPUT_FLAGS`] and the usage
/// [`Self::usage`] gives, shared `CONCURRENT` between the families
/// [`Self::families`] gives (exclusive where that is one), and bound to
/// memory. Where `decode_target` is set, a decode writes it, and its
/// create info chains a `VkVideoProfileListInfoKHR` of `profile` alone
/// ([`VideoProfile::with_list`]); else a copy does, and it has none.
/// [`Self::with_create_info`] builds that create info. Planes in a buffer
/// are of the same format, a sample [`VideoProfile::sample_bytes`] bytes.
///
/// **Layouts.** Each picture takes its output from `UNDEFINED`, dropping
/// what it held, and leaves it in `VK_IMAGE_LAYOUT_GENERAL`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SurfaceSpec {
    /// The extent of an output image: the coded extent each picture is
    /// decoded at, rounded up to the profile's picture access granularity.
    pub extent: vk::Extent2D,
    /// The stream's profile.
    pub profile: VideoProfile,
    /// Whether a decode writes the output itself
    /// (`DPB_AND_OUTPUT_DISTINCT`), rather than a copy out of the DPB.
    pub decode_target: bool,
}

impl SurfaceSpec {
    /// The output's format: NV12, or P010 at 10 bits.
    pub fn format(&self) -> vk::Format {
        self.profile.format()
    }

    /// The output's usage: [`OUTPUT_USAGE`], with `VIDEO_DECODE_DST`
    /// where a decode writes it.
    pub fn usage(&self) -> vk::ImageUsageFlags {
        if self.decode_target {
            OUTPUT_USAGE | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
        } else {
            OUTPUT_USAGE
        }
    }

    /// The families an output is shared between on `device`: the
    /// graphics family, the decode family and the copy family, each once.
    pub fn families(device: &VideoDevice) -> Vec<u32> {
        device.output_families()
    }

    /// Calls `f` with an output's create info on `device`, its chain
    /// alive for the call.
    pub fn with_create_info<R>(
        &self,
        device: &VideoDevice,
        f: impl FnOnce(&vk::ImageCreateInfo<'_>) -> R,
    ) -> R {
        let families = Self::families(device);
        let info = vk::ImageCreateInfo::default()
            .flags(OUTPUT_FLAGS)
            .image_type(vk::ImageType::TYPE_2D)
            .format(self.format())
            .extent(vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(self.usage())
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let info = if families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        if self.decode_target {
            self.profile.with_list(|list| f(&info.push_next(list)))
        } else {
            f(&info)
        }
    }
}

/// An output a decoder's picture lands in.
pub trait Surface: Send + Sync + 'static {
    /// Where the picture lands.
    fn landing(&self) -> Landing;
}

/// Where a picture lands.
#[derive(Clone, Copy)]
pub enum Landing {
    /// An image made as [`SurfaceSpec`] says, which holds the whole coded
    /// frame from its origin.
    Image(vk::Image),
    /// The frame's two planes, cropped to the SPS's rectangle, in a buffer
    /// of the caller's, written by one GPU copy out of the picture.
    Buffer(BufferPlanes),
}

/// NV12 or P010 planes in a buffer: the luma plane's rows of the crop's
/// width, then the interleaved chroma plane's half as many rows of the
/// same width in bytes, each plane from its first byte, one row each pitch
/// after the last. A sample is a byte in NV12, and two in P010, each a
/// little-endian 16-bit value whose high 10 bits are the sample.
///
/// The buffer has `TRANSFER_DST` usage and is bound to memory; it is
/// shared `CONCURRENT` between the families [`SurfaceSpec::families`]
/// gives where they are more than one, as an output image is. Each
/// offset is a multiple of 4, and each pitch at least the crop's row in
/// bytes and a whole number of the plane's texels (a sample of luma, a
/// pair of chroma), as Vulkan's copies take them where the queue that
/// copies neither draws nor computes; and each plane's rows of the crop
/// lie in the buffer's `size` bytes, which a picture whose crop does not
/// fit is refused for.
///
/// The copy makes its writes available to the host before the picture's
/// value is signalled, so that a host that has waited for it reads them
/// from coherent memory.
#[derive(Clone, Copy)]
pub struct BufferPlanes {
    /// The buffer.
    pub buffer: vk::Buffer,
    /// The luma plane: its first byte, and the bytes from one row to the
    /// next.
    pub luma: (u64, u32),
    /// The chroma plane: its first byte, and the bytes from one row to the
    /// next.
    pub chroma: (u64, u32),
    /// The buffer's bytes.
    pub size: u64,
}

impl BufferPlanes {
    /// Whether planes cropped to `width` by `height`, of samples of
    /// `sample_bytes` bytes, fit as Vulkan copies them, and in the buffer,
    /// or why not.
    pub(crate) fn check(&self, width: u32, height: u32, sample_bytes: u32) -> Result<(), Error> {
        let (luma, chroma) = (self.luma, self.chroma);
        if luma.0 % 4 != 0 || chroma.0 % 4 != 0 {
            return Err(Error::Unsupported(format!(
                "planes at {} and {}: each offset is a multiple of 4",
                luma.0, chroma.0
            )));
        }
        let sample = sample_bytes.max(1);
        let pair = 2 * sample;
        // A row of the crop in bytes, chroma a whole number of pairs.
        let (Some(luma_width), Some(chroma_width)) = (
            width.checked_mul(sample),
            width.div_ceil(2).checked_mul(pair),
        ) else {
            return Err(Error::Unsupported(format!("a frame {width} wide")));
        };
        if luma.1 < luma_width
            || chroma.1 < chroma_width
            || luma.1 % sample != 0
            || chroma.1 % pair != 0
        {
            return Err(Error::Unsupported(format!(
                "pitches {} and {} for a frame {width} wide of {sample}-byte samples: each at \
                 least its row, of whole samples, the chroma pitch of whole pairs",
                luma.1, chroma.1
            )));
        }
        // Where a plane's last row of the crop ends.
        let end = |(offset, pitch): (u64, u32), rows: u32, row: u32| {
            let before_last = u64::from(pitch).checked_mul(u64::from(rows.saturating_sub(1)))?;
            offset.checked_add(before_last)?.checked_add(u64::from(row))
        };
        let fits = |plane, rows, row| end(plane, rows, row).is_some_and(|end| end <= self.size);
        if !fits(luma, height, luma_width) || !fits(chroma, height.div_ceil(2), chroma_width) {
            return Err(Error::Unsupported(format!(
                "a frame {width} by {height} at {} and {} past a buffer of {} bytes",
                luma.0, chroma.0, self.size
            )));
        }
        Ok(())
    }
}

/// Where a decoder's outputs come from: one a picture, as the decoder
/// starts it.
///
/// An output is written once, by its picture's decode or copy, and read
/// while that picture lives: a source gives no output a live picture holds.
pub trait SurfaceSource: Send {
    /// What it gives.
    type Surface: Surface;

    /// An output for a picture of `spec`: an image made as it says, or
    /// planes in a buffer.
    fn surface(&mut self, spec: &SurfaceSpec) -> Result<Self::Surface, Error>;
}

/// A [`SurfaceSource`] that makes a new [`DecodeImage`] for each picture,
/// for the tests and the tools.
pub struct DecodeImages {
    device: Arc<VideoDevice>,
}

impl DecodeImages {
    /// Images on `device`.
    pub fn new(device: &Arc<VideoDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }
}

impl SurfaceSource for DecodeImages {
    type Surface = DecodeImage;

    fn surface(&mut self, spec: &SurfaceSpec) -> Result<DecodeImage, Error> {
        DecodeImage::new(&self.device, spec)
    }
}

/// An image made as [`SurfaceSpec`] says, with its own memory.
pub struct DecodeImage {
    device: Arc<VideoDevice>,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl DecodeImage {
    /// An output made as `spec` says, in `UNDEFINED` layout.
    pub fn new(device: &Arc<VideoDevice>, spec: &SurfaceSpec) -> Result<Self, Error> {
        let image = spec.with_create_info(device, |info| {
            // SAFETY: `info` and its chain live through the call.
            unsafe { device.device().create_image(info, None) }.named("vkCreateImage")
        })?;
        let mut made = Self {
            device: Arc::clone(device),
            image,
            memory: vk::DeviceMemory::null(),
        };
        made.memory = bind_image_memory(device, image)?;
        Ok(made)
    }
}

/// Binds dedicated device-local memory to `image`, which has none.
pub(crate) fn bind_image_memory(
    device: &VideoDevice,
    image: vk::Image,
) -> Result<vk::DeviceMemory, Error> {
    let vk_device = device.device();
    // SAFETY: the image is the device's.
    let requirements = unsafe { vk_device.get_image_memory_requirements(image) };
    let memory = device.allocate(
        &requirements,
        &[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        ],
        Some(image),
    )?;
    // SAFETY: the memory was allocated for the image, which is bound to
    // none yet.
    if let Err(error) = unsafe { vk_device.bind_image_memory(image, memory, 0) } {
        // SAFETY: the memory is bound to nothing.
        unsafe { vk_device.free_memory(memory, None) };
        return Err(Error::Vulkan("vkBindImageMemory", error));
    }
    Ok(memory)
}

impl Surface for DecodeImage {
    fn landing(&self) -> Landing {
        Landing::Image(self.image)
    }
}

impl Drop for DecodeImage {
    fn drop(&mut self) {
        // SAFETY: the image is the device's, and no GPU work uses it:
        // a decoder lets its outputs go only once the work that wrote them
        // is done.
        unsafe { self.device.device().destroy_image(self.image, None) };
        // SAFETY: its memory, now bound to nothing; a null handle is
        // nothing.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}

/// A [`SurfaceSource`] of host-visible buffers, one a picture, for the
/// tests and tools: each holds the luma plane at byte 256 and the chroma
/// plane after it, each row 64 bytes past the coded row's bytes rounded up
/// to 64, so that the planes' offsets and pitches are not the frame's own.
pub struct HostBuffers {
    device: Arc<VideoDevice>,
}

impl HostBuffers {
    /// Buffers on `device`.
    pub fn new(device: &Arc<VideoDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }
}

impl SurfaceSource for HostBuffers {
    type Surface = HostBuffer;

    fn surface(&mut self, spec: &SurfaceSpec) -> Result<HostBuffer, Error> {
        HostBuffer::new(&self.device, spec.extent, spec.profile.sample_bytes())
    }
}

/// A host-visible, host-coherent buffer of NV12 or P010 planes, mapped
/// while it lives.
pub struct HostBuffer {
    device: Arc<VideoDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: NonNull<u8>,
    size: usize,
    planes: BufferPlanes,
}

// SAFETY: the mapping is the buffer's memory alone, which GPU work writes
// once and the holder then reads.
unsafe impl Send for HostBuffer {}
// SAFETY: as above; `bytes` reads through `&self` alone.
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    /// Planes for frames of `extent`, cropped or not, of samples of
    /// `sample_bytes` bytes.
    pub fn new(
        device: &Arc<VideoDevice>,
        extent: vk::Extent2D,
        sample_bytes: u32,
    ) -> Result<Self, Error> {
        let pitch = (extent.width * sample_bytes.max(1)).next_multiple_of(64) + 64;
        let rows = u64::from(extent.height);
        let luma = 256u64;
        let chroma = luma + rows * u64::from(pitch) + 128;
        let size = chroma + rows.div_ceil(2) * u64::from(pitch);
        let families = SurfaceSpec::families(device);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        let info = if families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        let vk_device = device.device();
        // SAFETY: `info` lives through the call.
        let buffer = unsafe { vk_device.create_buffer(&info, None) }.named("vkCreateBuffer")?;
        let mut made = Self {
            device: Arc::clone(device),
            buffer,
            memory: vk::DeviceMemory::null(),
            mapped: NonNull::dangling(),
            size: usize::try_from(size)
                .map_err(|_| Error::Unsupported("a buffer past the address space".into()))?,
            planes: BufferPlanes {
                buffer,
                luma: (luma, pitch),
                chroma: (chroma, pitch),
                size,
            },
        };
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        made.memory = device.allocate(
            &requirements,
            &[host | vk::MemoryPropertyFlags::HOST_CACHED, host],
            None,
        )?;
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

    /// Where the planes are.
    pub fn planes(&self) -> BufferPlanes {
        self.planes
    }

    /// The buffer's bytes: the picture's planes once its value is
    /// reached ([`crate::Picture::wait`]).
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is `size` bytes of host-coherent memory,
        // which GPU work writes before the picture's value alone.
        unsafe { std::slice::from_raw_parts(self.mapped.as_ptr(), self.size) }
    }
}

impl Surface for HostBuffer {
    fn landing(&self) -> Landing {
        Landing::Buffer(self.planes)
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        // SAFETY: the buffer is the device's, and no GPU work uses it: a
        // decoder lets its outputs go only once the work that wrote them
        // is done. A null handle is nothing.
        unsafe { vk_device.destroy_buffer(self.buffer, None) };
        // SAFETY: its memory, unmapped as it is freed.
        unsafe { vk_device.free_memory(self.memory, None) };
    }
}
