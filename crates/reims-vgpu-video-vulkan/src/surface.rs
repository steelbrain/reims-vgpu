//! Where a decoder's pictures land: outputs its caller supplies, one a
//! picture. In the product each is the GPU copy of a guest IOSurface,
//! which the guest's Metal then samples per plane, through R8 and RG8
//! views. The decoder's references are DPB pictures of its own, never an
//! output: a guest recycles an output once it has shown it, whether or not
//! the picture is still a reference.

use std::sync::Arc;

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::profile::{H264Profile, OUTPUT_FLAGS, OUTPUT_USAGE, PICTURE_FORMAT};

/// How a decoder's outputs are made, which it gives its
/// [`SurfaceSource`] with each request.
///
/// An output is 2D, one mip level and one layer,
/// `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`, optimally tiled, at least
/// `extent`, with [`OUTPUT_FLAGS`] and the usage [`Self::usage`] gives,
/// shared `CONCURRENT` between the families
/// [`Self::families`] gives (exclusive where that is one), and bound to
/// memory. Where `decode_target` is set, a decode writes it, and its
/// create info chains a `VkVideoProfileListInfoKHR` of `profile` alone
/// ([`H264Profile::with_list`]); else a copy does, and it has none.
/// [`Self::with_create_info`] builds that create info.
///
/// **Layouts.** Each picture takes its output from `UNDEFINED`, dropping
/// what it held, and leaves it in `VK_IMAGE_LAYOUT_GENERAL`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SurfaceSpec {
    /// The coded extent each picture is decoded at.
    pub extent: vk::Extent2D,
    /// The stream's profile.
    pub profile: H264Profile,
    /// Whether a decode writes the output itself
    /// (`DPB_AND_OUTPUT_DISTINCT`), rather than a copy out of the DPB.
    pub decode_target: bool,
}

impl SurfaceSpec {
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
            .format(PICTURE_FORMAT)
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

/// An output a decoder's picture lands in, made as [`SurfaceSpec`] says.
pub trait SurfaceImage: Send + Sync + 'static {
    /// The image.
    fn image(&self) -> vk::Image;
}

/// Where a decoder's outputs come from: one a picture, as the decoder
/// starts it.
///
/// An output is written once, by its picture's decode or copy, and read
/// while that picture lives: a source gives no image a live picture holds.
pub trait SurfaceSource: Send {
    /// What it gives.
    type Surface: SurfaceImage;

    /// An output made as `spec` says.
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

impl SurfaceImage for DecodeImage {
    fn image(&self) -> vk::Image {
        self.image
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
