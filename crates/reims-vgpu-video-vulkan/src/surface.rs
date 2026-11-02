//! What a decoder decodes into: images its caller supplies, one a
//! picture, which the decoder never copies. In the product each is the GPU
//! copy of a guest IOSurface, which the guest's Metal then samples per
//! plane, through R8 and RG8 views.

use std::sync::Arc;

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::profile::{H264Profile, PICTURE_FORMAT, SURFACE_FLAGS, SURFACE_USAGE};

/// An image a decoder decodes one picture into, which is also that
/// picture's DPB slot (`VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR`).
///
/// It is made as [`DecodeImage::new`] makes one: 2D, one mip level and
/// one layer, `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`, optimal tiling, at
/// least the extent asked for; usage `VIDEO_DECODE_DST`, `VIDEO_DECODE_DPB`,
/// `SAMPLED` and `TRANSFER_SRC`, flags `MUTABLE_FORMAT` and
/// `EXTENDED_USAGE`; a `VkVideoProfileListInfoKHR` of the profile asked
/// for in its chain; shared `CONCURRENT` between the device's graphics and
/// decode families (exclusive where they are one), and bound to memory.
///
/// **Layouts.** Each decode takes its target from `UNDEFINED`, dropping
/// what it held. Where the device has `unifiedImageLayoutsVideo` enabled
/// ([`VideoDevice::unified_image_layouts`]) the decode leaves it in
/// `GENERAL`, and it may be sampled and copied there. Elsewhere it leaves
/// it in `VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR`, and every image a later
/// decode predicts from must be in that layout when that decode runs: a
/// reader that moves it to another layout moves it back first.
pub trait SurfaceImage: Send + Sync + 'static {
    /// The image.
    fn image(&self) -> vk::Image;
}

/// Where a decoder's surfaces come from: one a picture, as the decoder
/// starts it, of the extent and profile the stream's session decodes.
///
/// A surface is decoded into once, when it is given, and read while its
/// picture lives: a source gives no image that a live picture holds.
pub trait SurfaceSource: Send {
    /// What it gives.
    type Surface: SurfaceImage;

    /// A surface of at least `extent`, for `profile`.
    fn surface(
        &mut self,
        extent: vk::Extent2D,
        profile: H264Profile,
    ) -> Result<Self::Surface, Error>;
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

    fn surface(
        &mut self,
        extent: vk::Extent2D,
        profile: H264Profile,
    ) -> Result<DecodeImage, Error> {
        DecodeImage::new(&self.device, extent, profile)
    }
}

/// An image made as [`SurfaceImage`] says, with its own memory.
pub struct DecodeImage {
    device: Arc<VideoDevice>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
}

impl DecodeImage {
    /// An image of `extent` for `profile`, in `UNDEFINED` layout.
    pub fn new(
        device: &Arc<VideoDevice>,
        extent: vk::Extent2D,
        profile: H264Profile,
    ) -> Result<Self, Error> {
        let families = device.sharing_families();
        let image = profile.with_list(|list| {
            let mut info = vk::ImageCreateInfo::default()
                .flags(SURFACE_FLAGS)
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
                .usage(SURFACE_USAGE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(list);
            info = if families.len() > 1 {
                info.sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families)
            } else {
                info.sharing_mode(vk::SharingMode::EXCLUSIVE)
            };
            // SAFETY: `info` and its chain live through the call.
            unsafe { device.device().create_image(&info, None) }.named("vkCreateImage")
        })?;
        let made = Self {
            device: Arc::clone(device),
            image,
            memory: vk::DeviceMemory::null(),
            extent,
        };
        made.bind()
    }

    /// Binds memory of its own to the image.
    fn bind(mut self) -> Result<Self, Error> {
        let vk_device = self.device.device();
        // SAFETY: the image is the device's.
        let requirements = unsafe { vk_device.get_image_memory_requirements(self.image) };
        self.memory = self.device.allocate(
            &requirements,
            &[
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            ],
            Some(self.image),
        )?;
        // SAFETY: the memory was allocated for the image, which is bound to
        // none yet.
        unsafe { vk_device.bind_image_memory(self.image, self.memory, 0) }
            .named("vkBindImageMemory")?;
        Ok(self)
    }

    /// The extent it was made at.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl SurfaceImage for DecodeImage {
    fn image(&self) -> vk::Image {
        self.image
    }
}

impl Drop for DecodeImage {
    fn drop(&mut self) {
        // SAFETY: the image is the device's, and no GPU work uses it:
        // a decoder lets its surfaces go only once the work that used them
        // is done.
        unsafe { self.device.device().destroy_image(self.image, None) };
        // SAFETY: its memory, now bound to nothing.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}
