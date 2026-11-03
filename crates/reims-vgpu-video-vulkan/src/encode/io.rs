//! What an encoder reads and writes: each frame's source, which it reads
//! where it lies ([`EncodeSource`]), and each picture's bitstream, which it
//! writes straight into a buffer its caller supplies ([`BitstreamSource`]).
//!
//! **Sources.** A source is an image the caller made as [`SourceSpec`]
//! says, which the encoder reads in place: no copy. In the product it is
//! the GPU copy of a guest IOSurface. Or it is two planes in a buffer
//! ([`crate::BufferPlanes`]), such as an IOSurface's own pages imported by
//! their pointer ([`ImportedSource`]): devices encode only from optimally
//! tiled images (the RTX 5080 offers NV12 and P010 sources `OPTIMAL` alone,
//! 11-03), so one GPU copy, in the encode's own command buffer where the
//! encode family copies, takes the planes into a picture of the encoder's.
//! Neither costs a copy on the CPU.
//!
//! **Bitstreams.** Each picture's NAL units go into a buffer the caller's
//! [`BitstreamSource`] gives for it, from an offset the device's alignment
//! allows: host pages imported by their pointer ([`ImportedBitstream`]), so
//! that the guest reads them where the GPU wrote them, or any buffer made as
//! [`BitstreamSpec`] says.

use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use crate::device::VideoDevice;
use crate::encode::caps::EncodeProfile;
use crate::error::{Error, Named};
use crate::import::{AnyProfile, BufferUse, HostImport, ImportBuffer, buffer_over, plane_span};
use crate::surface::BufferPlanes;

/// How a source image an encoder reads in place is made, which its
/// caller asks the encoder for ([`crate::H264Encoder::source_spec`]).
///
/// A source image is 2D, one mip level and one layer, of the profile's
/// format ([`Self::format`]: NV12, or P010 at 10 bits), optimally tiled, at
/// least `extent`, with `VIDEO_ENCODE_SRC` usage among its usages, shared
/// `CONCURRENT` between the families [`Self::families`] gives (exclusive
/// where that is one), and bound to memory. Where `profile_independent` is
/// set it is made with `VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR`,
/// which serves every encoder; else its create info chains a
/// `VkVideoProfileListInfoKHR` of `profile` alone. [`Self::with_create_info`]
/// builds that create info.
///
/// It holds the frame from its origin, the padding to `extent` below and
/// to the right of it, which the encoder reads too, as the caller left it.
/// It is in `VK_IMAGE_LAYOUT_GENERAL` when the frame is encoded, and is left
/// there.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SourceSpec {
    /// The extent the encoder reads: the coded size, rounded up to the
    /// device's input and picture access granularities.
    pub extent: vk::Extent2D,
    /// The encoder's profile.
    pub profile: EncodeProfile,
    /// Whether it is made for any profile (`videoMaintenance1`).
    pub profile_independent: bool,
}

impl SourceSpec {
    /// The source's format: NV12, or P010 at 10 bits.
    pub fn format(&self) -> vk::Format {
        self.profile.format()
    }

    /// The usage an encoder needs: `VIDEO_ENCODE_SRC`.
    pub fn usage(&self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR
    }

    /// The flags an encoder needs: `VIDEO_PROFILE_INDEPENDENT` where the
    /// source is made for any profile.
    pub fn flags(&self) -> vk::ImageCreateFlags {
        if self.profile_independent {
            vk::ImageCreateFlags::VIDEO_PROFILE_INDEPENDENT_KHR
        } else {
            vk::ImageCreateFlags::empty()
        }
    }

    /// The families a source is shared between on `device`: the graphics
    /// family and the encode family, each once.
    pub fn families(device: &VideoDevice) -> Vec<u32> {
        device.encode_families()
    }

    /// Calls `f` with a source's create info on `device`, with `usage`
    /// and `flags` besides the encoder's own, its chain alive for the call.
    pub fn with_create_info<R>(
        &self,
        device: &VideoDevice,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
        f: impl FnOnce(&vk::ImageCreateInfo<'_>) -> R,
    ) -> R {
        let families = Self::families(device);
        let info = vk::ImageCreateInfo::default()
            .flags(self.flags() | flags)
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
            .usage(self.usage() | usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let info = if families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        if self.profile_independent {
            f(&info)
        } else {
            self.profile.with_list(|list| f(&info.push_next(list)))
        }
    }
}

/// A frame to encode, which the encoder holds until the picture that
/// reads it is done.
pub trait EncodeSource: Send + Sync + 'static {
    /// Where the frame lies.
    fn picture(&self) -> SourcePicture;
}

/// Where a frame to encode lies.
#[derive(Clone, Copy)]
pub enum SourcePicture {
    /// An image made as [`SourceSpec`] says, in `GENERAL`, read in place.
    Image(vk::Image),
    /// Its two planes in a buffer, of the frame's own size (not the coded
    /// size), each as [`BufferPlanes`] lays it out: one GPU copy takes them
    /// into a picture of the encoder's, the last row and column repeated
    /// into the padding to [`SourceSpec::extent`]. The buffer has
    /// `TRANSFER_SRC` usage and is shared with the encode family
    /// (`CONCURRENT` between the families [`SourceSpec::families`] gives).
    Buffer(BufferPlanes),
}

/// How a bitstream buffer is made, which the encoder gives its
/// [`BitstreamSource`] with each request.
///
/// A bitstream buffer has `VIDEO_ENCODE_DST` and `TRANSFER_DST` usage
/// ([`Self::usage`]: the GPU writes in-band parameter sets in front of the
/// picture), is shared `CONCURRENT` between the families
/// [`Self::families`] gives (exclusive where that is one), and is bound to
/// memory. Where `profile_independent` is set it is made with
/// `VK_BUFFER_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR`; else its create info
/// chains a `VkVideoProfileListInfoKHR` of `profile` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitstreamSpec {
    /// The bytes a picture may take at most: its target holds at least
    /// this many from its offset, else a picture that needs more may be
    /// cut short ([`crate::EncodeStatus`]).
    pub size: u64,
    /// The offset alignment the device takes: the encoder writes from the
    /// target's offset rounded up to it.
    pub offset_alignment: u64,
    /// The encoder's profile.
    pub profile: EncodeProfile,
    /// Whether it is made for any profile.
    pub profile_independent: bool,
}

impl BitstreamSpec {
    /// A bitstream buffer's usage.
    pub fn usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR | vk::BufferUsageFlags::TRANSFER_DST
    }

    /// Its flags: `VIDEO_PROFILE_INDEPENDENT` where it is made for any
    /// profile.
    pub fn flags(&self) -> vk::BufferCreateFlags {
        if self.profile_independent {
            vk::BufferCreateFlags::VIDEO_PROFILE_INDEPENDENT_KHR
        } else {
            vk::BufferCreateFlags::empty()
        }
    }

    /// The families it is shared between on `device`: as a source's.
    pub fn families(device: &VideoDevice) -> Vec<u32> {
        device.encode_families()
    }

    /// The profile its create info names, where it names one.
    pub(crate) fn any_profile(&self) -> Option<AnyProfile> {
        (!self.profile_independent).then_some(AnyProfile::Encode(self.profile))
    }

    /// Calls `f` with the create info of a buffer of `size` bytes on
    /// `device`, its chain alive for the call.
    pub fn with_create_info<R>(
        &self,
        device: &VideoDevice,
        size: u64,
        f: impl FnOnce(&vk::BufferCreateInfo<'_>) -> R,
    ) -> R {
        let families = Self::families(device);
        let info = vk::BufferCreateInfo::default()
            .flags(self.flags())
            .size(size)
            .usage(Self::usage());
        let info = if families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        if self.profile_independent {
            f(&info)
        } else {
            self.profile.with_list(|list| f(&info.push_next(list)))
        }
    }
}

/// Where a picture's bitstream goes: `size` bytes of `buffer` from
/// `offset`, a buffer made as [`BitstreamSpec`] says.
#[derive(Clone, Copy, Debug)]
pub struct BitstreamBuffer {
    /// The buffer.
    pub buffer: vk::Buffer,
    /// Where the bytes the picture may take start.
    pub offset: u64,
    /// How many there are.
    pub size: u64,
}

/// A buffer a picture's bitstream goes into, which the picture holds until
/// it is dropped.
pub trait BitstreamTarget: Send + Sync + 'static {
    /// Where the bitstream goes.
    fn bitstream(&self) -> BitstreamBuffer;
}

/// Where an encoder's bitstream buffers come from: one a picture.
///
/// A target is written once, by its picture's encode, and read while that
/// picture lives: a source gives no target a live picture holds.
pub trait BitstreamSource: Send {
    /// What it gives.
    type Target: BitstreamTarget;

    /// A target for a picture of `spec`.
    fn target(&mut self, spec: &BitstreamSpec) -> Result<Self::Target, Error>;
}

/// A [`BitstreamSource`] of host-visible buffers, one a picture, for the
/// tests and tools: each holds `spec.size` bytes from offset 512, so that
/// the target's offset is not the buffer's start.
pub struct HostBitstreams {
    device: Arc<VideoDevice>,
}

impl HostBitstreams {
    /// Buffers on `device`.
    pub fn new(device: &Arc<VideoDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }
}

impl BitstreamSource for HostBitstreams {
    type Target = HostBitstream;

    fn target(&mut self, spec: &BitstreamSpec) -> Result<HostBitstream, Error> {
        HostBitstream::new(&self.device, spec)
    }
}

/// A host-visible, host-coherent bitstream buffer, mapped while it lives.
pub struct HostBitstream {
    device: Arc<VideoDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: NonNull<u8>,
    size: usize,
    target: BitstreamBuffer,
}

// SAFETY: the mapping is the buffer's memory alone, which GPU work writes
// once and the holder then reads.
unsafe impl Send for HostBitstream {}
// SAFETY: as above; `bytes` reads through `&self` alone.
unsafe impl Sync for HostBitstream {}

impl HostBitstream {
    /// The offset of its target from the buffer's start.
    pub const OFFSET: u64 = 512;

    /// A buffer of `spec.size` bytes after [`Self::OFFSET`].
    pub fn new(device: &Arc<VideoDevice>, spec: &BitstreamSpec) -> Result<Self, Error> {
        let size = Self::OFFSET + spec.size;
        let vk_device = device.device();
        let buffer = spec.with_create_info(device, size, |info| {
            // SAFETY: `info` and its chain live through the call.
            unsafe { vk_device.create_buffer(info, None) }.named("vkCreateBuffer")
        })?;
        let mut made = Self {
            device: Arc::clone(device),
            buffer,
            memory: vk::DeviceMemory::null(),
            mapped: NonNull::dangling(),
            size: usize::try_from(size)
                .map_err(|_| Error::Unsupported("a buffer past the address space".into()))?,
            target: BitstreamBuffer {
                buffer,
                offset: Self::OFFSET,
                size: spec.size,
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

    /// The buffer's bytes: the picture's NAL units at the offsets its
    /// feedback gives, once its value is reached
    /// ([`crate::EncodedFrame::wait`]).
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is `size` bytes of host-coherent memory,
        // which GPU work writes before the picture's value alone.
        unsafe { std::slice::from_raw_parts(self.mapped.as_ptr(), self.size) }
    }
}

impl BitstreamTarget for HostBitstream {
    fn bitstream(&self) -> BitstreamBuffer {
        self.target
    }
}

impl Drop for HostBitstream {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        // SAFETY: the buffer is the device's, and no GPU work uses it: an
        // encoder lets its targets go only once the work that wrote them is
        // done. A null handle is nothing.
        unsafe { vk_device.destroy_buffer(self.buffer, None) };
        // SAFETY: its memory, unmapped as it is freed.
        unsafe { vk_device.free_memory(self.memory, None) };
    }
}

/// A bitstream target in imported host pages: the guest's buffer, which
/// the encode writes in place.
pub struct ImportedBitstream {
    target: BitstreamBuffer,
    _buffer: ImportBuffer,
}

impl ImportedBitstream {
    /// `size` bytes of `import` from its offset plus `offset`, as `spec`'s
    /// encoder writes them; `None` where they lie outside the import or the
    /// device takes no buffer over them.
    ///
    /// # Safety
    ///
    /// `import` is memory of `device`'s, imported as it says, valid while
    /// `keep` lives; the bytes are the encoder's to write meanwhile.
    pub unsafe fn new(
        device: &Arc<VideoDevice>,
        spec: &BitstreamSpec,
        import: &HostImport,
        offset: u64,
        size: u64,
        keep: Arc<dyn Send + Sync>,
    ) -> Result<Option<Self>, Error> {
        let (Some(from), Some(to)) = (
            import.offset.checked_add(offset),
            import
                .offset
                .checked_add(offset)
                .and_then(|from| from.checked_add(size)),
        ) else {
            return Ok(None);
        };
        let families = BitstreamSpec::families(device);
        let made_for = BufferUse {
            usage: BitstreamSpec::usage(),
            flags: spec.flags(),
            families: &families,
            profile: spec.any_profile(),
        };
        let Some((buffer, base)) = buffer_over(device, import, from, to, made_for)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            target: BitstreamBuffer {
                buffer,
                offset: from - base,
                size,
            },
            _buffer: ImportBuffer::new(device, buffer, keep),
        }))
    }
}

impl BitstreamTarget for ImportedBitstream {
    fn bitstream(&self) -> BitstreamBuffer {
        self.target
    }
}

/// A frame's NV12 or P010 planes in imported host pages: an IOSurface's
/// own pages, which one GPU copy takes into a picture of the encoder's.
pub struct ImportedSource {
    planes: BufferPlanes,
    _buffer: ImportBuffer,
}

impl ImportedSource {
    /// The planes `luma` and `chroma` (each its first byte from `import`'s
    /// offset, the bytes from one of its rows to the next, its rows, and
    /// the bytes of a row) in `import`, each ending at its last row's last
    /// byte ([`plane_span`]); `None` where a plane lies outside the import
    /// or the device takes no buffer over it.
    ///
    /// # Safety
    ///
    /// `import` is memory of `device`'s, imported as it says, valid while
    /// `keep` lives; nothing writes the planes until the encoder is done
    /// with the source.
    pub unsafe fn new(
        device: &Arc<VideoDevice>,
        import: &HostImport,
        luma: (u64, u32, u32, u32),
        chroma: (u64, u32, u32, u32),
        keep: Arc<dyn Send + Sync>,
    ) -> Result<Option<Self>, Error> {
        let place = |plane| plane_span(import.offset, plane);
        let (Some(luma_at), Some(chroma_at)) = (place(luma), place(chroma)) else {
            return Ok(None);
        };
        let (from, to) = (luma_at.0.min(chroma_at.0), luma_at.1.max(chroma_at.1));
        let families = SourceSpec::families(device);
        let made_for = BufferUse {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            flags: vk::BufferCreateFlags::empty(),
            families: &families,
            profile: None,
        };
        let Some((buffer, base)) = buffer_over(device, import, from, to, made_for)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            planes: BufferPlanes {
                buffer,
                luma: (luma_at.0 - base, luma.1),
                chroma: (chroma_at.0 - base, chroma.1),
                size: to - base,
            },
            _buffer: ImportBuffer::new(device, buffer, keep),
        }))
    }
}

impl EncodeSource for ImportedSource {
    fn picture(&self) -> SourcePicture {
        SourcePicture::Buffer(self.planes)
    }
}
