//! Buffers of the crate's own over host pages the device imported by their
//! pointer (`VK_EXT_external_memory_host`), as metal2vulkan's
//! `Buffer::vk_host_memory` answers for a buffer over them: where a decode
//! reads a sample in place ([`crate::SampleMemory`]), and where a picture
//! lands in an IOSurface's planes ([`ImportedPlanes`]). The import is the
//! runtime's: the buffers bind to it and never free it.

use std::sync::Arc;

use ash::vk;

use crate::device::VideoDevice;
use crate::encode::caps::EncodeProfile;
use crate::error::{Error, Named};
use crate::profile::VideoProfile;
use crate::surface::{BufferPlanes, Landing, Surface};

/// A sample's bytes in device memory, which a decoder's `decode_in_place`
/// ([`crate::H264Decoder::decode_in_place`],
/// [`crate::H265Decoder::decode_in_place`]) reads in place: host pages the
/// device imported by their pointer, as metal2vulkan's
/// `Buffer::vk_host_memory` answers for a buffer over them.
pub struct SampleMemory {
    /// Where the sample lies: its first byte at the import's offset.
    pub import: HostImport,
    /// What keeps the memory alive, held until every decode that reads
    /// the sample in place is done.
    pub keep: Arc<dyn Send + Sync>,
}

/// Where bytes lie in an import of host pages, as metal2vulkan's
/// `VkHostMemory` answers.
#[derive(Clone, Copy)]
pub struct HostImport {
    /// The import.
    pub memory: vk::DeviceMemory,
    /// Where the bytes' first lies in it.
    pub offset: u64,
    /// The bytes of the import from `offset` a buffer may lie over.
    pub size: u64,
    /// The import's memory type.
    pub type_index: u32,
    /// The handle type it was imported as, which a buffer bound to it
    /// names.
    pub handle_type: vk::ExternalMemoryHandleTypeFlags,
}

/// What a buffer over an import is for.
#[derive(Clone, Copy)]
pub(crate) struct BufferUse<'a> {
    /// Its usage.
    pub(crate) usage: vk::BufferUsageFlags,
    /// Its create flags.
    pub(crate) flags: vk::BufferCreateFlags,
    /// The families it is shared between, `CONCURRENT` where more than
    /// one.
    pub(crate) families: &'a [u32],
    /// The video profile a decode's or encode's use of it names, if any.
    pub(crate) profile: Option<AnyProfile>,
}

/// A decode or an encode profile, as a buffer's create info names it.
#[derive(Clone, Copy)]
pub(crate) enum AnyProfile {
    Decode(VideoProfile),
    Encode(EncodeProfile),
}

impl AnyProfile {
    /// Calls `f` with a `VkVideoProfileListInfoKHR` of the profile alone.
    fn with_list<R>(&self, f: impl FnOnce(&mut vk::VideoProfileListInfoKHR<'_>) -> R) -> R {
        match self {
            Self::Decode(profile) => profile.with_list(f),
            Self::Encode(profile) => profile.with_list(f),
        }
    }
}

/// A buffer over `import`'s memory from `from` (an offset in the memory)
/// to `to`, bound at the multiple of the buffer's alignment at or below
/// `from`, and that multiple; `None`, and no buffer, where the memory's
/// type does not take such a buffer, it binds past the import's bytes, or
/// its bind fails.
pub(crate) fn buffer_over(
    device: &VideoDevice,
    import: &HostImport,
    from: u64,
    to: u64,
    made_for: BufferUse<'_>,
) -> Result<Option<(vk::Buffer, u64)>, Error> {
    let end = import.offset.saturating_add(import.size);
    if from < import.offset || to < from || to > end {
        return Ok(None);
    }
    // Made from `from` first; again from the alignment's multiple below it
    // where `from` is not one.
    let Some((buffer, alignment, _)) = unbound(device, import, to - from, made_for)? else {
        return Ok(None);
    };
    // SAFETY: the buffer is the device's, bound to nothing and used by
    // nothing.
    unsafe { device.device().destroy_buffer(buffer, None) };
    let base = from - from % alignment;
    let Some((buffer, again, size)) = unbound(device, import, to - base, made_for)? else {
        return Ok(None);
    };
    if !base.is_multiple_of(again) || base.saturating_add(size) > end {
        // SAFETY: as above.
        unsafe { device.device().destroy_buffer(buffer, None) };
        return Ok(None);
    }
    // SAFETY: the buffer is the device's, made for the import's handle
    // type, its memory type among those it takes, and `base` a multiple of
    // its alignment from which its size lies in the import.
    let bound = unsafe {
        device
            .device()
            .bind_buffer_memory(buffer, import.memory, base)
    };
    if let Err(result) = bound {
        // SAFETY: as above: bound to nothing, used by nothing.
        unsafe { device.device().destroy_buffer(buffer, None) };
        log::debug!("a buffer over imported host pages: vkBindBufferMemory: {result}");
        return Ok(None);
    }
    Ok(Some((buffer, base)))
}

/// A buffer of `size` bytes made for `made_for` over `import`'s memory,
/// not yet bound, its alignment and the bytes it binds; `None`, and no
/// buffer, where the memory's type does not take it.
fn unbound(
    device: &VideoDevice,
    import: &HostImport,
    size: u64,
    made_for: BufferUse<'_>,
) -> Result<Option<(vk::Buffer, u64, u64)>, Error> {
    let vk_device = device.device();
    let mut external =
        vk::ExternalMemoryBufferCreateInfo::default().handle_types(import.handle_type);
    let info = vk::BufferCreateInfo::default()
        .flags(made_for.flags)
        .size(size.max(1))
        .usage(made_for.usage)
        .push_next(&mut external);
    let info = if made_for.families.len() > 1 {
        info.sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(made_for.families)
    } else {
        info.sharing_mode(vk::SharingMode::EXCLUSIVE)
    };
    let buffer = match made_for.profile {
        Some(profile) => profile.with_list(|list| {
            // SAFETY: `info` and its chain live through the call.
            unsafe { vk_device.create_buffer(&info.push_next(list), None) }
        }),
        // SAFETY: as above.
        None => unsafe { vk_device.create_buffer(&info, None) },
    }
    .named("vkCreateBuffer")?;
    // SAFETY: the buffer is the device's.
    let requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };
    if requirements.memory_type_bits & (1 << import.type_index) == 0 {
        // SAFETY: the buffer is the device's, bound to nothing, used by
        // nothing.
        unsafe { vk_device.destroy_buffer(buffer, None) };
        log::debug!(
            "a buffer over imported host pages: memory type {} of {:#x}, {} bytes for {size}",
            import.type_index,
            requirements.memory_type_bits,
            requirements.size,
        );
        return Ok(None);
    }
    Ok(Some((
        buffer,
        requirements.alignment.max(1),
        requirements.size,
    )))
}

/// A buffer of the crate's over imported host pages, destroyed as it goes,
/// with what keeps the pages.
pub(crate) struct ImportBuffer {
    device: Arc<VideoDevice>,
    pub(crate) buffer: vk::Buffer,
    _keep: Arc<dyn Send + Sync>,
}

impl ImportBuffer {
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        buffer: vk::Buffer,
        keep: Arc<dyn Send + Sync>,
    ) -> Self {
        Self {
            device: Arc::clone(device),
            buffer,
            _keep: keep,
        }
    }
}

impl Drop for ImportBuffer {
    fn drop(&mut self) {
        // SAFETY: the buffer is the device's, and the work that used it is
        // done: its holders retire it until then.
        unsafe { self.device.device().destroy_buffer(self.buffer, None) };
    }
}

/// Where a plane `(offset, pitch, rows, row)` from `base` lies: its first
/// byte, and the end of its last row, `row` bytes past that row's start;
/// `None` for no rows, or past 2^64.
pub(crate) fn plane_span(
    base: u64,
    (offset, pitch, rows, row): (u64, u32, u32, u32),
) -> Option<(u64, u64)> {
    let start = base.checked_add(offset)?;
    let before_last = u64::from(pitch).checked_mul(u64::from(rows.checked_sub(1)?))?;
    Some((
        start,
        start
            .checked_add(before_last)?
            .checked_add(u64::from(row))?,
    ))
}

/// NV12 or P010 planes in imported host pages, as a picture's output
/// ([`Landing::Buffer`]): an IOSurface's planes in the guest's memory, where
/// one GPU copy out of the decoder's picture writes them, cropped.
pub struct ImportedPlanes {
    planes: BufferPlanes,
    _buffer: ImportBuffer,
}

impl ImportedPlanes {
    /// The planes `luma` and `chroma` (each its first byte from `import`'s
    /// offset, the bytes from one of its rows to the next, its rows, and
    /// the bytes of a row) in `import`, each ending at its last row's last
    /// byte; `None` where a plane lies outside the import or the device
    /// takes no buffer over it.
    ///
    /// # Safety
    ///
    /// `import` is memory of `device`'s, imported as it says, valid while
    /// `keep` lives; the planes' bytes are the caller's to write meanwhile.
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
        let families = device.output_families();
        let made_for = BufferUse {
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            flags: vk::BufferCreateFlags::empty(),
            families: &families,
            profile: None,
        };
        let Some((buffer, base)) = buffer_over(device, import, from, to, made_for)? else {
            return Ok(None);
        };
        let planes = BufferPlanes {
            buffer,
            luma: (luma_at.0 - base, luma.1),
            chroma: (chroma_at.0 - base, chroma.1),
            size: to - base,
        };
        Ok(Some(Self {
            planes,
            _buffer: ImportBuffer::new(device, buffer, keep),
        }))
    }
}

impl Surface for ImportedPlanes {
    fn landing(&self) -> Landing {
        Landing::Buffer(self.planes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane's span, a decode's output's ([`ImportedPlanes`]) or an encode's source's
    /// ([`crate::ImportedSource`]), runs from its first byte to its last row's last, `row`
    /// bytes past that row's start, not a whole pitch: a tile's chroma at (256, 512) of a
    /// 2048-byte pitch plane at 3145728 in 4718592 bytes, 256 rows of 512 bytes, ends inside
    /// them, where 256 whole rows would end 512 bytes past; no rows is no span.
    #[test]
    fn a_plane_s_span_ends_at_its_last_row_s_last_byte() {
        let tile = 3_145_728 + 512 * 2048 + 256 * 2;
        assert_eq!(
            plane_span(0, (tile, 2048, 256, 512)),
            Some((tile, 4_718_592 - 1024))
        );
        assert_eq!(tile + 256 * 2048, 4_718_592 + 512);
        assert_eq!(
            plane_span(4096, (512, 2048, 3, 1024)),
            Some((4608, 4608 + 2 * 2048 + 1024))
        );
        assert_eq!(plane_span(0, (0, 2048, 1, 2048)), Some((0, 2048)));
        assert_eq!(plane_span(0, (0, 2048, 0, 2048)), None);
        assert_eq!(plane_span(u64::MAX, (1, 2048, 1, 1)), None);
    }
}
