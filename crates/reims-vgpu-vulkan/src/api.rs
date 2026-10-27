//! What the Rust API (`reims-vgpu`, feature `vulkan`) asks of this platform beyond core's
//! `Platform`: a frame target's pixels ([`read_pixels`]), and a display's frame where the CPU
//! reads it in place ([`read_in_place`]).

use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::surface::Surface as _;

use crate::answer::{get, plain};
use crate::gpu::resource::VulkanTexture;

/// Whether `t`'s texels are its rows of 4 bytes a texel as they lie in its bytes: a 2D texture of
/// one level, slice and sample, not rotated, in an 8-bit RGBA or BGRA format, made so or a view
/// of the whole of such a texture (its level 0 and slice 0, of its size), itself one of these.
fn whole_in_place(t: &rt::Texture) -> bool {
    let four_bytes = matches!(
        t.pixel_format(),
        PixelFormat::BGRA8Unorm
            | PixelFormat::BGRA8UnormSRGB
            | PixelFormat::RGBA8Unorm
            | PixelFormat::RGBA8UnormSRGB
    );
    let plain = four_bytes
        && t.texture_type() == TextureType::Type2D
        && t.mipmap_level_count() == 1
        && t.array_length() == 1
        && t.sample_count() == 1
        && t.rotation() == 0;
    match t.parent_texture() {
        None => plain,
        Some(parent) => {
            plain
                && t.parent_relative_level() == 0
                && t.parent_relative_slice() == 0
                && parent.width() == t.width()
                && parent.height() == t.height()
                && whole_in_place(parent)
        }
    }
}

/// Calls `read` with the bytes of `texture`'s texels in the memory of the IOSurface it was made
/// on (`newTextureWithDescriptor:iosurface:plane:`): the first byte of its plane, the bytes from
/// there to the end of its last row's `width * 4`, and the plane's bytes per row. They are the
/// bytes `getBytes` reads ([`read_pixels`]): the runtime's buffer under such a texture lies over
/// the surface's memory, and a texture the GPU renders into is copied back into it after each
/// encoder that writes it. A view of such a texture in another 4-byte format, of the whole of it,
/// has the same bytes (a guest display's present is one: `present_surface`), and so does a view of
/// such a view. False, without calling it, for a texture of any other kind: not made on a surface
/// or a view of part of one, rotated, of other than one level, slice and sample, 2D and 4 bytes a
/// texel, on a surface whose GPU writes never reach its memory, without a base address
/// (protected), or whose rows run past the surface's allocation.
pub fn read_in_place(
    texture: &VulkanTexture,
    read: &mut dyn FnMut(*const u8, usize, usize),
) -> bool {
    let (Some(surface), Some(plane)) = (&texture.surface, texture.plane) else {
        return false;
    };
    let t = &texture.raw;
    if !whole_in_place(t) || texture.gpu_read_only() {
        return false;
    }
    let base = surface.base_address().as_ptr::<u8>();
    if base.is_null() {
        return false;
    }
    // The bytes from the base address to the allocation's end: the base lies the surface's
    // offset into it (`kIOSurfaceOffset`).
    let span = surface
        .alloc_size()
        .saturating_sub(crate::os::surface_offset(surface));
    let row = t.width().checked_mul(4);
    let last = t.height().checked_sub(1);
    let end = row.zip(last).and_then(|(row, last)| {
        last.checked_mul(plane.bytes_per_row)?
            .checked_add(row)?
            .checked_add(plane.offset)
    });
    let (Some(row), Some(end)) = (row, end) else {
        return false;
    };
    if plane.bytes_per_row < row || end > span {
        return false;
    }
    // SAFETY: the plane's bytes lie in the surface's mapping from its base address (`span`,
    // checked above), which the texture's surface reference keeps mapped while the caller holds
    // the texture.
    let first = unsafe { base.add(plane.offset.to_usize()) };
    read(
        first,
        (end - plane.offset).to_usize(),
        plane.bytes_per_row.to_usize(),
    );
    true
}

/// Copies the whole of `texture`'s first level and slice into `dst`: `height` rows of `width * 4`
/// bytes, each `bytes_per_row` after the last, the bytes between rows left as they are
/// (`getBytes:bytesPerRow:fromRegion:mipmapLevel:`). The caller has checked that `dst` holds
/// them. A refusal of the runtime's dies at the selector, as a raise in Metal's `getBytes` dies
/// at its send.
pub fn read_pixels(texture: &VulkanTexture, dst: &mut [u8], bytes_per_row: usize) {
    let sel = "getBytes:bytesPerRow:fromRegion:mipmapLevel:";
    let t = texture.copied(sel);
    let region = rt::Region::new_2d(0, 0, plain(|| t.width()), plain(|| t.height()));
    get(sel, || {
        t.get_bytes(dst, bytes_per_row.to_u64(), 0, region, 0, 0)
    });
}

#[cfg(test)]
mod tests {
    use metal2vulkan::metal::PixelFormat;
    use reims_vgpu_core::platform::Handle as _;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::TextureObject;
    use crate::platform::Obj;

    /// A BGRA8 texture of `width` by `height` (shared storage, the descriptor's default), holding
    /// `px` (rows of `width * 4` bytes).
    fn texture(width: u64, height: u64, px: &[u8]) -> VulkanTexture {
        let d = dev();
        let desc = rt::TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
        let raw = d.raw.new_texture(&desc).unwrap();
        let all = rt::Region::new_2d(0, 0, width, height);
        raw.replace_region(all, 0, 0, px, width * 4, 0).unwrap();
        Obj::new(TextureObject {
            raw,
            device: d.clone_ref(),
            surface: None,
            plane: None,
            under: None,
            private_usage: 0,
            space: std::sync::Arc::new(crate::gpu::metal_space::Hold::NONE),
            gpu_read_only: false,
        })
    }

    /// Every pixel of the texture, row by row; the bytes past each row's pixels are left alone
    #[test]
    fn read_pixels_reads_the_whole_texture() {
        let (w, h) = (3u64, 2u64);
        let px: Vec<u8> = (0u8..24).collect();
        let t = texture(w, h, &px);
        let mut tight = vec![0xaa; 24];
        read_pixels(&t, &mut tight, 12);
        assert_eq!(tight, px);
        let mut padded = vec![0xaa; 16 + 12];
        read_pixels(&t, &mut padded, 16);
        assert_eq!(&padded[..12], &px[..12]);
        assert_eq!(&padded[12..16], &[0xaa; 4]);
        assert_eq!(&padded[16..], &px[12..]);
    }

    /// A texture made on a surface, of its own memory or over the guest's pages at an offset, and
    /// a view of the whole of it in another 4-byte format, are read in place, the bytes
    /// `getBytes` reads; a texture of its own memory is not.
    #[test]
    fn a_texture_on_a_surface_is_read_in_place_as_get_bytes_reads_it() {
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Texture as _;
        use reims_vgpu_core::raise::OnFault;

        use crate::gpu::resource::tests::bgra;

        let d = dev();
        let (w, h) = (16usize, 8usize);
        let surface = crate::os::tests::bgra_surface(16, 8);
        let (over, pages, length) = crate::os::tests::bgra_surface_over_at(16, 8, 0x40);
        let px: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 % 251) as u8).collect();
        // The texture's rows as a read in place gives them, if it does.
        let in_place = |t: &VulkanTexture| {
            let mut all = Vec::new();
            let read = read_in_place(t, &mut |first, len, bytes_per_row| {
                assert!(bytes_per_row >= w * 4 && len >= (h - 1) * bytes_per_row + w * 4);
                for y in 0..h {
                    let mut row = vec![0u8; w * 4];
                    // SAFETY: row `y` lies in the `len` bytes from `first` (asserted above).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            first.add(y * bytes_per_row),
                            row.as_mut_ptr(),
                            w * 4,
                        );
                    }
                    all.extend_from_slice(&row);
                }
            });
            read.then_some(all)
        };
        for s in [&surface, &over] {
            let t = (d.new_texture_on_surface(&bgra(16, 8, 0), Some(s), 0))
                .unwrap()
                .unwrap();
            let all = rt::Region::new_2d(0, 0, 16, 8);
            t.raw.replace_region(all, 0, 0, &px, 64, 0).unwrap();
            let mut copy = vec![0u8; px.len()];
            read_pixels(&t, &mut copy, w * 4);
            assert_eq!(copy, px);
            assert_eq!(in_place(&t), Some(copy));
            // A view of the whole texture in another 4-byte format (a guest display's present):
            // the same bytes.
            let view = t.new_texture_view(70).unwrap().unwrap();
            let mut through_view = vec![0u8; px.len()];
            read_pixels(&view, &mut through_view, w * 4);
            assert_eq!(through_view, px);
            assert_eq!(in_place(&view), Some(through_view));
        }
        drop(over);
        crate::os::tests::free(pages, length);
        let own = d
            .new_texture(OnFault::Die, &bgra(16, 8, 0))
            .unwrap()
            .unwrap();
        assert!(!read_in_place(&own, &mut |_, _, _| panic!("read in place")));
    }
}
