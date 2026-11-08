//! What the Rust API (`reims-vgpu`) asks of this platform beyond core's `Platform`: a frame
//! target's pixels ([`read_pixels`]).

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;

use crate::answer::{get, plain};
use crate::gpu::resource::VulkanTexture;

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
}
