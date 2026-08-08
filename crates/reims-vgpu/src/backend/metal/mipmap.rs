//! Metal-filtered `generateMipmapsForTexture:` for multi-level 2D textures.
//!
//! Builds a temporary Shared storage texture, uploads level 0 in the guest
//! native pixel format, runs the Metal blit encoder filter, and reads back
//! every level as tightly packed native rows.
//!
//! What is left here is exactly the part that needs a device. The request's
//! argument ladder, the filterable-format set and [`MetalMipmapError`] itself
//! are arithmetic over guest numbers, so they live in [`crate::contract::mipmap`]
//! where they can be executed on a host with no Apple linker — which is where
//! five of this module's six tests went with them. The one that stayed
//! (`metal_generate_constant_rgba8_preserves_color`) asks Metal to filter real
//! pixels and cannot be answered anywhere else.

use crate::backend::metal::mtl_enum;
use crate::backend::metal::runtime::{system_device, thread_queue};
use crate::contract::mipmap::{filterable_bpp, plan_level0, MetalMipmapError};
use metal::{
    MTLCommandBufferStatus, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode,
    MTLTextureType, MTLTextureUsage, TextureDescriptor,
};

/// One mip level after Metal filter generation (tight native packing).
#[derive(Clone, Debug)]
pub struct MetalMipLevel {
    pub width: u32,
    pub height: u32,
    pub tight_bytes: Vec<u8>,
}

/// The `MTLPixelFormat` and bytes-per-pixel of a guest format Metal will filter.
///
/// The filterability decision is [`filterable_bpp`]'s; this adds the one step
/// that needs the `metal` crate. Every format that predicate accepts is one of
/// this device's own named constants, so the conversion cannot decline here —
/// it is written as a `?` rather than an unwrap because the caller's `None`
/// already means "this device will not filter that format", which is the right
/// answer for a code naming no format at all.
pub fn filterable_format(format: u16) -> Option<(MTLPixelFormat, u32)> {
    let bpp = filterable_bpp(format)?;
    Some((mtl_enum::pixel_format(format as u32)?, bpp))
}

/// Upload L0, run Metal-filtered mip generation, return levels `[0..levels)`.
///
/// `level0` must be tightly packed native rows (`width * bpp` per row). `levels`
/// must be `> 1`. Level 0 in the result is a copy of the input; levels 1.. are
/// Metal-filtered.
pub fn generate_mipmaps_filtered(
    format: u16,
    width: u32,
    height: u32,
    levels: u32,
    level0: &[u8],
) -> Result<Vec<MetalMipLevel>, MetalMipmapError> {
    // The whole argument ladder, in one call, so that its order stays a fact
    // some host can execute rather than one only an Apple machine can.
    let plan = plan_level0(format, width, height, levels, level0.len())?;
    let bpp = plan.bpp;
    // Re-asked only for the `MTLPixelFormat`: `plan_level0` has already accepted
    // the format, so a `None` here is the enum table missing a code the
    // filterable set names. That is this device disagreeing with itself, and
    // `UnsupportedFormat` is the honest report of it — never a cast.
    let (mtl_fmt, _) =
        filterable_format(format).ok_or(MetalMipmapError::UnsupportedFormat { format })?;

    let device = system_device().ok_or(MetalMipmapError::NoDevice)?;
    let queue = thread_queue(device);

    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(mtl_fmt);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_mipmap_level_count(levels as u64);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    // ShaderRead is the documented usage for filterable sampled textures;
    // generateMipmapsForTexture operates on filterable color textures.
    descriptor.set_usage(MTLTextureUsage::ShaderRead);
    // Before the level-count check, not after: an unchecked nil answers
    // `mipmapLevelCount` with 0, so an exhausted device used to be reported as
    // one that rejected the level count.
    let Some(texture) = crate::backend::metal::raw_metal::new_texture(device, &descriptor) else {
        return Err(MetalMipmapError::TextureAllocationFailed {
            width,
            height,
            levels,
        });
    };
    if texture.mipmap_level_count() < levels as u64 {
        return Err(MetalMipmapError::LevelCountRejected {
            requested: levels,
            actual: texture.mipmap_level_count(),
        });
    }

    let region0 = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.replace_region(region0, 0, level0.as_ptr() as *const _, plan.bytes_per_row);

    let command_buffer = crate::backend::metal::raw_metal::new_command_buffer(&queue)
        .ok_or(MetalMipmapError::CommandBufferFailed)?
        .to_owned();
    let blit = crate::backend::metal::raw_metal::new_blit_command_encoder(&command_buffer)
        .ok_or(MetalMipmapError::CommandBufferFailed)?;
    blit.generate_mipmaps(&texture);
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalMipmapError::CommandBufferFailed);
    }

    let mut out = Vec::with_capacity(levels as usize);
    for level in 0..levels {
        let w = crate::contract::extent::mip_extent(width, level);
        let h = crate::contract::extent::mip_extent(height, level);
        // Both factors are u32, so their product always fits in u64.
        let bpr = (w as u64) * (bpp as u64);
        let need = bpr
            .checked_mul(h as u64)
            .ok_or(MetalMipmapError::LevelSpanOverflow {
                level,
                row_bytes: bpr,
                height: h,
            })?;
        let mut tight = vec![0u8; need as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as u64,
                height: h as u64,
                depth: 1,
            },
        };
        texture.get_bytes(tight.as_mut_ptr() as *mut _, bpr, region, level as u64);
        out.push(MetalMipLevel {
            width: w,
            height: h,
            tight_bytes: tight,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;

    #[test]
    fn metal_generate_constant_rgba8_preserves_color() {
        // 4×4 solid (200, 10, 20, 255) → filtered mips stay that color.
        let w = 4u32;
        let h = 4u32;
        let levels = 3u32;
        let mut l0 = vec![0u8; (w * h * 4) as usize];
        for px in l0.chunks_exact_mut(4) {
            px[0] = 200;
            px[1] = 10;
            px[2] = 20;
            px[3] = 255;
        }
        let chain = generate_mipmaps_filtered(MTL_FORMAT_RGBA8_UNORM, w, h, levels, &l0)
            .expect("metal generate");
        assert_eq!(chain.len(), 3);
        assert_eq!((chain[0].width, chain[0].height), (4, 4));
        assert_eq!((chain[1].width, chain[1].height), (2, 2));
        assert_eq!((chain[2].width, chain[2].height), (1, 1));
        for level in &chain {
            for px in level.tight_bytes.chunks_exact(4) {
                assert_eq!(px, &[200, 10, 20, 255], "mip {} color", level.width);
            }
        }
    }
}
