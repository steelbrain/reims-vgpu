//! The pixel formats Metal describes and makes no texture of: `MTLPixelFormatUnspecialized` (263),
//! and `MTLPixelFormatRGBA16Float_XR` (556), `MTLPixelFormatAnyDepth` (651) and
//! `MTLPixelFormatAnyStencil` (652) ([`FORMATS`]; the runtime describes each as Metal does,
//! [`crate::gpu::format_info`]). The runtime names each and makes no texture of any either. A
//! texture descriptor takes each at `setPixelFormat:` and answers it
//! ([`crate::gpu::desc::TextureFields`]), and every creator answers a descriptor of one as Metal
//! does: the runtime, from the descriptor [`descriptor`] makes, as it answers a two-plane YCbCr or
//! RGB format ([`crate::gpu::ycbcr::two_plane`]), and this platform at
//! `newSharedTextureWithDescriptor:`, whose nil Metal answers before its validation. A view in
//! one of them, or in another format Metal makes no view in, is nil ([`no_view`]).
//!
//! Measured on Metal on this host (M3 Max, macOS 26.7, without API validation, one case per
//! process), with 1D, 1D array, 2D, 2D array, cube, cube array, 3D, texture buffer, 2D
//! multisample and 2D multisample array descriptors, at sizes from 1 by 1 to the largest, of every
//! level count the size takes, in every storage mode, of every usage, swizzled, `framebufferOnly`,
//! rotated and with `writeSwizzleEnabled`:
//! - `newSharedTextureWithDescriptor:` answers nil for every descriptor, first: one its validation
//!   aborts at for another format too;
//! - every other creator validates the descriptor and aborts where that refuses it, then answers
//!   nil: `newTextureWithDescriptor:`, a heap's `newTextureWithDescriptor:offset:`, a buffer's
//!   `newTextureWithDescriptor:offset:bytesPerRow:` and
//!   `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, and
//!   `newTextureWithDescriptor:iosurface:plane:`. `heapTextureSizeAndAlignWithDescriptor:`
//!   answers 0 bytes there, at an alignment of the type alone: 128 for a 1D, 2D or 2D multisample
//!   texture, 16 for a texture buffer, and 16 KiB for every other type (128 at macOS 27.0.1, the
//!   runtime's answer at that level). The nil comes before each creator's own checks, which a
//!   texture of another format aborts or is nil at:
//!   `framebufferOnly` where the creator asserts it is not set, a rotation of a texture not over an
//!   IOSurface, a storage mode other than the heap's, an offset or rows off their alignment or
//!   short of a row, an offset past the heap's end, a plane the surface lacks, and a type other
//!   than 2D over a surface;
//! - the validation is the one every format meets (as [`crate::gpu::ycbcr`]'s, and a usage bit
//!   Metal does not define, [`crate::gpu::desc`]), and the formats' own: none takes `ShaderAtomic`
//!   usage ("is not a valid format to be used with texture atomics"); `AnyDepth` and `AnyStencil`
//!   take no 1D, 1D array, 3D, texture buffer or multisample type ("depth/stencil pixel formats
//!   require texture type of MTLTextureType2D\[Array\] or MTLTextureTypeCube\[Array\]", "does not
//!   support multisample textures"); `Unspecialized` takes no multisample type; `RGBA16Float_XR`
//!   takes both, of 2 or 4 samples, as a format Metal makes multisample textures of does;
//! - the runtime answers each of those creators and the size query as Metal does, its nil and its 0
//!   bytes at Metal's alignment, and its validation stops what Metal's aborts at, but for usage
//!   bits Metal does not define, at which [`descriptor`] dies first ([`runtime_passes`]; it dies
//!   first at two more, which the runtime's validation stops too). Its nil comes before what it
//!   does not build yet, as Metal's does: a texture buffer of more than one level, within the
//!   chain, and a swizzle under `ShaderWrite` or `RenderTarget` usage of a descriptor of
//!   `writeSwizzleEnabled`. Compared at every case of the tests for each of the 73 formats, these
//!   and the two-plane ones: the runtime validates before it answers nil, and before each
//!   creator's own checks, but at `newSharedTextureWithDescriptor:`, which validates first;
//! - a view of a texture of any format Metal makes one of on this host, made by any of the view
//!   selectors (with or without a resource index, and with any type, levels, slices or swizzle,
//!   the source's or not), in a format [`NO_VIEW`] lists is nil, before the view's swizzle, type,
//!   levels and slices are checked (a swizzle channel of 9, type 10 or 99, and levels whose end
//!   passes `u64::MAX` among them): the formats here, and every two-plane YCbCr or RGB format Metal has
//!   and the runtime lacks (those of `_2P` in Metal's names, 500 to 620) but `RGB10A8_2P` and its
//!   three siblings (624 to 627), whose views are made from some sources. A view in a one-plane
//!   YCbCr format (`_1P`, as 501) or a plane format (as 576, `MTLPixelFormatR10Unorm_X6`) aborts
//!   from every source measured. A view of a texture of the two-plane YCbCr family the runtime
//!   makes over a surface ([`crate::gpu::ycbcr`]) in its own format, or in its `_sRGB` twin, is
//!   not nil: Metal makes it, and so does the runtime (`own_format_views_are_metals`).

use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime as rt;

use crate::answer::metal_aborts;
use crate::gpu::desc::TextureFields;

/// The formats here a descriptor keeps raw, by Metal's raw values: `RGBA16Float_XR`, `AnyDepth`
/// and `AnyStencil`. `MTLPixelFormatUnspecialized` is held as the runtime's format.
pub const FORMATS: [u64; 3] = [RGBA16_FLOAT_XR, ANY_DEPTH, ANY_STENCIL];

/// `MTLPixelFormatUnspecialized`, which the runtime has.
const UNSPECIALIZED: u64 = 263;

/// `MTLPixelFormatRGBA16Float_XR`.
const RGBA16_FLOAT_XR: u64 = 556;

/// `MTLPixelFormatAnyDepth`.
const ANY_DEPTH: u64 = 651;

/// `MTLPixelFormatAnyStencil`.
const ANY_STENCIL: u64 = 652;

/// The formats a view of a texture of a format the runtime has is nil in, by Metal's raw values,
/// as inclusive ranges: `MTLPixelFormatUnspecialized`, the two-plane YCbCr and RGB formats from
/// `MTLPixelFormatYCBCR8_420_2P` (500) to `MTLPixelFormatYCBCR12_444_2P_PACKED_HLG` (620) that
/// Metal on this host has, `RGBA16Float_XR`, `AnyDepth` and `AnyStencil` (measured: the module's
/// docs).
pub const NO_VIEW: &[(u64, u64)] = &[
    (263, 263),
    (500, 500),
    (502, 503),
    (505, 510),
    (520, 520),
    (522, 523),
    (525, 530),
    (540, 548),
    (556, 556),
    (564, 575),
    (580, 585),
    (591, 602),
    (608, 613),
    (615, 620),
    (651, 652),
];

/// The largest width of a texture buffer, measured.
const MAX_BUFFER_WIDTH: u64 = 268_435_456;

/// The sample counts of a multisample texture the device takes, measured (1, 3, 8 and 16 abort).
const SAMPLE_COUNTS: [u64; 2] = [2, 4];

/// Whether a descriptor's `setPixelFormat:` takes `raw` as a format here.
pub fn taken(raw: u64) -> bool {
    raw == UNSPECIALIZED || FORMATS.contains(&raw)
}

/// Whether a view in `raw` of a texture of a format the runtime has is nil ([`NO_VIEW`]).
pub fn no_view(raw: u64) -> bool {
    NO_VIEW
        .iter()
        .any(|&(first, last)| (first..=last).contains(&raw))
}

/// The runtime's descriptor of `fields`, which holds `pf`, a format Metal makes no texture of
/// from a descriptor alone ([`TextureFields::formatless`]), for `sel`, a creator or the heap's
/// size query the runtime answers as Metal does for it (the module's docs). Dies first where
/// Metal's validation aborts and the runtime's passes the descriptor ([`runtime_passes`]). Its
/// fields are those set, `framebufferOnly`, the rotation, `isDrawable` and
/// `writeSwizzleEnabled` among them, with none of the checks Metal makes after its nil.
pub(crate) fn descriptor(
    sel: &'static str,
    fields: &TextureFields,
    pf: PixelFormat,
) -> rt::TextureDescriptor {
    if let Some(why) = runtime_passes(fields) {
        metal_aborts(sel, &format!("a texture of {}: {why}", pf.name()));
    }
    let mut desc = fields.desc;
    desc.pixel_format = pf;
    desc.framebuffer_only = fields.framebuffer_only == Some(true);
    desc.rotation = fields.rotation;
    desc.is_drawable = fields.is_drawable == Some(true);
    desc.write_swizzle_enabled = fields.write_swizzle_enabled == Some(true);
    desc
}

/// What Metal's validation refuses of a descriptor of a format it makes no texture of from a
/// descriptor alone, whose fields `fields` hold, and the runtime's passes, or `None` (measured
/// with every format here, and with 505 and 543 of the two-plane ones, which stand for them all:
/// the module's docs, and [`crate::gpu::ycbcr`]'s; the runtime's answers compared for each of
/// the 73 formats): usage bits Metal does not define, which the runtime's descriptor does not
/// hold. It names two more Metal aborts at, which the runtime's validation stops as well, at the
/// same op: a texture buffer wider than Metal's largest (2^28 texels), and a multisample texture
/// of other than 2 or 4 samples (the runtime's stops 1, 3, 8 and 16, as Metal's, and any count
/// not a power of two). A cube array of more than 341 cubes the runtime's validation stops, as
/// Metal's aborts at it.
fn runtime_passes(fields: &TextureFields) -> Option<&'static str> {
    use TextureType as T;
    let d = &fields.desc;
    if fields.undefined_usage() != 0 {
        return Some("usage bits Metal does not define");
    }
    match d.texture_type {
        T::TextureBuffer if d.width > MAX_BUFFER_WIDTH => Some("a size past the largest"),
        T::Type2DMultisample | T::Type2DMultisampleArray
            if !SAMPLE_COUNTS.contains(&d.sample_count) =>
        {
            Some("a sample count the type does not take")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::core::platform::Platform;
    use crate::core::platform::descriptor::TextureDescriptor as _;
    use crate::core::platform::values::TextureDesc;

    use crate::platform::tests::platform;

    /// A 2D descriptor of `format`, 16 x 16, one level, usage `ShaderRead`, storage `Private`,
    /// then `change`.
    fn of(format: u64, change: fn(&mut TextureDesc)) -> TextureDesc {
        let mut desc = TextureDesc {
            pixel_format: format,
            width: 16,
            height: 16,
            usage: 1,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        change(&mut desc);
        desc
    }

    /// A descriptor keeps a format here until another is set, which it then answers
    #[test]
    fn descriptors_keep_formats_metal_makes_none_of() {
        let p = platform();
        let desc = p.texture_descriptor(&of(556, |_| {}));
        for format in [651, 80, 652, 263, 505, 556] {
            desc.set_pixel_format(format);
            assert_eq!(desc.pixel_format(), format);
        }
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
