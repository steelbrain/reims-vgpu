//! What the private fills write, and where: `fillBuffer:range:pattern4:`,
//! `fillTexture:level:slice:region:bytes:length:` and
//! `fillTexture:level:slice:region:color:pixelFormat:`. The blit encoder's texture fills are the
//! runtime's ([`crate::gpu::encode_blit`]), where [`place`] says: a bytes fill its `fill_texture`,
//! a colour fill its `fill_texture_color`, which packs the colour as Metal does (below) and fills
//! the texel it packs. The pattern fill is the runtime's (`fill_buffer_pattern4`), which writes it
//! as Metal does (below).
//!
//! Metal's answers, measured on `AppleM3Max-40c-25G229` with no validation:
//! - **The pattern** of `pattern4:` is its four bytes, low first, repeated from the range's start
//!   up to the first offset that is a multiple of 16 (the head), and from that offset afresh to
//!   the range's end: a fill of 7 bytes at 41 writes `01 02 03 04 01 02 03`, then `01 02 ..` from
//!   48. An empty range writes nothing (`encode_blit`'s `pattern_fills_are_the_metal_platforms`).
//! - **A texel** of either texture fill is its pattern (the bytes, or the colour's bytes) repeated
//!   from its start and cut at the texture format's block bytes: bytes past a block go unread,
//!   and a colour of fewer bytes repeats (an R8Unorm colour fills each byte of an RGBA32Uint
//!   texel).
//! - **A colour** is packed in the call's pixel format, whatever the texture's, by the runtime's
//!   rules (its `fill_value.rs`), which are Metal's. Measured: colours filled by this platform's
//!   blit encoder and by Metal's into an `RGBA32Uint` texel each, every byte compared, in each
//!   format of [`PACKED`] (`tests/metal_tests.rs`): a sweep of each value in each component
//!   (zeroes, ones, NaNs, infinities, denormals, the integer types' bounds and past them, the
//!   halves between every step of the narrow normalized codes, a ramp past [-1, 1] and randomly
//!   drawn values); the least value of each 8-bit sRGB code and of each extended-range code, and
//!   the values beside each; the halfway points of RG11B10Float's and RGB9E5Float's mantissas at
//!   every exponent and below the least normal, and the values beside each; 36 NaNs, quiet and
//!   signalling, of either sign, whose payloads set the bits a half, a float and a bfloat keep
//!   and those below them; and doubles on, beside and between the 8-, 16- and 32-bit integer
//!   edges. A colour in a format of [`OWN_FORMAT_PACKED`] (the bfloat formats, the
//!   extended-range `_sRGB` twins, the RGBX formats) is compared into a texture of its own format
//!   (`encode_blit`'s `colours_the_runtime_packs_are_the_metal_platforms`); into a texture of
//!   another format nothing compared it ([`crate::gpu::encode_blit`]).
//! - **A colour in a format Metal packs none in** (every format outside [`PACKED`] and
//!   [`OWN_FORMAT_PACKED`]: Invalid, the depth and stencil formats and the two-plane XR10
//!   formats, 550 and 551, [`LEFTOVER_FORMATS`]; the compressed, 4:2:2, YCbCr and padded ones)
//!   writes no byte of the colour, but the 16 bytes of the calling thread's stack where Metal's
//!   packing writes a colour in a format it packs, which hold what the last call to use them
//!   left (measured on the M3 Max, macOS 27.0.1, 2026-11-07): a fill sent right after an
//!   RGBA32Uint fill of (7, 9, 3, 1) into another texture, with no call between the two, writes
//!   `07 00 00 00 09 00 00 00 03 00 00 00 01 00 00 00` into an RGBA32Uint texel, cut at a
//!   narrower one, and one sent right after a call that set the stack below to one byte writes
//!   that byte. A call between them leaves its own bytes: an encoder's end and the next
//!   encoder's start leave other bytes, and objc2's check of the method before each send in a
//!   dev build, which this library's sends to Metal made, leaves the encoder's class and the
//!   selector, as pointers (measured: `80 ef 0b 82 02 00 00 00 42 a2 3d fe 01 00 00 00`),
//!   whatever came before. On the AMD Radeon Pro 5300M and the Intel UHD Graphics 630 (macOS
//!   26.7.1) every colour fill raises `NSInvalidArgumentException`: Metal's colour fill sends
//!   `fillTexture:level:slice:region:bytes:length:`, which their encoders do not implement. The
//!   runtime's colour fill holds the command buffer's last packed texel over zeros and fills
//!   those bytes, Metal's for a fill sent right after another
//!   (`leftover_colours_fill_an_earlier_colours_bytes_as_metals`, each of these formats), and
//!   announces `UnpackedFillColor`, one of Metal's answers, relayed.
//! - **Where** ([`place`]): a level past the texture's writes nothing; the region is cut at the
//!   level's extent. A 3D texture takes the region's depth and ignores the slice; every other
//!   type ignores the region's `z` and depth. A 1D or 2D texture ignores the slice; an array
//!   writes a slice it has and nothing past them; a cube writes face `slice % 6`, a cube array
//!   slice `slice` while it has it. A 2D multisample texture ignores the slice, as a 2D one
//!   does, and a multisample array writes a slice it has, every sample of the region's texels
//!   (the runtime's fill, below). Metal's work grows with the region before the cut (2^32
//!   texels: 23 ms; 2^64 did not end in 10 minutes), so a region past 2^32 texels dies, as
//!   nothing measured how it ends. A texture buffer's fill skips `place` and goes to the runtime
//!   as the call gives it ([`texture_buffer_writes`]): Metal writes it as bytes of the buffer,
//!   from the texture's offset plus the region's `x` texels, one row whatever the level, slice,
//!   row, height and depth, the pattern running across the region's bytes, not started again at
//!   each texel, past the width too and cut at the buffer's end, as the runtime's fill writes it
//!   (`texture_buffer_fills_are_the_metal_platforms`,
//!   `texture_buffer_fills_metal_streams_are_metals`), of a pattern of up to 32752 bytes; a
//!   longer one ends Metal's process in the call, whatever the region, and dies there, naming the
//!   runtime's refusal of it ([`streamed_pattern`]). Where a 16-byte chunk of Metal's reads
//!   bytes past the pattern's whole copies, which are its stack's, the runtime's reads zeros and
//!   announces `TextureBufferFillLeftover`, one of Metal's answers, relayed.
//! - **A multisample texture's** fill writes every sample of the region's texels and no other,
//!   each sample its own bytes of the pattern where a texel's samples hold 16 bytes or fewer,
//!   else every sample the pattern cut at the texel, as the runtime's fill writes them
//!   (`encode_blit`'s `multisample_fills_are_the_metal_platforms`). The runtime refuses one only
//!   where the device cannot run its sample copies, which dies naming the refusal.

use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::types::MtlRegion;

use crate::answer::unrepresentable;

/// The most texels a fill region may span before the cut: Metal took 23 ms for this many.
const MEASURED_TEXELS: u64 = 1 << 32;

/// The most bytes of its pattern a fill of a texture with bytes reads, but a texture buffer's: a
/// block of the largest format Metal describes at any level the runtime measured
/// (`RGBA32Float` and the 16-byte compressed blocks), as the runtime's fill cuts a texel from
/// the pattern at its format's block, the bytes past it unread.
pub const MOST_PATTERN_BYTES: usize = 16;

/// The most bytes of its pattern a fill of a texture buffer with bytes reads on Metal, which runs
/// the whole pattern across the buffer's bytes, its length deciding which of its bytes lands
/// where (the module's docs), as the runtime's fill streams it, byte for byte up to this length
/// (its probes on the M3 Max under macOS 27.0.1: `R8Uint`, `R32Uint` and `RGBA32Uint` texture
/// buffers): a longer pattern ends Metal's process in the call ([`streamed_pattern`]).
pub const MOST_STREAMED_PATTERN_BYTES: usize = 32752;

/// The formats Metal packs no colour in, for which a colour fill writes bytes of the calling
/// thread's stack (the module's docs).
pub const LEFTOVER_FORMATS: [PixelFormat; 10] = [
    PixelFormat::Invalid,
    PixelFormat::Depth16Unorm,
    PixelFormat::Depth32Float,
    PixelFormat::Stencil8,
    PixelFormat::Depth24UnormStencil8,
    PixelFormat::Depth32FloatStencil8,
    PixelFormat::X32Stencil8,
    PixelFormat::X24Stencil8,
    PixelFormat::RGB10A82PXR10,
    PixelFormat::RGB10A82PXR10SRGB,
];

/// The pixel formats Metal packs a fill's colour in, which the runtime's colour fill packs as
/// Metal does into a texture of any format (the module's docs).
pub const PACKED: [PixelFormat; 54] = [
    PixelFormat::A8Unorm,
    PixelFormat::R8Unorm,
    PixelFormat::R8UnormSRGB,
    PixelFormat::R8Snorm,
    PixelFormat::R8Uint,
    PixelFormat::R8Sint,
    PixelFormat::R16Unorm,
    PixelFormat::R16Snorm,
    PixelFormat::R16Uint,
    PixelFormat::R16Sint,
    PixelFormat::R16Float,
    PixelFormat::RG8Unorm,
    PixelFormat::RG8UnormSRGB,
    PixelFormat::RG8Snorm,
    PixelFormat::RG8Uint,
    PixelFormat::RG8Sint,
    PixelFormat::B5G6R5Unorm,
    PixelFormat::A1BGR5Unorm,
    PixelFormat::ABGR4Unorm,
    PixelFormat::BGR5A1Unorm,
    PixelFormat::R32Uint,
    PixelFormat::R32Sint,
    PixelFormat::R32Float,
    PixelFormat::RG16Unorm,
    PixelFormat::RG16Snorm,
    PixelFormat::RG16Uint,
    PixelFormat::RG16Sint,
    PixelFormat::RG16Float,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8UnormSRGB,
    PixelFormat::RGBA8Snorm,
    PixelFormat::RGBA8Uint,
    PixelFormat::RGBA8Sint,
    PixelFormat::ARGB8Unorm,
    PixelFormat::BGRA8Unorm,
    PixelFormat::BGRA8UnormSRGB,
    PixelFormat::BGRA10XR,
    PixelFormat::BGR10XR,
    PixelFormat::RG11B10Float,
    PixelFormat::RGB9E5Float,
    PixelFormat::RGB10A2Unorm,
    PixelFormat::RGB10A2Uint,
    PixelFormat::BGR10A2Unorm,
    PixelFormat::RG32Uint,
    PixelFormat::RG32Sint,
    PixelFormat::RG32Float,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA16Snorm,
    PixelFormat::RGBA16Uint,
    PixelFormat::RGBA16Sint,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Uint,
    PixelFormat::RGBA32Sint,
    PixelFormat::RGBA32Float,
];

/// The pixel formats Metal packs a fill's colour in whose fills the runtime's colour fill writes
/// as Metal does into a texture of the format itself, and which nothing compared in a texture of
/// another format: `R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`, `BGRA10_XR_sRGB`, `BGR10_XR_sRGB`
/// and the RGBX formats ([`RGBX`]; the module's docs).
pub const OWN_FORMAT_PACKED: [PixelFormat; 16] = [
    PixelFormat::R16Bfloat,
    PixelFormat::RG16Bfloat,
    PixelFormat::RGBA16Bfloat,
    PixelFormat::BGRA10XRSRGB,
    PixelFormat::BGR10XRSRGB,
    RGBX[0],
    RGBX[1],
    RGBX[2],
    RGBX[3],
    RGBX[4],
    RGBX[5],
    RGBX[6],
    RGBX[7],
    RGBX[8],
    RGBX[9],
    RGBX[10],
];

/// The RGBX formats ([`OWN_FORMAT_PACKED`]).
pub const RGBX: [PixelFormat; 11] = [
    PixelFormat::RGBX8Unorm,
    PixelFormat::RGBX8UnormSRGB,
    PixelFormat::RGBX8Snorm,
    PixelFormat::RGBX8Uint,
    PixelFormat::RGBX8Sint,
    PixelFormat::RGBX16Uint,
    PixelFormat::RGBX16Sint,
    PixelFormat::RGBX16Float,
    PixelFormat::RGBX32Uint,
    PixelFormat::RGBX32Sint,
    PixelFormat::RGBX32Float,
];

/// The pixel format `raw` a colour fill for `sel` packs its colour in: any format, one of
/// [`PACKED`] or [`OWN_FORMAT_PACKED`] packing it and any other taking the bytes of the calling
/// thread's stack on Metal (the module's docs). A value no format is dies.
pub fn colour_format(sel: &'static str, raw: u64) -> PixelFormat {
    let format = u32::try_from(raw)
        .ok()
        .and_then(|raw| PixelFormat::try_from(raw).ok());
    format.unwrap_or_else(|| {
        unrepresentable(
            sel,
            &format!("a colour in pixel format {raw}, which no format is"),
        )
    })
}

/// What a fill can write in a texture: its type, its first level's extent, its levels and its
/// array length.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// `textureType`.
    pub texture_type: TextureType,
    /// `width`, `height` and `depth`.
    pub extent: [u64; 3],
    /// `mipmapLevelCount`.
    pub levels: u64,
    /// `arrayLength`.
    pub array_length: u64,
}

/// Where a fill writes: the slice, the level, the origin and the size, inside the level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Place {
    /// The slice (a cube's face).
    pub slice: u64,
    /// The level.
    pub level: u64,
    /// The first texel.
    pub origin: [u64; 3],
    /// The texels across, down and deep.
    pub size: [u64; 3],
}

/// Whether a fill of `region` of a texture buffer writes a byte, given to the runtime's fill as
/// it is, which writes the buffer's bytes as Metal's does (the module's docs): a region of no
/// texels across writes nothing, as Metal's. A region past 2^32 texels across and down dies for
/// `sel`, as [`place`]'s.
pub fn texture_buffer_writes(sel: &'static str, region: MtlRegion) -> bool {
    measured(sel, region, 1);
    region.size.width != 0
}

/// Dies for `sel` at a fill of a texture buffer with a pattern of `length` bytes, past
/// [`MOST_STREAMED_PATTERN_BYTES`], where Metal's process ends in the call whatever the region (of
/// no texels and past the buffer too), naming the runtime's refusal of it
/// (`LongTextureBufferFillPattern`), before the region is read: the runtime refuses it only for a
/// region it is given, and this platform gives none of no texels across.
pub fn streamed_pattern(sel: &'static str, length: u64) {
    use reims_vgpu_core::num::ToU64 as _;
    if length > MOST_STREAMED_PATTERN_BYTES.to_u64() {
        let problem = rt::Problem::LongTextureBufferFillPattern { length };
        let e = rt::Error::Invalid(rt::Invalid::new("fill_texture", problem));
        crate::answer::refused(sel, &e);
    }
}

/// Dies for `sel` at a fill of `region`, of `depth` images, past [`MEASURED_TEXELS`].
fn measured(sel: &'static str, region: MtlRegion, depth: u64) {
    let texels = region
        .size
        .width
        .checked_mul(region.size.height)
        .and_then(|area| area.checked_mul(depth));
    if texels.is_none_or(|texels| texels > MEASURED_TEXELS) {
        unrepresentable(
            sel,
            &format!("a fill of the region {region:?}, past what was measured to end on Metal"),
        );
    }
}

/// Where a fill of `region` at `level` and `slice` writes in a texture of `shape`, as Metal
/// writes it; `None` where it writes nothing. A texture buffer (whose fill is
/// [`texture_buffer_writes`]'s), and a region past 2^32 texels, die for `sel`, as nothing
/// measured them here.
pub fn place(
    sel: &'static str,
    shape: &Shape,
    level: u64,
    slice: u64,
    region: MtlRegion,
) -> Option<Place> {
    use TextureType as T;
    let deep = shape.texture_type == T::Type3D;
    let slice = match shape.texture_type {
        T::Type1D | T::Type2D | T::Type2DMultisample | T::Type3D => 0,
        T::Type1DArray | T::Type2DArray | T::Type2DMultisampleArray => {
            (slice < shape.array_length).then_some(slice)?
        }
        T::Cube => slice % 6,
        T::CubeArray => (slice < shape.array_length.saturating_mul(6)).then_some(slice)?,
        other => unrepresentable(
            sel,
            &format!("a fill of a {other:?} texture, which nothing measured"),
        ),
    };
    let (z, depth) = if deep {
        (region.origin.z, region.size.depth)
    } else {
        (0, 1)
    };
    measured(sel, region, depth);
    if level >= shape.levels {
        return None;
    }
    let extent = shape.extent.map(|e| (e >> level.min(63)).max(1));
    let extent = [extent[0], extent[1], if deep { extent[2] } else { 1 }];
    let origin = [region.origin.x, region.origin.y, z];
    let size = [region.size.width, region.size.height, depth];
    let mut cut = [0; 3];
    for i in 0..3 {
        if size[i] == 0 || origin[i] >= extent[i] {
            return None;
        }
        cut[i] = size[i].min(extent[i] - origin[i]);
    }
    Some(Place {
        slice,
        level,
        origin,
        size: cut,
    })
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize};
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;

    /// A colour fill takes a format Metal packs its colour in, and one it packs none in (Invalid,
    /// depth, stencil, two-plane, compressed, 4:2:2, YCbCr, padded), and dies at a value no
    /// format is
    #[test]
    fn colour_formats() {
        assert_eq!(colour_format("s", 70), PixelFormat::RGBA8Unorm);
        assert_eq!(colour_format("s", 93), PixelFormat::RGB9E5Float);
        assert_eq!(colour_format("s", 553), PixelFormat::BGRA10XRSRGB);
        assert_eq!(colour_format("s", 300), PixelFormat::RGBX8Unorm);
        for format in [0, 252, 260, 550, 130, 186, 240, 505, 576] {
            let raw = u32::try_from(format).unwrap();
            assert_eq!(colour_format("s", format).as_raw(), raw);
        }
        for format in [9999, 1 << 32] {
            let site = catch_death(|| colour_format("s", format)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "s"), "{format}");
        }
    }

    /// No format Metal describes holds a block of more bytes than a fill's copy of its pattern
    /// keeps ([`MOST_PATTERN_BYTES`]), so every texel a fill writes is cut from bytes it copied
    #[test]
    fn no_block_is_longer_than_a_fills_pattern() {
        let most = u64::try_from(MOST_PATTERN_BYTES).unwrap();
        let d = crate::gpu::device::tests::dev();
        for raw in 0..2048 {
            let [bytes, ..] = d.raw.pixel_format_info(raw).words;
            assert!(bytes <= most, "format {raw}: {bytes} bytes");
        }
    }

    /// Where a fill writes, as measured on Metal: levels past the texture's write nothing; the
    /// region is cut at the level; 1D and 2D ignore the slice and every type but 3D the region's
    /// depth; arrays write their slices, a multisample array too; a cube writes face `slice %
    /// 6`. A texture buffer's fill
    /// writes but for a region of no texels across, and dies at a pattern past the bytes Metal
    /// fills, naming the runtime's refusal
    #[test]
    fn places_are_metals() {
        let shape = |texture_type, extent, levels, array_length| Shape {
            texture_type,
            extent,
            levels,
            array_length,
        };
        let r = |o: [u64; 3], s: [u64; 3]| {
            MtlRegion::new(
                MtlOrigin::new(o[0], o[1], o[2]),
                MtlSize::new(s[0], s[1], s[2]),
            )
        };
        let place_of =
            |shape: &Shape, level, slice, region| place("s", shape, level, slice, region);
        let at = |slice, level, origin, size| {
            Some(Place {
                slice,
                level,
                origin,
                size,
            })
        };
        let flat = shape(TextureType::Type2D, [4, 4, 1], 2, 1);
        assert_eq!(
            place_of(&flat, 0, 5, r([3, 0, 5], [4, 1, 0])),
            at(0, 0, [3, 0, 0], [1, 1, 1])
        );
        assert_eq!(
            place_of(&flat, 1, 0, r([1, 0, 0], [4, 4, 1])),
            at(0, 1, [1, 0, 0], [1, 2, 1])
        );
        assert_eq!(place_of(&flat, 2, 0, r([0, 0, 0], [1, 1, 1])), None);
        assert_eq!(place_of(&flat, 0, 0, r([4, 0, 0], [1, 1, 1])), None);
        assert_eq!(place_of(&flat, 0, 0, r([0, 0, 0], [0, 1, 1])), None);
        let array = shape(TextureType::Type2DArray, [2, 1, 1], 1, 2);
        assert_eq!(
            place_of(&array, 0, 1, r([0, 0, 1], [1, 1, 3])),
            at(1, 0, [0, 0, 0], [1, 1, 1])
        );
        assert_eq!(place_of(&array, 0, 2, r([0, 0, 0], [1, 1, 1])), None);
        let cube = shape(TextureType::Cube, [1, 1, 1], 1, 1);
        assert_eq!(
            place_of(&cube, 0, 100, r([0, 0, 0], [1, 1, 1])),
            at(4, 0, [0, 0, 0], [1, 1, 1])
        );
        let cubes = shape(TextureType::CubeArray, [1, 1, 1], 1, 2);
        assert_eq!(
            place_of(&cubes, 0, 7, r([0, 0, 0], [1, 1, 1])),
            at(7, 0, [0, 0, 0], [1, 1, 1])
        );
        assert_eq!(place_of(&cubes, 0, 12, r([0, 0, 0], [1, 1, 1])), None);
        let deep = shape(TextureType::Type3D, [2, 2, 2], 1, 1);
        assert_eq!(
            place_of(&deep, 0, 1, r([0, 0, 0], [1, 1, 3])),
            at(0, 0, [0, 0, 0], [1, 1, 2])
        );
        assert_eq!(place_of(&deep, 0, 0, r([0, 0, 0], [1, 1, 0])), None);
        let huge = r([0, 0, 0], [1 << 16, (1 << 16) + 1, 1]);
        let site = catch_death(|| place_of(&flat, 0, 0, huge)).unwrap_err();
        assert_eq!(site.file, "s");
        let many = shape(TextureType::Type2DMultisample, [4, 4, 1], 1, 1);
        assert_eq!(
            place_of(&many, 0, 3, r([3, 2, 1], [4, 4, 2])),
            at(0, 0, [3, 2, 0], [1, 2, 1])
        );
        let many_array = shape(TextureType::Type2DMultisampleArray, [4, 4, 1], 1, 2);
        assert_eq!(
            place_of(&many_array, 0, 1, r([0, 0, 0], [1, 1, 1])),
            at(1, 0, [0, 0, 0], [1, 1, 1])
        );
        assert_eq!(place_of(&many_array, 0, 2, r([0, 0, 0], [1, 1, 1])), None);
        let buffer = shape(TextureType::TextureBuffer, [8, 1, 1], 1, 1);
        assert!(catch_death(|| place_of(&buffer, 0, 0, r([0, 0, 0], [1, 1, 1]))).is_err());
        let writes = |region| texture_buffer_writes("s", region);
        for region in [
            r([4, 5, 3], [4, 2, 2]),
            r([7, 0, 0], [2, 1, 1]),
            r([8, 0, 0], [1, 1, 1]),
            r([0, 0, 0], [1, 0, 1]),
            r([0, 0, 0], [1, 1, 0]),
        ] {
            assert!(writes(region), "{region:?}");
        }
        assert!(!writes(r([9, 0, 0], [0, 1, 1])));
        let site = catch_death(|| writes(r([0, 0, 0], [1 << 16, (1 << 16) + 1, 1]))).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "s"));
        streamed_pattern("s", 32752);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| streamed_pattern("s", 32753)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "s"));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(
            line.contains("LongTextureBufferFillPattern { length: 32753 }"),
            "{line}"
        );
        assert_eq!(
            place_of(&flat, 0, 0, r([0, 0, 0], [1 << 16, 1 << 16, 1])),
            at(0, 0, [0, 0, 0], [4, 4, 1])
        );
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
