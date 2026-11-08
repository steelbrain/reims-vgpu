//! `MTLPixelFormatGetInfoForDevice()` and `MTLGetTextureLevelInfoForDeviceWithOptions()`, the
//! runtime's answers, at the one macOS release whose Metal every device answers as
//! (`rt::MACOS_RELEASE`, 27.0.1).
//!
//! **A pixel format's description** ([`description`]) is the runtime's
//! (`Device::pixel_format_info`): Apple's private description, from the runtime's own
//! measurements of plain Metal at every value from 0 to 2047 on an Apple M3 Max under macOS
//! 27.0.1 (26A434), which describes 264 formats (an Apple M4 describes 630 to 645 too, which the
//! runtime does not). A value Metal describes no format for is described as Metal describes it:
//! `MTLPixelFormatInvalid`, every word 0. Metal describes neither `Depth24Unorm_Stencil8` (255)
//! nor `X24_Stencil8` (262), as no Apple GPU has them. A format's description does not hang on
//! whether the runtime makes textures of it: a descriptor takes one it lacks, and a texture of it
//! dies where it is made ([`crate::gpu::desc::lacked`]) or is nil ([`crate::gpu::no_texture`]).
//! The descriptions are the Metal platform's on this host at every value
//! (`descriptions_are_the_metal_platforms_here`).
//!
//! **A texture level's layout** ([`level_layout`]), which core checks a texture on an IOSurface
//! against (its rows' bytes, its rows, its bytes a pixel), is the runtime's
//! (`Device::texture_level_info`): computed from the description by the rule the runtime fitted,
//! with no miss, to 34,306 of Metal's answers on the M3 Max and 34,626 on the M4, over
//! every described format at 20 sizes, levels (the sixth argument) from 0 to 2^32, depths, array
//! lengths, plane arguments (the seventh), values no format is, and zero sizes, which Metal
//! answers as sizes of 1. `levels_are_the_metal_platforms_here` compares it with the Metal
//! platform on this host, where it wraps at 2^64 as Metal does: it counts an extent's blocks as
//! `(extent + block - 1) / block` with an add that wraps, so a width of `u64::MAX` has no blocks
//! 4 wide (`wrapped_block_counts_are_metals`), and its products of blocks, bytes, depths and
//! array lengths wrap (at extents of 2^62 + 3 and `u64::MAX`, and depths and array lengths of
//! 2^32 + 5 and 2^62 + 3). Core asks none of these: its extents are the guest's 32-bit width and
//! height, at depth 1 and array length 1.
//!
//! **What this platform's own rules read of a description** ([`block`]: a texture's bytes by
//! Apple's layout, a CPU write's unit) is the description's compressed bit and block words.
//!
//! **Which values Metal's setter takes** ([`setter_takes`]): every value of a format the runtime
//! has, `MTLPixelFormatInvalid` among them, and **which formats Metal describes** ([`described`]):
//! those its description names.
//!
//! The answers need a device, any of the process's, as every device answers alike
//! ([`crate::platform::answering_device`]).

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::types::{PixelFormatInfo, TextureLevelInfo, TextureLevelQuery};

/// Metal's description of `raw`, asked by `sel` of the process's device
/// ([`crate::platform::answering_device`]).
fn info(sel: &'static str, raw: u64) -> rt::PixelFormatInfo {
    crate::platform::answering_device(sel).pixel_format_info(raw)
}

/// Whether Metal describes `raw` as a format: a format the runtime has that Metal's description
/// names, which `Depth24Unorm_Stencil8` (255) and `X24_Stencil8` (262) are not, as no Apple GPU
/// has them. These are the 264 formats the Metal platform describes on this host
/// (`the_runtime_has_every_format_metal_describes`).
pub(crate) fn described(sel: &'static str, raw: u64) -> bool {
    crate::gpu::runtime_pixel_format(raw).is_some() && info(sel, raw).describes_a_format()
}

/// Whether Metal's `setPixelFormat:` takes `raw`: a value of a format the runtime has,
/// `MTLPixelFormatInvalid` among them. Metal's setter takes the formats Metal describes,
/// `MTLPixelFormatInvalid`, 255, 262, 630 to 650 and the 13 three-channel formats, and asserts at
/// every other value ("pixelFormat (2000) is not a valid MTLPixelFormat"); every texture creator
/// asserts at a descriptor of one of those it takes but describes none for ("MTLTextureDescriptor
/// has invalid pixelFormat (630)"), but `newSharedTextureWithDescriptor:`, which answers nil for
/// one at a storage other than `Private`, and at `Private` for 0 and 630 to 650
/// (`formats_the_runtime_lacks_die_where_textures_are_made`).
pub(crate) fn setter_takes(raw: u64) -> bool {
    crate::gpu::runtime_pixel_format(raw).is_some()
}

/// `flags` bit 10: a format of compressed blocks (core's `KIND_COMPRESSED`), whose block words
/// are its block's width and height.
pub(crate) const COMPRESSED: u64 = 0x400;

/// `MTLPixelFormatGetInfoForDevice()` of `raw` on `device`: Metal's description, the runtime's
/// (`Device::pixel_format_info`), field for field.
pub(crate) fn description(device: &rt::Device, raw: u64) -> PixelFormatInfo {
    let info = device.pixel_format_info(raw);
    let [block_bytes, block_width, block_height, block_depth] = info.words;
    PixelFormatInfo {
        name: Some(info.name.to_owned()),
        flags: info.flags,
        cast_class: info.cast_class,
        block_bytes,
        block_width,
        block_height,
        block_depth,
    }
}

/// `MTLGetTextureLevelInfoForDeviceWithOptions()` of `q` on `device`: Metal's layout of the
/// level, the runtime's (`Device::texture_level_info`), word for word. Core's sixth and seventh
/// arguments (`options` and `flags`) are the runtime's level and plane argument, as its probes
/// found Metal reads them.
pub(crate) fn level_layout(device: &rt::Device, q: &TextureLevelQuery) -> TextureLevelInfo {
    let query = rt::TextureLevelQuery {
        pixel_format: q.pixel_format,
        width: q.width,
        height: q.height,
        depth: q.depth,
        array_length: q.array_length,
        level: q.options,
        planes: q.flags,
    };
    let [
        field_0,
        field_8,
        field_10,
        field_18,
        field_20,
        field_28,
        field_30,
        aligned_height,
        field_40,
        bytes_per_pixel,
        bytes_per_row,
        field_58,
        field_60,
    ] = device.texture_level_info(&query).words();
    TextureLevelInfo {
        field_0,
        field_8,
        field_10,
        field_18,
        field_20,
        field_28,
        field_30,
        aligned_height,
        field_40,
        bytes_per_pixel,
        bytes_per_row,
        field_58,
        field_60,
    }
}

/// Metal's name for `raw`, its description's (`MTLPixelFormatInvalid` for a value it describes
/// no format for), for diagnostics, asked by `sel`.
pub(crate) fn name(sel: &'static str, raw: u64) -> &'static str {
    info(sel, raw).name
}

/// What this platform's own rules read of Metal's description of a format ([`block`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    /// Whether the format is one of blocks: flags bit 10 ([`COMPRESSED`]).
    compressed: bool,
    /// The description's four words.
    words: [u64; 4],
}

impl Block {
    /// Whether the format is one of blocks: flags bit 10 ([`COMPRESSED`]), the compressed formats
    /// and the 4:2:2 `GBGR422` and `BGRG422`.
    pub(crate) fn compressed(self) -> bool {
        self.compressed
    }

    /// A block's bytes (the first word): a texel's for a format not of blocks, 0 for a value Metal
    /// describes no format for.
    pub(crate) fn bytes(self) -> u64 {
        let [bytes, ..] = self.words;
        bytes
    }

    /// The block's width and height in texels (the second and third words) for a format of
    /// blocks; one texel for any other.
    pub(crate) fn extent(self) -> [u64; 2] {
        let [_, width, height, _] = self.words;
        match self.compressed {
            true => [width, height],
            false => [1, 1],
        }
    }
}

/// What this platform's own rules read of Metal's description of `raw` (a texture's bytes by
/// Apple's layout, [`crate::gpu::metal_space`]; a CPU write's unit, [`crate::gpu::resource`]),
/// asked by `sel`.
pub(crate) fn block(sel: &'static str, raw: u64) -> Block {
    let info = info(sel, raw);
    Block {
        compressed: info.flags & COMPRESSED != 0,
        words: info.words,
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use metal2vulkan::metal::PixelFormat;

    use super::*;

    /// The selector a description is asked by here.
    const INFO: &str = "MTLPixelFormatGetInfoForDevice()";

    /// Every value Metal describes, from 0 to 2047.
    fn described_values() -> Vec<u64> {
        let d = crate::gpu::device::tests::dev();
        (0..2048)
            .filter(|&raw| d.raw.pixel_format_info(raw).describes_a_format())
            .collect()
    }

    /// The runtime has a format of every value Metal describes, the 264 formats Metal describes on
    /// the measured host, so a value this platform keeps raw ([`crate::gpu::held_format`]) is one
    /// Metal describes no format for: the death of a texture, view or state of one is Metal's
    /// assertion ([`crate::gpu::desc::lacked_as`]). Metal's setter takes what Metal describes,
    /// `MTLPixelFormatInvalid`, 255, 262, 630 to 650 and the 13 three-channel formats, measured,
    /// and Metal describes neither of the two formats no Apple GPU has, `Depth24Unorm_Stencil8`
    /// (255) and `X24_Stencil8` (262)
    #[test]
    fn the_runtime_has_every_format_metal_describes() {
        let values = described_values();
        assert_eq!(values.len(), 264);
        for raw in 0..2048 {
            let measured = values.contains(&raw);
            assert_eq!(described(INFO, raw), measured, "{raw}");
            let taken = measured
                || matches!(
                    raw,
                    0 | 255 | 262 | 630..=650 | 45..=48 | 95..=100 | 120..=122
                );
            assert_eq!(setter_takes(raw), taken, "{raw}");
        }
    }

    /// The extents a level's layout is compared at: zeros, which Metal answers as 1, sizes that
    /// round to blocks and that do not, and sizes whose block counts or bytes wrap.
    #[cfg(target_os = "macos")]
    const EXTENTS: [u64; 12] = [
        0,
        1,
        3,
        4,
        5,
        16,
        17,
        100,
        4096,
        16384,
        (1 << 62) + 3,
        u64::MAX,
    ];

    /// Metal's layouts where a level's block count wraps, as the M3 Max answered them: Metal counts an extent's blocks as `(extent + block - 1) / block` with an add that
    /// wraps at 2^64, so a width or height past 2^64 - block has no blocks (`u64::MAX` has none 4
    /// wide), and the extent they cover and the bytes of a row, an image and the level follow from
    /// the count. The runtime answers each so (`levels_are_the_metal_platforms_here` compares every
    /// format so on the device). Core never asks one: its extents are the guest's
    /// 32-bit width and height
    #[test]
    fn wrapped_block_counts_are_metals() {
        const MAX: u64 = u64::MAX;
        const PAST_2_62: u64 = (1 << 62) + 3;
        // (format, width, height, Metal's 13 words): BC1_RGBA (4 by 4 blocks of 8 bytes),
        // ASTC_12x10_LDR (12 by 10 of 16) and GBGR422 (2 by 1 of 4).
        let measured: [(u64, u64, u64, [u64; 13]); 4] = [
            (130, MAX, 17, [MAX, 17, 1, 0, 5, 1, 0, 20, 1, 8, 0, 0, 0]),
            (130, 17, MAX, [17, MAX, 1, 5, 0, 1, 20, 0, 1, 8, 40, 0, 0]),
            (
                217,
                PAST_2_62,
                MAX,
                [
                    PAST_2_62,
                    MAX,
                    1,
                    384_307_168_202_282_326,
                    0,
                    1,
                    4_611_686_018_427_387_912,
                    0,
                    1,
                    16,
                    6_148_914_691_236_517_216,
                    0,
                    0,
                ],
            ),
            (240, MAX, 5, [MAX, 5, 1, 0, 5, 1, 0, 5, 1, 4, 0, 0, 0]),
        ];
        let d = crate::gpu::device::tests::dev();
        for (format, width, height, metal) in measured {
            let q = rt::TextureLevelQuery::plane_2d(format, width, height);
            let runtime = d.raw.texture_level_info(&q);
            assert_eq!(runtime.words(), metal, "{format}, {width} by {height}");
        }
    }

    /// What this platform's rules read of a description: of `BGRA8Unorm` a texel of 4 bytes, of
    /// `ASTC_12x10_LDR` blocks of 16 bytes 12 by 10, and at 630 to 645, which Metal does not
    /// describe (an Apple M4 does), no format, named `MTLPixelFormatInvalid`
    #[test]
    fn rules_read_metals_description() {
        let bgra = block(INFO, 80);
        assert_eq!(
            (bgra.compressed(), bgra.bytes(), bgra.extent()),
            (false, 4, [1, 1])
        );
        let astc = block(INFO, 217);
        assert_eq!(
            (astc.compressed(), astc.bytes(), astc.extent()),
            (true, 16, [12, 10])
        );
        for raw in 630..=645 {
            let none = block(INFO, raw);
            assert_eq!((none.bytes(), none.extent()), (0, [1, 1]), "{raw}");
            assert_eq!(name(INFO, raw), "MTLPixelFormatInvalid", "{raw}");
        }
        assert_eq!(name(INFO, 80), "MTLPixelFormatBGRA8Unorm");
    }

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
