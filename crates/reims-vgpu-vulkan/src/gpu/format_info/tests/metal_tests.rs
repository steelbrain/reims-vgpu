//! The tests of `format_info` that compare with Metal on this host: macOS only.

use super::*;

/// Core's query of a level `width` by `height` in `format`.
fn level(format: u64, width: u64, height: u64) -> TextureLevelQuery {
    TextureLevelQuery {
        pixel_format: format,
        width,
        height,
        depth: 1,
        array_length: 1,
        options: 0,
        flags: 0,
    }
}

/// A level's layout is Metal's on the M3 Max, word for word: for every value from
/// 0 to 2047 and past 2^32, as core asks it, at every pair of [`EXTENTS`] (whose block counts
/// and bytes wrap at 2^62 + 3 and `u64::MAX` as Metal's do); and for every format the runtime
/// has, at queries core never makes, of other depths and array lengths (2^32 + 5, and
/// 2^62 + 3, where the bytes of a level wrap), levels (the sixth argument, past 63 and 2^32
/// among them) and plane arguments (the seventh), which the device answers by the runtime's
/// rule where this module died before
#[test]
fn levels_are_metals() {
    use crate::core::platform::device::GpuDevice as _;

    use crate::gpu::device::tests::dev;

    let d = dev();
    let bc1 = d.texture_level_info(&level(130, 17, 5));
    assert_eq!(
        (bc1.aligned_height, bc1.bytes_per_pixel, bc1.bytes_per_row),
        (8, 8, 40)
    );
    let zero = d.texture_level_info(&level(80, 0, 0));
    assert_eq!(
        (zero.field_0, zero.aligned_height, zero.bytes_per_row),
        (1, 1, 4)
    );
    let values = (0..2048).chain([9999, u64::from(u32::MAX) + 70, (1 << 32) + 80]);
    let mut differ = Vec::new();
    // A format's levels, ours checked against Metal's, recorded a format at a time.
    let mut compare = |key: String, queries: Vec<TextureLevelQuery>| {
        let ours: Vec<_> = queries.iter().map(|q| d.texture_level_info(q)).collect();
        if !crate::metal_matches_raw!(key, ours) {
            differ.push(format!("{key}: ours differ from Metal's"));
        }
    };
    for format in values {
        let mut queries = Vec::new();
        for width in EXTENTS {
            for height in EXTENTS {
                queries.push(level(format, width, height));
            }
        }
        compare(format!("format {format}"), queries);
    }
    for pf in PixelFormat::ALL {
        let mut queries = Vec::new();
        for (width, height) in [(17, 9), (1000, 7)] {
            for depth in [1, 3, (1 << 32) + 5, (1 << 62) + 3] {
                for array_length in [0, 1, 6, (1 << 32) + 5, (1 << 62) + 3] {
                    for options in [0, 1, 5, 63, 64, 1000, 1 << 32] {
                        for flags in [0, 1, 2, 3, 7, 1 << 31, u64::MAX] {
                            queries.push(TextureLevelQuery {
                                pixel_format: u64::from(pf.as_raw()),
                                width,
                                height,
                                depth,
                                array_length,
                                options,
                                flags,
                            });
                        }
                    }
                }
            }
        }
        compare(format!("{pf:?} at other queries"), queries);
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// Every format's description, and those of values no format is, are Metal's on the M3 Max,
/// field for field, at every value from 0 to 2047 and past 2^32 (2^32 + 80 among
/// them, which Metal would answer as BGRA8Unorm if it read the value's low 32 bits alone), as
/// the device answers them; a format is named as the runtime names it
#[test]
fn descriptions_are_metals() {
    use crate::core::platform::device::GpuDevice as _;

    use crate::gpu::device::tests::dev;

    let d = dev();
    let bgra = d.pixel_format_info(80);
    assert_eq!(
        (bgra.name.as_deref(), bgra.flags, bgra.block_bytes),
        (Some("MTLPixelFormatBGRA8Unorm"), 0x703_019f, 4)
    );
    let ycbcr8 = d.pixel_format_info(500);
    assert_eq!(
        (ycbcr8.name.as_deref(), ycbcr8.flags, ycbcr8.cast_class),
        (Some("MTLPixelFormatYCBCR8_420_2P"), 0x783_001d, 46)
    );
    let invalid = d.pixel_format_info(9999);
    assert_eq!(invalid.name.as_deref(), Some("MTLPixelFormatInvalid"));
    assert_eq!((invalid.flags, invalid.block_bytes), (0, 0));
    for pf in PixelFormat::ALL {
        let info = d.pixel_format_info(u64::from(pf.as_raw()));
        if info != invalid {
            assert_eq!(info.name.as_deref(), Some(pf.name()), "{pf:?}");
        }
    }
    let values = (0..2048).chain([9999, u64::from(u32::MAX) + 70, (1 << 32) + 80]);
    for format in values {
        let key = format!("format {format}");
        crate::metal_check!(key, d.pixel_format_info(format));
    }
}
