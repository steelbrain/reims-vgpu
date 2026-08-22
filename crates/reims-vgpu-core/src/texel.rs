//! Backend-independent conversion between stored texels and host RGBA8 frames.

use reims_vgpu_protocol::TexelLayout;
use std::sync::OnceLock;

const UNORM8_MAX: u8 = u8::MAX;
const RGBA8_BYTES: usize = 4;
const COMPONENT_R: usize = 0;
const COMPONENT_G: usize = 1;
const COMPONENT_B: usize = 2;
const COMPONENT_A: usize = 3;

const F16_SIGN_MASK: u16 = 0x8000;
const F16_EXP_SHIFT: u32 = 10;
const F16_EXP_MASK: u32 = 0x1f;
const F16_MANT_MASK: u32 = 0x03ff;
const F16_HIDDEN_BIT: u32 = 0x0400;
const F16_EXP_BIAS: i32 = 15;
const F16_INF_BITS: u16 = 0x7c00;
const F16_SUBNORMAL_EXP_MIN: i32 = -10;
const F16_SUBNORMAL_SHIFT_BASE: i32 = 14;
const F16_F32_SIGN_SHIFT: u32 = 16;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_MASK: u32 = 0xff;
const F32_MANT_MASK: u32 = 0x007f_ffff;
const F32_HIDDEN_BIT: u32 = 0x0080_0000;
const F32_EXP_BIAS: i32 = 127;
const F32_INF_BITS: u32 = 0x7f80_0000;
const F16_F32_MANT_SHIFT: u32 = 13;
const F32_TO_F16_ROUND_BIT: u32 = 0x1000;

/// Decode one IEEE binary16 bit pattern.
pub fn f16_to_f32(half_bits: u16) -> f32 {
    let sign = (u32::from(half_bits & F16_SIGN_MASK)) << F16_F32_SIGN_SHIFT;
    let exp = (u32::from(half_bits) >> F16_EXP_SHIFT) & F16_EXP_MASK;
    let mut mant = u32::from(half_bits) & F16_MANT_MASK;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut normal_exp: i32 = 1;
            while (mant & F16_HIDDEN_BIT) == 0 {
                mant <<= 1;
                normal_exp -= 1;
            }
            mant &= F16_MANT_MASK;
            sign | (((normal_exp - F16_EXP_BIAS + F32_EXP_BIAS) as u32) << F32_EXP_SHIFT)
                | (mant << F16_F32_MANT_SHIFT)
        }
    } else if exp == F16_EXP_MASK {
        sign | F32_INF_BITS | (mant << F16_F32_MANT_SHIFT)
    } else {
        let f32_exp = (exp as i32 - F16_EXP_BIAS + F32_EXP_BIAS) as u32;
        sign | (f32_exp << F32_EXP_SHIFT) | (mant << F16_F32_MANT_SHIFT)
    };
    f32::from_bits(bits)
}

fn build_f16_to_unorm8() -> Box<[u8; 65536]> {
    let mut table = Box::new([0; 65536]);
    for bits in 0..=u16::MAX {
        let value = f16_to_f32(bits);
        table[bits as usize] =
            if !matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
                0
            } else if value >= 1.0 {
                UNORM8_MAX
            } else {
                (value * f32::from(UNORM8_MAX) + 0.5) as u8
            };
    }
    table
}

/// Convert binary16 to unorm8 using the shared clamp-and-round policy.
pub fn f16_to_unorm8(bits: u16) -> u8 {
    static TABLE: OnceLock<Box<[u8; 65536]>> = OnceLock::new();
    TABLE.get_or_init(build_f16_to_unorm8)[bits as usize]
}

fn unorm8_to_f16_slow(value: u8) -> u16 {
    let value = f32::from(value) / f32::from(UNORM8_MAX);
    let bits = value.to_bits();
    let sign = ((bits >> F16_F32_SIGN_SHIFT) as u16) & F16_SIGN_MASK;
    let exponent = ((bits >> F32_EXP_SHIFT) & F32_EXP_MASK) as i32 - F32_EXP_BIAS + F16_EXP_BIAS;
    let mut mantissa = bits & F32_MANT_MASK;

    if value <= 0.0 {
        return sign;
    }
    if exponent >= F16_EXP_MASK as i32 {
        return sign | F16_INF_BITS;
    }
    if exponent <= 0 {
        if exponent < F16_SUBNORMAL_EXP_MIN {
            return sign;
        }
        mantissa |= F32_HIDDEN_BIT;
        let shift = (F16_SUBNORMAL_SHIFT_BASE - exponent) as u32;
        let mut half_mantissa = mantissa >> shift;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half_mantissa += 1;
        }
        return sign | half_mantissa as u16;
    }

    let mut half = sign
        | (((exponent as u32) << F16_EXP_SHIFT) as u16)
        | ((mantissa >> F16_F32_MANT_SHIFT) as u16);
    if (mantissa & F32_TO_F16_ROUND_BIT) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

/// Convert unorm8 to binary16 using the shared rounding policy.
pub fn unorm8_to_f16(value: u8) -> u16 {
    static TABLE: OnceLock<[u16; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0; 256];
        for value in 0..=u8::MAX {
            table[value as usize] = unorm8_to_f16_slow(value);
        }
        table
    })[value as usize]
}

/// Restate semantic RGBA8 pixels as tightly packed `layout` texels.
pub fn expand_rgba8_to_texel(
    layout: TexelLayout,
    src_rgba: &[u8],
    pixels: u32,
    dst: &mut [u8],
) -> bool {
    let pixels = pixels as usize;
    let Some(dst_len) = pixels.checked_mul(layout.bytes_per_texel() as usize) else {
        return false;
    };
    let Some(src_len) = pixels.checked_mul(RGBA8_BYTES) else {
        return false;
    };
    if src_rgba.len() < src_len || dst.len() < dst_len {
        return false;
    }
    match layout {
        TexelLayout::Rgba8 => dst[..src_len].copy_from_slice(&src_rgba[..src_len]),
        TexelLayout::Bgra8 => {
            for pixel in 0..pixels {
                let offset = pixel * RGBA8_BYTES;
                dst[offset] = src_rgba[offset + COMPONENT_B];
                dst[offset + 1] = src_rgba[offset + COMPONENT_G];
                dst[offset + 2] = src_rgba[offset + COMPONENT_R];
                dst[offset + 3] = src_rgba[offset + COMPONENT_A];
            }
        }
        TexelLayout::Rgba16Float => {
            for pixel in 0..pixels {
                let src_offset = pixel * RGBA8_BYTES;
                let dst_offset = pixel * 8;
                for channel in 0..4 {
                    dst[dst_offset + channel * 2..dst_offset + channel * 2 + 2].copy_from_slice(
                        &unorm8_to_f16(src_rgba[src_offset + channel]).to_le_bytes(),
                    );
                }
            }
        }
        TexelLayout::R16Float | TexelLayout::Rg16Float => {
            let channels = layout.bytes_per_texel() as usize / 2;
            for pixel in 0..pixels {
                let src_offset = pixel * RGBA8_BYTES;
                let dst_offset = pixel * layout.bytes_per_texel() as usize;
                for channel in 0..channels {
                    dst[dst_offset + channel * 2..dst_offset + channel * 2 + 2].copy_from_slice(
                        &unorm8_to_f16(src_rgba[src_offset + channel]).to_le_bytes(),
                    );
                }
            }
        }
        TexelLayout::R8 => {
            for (pixel, value) in dst[..pixels].iter_mut().enumerate() {
                *value = src_rgba[pixel * RGBA8_BYTES + COMPONENT_R];
            }
        }
        TexelLayout::Rg8
        | TexelLayout::R32Float
        | TexelLayout::R16Unorm
        | TexelLayout::Rg16Unorm
        | TexelLayout::Rgba16Unorm
        | TexelLayout::Rgb10a2Unorm
        | TexelLayout::Bgr10a2Unorm
        | TexelLayout::Rg11b10Float => return false,
    }
    true
}

/// Restate tightly packed `layout` texels as semantic RGBA8 pixels.
pub fn narrow_texel_to_rgba8(
    layout: TexelLayout,
    src: &[u8],
    pixels: u32,
    dst_rgba: &mut [u8],
) -> bool {
    let pixels = pixels as usize;
    let Some(src_len) = pixels.checked_mul(layout.bytes_per_texel() as usize) else {
        return false;
    };
    let Some(dst_len) = pixels.checked_mul(RGBA8_BYTES) else {
        return false;
    };
    if src.len() < src_len || dst_rgba.len() < dst_len {
        return false;
    }
    match layout {
        TexelLayout::Rgba8 => dst_rgba[..dst_len].copy_from_slice(&src[..dst_len]),
        TexelLayout::Bgra8 => {
            for pixel in 0..pixels {
                let offset = pixel * RGBA8_BYTES;
                dst_rgba[offset] = src[offset + COMPONENT_B];
                dst_rgba[offset + 1] = src[offset + COMPONENT_G];
                dst_rgba[offset + 2] = src[offset + COMPONENT_R];
                dst_rgba[offset + 3] = src[offset + COMPONENT_A];
            }
        }
        TexelLayout::Rgba16Float => {
            for pixel in 0..pixels {
                let src_offset = pixel * 8;
                let dst_offset = pixel * RGBA8_BYTES;
                for channel in 0..4 {
                    let bits = u16::from_le_bytes([
                        src[src_offset + channel * 2],
                        src[src_offset + channel * 2 + 1],
                    ]);
                    dst_rgba[dst_offset + channel] = f16_to_unorm8(bits);
                }
            }
        }
        TexelLayout::R16Float | TexelLayout::Rg16Float => {
            let channels = layout.bytes_per_texel() as usize / 2;
            for pixel in 0..pixels {
                let src_offset = pixel * layout.bytes_per_texel() as usize;
                let dst_offset = pixel * RGBA8_BYTES;
                for channel in 0..channels {
                    let bits = u16::from_le_bytes([
                        src[src_offset + channel * 2],
                        src[src_offset + channel * 2 + 1],
                    ]);
                    dst_rgba[dst_offset + channel] = f16_to_unorm8(bits);
                }
                dst_rgba[dst_offset + channels..dst_offset + COMPONENT_A].fill(0);
                dst_rgba[dst_offset + COMPONENT_A] = UNORM8_MAX;
            }
        }
        TexelLayout::R8 => {
            for (pixel, value) in src[..pixels].iter().copied().enumerate() {
                let offset = pixel * RGBA8_BYTES;
                dst_rgba[offset + COMPONENT_R] = value;
                dst_rgba[offset + COMPONENT_G] = 0;
                dst_rgba[offset + COMPONENT_B] = 0;
                dst_rgba[offset + COMPONENT_A] = UNORM8_MAX;
            }
        }
        TexelLayout::Rg8
        | TexelLayout::R32Float
        | TexelLayout::R16Unorm
        | TexelLayout::Rg16Unorm
        | TexelLayout::Rgba16Unorm
        | TexelLayout::Rgb10a2Unorm
        | TexelLayout::Bgr10a2Unorm
        | TexelLayout::Rg11b10Float => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_conversion_round_trips_supported_layouts() {
        let rgba = [0, 17, 128, 255, 255, 64, 32, 1];
        for layout in [
            TexelLayout::Rgba8,
            TexelLayout::Bgra8,
            TexelLayout::Rgba16Float,
        ] {
            let mut texels = vec![0; 2 * layout.bytes_per_texel() as usize];
            let mut round_trip = [0; 8];
            assert!(expand_rgba8_to_texel(layout, &rgba, 2, &mut texels));
            assert!(narrow_texel_to_rgba8(layout, &texels, 2, &mut round_trip));
            assert_eq!(round_trip, rgba);
        }
    }

    #[test]
    fn one_channel_layout_defines_missing_channels() {
        let rgba = [17, 23, 42, 3];
        let mut texel = [0; 2];
        let mut round_trip = [9; 4];
        assert!(expand_rgba8_to_texel(
            TexelLayout::R16Float,
            &rgba,
            1,
            &mut texel
        ));
        assert!(narrow_texel_to_rgba8(
            TexelLayout::R16Float,
            &texel,
            1,
            &mut round_trip
        ));
        assert_eq!(round_trip, [17, 0, 0, 255]);
    }

    #[test]
    fn unsupported_layout_refuses_without_mutating_output() {
        let mut output = [0xaa; 4];
        assert!(!expand_rgba8_to_texel(
            TexelLayout::Rg8,
            &[1, 2, 3, 4],
            1,
            &mut output
        ));
        assert_eq!(output, [0xaa; 4]);
    }
}
