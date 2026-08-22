//! Semantic vertex-attribute formats decoded from guest API ordinals.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VertexAttributeFormat {
    UChar2,
    UChar3,
    UChar4,
    Char2,
    Char3,
    Char4,
    UChar2Normalized,
    UChar3Normalized,
    UChar4Normalized,
    Char2Normalized,
    Char3Normalized,
    Char4Normalized,
    UShort2,
    UShort3,
    UShort4,
    Short2,
    Short3,
    Short4,
    UShort2Normalized,
    UShort3Normalized,
    UShort4Normalized,
    Short2Normalized,
    Short3Normalized,
    Short4Normalized,
    Half2,
    Half3,
    Half4,
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    Int1010102Normalized,
    UInt1010102Normalized,
    UChar4NormalizedBgra,
    UChar,
    Char,
    UCharNormalized,
    CharNormalized,
    UShort,
    Short,
    UShortNormalized,
    ShortNormalized,
    Half,
    FloatRg11B10,
    FloatRgb9E5,
}

impl VertexAttributeFormat {
    pub const fn byte_size(self) -> u32 {
        use VertexAttributeFormat as F;
        match self {
            F::UChar | F::Char | F::UCharNormalized | F::CharNormalized => 1,
            F::UChar2
            | F::Char2
            | F::UChar2Normalized
            | F::Char2Normalized
            | F::UShort
            | F::Short
            | F::UShortNormalized
            | F::ShortNormalized
            | F::Half => 2,
            F::UChar3 | F::Char3 | F::UChar3Normalized | F::Char3Normalized => 3,
            F::UChar4
            | F::Char4
            | F::UChar4Normalized
            | F::Char4Normalized
            | F::UChar4NormalizedBgra
            | F::UShort2
            | F::Short2
            | F::UShort2Normalized
            | F::Short2Normalized
            | F::Half2
            | F::Float
            | F::Int
            | F::UInt
            | F::Int1010102Normalized
            | F::UInt1010102Normalized
            | F::FloatRg11B10
            | F::FloatRgb9E5 => 4,
            F::UShort3 | F::Short3 | F::UShort3Normalized | F::Short3Normalized | F::Half3 => 6,
            F::UShort4
            | F::Short4
            | F::UShort4Normalized
            | F::Short4Normalized
            | F::Half4
            | F::Float2
            | F::Int2
            | F::UInt2 => 8,
            F::Float3 | F::Int3 | F::UInt3 => 12,
            F::Float4 | F::Int4 | F::UInt4 => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VertexFormatDecodeError(pub u32);

impl VertexFormatDecodeError {
    pub const fn raw(self) -> u32 {
        self.0
    }

    pub const fn slug(self) -> &'static str {
        "unknown_vertex_format"
    }
}

pub const fn decode_vertex_attribute_format(
    raw: u32,
) -> Result<VertexAttributeFormat, VertexFormatDecodeError> {
    use VertexAttributeFormat as F;
    Ok(match raw {
        1 => F::UChar2,
        2 => F::UChar3,
        3 => F::UChar4,
        4 => F::Char2,
        5 => F::Char3,
        6 => F::Char4,
        7 => F::UChar2Normalized,
        8 => F::UChar3Normalized,
        9 => F::UChar4Normalized,
        10 => F::Char2Normalized,
        11 => F::Char3Normalized,
        12 => F::Char4Normalized,
        13 => F::UShort2,
        14 => F::UShort3,
        15 => F::UShort4,
        16 => F::Short2,
        17 => F::Short3,
        18 => F::Short4,
        19 => F::UShort2Normalized,
        20 => F::UShort3Normalized,
        21 => F::UShort4Normalized,
        22 => F::Short2Normalized,
        23 => F::Short3Normalized,
        24 => F::Short4Normalized,
        25 => F::Half2,
        26 => F::Half3,
        27 => F::Half4,
        28 => F::Float,
        29 => F::Float2,
        30 => F::Float3,
        31 => F::Float4,
        32 => F::Int,
        33 => F::Int2,
        34 => F::Int3,
        35 => F::Int4,
        36 => F::UInt,
        37 => F::UInt2,
        38 => F::UInt3,
        39 => F::UInt4,
        40 => F::Int1010102Normalized,
        41 => F::UInt1010102Normalized,
        42 => F::UChar4NormalizedBgra,
        45 => F::UChar,
        46 => F::Char,
        47 => F::UCharNormalized,
        48 => F::CharNormalized,
        49 => F::UShort,
        50 => F::Short,
        51 => F::UShortNormalized,
        52 => F::ShortNormalized,
        53 => F::Half,
        54 => F::FloatRg11B10,
        55 => F::FloatRgb9E5,
        other => return Err(VertexFormatDecodeError(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_vertex_format_decodes_with_its_exact_width() {
        for raw in (1..=42).chain(45..=55) {
            assert!(decode_vertex_attribute_format(raw).unwrap().byte_size() > 0);
        }
        for raw in [0, 43, 44, 56, u32::MAX] {
            assert_eq!(
                decode_vertex_attribute_format(raw),
                Err(VertexFormatDecodeError(raw))
            );
        }
    }
}
