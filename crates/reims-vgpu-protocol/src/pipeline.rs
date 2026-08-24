//! Semantic pipeline state decoded from guest API ordinals.
//!
//! These values belong to the guest contract. Backends translate the semantic
//! values into native state; they do not decide what a guest ordinal means.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VisibilityResultMode {
    Boolean,
    Counting,
}

impl VisibilityResultMode {
    pub const fn guest_ordinal(self) -> u32 {
        match self {
            Self::Boolean => 1,
            Self::Counting => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum FillMode {
    #[default]
    Fill,
    Lines,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DepthClipMode {
    #[default]
    Clip,
    Clamp,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PrimitiveTopology {
    Point,
    Line,
    LineStrip,
    #[default]
    Triangle,
    TriangleStrip,
}

impl PrimitiveTopology {
    pub const fn guest_ordinal(self) -> u32 {
        match self {
            Self::Point => 0,
            Self::Line => 1,
            Self::LineStrip => 2,
            Self::Triangle => 3,
            Self::TriangleStrip => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexType {
    U16,
    U32,
}

impl IndexType {
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }

    pub const fn guest_ordinal(self) -> u32 {
        match self {
            Self::U16 => 0,
            Self::U32 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexTypeDecodeError {
    pub raw: u32,
}

pub const fn decode_index_type(raw: u32) -> Result<IndexType, IndexTypeDecodeError> {
    match raw {
        0 => Ok(IndexType::U16),
        1 => Ok(IndexType::U32),
        other => Err(IndexTypeDecodeError { raw: other }),
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerFilter {
    #[default]
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerMipFilter {
    #[default]
    NotMipmapped,
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerAddressMode {
    #[default]
    ClampToEdge,
    MirrorClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToZero,
    ClampToBorderColor,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerBorderColor {
    #[default]
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerCompareFunction {
    #[default]
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

impl BlendFactor {
    pub const fn is_dual_source(self) -> bool {
        matches!(
            self,
            Self::Src1Color | Self::OneMinusSrc1Color | Self::Src1Alpha | Self::OneMinusSrc1Alpha
        )
    }

    pub const fn uses_blend_constant(self) -> bool {
        matches!(
            self,
            Self::ConstantColor
                | Self::OneMinusConstantColor
                | Self::ConstantAlpha
                | Self::OneMinusConstantAlpha
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendStateResource {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
}

/// An ordinal outside the corresponding guest API enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineStateDecodeError {
    PrimitiveTopology(u32),
    CullMode(u32),
    FillMode(u32),
    DepthClipMode(u32),
    Winding(u32),
    CompareFunction(u32),
    StencilOperation(u32),
    IndexType(u32),
    VisibilityResultMode(u32),
    SamplerFilter(u32),
    SamplerMipFilter(u32),
    SamplerAddressMode(u32),
    SamplerBorderColor(u32),
    BlendFactor(u32),
    BlendOperation(u32),
}

impl PipelineStateDecodeError {
    /// The guest ordinal rejected by the semantic decoder.
    pub const fn raw(self) -> u32 {
        match self {
            Self::PrimitiveTopology(raw)
            | Self::CullMode(raw)
            | Self::FillMode(raw)
            | Self::DepthClipMode(raw)
            | Self::Winding(raw)
            | Self::CompareFunction(raw)
            | Self::StencilOperation(raw)
            | Self::IndexType(raw)
            | Self::VisibilityResultMode(raw)
            | Self::SamplerFilter(raw)
            | Self::SamplerMipFilter(raw)
            | Self::SamplerAddressMode(raw)
            | Self::SamplerBorderColor(raw)
            | Self::BlendFactor(raw)
            | Self::BlendOperation(raw) => raw,
        }
    }

    /// Stable semantic name for diagnostics at any adapter boundary.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::PrimitiveTopology(_) => "unknown_primitive_type",
            Self::CullMode(_) => "unknown_cull_mode",
            Self::FillMode(_) => "unknown_fill_mode",
            Self::DepthClipMode(_) => "unknown_depth_clip_mode",
            Self::Winding(_) => "unknown_winding",
            Self::CompareFunction(_) => "unknown_compare_function",
            Self::StencilOperation(_) => "unknown_stencil_operation",
            Self::IndexType(_) => "unknown_index_type",
            Self::VisibilityResultMode(_) => "unknown_visibility_result_mode",
            Self::SamplerFilter(_) => "unknown_sampler_filter",
            Self::SamplerMipFilter(_) => "unknown_sampler_mip_filter",
            Self::SamplerAddressMode(_) => "unknown_sampler_address_mode",
            Self::SamplerBorderColor(_) => "unknown_sampler_border_color",
            Self::BlendFactor(_) => "unknown_blend_factor",
            Self::BlendOperation(_) => "unknown_blend_operation",
        }
    }
}

pub const fn primitive_topology(raw: u32) -> Result<PrimitiveTopology, PipelineStateDecodeError> {
    Ok(match raw {
        0 => PrimitiveTopology::Point,
        1 => PrimitiveTopology::Line,
        2 => PrimitiveTopology::LineStrip,
        3 => PrimitiveTopology::Triangle,
        4 => PrimitiveTopology::TriangleStrip,
        other => return Err(PipelineStateDecodeError::PrimitiveTopology(other)),
    })
}

pub const fn cull_mode(raw: u32) -> Result<CullMode, PipelineStateDecodeError> {
    Ok(match raw {
        0 => CullMode::None,
        1 => CullMode::Front,
        2 => CullMode::Back,
        other => return Err(PipelineStateDecodeError::CullMode(other)),
    })
}

pub const fn fill_mode(raw: u32) -> Result<FillMode, PipelineStateDecodeError> {
    Ok(match raw {
        0 => FillMode::Fill,
        1 => FillMode::Lines,
        other => return Err(PipelineStateDecodeError::FillMode(other)),
    })
}

pub const fn depth_clip_mode(raw: u32) -> Result<DepthClipMode, PipelineStateDecodeError> {
    Ok(match raw {
        0 => DepthClipMode::Clip,
        1 => DepthClipMode::Clamp,
        other => return Err(PipelineStateDecodeError::DepthClipMode(other)),
    })
}

pub const fn front_face_ccw(raw: u32) -> Result<bool, PipelineStateDecodeError> {
    match raw {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(PipelineStateDecodeError::Winding(other)),
    }
}

pub const fn compare_function(
    raw: u32,
) -> Result<SamplerCompareFunction, PipelineStateDecodeError> {
    Ok(match raw {
        0 => SamplerCompareFunction::Never,
        1 => SamplerCompareFunction::Less,
        2 => SamplerCompareFunction::Equal,
        3 => SamplerCompareFunction::LessEqual,
        4 => SamplerCompareFunction::Greater,
        5 => SamplerCompareFunction::NotEqual,
        6 => SamplerCompareFunction::GreaterEqual,
        7 => SamplerCompareFunction::Always,
        other => return Err(PipelineStateDecodeError::CompareFunction(other)),
    })
}

pub const fn stencil_operation(raw: u32) -> Result<StencilOp, PipelineStateDecodeError> {
    Ok(match raw {
        0 => StencilOp::Keep,
        1 => StencilOp::Zero,
        2 => StencilOp::Replace,
        3 => StencilOp::IncrementClamp,
        4 => StencilOp::DecrementClamp,
        5 => StencilOp::Invert,
        6 => StencilOp::IncrementWrap,
        7 => StencilOp::DecrementWrap,
        other => return Err(PipelineStateDecodeError::StencilOperation(other)),
    })
}

pub const fn index_type(raw: u32) -> Result<IndexType, PipelineStateDecodeError> {
    match decode_index_type(raw) {
        Ok(value) => Ok(value),
        Err(error) => Err(PipelineStateDecodeError::IndexType(error.raw)),
    }
}

pub const fn visibility_result_mode(
    raw: u32,
) -> Result<Option<VisibilityResultMode>, PipelineStateDecodeError> {
    Ok(match raw {
        0 => None,
        1 => Some(VisibilityResultMode::Boolean),
        2 => Some(VisibilityResultMode::Counting),
        other => return Err(PipelineStateDecodeError::VisibilityResultMode(other)),
    })
}

pub const fn sampler_filter(raw: u32) -> Result<SamplerFilter, PipelineStateDecodeError> {
    Ok(match raw {
        0 => SamplerFilter::Nearest,
        1 => SamplerFilter::Linear,
        other => return Err(PipelineStateDecodeError::SamplerFilter(other)),
    })
}

pub const fn sampler_mip_filter(raw: u32) -> Result<SamplerMipFilter, PipelineStateDecodeError> {
    Ok(match raw {
        0 => SamplerMipFilter::NotMipmapped,
        1 => SamplerMipFilter::Nearest,
        2 => SamplerMipFilter::Linear,
        other => return Err(PipelineStateDecodeError::SamplerMipFilter(other)),
    })
}

pub const fn sampler_address_mode(
    raw: u32,
) -> Result<SamplerAddressMode, PipelineStateDecodeError> {
    Ok(match raw {
        0 => SamplerAddressMode::ClampToEdge,
        1 => SamplerAddressMode::MirrorClampToEdge,
        2 => SamplerAddressMode::Repeat,
        3 => SamplerAddressMode::MirrorRepeat,
        4 => SamplerAddressMode::ClampToZero,
        5 => SamplerAddressMode::ClampToBorderColor,
        other => return Err(PipelineStateDecodeError::SamplerAddressMode(other)),
    })
}

pub const fn sampler_border_color(
    raw: u32,
) -> Result<SamplerBorderColor, PipelineStateDecodeError> {
    Ok(match raw {
        0 => SamplerBorderColor::TransparentBlack,
        1 => SamplerBorderColor::OpaqueBlack,
        2 => SamplerBorderColor::OpaqueWhite,
        other => return Err(PipelineStateDecodeError::SamplerBorderColor(other)),
    })
}

pub const fn blend_factor(raw: u32) -> Result<BlendFactor, PipelineStateDecodeError> {
    Ok(match raw {
        0 => BlendFactor::Zero,
        1 => BlendFactor::One,
        2 => BlendFactor::SrcColor,
        3 => BlendFactor::OneMinusSrcColor,
        4 => BlendFactor::SrcAlpha,
        5 => BlendFactor::OneMinusSrcAlpha,
        6 => BlendFactor::DstColor,
        7 => BlendFactor::OneMinusDstColor,
        8 => BlendFactor::DstAlpha,
        9 => BlendFactor::OneMinusDstAlpha,
        10 => BlendFactor::SrcAlphaSaturated,
        11 => BlendFactor::ConstantColor,
        12 => BlendFactor::OneMinusConstantColor,
        13 => BlendFactor::ConstantAlpha,
        14 => BlendFactor::OneMinusConstantAlpha,
        15 => BlendFactor::Src1Color,
        16 => BlendFactor::OneMinusSrc1Color,
        17 => BlendFactor::Src1Alpha,
        18 => BlendFactor::OneMinusSrc1Alpha,
        other => return Err(PipelineStateDecodeError::BlendFactor(other)),
    })
}

pub const fn blend_operation(raw: u32) -> Result<BlendOp, PipelineStateDecodeError> {
    Ok(match raw {
        0 => BlendOp::Add,
        1 => BlendOp::Subtract,
        2 => BlendOp::ReverseSubtract,
        3 => BlendOp::Min,
        4 => BlendOp::Max,
        other => return Err(PipelineStateDecodeError::BlendOperation(other)),
    })
}

pub fn blend_state(
    attachment: &crate::resource::PipelineColorAttachment,
) -> Result<BlendStateResource, PipelineStateDecodeError> {
    Ok(BlendStateResource {
        src_color: blend_factor(attachment.src_rgb)?,
        dst_color: blend_factor(attachment.dst_rgb)?,
        color_op: blend_operation(attachment.op_rgb)?,
        src_alpha: blend_factor(attachment.src_alpha)?,
        dst_alpha: blend_factor(attachment.dst_alpha)?,
        alpha_op: blend_operation(attachment.op_alpha)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_pipeline_ordinals_decode_and_unknown_values_refuse() {
        assert_eq!(primitive_topology(4), Ok(PrimitiveTopology::TriangleStrip));
        assert_eq!(cull_mode(2), Ok(CullMode::Back));
        assert_eq!(compare_function(7), Ok(SamplerCompareFunction::Always));
        assert_eq!(stencil_operation(7), Ok(StencilOp::DecrementWrap));
        assert_eq!(index_type(1), Ok(IndexType::U32));
        assert_eq!(visibility_result_mode(0), Ok(None));
        assert_eq!(sampler_filter(1), Ok(SamplerFilter::Linear));
        assert_eq!(sampler_mip_filter(2), Ok(SamplerMipFilter::Linear));
        assert_eq!(
            sampler_address_mode(5),
            Ok(SamplerAddressMode::ClampToBorderColor)
        );
        assert_eq!(sampler_border_color(2), Ok(SamplerBorderColor::OpaqueWhite));
        assert_eq!(blend_factor(18), Ok(BlendFactor::OneMinusSrc1Alpha));
        assert_eq!(blend_operation(4), Ok(BlendOp::Max));
        assert_eq!(
            primitive_topology(5),
            Err(PipelineStateDecodeError::PrimitiveTopology(5))
        );
        assert_eq!(
            sampler_address_mode(6),
            Err(PipelineStateDecodeError::SamplerAddressMode(6))
        );
    }
}
