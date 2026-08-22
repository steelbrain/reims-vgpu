//! Backend-independent texel storage vocabulary.

/// The byte layout of one guest texel, independent of any host graphics API.
///
/// This is a storage contract, not a rendering-backend format. Backends map it
/// into their own format vocabulary at the point where they create or access a
/// resident resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLayout {
    /// Four unorm8 channels in red, green, blue, alpha byte order.
    Rgba8,
    /// Four unorm8 channels in blue, green, red, alpha byte order.
    Bgra8,
    /// One unorm8 channel.
    R8,
    /// Two unorm8 channels.
    Rg8,
    /// One IEEE binary16 channel.
    R16Float,
    /// One IEEE binary32 channel.
    R32Float,
    /// One sixteen-bit normalized channel.
    R16Unorm,
    /// Two sixteen-bit normalized channels.
    Rg16Unorm,
    /// Four IEEE binary16 channels.
    Rgba16Float,
    /// Two IEEE binary16 channels.
    Rg16Float,
    /// Four sixteen-bit normalized channels.
    Rgba16Unorm,
    /// Packed 10-bit RGB and 2-bit alpha, red in the low bits.
    Rgb10a2Unorm,
    /// Packed 10-bit BGR and 2-bit alpha, blue in the low bits.
    Bgr10a2Unorm,
    /// Packed 11-bit red and green plus 10-bit blue floating-point channels.
    Rg11b10Float,
}

/// Fixed-function interpretation applied when an image is sampled or written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    #[default]
    Linear,
    Srgb,
}

/// Backend-independent image-view format.
///
/// Stored bytes and their fixed-function transfer are distinct: linear and
/// sRGB views may name the same allocation without naming the same rendering
/// operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageFormat {
    layout: TexelLayout,
    transfer: TransferFunction,
}

impl ImageFormat {
    pub const fn linear(layout: TexelLayout) -> Self {
        Self {
            layout,
            transfer: TransferFunction::Linear,
        }
    }

    pub fn srgb(layout: TexelLayout) -> Option<Self> {
        layout.has_srgb_encoding().then_some(Self {
            layout,
            transfer: TransferFunction::Srgb,
        })
    }

    pub fn with_transfer(layout: TexelLayout, transfer: TransferFunction) -> Option<Self> {
        match transfer {
            TransferFunction::Linear => Some(Self::linear(layout)),
            TransferFunction::Srgb => Self::srgb(layout),
        }
    }

    pub const fn layout(self) -> TexelLayout {
        self.layout
    }

    pub const fn transfer(self) -> TransferFunction {
        self.transfer
    }

    pub const fn is_srgb(self) -> bool {
        matches!(self.transfer, TransferFunction::Srgb)
    }
}

/// Source selected for one output channel of a texture view.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    Zero = 0,
    One = 1,
    R = 2,
    G = 3,
    B = 4,
    A = 5,
}

/// Semantic texture-view channel mapping, independent of host image APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwizzlePlan {
    pub source: [SwizzleSource; 4],
}

impl Default for SwizzlePlan {
    fn default() -> Self {
        swizzle_identity()
    }
}

impl SwizzlePlan {
    pub fn is_identity(&self) -> bool {
        *self == swizzle_identity()
    }

    /// Apply this mapping after `inner`, folding two view mappings into one.
    pub fn after(self, inner: &Self) -> Self {
        let mut source = self.source;
        for slot in &mut source {
            *slot = match *slot {
                SwizzleSource::Zero => SwizzleSource::Zero,
                SwizzleSource::One => SwizzleSource::One,
                SwizzleSource::R => inner.source[0],
                SwizzleSource::G => inner.source[1],
                SwizzleSource::B => inner.source[2],
                SwizzleSource::A => inner.source[3],
            };
        }
        Self { source }
    }
}

pub const fn swizzle_identity() -> SwizzlePlan {
    SwizzlePlan {
        source: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
    }
}

const fn swizzle_selector_source(selector: u8) -> Option<SwizzleSource> {
    Some(match selector {
        0 => SwizzleSource::Zero,
        1 => SwizzleSource::One,
        2 => SwizzleSource::R,
        3 => SwizzleSource::G,
        4 => SwizzleSource::B,
        5 => SwizzleSource::A,
        _ => return None,
    })
}

pub fn swizzle_plan(raw: &[u8; 4]) -> Option<SwizzlePlan> {
    let mut source = [SwizzleSource::Zero; 4];
    for (destination, selector) in source.iter_mut().zip(raw) {
        *destination = swizzle_selector_source(*selector)?;
    }
    Some(SwizzlePlan { source })
}

pub fn swizzle_is_identity(plan: &SwizzlePlan) -> bool {
    plan.is_identity()
}

pub fn apply_swizzle_rgba8(plan: &SwizzlePlan, input: [u8; 4]) -> [u8; 4] {
    let mut output = [0; 4];
    for (component, source) in output.iter_mut().zip(plan.source) {
        *component = match source {
            SwizzleSource::Zero => 0,
            SwizzleSource::One => u8::MAX,
            SwizzleSource::R => input[0],
            SwizzleSource::G => input[1],
            SwizzleSource::B => input[2],
            SwizzleSource::A => input[3],
        };
    }
    output
}

/// Typed texel format carried by semantic sampled and storage image requests.
///
/// Access and capability requirements are separate from this vocabulary: the
/// same stored format can be sampled on a host which cannot expose it for
/// storage writes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum StorageImageFormat {
    #[default]
    Rgba32Float,
    Rgba16Float,
    R16Float,
    R16Uint,
    Rgba16Uint,
    Rgba8Uint,
    Rgba8Sint,
    Rgba8Unorm,
    Bgra8Unorm,
    Rg16Float,
    R8Unorm,
    Rg8Unorm,
    Rgba32Uint,
    Rgba32Sint,
    R32Uint,
    R32Sint,
    R32Float,
    Rgb9e5Ufloat,
    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,
    Rgb10a2Unorm,
    Bgr10a2Unorm,
    Rg11b10Float,
}

impl StorageImageFormat {
    /// Bytes occupied by one stored texel.
    pub const fn bytes_per_texel(self) -> usize {
        match self {
            Self::Rgba32Float | Self::Rgba32Uint | Self::Rgba32Sint => 16,
            Self::Rgba16Float | Self::Rgba16Uint | Self::Rgba16Unorm => 8,
            Self::Rg16Float | Self::Rg16Unorm => 4,
            Self::R16Float | Self::R16Uint | Self::Rg8Unorm | Self::R16Unorm => 2,
            Self::R8Unorm => 1,
            Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rgba8Unorm
            | Self::Bgra8Unorm
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => 4,
        }
    }
}

impl From<TexelLayout> for StorageImageFormat {
    fn from(layout: TexelLayout) -> Self {
        match layout {
            TexelLayout::Rgba8 => Self::Rgba8Unorm,
            TexelLayout::Bgra8 => Self::Bgra8Unorm,
            TexelLayout::R8 => Self::R8Unorm,
            TexelLayout::Rg8 => Self::Rg8Unorm,
            TexelLayout::R16Float => Self::R16Float,
            TexelLayout::R32Float => Self::R32Float,
            TexelLayout::R16Unorm => Self::R16Unorm,
            TexelLayout::Rg16Unorm => Self::Rg16Unorm,
            TexelLayout::Rgba16Float => Self::Rgba16Float,
            TexelLayout::Rg16Float => Self::Rg16Float,
            TexelLayout::Rgba16Unorm => Self::Rgba16Unorm,
            TexelLayout::Rgb10a2Unorm => Self::Rgb10a2Unorm,
            TexelLayout::Bgr10a2Unorm => Self::Bgr10a2Unorm,
            TexelLayout::Rg11b10Float => Self::Rg11b10Float,
        }
    }
}

impl TexelLayout {
    /// Every layout in stable table-index order.
    pub const ALL: &'static [Self] = &[
        Self::Rgba8,
        Self::Bgra8,
        Self::R8,
        Self::Rg8,
        Self::R16Float,
        Self::R32Float,
        Self::R16Unorm,
        Self::Rg16Unorm,
        Self::Rgba16Float,
        Self::Rg16Float,
        Self::Rgba16Unorm,
        Self::Rgb10a2Unorm,
        Self::Bgr10a2Unorm,
        Self::Rg11b10Float,
    ];

    /// This layout's position in [`Self::ALL`].
    pub fn index(self) -> usize {
        match self {
            Self::Rgba8 => 0,
            Self::Bgra8 => 1,
            Self::R8 => 2,
            Self::Rg8 => 3,
            Self::R16Float => 4,
            Self::R32Float => 5,
            Self::R16Unorm => 6,
            Self::Rg16Unorm => 7,
            Self::Rgba16Float => 8,
            Self::Rg16Float => 9,
            Self::Rgba16Unorm => 10,
            Self::Rgb10a2Unorm => 11,
            Self::Bgr10a2Unorm => 12,
            Self::Rg11b10Float => 13,
        }
    }

    /// Bytes occupied by one texel in guest linear storage.
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R8 => 1,
            Self::Rg8 | Self::R16Float | Self::R16Unorm => 2,
            Self::Rgba8
            | Self::Bgra8
            | Self::R32Float
            | Self::Rg16Unorm
            | Self::Rg16Float
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => 4,
            Self::Rgba16Float | Self::Rgba16Unorm => 8,
        }
    }

    /// Whether this is one of the two byte-addressable four-channel layouts.
    pub fn is_four_byte_color(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }

    /// Whether the shared CPU conversion contract defines an RGBA8 loader.
    pub fn has_cpu_loader_arm(self) -> bool {
        matches!(
            self,
            Self::Rgba8 | Self::Bgra8 | Self::R8 | Self::Rg8 | Self::Rgba16Float | Self::Rg16Float
        )
    }

    /// Whether that CPU loader necessarily loses guest-visible precision.
    pub fn cpu_loader_arm_is_lossy(self) -> bool {
        matches!(self, Self::Rgba16Float | Self::Rg16Float)
    }

    /// Whether this storage order also has an sRGB backend encoding.
    pub fn has_srgb_encoding(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_swizzle_rgba8, swizzle_identity, swizzle_plan, ImageFormat, StorageImageFormat,
        SwizzlePlan, TexelLayout,
    };

    #[test]
    fn all_is_a_total_unique_index() {
        let mut seen = [false; TexelLayout::ALL.len()];
        for &layout in TexelLayout::ALL {
            let index = layout.index();
            assert!(index < seen.len());
            assert!(!seen[index], "duplicate index {index} for {layout:?}");
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|present| present));
    }

    #[test]
    fn lossy_loader_is_always_an_existing_loader() {
        for &layout in TexelLayout::ALL {
            assert!(!layout.cpu_loader_arm_is_lossy() || layout.has_cpu_loader_arm());
        }
    }

    #[test]
    fn semantic_image_formats_report_their_storage_width() {
        assert_eq!(StorageImageFormat::R8Unorm.bytes_per_texel(), 1);
        assert_eq!(StorageImageFormat::Rgba8Uint.bytes_per_texel(), 4);
        assert_eq!(StorageImageFormat::Rgba16Float.bytes_per_texel(), 8);
        assert_eq!(StorageImageFormat::Rgba32Float.bytes_per_texel(), 16);
    }

    #[test]
    fn every_texel_layout_has_an_equally_wide_storage_format() {
        for &layout in TexelLayout::ALL {
            let storage = StorageImageFormat::from(layout);
            assert_eq!(storage.bytes_per_texel(), layout.bytes_per_texel() as usize);
        }
    }

    #[test]
    fn image_views_keep_storage_and_transfer_as_separate_semantic_facts() {
        let linear = ImageFormat::linear(TexelLayout::Bgra8);
        let srgb = ImageFormat::srgb(TexelLayout::Bgra8).unwrap();

        assert_eq!(linear.layout(), srgb.layout());
        assert_ne!(linear.transfer(), srgb.transfer());
        assert!(srgb.is_srgb());
        assert!(ImageFormat::srgb(TexelLayout::R16Float).is_none());
    }

    #[test]
    fn swizzles_are_semantic_and_composable() {
        let reverse = swizzle_plan(&[4, 3, 2, 5]).unwrap();
        assert_eq!(apply_swizzle_rgba8(&reverse, [1, 2, 3, 4]), [3, 2, 1, 4]);
        assert_eq!(reverse.after(&swizzle_identity()), reverse);
        assert_eq!(SwizzlePlan::default(), swizzle_identity());
        assert!(swizzle_plan(&[6, 2, 3, 4]).is_none());
    }
}
