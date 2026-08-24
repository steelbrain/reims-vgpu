//! Vulkan representation of backend-independent texel storage layouts.

use ash::vk;
use reims_vgpu_protocol::{
    ImageFormat, SampledImageFormat, StorageImageFormat, TexelLayout, TransferFunction,
};

/// Vulkan's linear format spelling for one guest texel layout.
pub fn vk_texel_layout(layout: TexelLayout) -> vk::Format {
    match layout {
        TexelLayout::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TexelLayout::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        TexelLayout::R8 => vk::Format::R8_UNORM,
        TexelLayout::Rg8 => vk::Format::R8G8_UNORM,
        TexelLayout::R16Float => vk::Format::R16_SFLOAT,
        TexelLayout::R32Float => vk::Format::R32_SFLOAT,
        TexelLayout::R16Unorm => vk::Format::R16_UNORM,
        TexelLayout::Rg16Unorm => vk::Format::R16G16_UNORM,
        TexelLayout::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        TexelLayout::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TexelLayout::Rg16Float => vk::Format::R16G16_SFLOAT,
        TexelLayout::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        TexelLayout::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        TexelLayout::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
    }
}

/// Vulkan's sRGB spelling for layouts that define one.
pub fn srgb_texel_layout(layout: TexelLayout) -> Option<vk::Format> {
    match layout {
        TexelLayout::Rgba8 => Some(vk::Format::R8G8B8A8_SRGB),
        TexelLayout::Bgra8 => Some(vk::Format::B8G8R8A8_SRGB),
        _ => None,
    }
}

/// Vulkan representation of a semantic image-view format.
pub fn vk_image_format(format: ImageFormat) -> vk::Format {
    match format.transfer() {
        TransferFunction::Linear => vk_texel_layout(format.layout()),
        TransferFunction::Srgb => srgb_texel_layout(format.layout())
            .expect("ImageFormat only constructs sRGB-capable layouts"),
    }
}

/// Vulkan representation of a semantic sampled or storage image format.
pub fn vk_storage_image(format: StorageImageFormat) -> vk::Format {
    match format {
        StorageImageFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        StorageImageFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        StorageImageFormat::R16Float => vk::Format::R16_SFLOAT,
        StorageImageFormat::R16Uint => vk::Format::R16_UINT,
        StorageImageFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        StorageImageFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        StorageImageFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        StorageImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        StorageImageFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        StorageImageFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        StorageImageFormat::R8Unorm => vk::Format::R8_UNORM,
        StorageImageFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        StorageImageFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        StorageImageFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        StorageImageFormat::R32Uint => vk::Format::R32_UINT,
        StorageImageFormat::R32Sint => vk::Format::R32_SINT,
        StorageImageFormat::R32Float => vk::Format::R32_SFLOAT,
        StorageImageFormat::Rgb9e5Ufloat => vk::Format::E5B9G9R9_UFLOAT_PACK32,
        StorageImageFormat::R16Unorm => vk::Format::R16_UNORM,
        StorageImageFormat::Rg16Unorm => vk::Format::R16G16_UNORM,
        StorageImageFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        StorageImageFormat::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        StorageImageFormat::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        StorageImageFormat::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
    }
}

/// Vulkan view format for a semantic sampled image.
pub fn vk_sampled_image(format: SampledImageFormat) -> vk::Format {
    match format.transfer() {
        TransferFunction::Linear => vk_storage_image(format.storage()),
        TransferFunction::Srgb => match format.storage() {
            StorageImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_SRGB,
            StorageImageFormat::Bgra8Unorm => vk::Format::B8G8R8A8_SRGB,
            _ => unreachable!("SampledImageFormat validates sRGB storage shapes"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{srgb_texel_layout, vk_image_format, vk_storage_image, vk_texel_layout};
    use ash::vk;
    use reims_vgpu_protocol::{StorageImageFormat, TexelLayout};

    #[test]
    fn every_semantic_layout_has_one_vulkan_storage_format() {
        for &layout in TexelLayout::ALL {
            assert_ne!(vk_texel_layout(layout), ash::vk::Format::UNDEFINED);
            assert_eq!(
                srgb_texel_layout(layout).is_some(),
                layout.has_srgb_encoding()
            );
        }
    }

    #[test]
    fn semantic_image_formats_map_without_an_undefined_fallback() {
        let formats = [
            StorageImageFormat::Rgba32Float,
            StorageImageFormat::Rgba8Unorm,
            StorageImageFormat::R16Unorm,
            StorageImageFormat::Bgr10a2Unorm,
        ];
        for format in formats {
            assert_ne!(vk_storage_image(format), vk::Format::UNDEFINED);
        }
    }

    #[test]
    fn image_view_transfer_selects_the_native_view_without_changing_storage() {
        let linear = reims_vgpu_protocol::ImageFormat::linear(TexelLayout::Bgra8);
        let srgb = reims_vgpu_protocol::ImageFormat::srgb(TexelLayout::Bgra8).unwrap();
        assert_eq!(vk_image_format(linear), vk::Format::B8G8R8A8_UNORM);
        assert_eq!(vk_image_format(srgb), vk::Format::B8G8R8A8_SRGB);
    }
}
