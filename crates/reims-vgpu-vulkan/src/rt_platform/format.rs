//! `MTLPixelFormatGetInfoForDevice()` and `MTLGetTextureLevelInfoForDeviceWithOptions()` on rt:
//! rt's answers (`Device::pixel_format_info` and `Device::texture_level_info`), mapped field for
//! field as the Vulkan platform maps its runtime's (`crate::gpu::format_info`).
//!
//! rt's description of a raw pixel format value is Metal's on an Apple M3 Max under macOS 27.0.1,
//! the same on every device and the old runtime's unchanged: a format described there may be one
//! the device makes no texture of, which rt refuses where the texture is made
//! (`Gap::TextureFormat`). A level's layout is computed from the description by the rule rt
//! fitted to Metal's answers. Neither asks Vulkan anything.

use metal2vulkan_rt as rt;

use crate::core::platform::types::{PixelFormatInfo, TextureLevelInfo, TextureLevelQuery};

/// `MTLPixelFormatGetInfoForDevice()` of `raw` on `device`: rt's description, field for field.
pub fn description(device: &rt::Device, raw: u64) -> PixelFormatInfo {
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

/// `MTLGetTextureLevelInfoForDeviceWithOptions()` of `q` on `device`: rt's layout of the level,
/// word for word. Core's sixth and seventh arguments (`options` and `flags`) are rt's level and
/// plane argument, as rt's probes found Metal reads them.
pub fn level_layout(device: &rt::Device, q: &TextureLevelQuery) -> TextureLevelInfo {
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
