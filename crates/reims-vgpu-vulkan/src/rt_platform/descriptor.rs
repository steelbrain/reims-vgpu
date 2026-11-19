//! The descriptors the rt platform keeps as objects: the texture descriptor, the render pass
//! descriptor and the rasterization rate map's.
//!
//! - **The texture descriptor** keeps Metal's raw values as they are set ([`TextureDesc`]) and
//!   reads them back as Metal's descriptor reads them. A texture's creation is what turns them into
//!   rt's descriptor. Metal also ends the process at a `setPixelFormat:` of a format it makes no
//!   texture of. The rt platform keeps any value, so it is the texture's creation that refuses it.
//! - **The render pass descriptor** is unbuilt ([`Unbuilt::RenderPass`]).
//! - **The rate map descriptor** is unbuilt ([`Unbuilt::RateMap`]): rt has no rate maps.

use crate::core::platform::Platform;
use crate::core::platform::descriptor::{RenderPassDescriptor, TextureDescriptor};
use crate::core::platform::rate_map::RateMapDescriptor;
use crate::core::platform::types::MtlSamplePosition;
use crate::core::platform::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
    TextureDesc,
};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::{Mutex, MutexGuard};
use metal2vulkan_rt as rt;

use super::RtObj;
use super::unbuilt::{Unbuilt, refuse};

/// `MTLTextureDescriptor`: the values set, as Metal's descriptor keeps them.
pub struct TextureFields(Mutex<TextureDesc>);

impl TextureFields {
    /// A descriptor of `desc`'s values.
    pub fn new(desc: TextureDesc) -> TextureFields {
        TextureFields(Mutex::new(desc))
    }

    /// The values set so far.
    pub fn desc(&self) -> TextureDesc {
        *self.0.lock()
    }

    /// The values, to set or read.
    fn lock(&self) -> MutexGuard<'_, TextureDesc> {
        self.0.lock()
    }
}

/// Apple's private texture usage 0x8, which Metal's texture drops.
const USAGE_0X8: u32 = 0x8;

/// rt's descriptor of a texture descriptor's values `d`: each raw value as rt's type of it, the
/// storage mode of bits 4 to 7 of the resource options (Shared before any set), and each channel
/// from itself before a swizzle is set. A usage of 0x8 alone is `ShaderRead`: Metal makes a texture
/// of 0x8 alone as one of `ShaderRead` alone in every respect it shows, its heap size included,
/// and not as one of `Unknown` usage, which rt takes as every usage (measured, `crate::gpu::desc`'s
/// docs); the texture answers the guest a usage of 0 all the same. The private values (`resourceIndex`, `protectionOptions`,
/// `framebufferOnly`, `isDrawable`, `writeSwizzleEnabled`, `allowGPUOptimizedContents` and
/// `rotation`) have no field in rt's descriptor and are not sent.
///
/// Err: the first value rt's descriptor has no value of, named.
pub fn rt_texture_descriptor(d: &TextureDesc) -> Result<rt::TextureDescriptor, String> {
    let word = |what: &str, raw: u64| u32::try_from(raw).map_err(|_| format!("{what} {raw}"));
    let texture_type = word("texture type", d.texture_type)?;
    let texture_type = rt::TextureType::try_from(texture_type)
        .map_err(|_| format!("texture type {texture_type}"))?;
    let pixel_format = word("pixel format", d.pixel_format)?;
    let pixel_format = rt::PixelFormat::try_from(pixel_format)
        .map_err(|_| format!("pixel format {pixel_format}"))?;
    let (width, height) = (word("width", d.width)?, word("height", d.height)?);
    let mut out = rt::TextureDescriptor::new(texture_type, pixel_format, width, height);
    out.depth = word("depth", d.depth)?;
    out.mipmap_level_count = word("mipmap level count", d.mipmap_level_count)?;
    out.sample_count = word("sample count", d.sample_count)?;
    out.array_length = word("array length", d.array_length)?;
    let usage = match word("texture usage", d.usage)? {
        USAGE_0X8 => rt::TextureUsage::SHADER_READ.as_raw(),
        usage => usage,
    };
    out.usage =
        rt::TextureUsage::try_from(usage).map_err(|_| format!("texture usage {usage:#x}"))?;
    out.storage_mode = match (d.resource_options.unwrap_or(0) >> 4) & 0xf {
        0 => rt::StorageMode::Shared,
        1 => rt::StorageMode::Managed,
        2 => rt::StorageMode::Private,
        3 => rt::StorageMode::Memoryless,
        mode => return Err(format!("storage mode {mode}")),
    };
    if let Some(s) = d.swizzle {
        out.swizzle = crate::gpu::runtime_swizzle(s).ok_or_else(|| {
            let channels = [s.red, s.green, s.blue, s.alpha];
            format!("texture swizzle {channels:?}")
        })?;
    }
    Ok(out)
}

/// The values of `texture2DDescriptorWithPixelFormat:width:height:mipmapped:`: a 2D texture of
/// the format and size, of every level down to 1×1 where it is mipmapped (the levels of the larger
/// side), and Metal's defaults for the rest.
pub fn texture_2d_desc(pixel_format: u64, width: u64, height: u64, mipmapped: bool) -> TextureDesc {
    let levels = if mipmapped {
        u64::from(u64::BITS - width.max(height).max(1).leading_zeros())
    } else {
        1
    };
    TextureDesc {
        pixel_format,
        width,
        height,
        mipmap_level_count: levels,
        ..TextureDesc::default()
    }
}

impl<P: Platform> TextureDescriptor<P> for RtObj<TextureFields> {
    fn set_resource_options(&self, resource_options: u64) {
        self.lock().resource_options = Some(resource_options);
    }
    fn set_rotation(&self, rotation: u64) {
        self.lock().rotation = Some(rotation);
    }
    fn set_usage(&self, usage: u64) {
        self.lock().usage = usage;
    }
    fn set_width(&self, width: u64) {
        self.lock().width = width;
    }
    fn set_height(&self, height: u64) {
        self.lock().height = height;
    }
    fn set_depth(&self, depth: u64) {
        self.lock().depth = depth;
    }
    fn set_texture_type(&self, texture_type: u64) {
        self.lock().texture_type = texture_type;
    }
    fn set_pixel_format(&self, pixel_format: u64) {
        self.lock().pixel_format = pixel_format;
    }
    /// As set, 0 before any set.
    fn rotation(&self) -> u64 {
        self.lock().rotation.unwrap_or(0)
    }
    /// Bits 4 to 7 of the resource options, as Metal's descriptor reads them: 0 before any set.
    fn storage_mode(&self) -> u64 {
        (self.lock().resource_options.unwrap_or(0) >> 4) & 0xf
    }
    fn pixel_format(&self) -> u64 {
        self.lock().pixel_format
    }
    fn texture_type(&self) -> u64 {
        self.lock().texture_type
    }
}

/// `MTLRenderPassDescriptor`, unbuilt ([`Unbuilt::RenderPass`]): it keeps nothing it is set, and
/// its attachments read as nil ones do.
pub struct PassFields;

impl<P: Platform> RenderPassDescriptor<P> for RtObj<PassFields> {
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()> {
        let _ = positions;
        refuse(Unbuilt::RenderPass, "setSamplePositions:count:");
        Ok(())
    }
    fn set_pass(&self, set: &PassSet<'_, P>) {
        let _ = set;
        refuse(Unbuilt::RenderPass, "MTLRenderPassDescriptor set");
    }
    fn set_color_attachment(
        &self,
        on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, P>,
    ) -> RaiseResult<()> {
        let _ = (on, index, set);
        refuse(Unbuilt::RenderPass, "colorAttachments set");
        Ok(())
    }
    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, P>) {
        let _ = set;
        refuse(Unbuilt::RenderPass, "depthAttachment set");
    }
    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, P>) {
        let _ = set;
        refuse(Unbuilt::RenderPass, "stencilAttachment set");
    }
    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState> {
        let _ = index;
        refuse(Unbuilt::RenderPass, "colorAttachments");
        Ok(AttachmentState::default())
    }
    fn depth_attachment_state(&self) -> AttachmentState {
        refuse(Unbuilt::RenderPass, "depthAttachment");
        AttachmentState::default()
    }
    fn stencil_attachment_state(&self) -> AttachmentState {
        refuse(Unbuilt::RenderPass, "stencilAttachment");
        AttachmentState::default()
    }
}

/// `MTLRasterizationRateMapDescriptor`, unbuilt ([`Unbuilt::RateMap`]): no layer exists to set
/// on it, and it reads as one with none.
pub struct RateMapFields;

impl<P: Platform> RateMapDescriptor<P> for RtObj<RateMapFields> {
    fn set_layer(
        &self,
        on: OnFault,
        layer: &P::RateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        let _ = (on, layer, layer_index);
        refuse(Unbuilt::RateMap, "setLayer:atIndex:");
        Ok(())
    }
    fn layer_count(&self) -> u64 {
        refuse(Unbuilt::RateMap, "layerCount");
        0
    }
    fn layer(
        &self,
        on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<P::RateMapLayerDescriptor>> {
        let _ = (on, layer_index);
        refuse(Unbuilt::RateMap, "layers");
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mipmapped 2D descriptor has the levels of its larger side down to 1×1; one not mipmapped
    /// has one, as Metal's class method makes them.
    #[test]
    fn a_2d_descriptor_has_the_levels_of_its_larger_side() {
        assert_eq!(texture_2d_desc(70, 1024, 512, true).mipmap_level_count, 11);
        assert_eq!(texture_2d_desc(70, 1000, 3, true).mipmap_level_count, 10);
        assert_eq!(texture_2d_desc(70, 1, 1, true).mipmap_level_count, 1);
        assert_eq!(texture_2d_desc(70, 1024, 512, false).mipmap_level_count, 1);
        let d = texture_2d_desc(80, 16, 8, false);
        assert_eq!((d.pixel_format, d.width, d.height), (80, 16, 8));
        assert_eq!(d.texture_type, TextureDesc::default().texture_type);
    }

    /// rt's descriptor takes each value as rt's type of it, the storage mode from the resource
    /// options, and names the first value it has no type of.
    #[test]
    fn rt_s_descriptor_takes_each_raw_value() {
        let d = rt_texture_descriptor(&TextureDesc::default()).unwrap();
        assert_eq!(d.texture_type, rt::TextureType::Type2D);
        assert_eq!(d.pixel_format, rt::PixelFormat::RGBA8Unorm);
        assert_eq!((d.width, d.height, d.depth), (1, 1, 1));
        assert_eq!(d.usage, rt::TextureUsage::SHADER_READ);
        assert_eq!(d.storage_mode, rt::StorageMode::Shared);
        let private = TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 32,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        let d = rt_texture_descriptor(&private).unwrap();
        assert_eq!(d.pixel_format, rt::PixelFormat::BGRA8Unorm);
        assert_eq!(d.storage_mode, rt::StorageMode::Private);
        let wide = TextureDesc {
            width: 1 << 32,
            ..TextureDesc::default()
        };
        assert_eq!(
            rt_texture_descriptor(&wide),
            Err(format!("width {}", 1u64 << 32))
        );
        let typed = TextureDesc {
            texture_type: 99,
            ..TextureDesc::default()
        };
        assert_eq!(
            rt_texture_descriptor(&typed),
            Err("texture type 99".to_owned())
        );
        let stored = TextureDesc {
            resource_options: Some(0x70),
            ..TextureDesc::default()
        };
        assert_eq!(
            rt_texture_descriptor(&stored),
            Err("storage mode 7".to_owned())
        );
    }
}
