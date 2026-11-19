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
}
