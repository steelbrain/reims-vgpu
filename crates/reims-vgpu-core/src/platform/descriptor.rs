//! Platform traits for the Metal descriptors core keeps as objects (D5): the texture descriptor and
//! the render pass descriptor. The descriptors core only fills in and hands to a creator are values
//! (`platform::values`), which the platform applies inside the creator.
//!
//! A texture descriptor and a render pass descriptor are objects because Metal checks what they
//! are given when it is set: a pixel format Metal refuses aborts at `setPixelFormat:`, and
//! `setSamplePositions:count:` refuses some counts. A render pass's attachments also retain their
//! textures from the record that sets them to the pass's release. Each is made where core builds
//! it (`Platform::texture_descriptor`, `new_render_pass_descriptor`), and later sets are typed
//! methods, each sending its setters in the order core sets them.
//!
//! Conventions (architecture 4.2): a setter or getter that cannot raise returns its value (a
//! platform dies at the send on a raise); one that can raise returns `RaiseResult<T>`.

use super::types::MtlSamplePosition;
use super::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use super::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};

/// `MTLTextureDescriptor` (with its private selectors), made from a `TextureDesc` value
/// (`Platform::texture_descriptor`) or by the class method
/// `texture2DDescriptorWithPixelFormat:…`. `P` is architecture 4.1's bound
/// (`TextureDescriptor<Self>`); no method names another handle.
pub trait TextureDescriptor<P: Platform>: Handle {
    // The properties core sets on a made descriptor, and the ones it reads back.
    /// sel: setResourceOptions: — the parent resource's options, set after the build.
    fn set_resource_options(&self, resource_options: u64);
    /// sel: setRotation: SPI
    fn set_rotation(&self, rotation: u64);
    /// sel: setUsage: — on a descriptor the class method made.
    fn set_usage(&self, usage: u64);
    /// sel: setWidth: — a later plane's, on the descriptor made for the first.
    fn set_width(&self, width: u64);
    /// sel: setHeight: — a later plane's, on the descriptor made for the first.
    fn set_height(&self, height: u64);
    /// sel: setDepth: — a later plane's, on the descriptor made for the first.
    fn set_depth(&self, depth: u64);
    /// sel: setTextureType: — a later plane's, on the descriptor made for the first.
    fn set_texture_type(&self, texture_type: u64);
    /// sel: setPixelFormat: — a later plane's, on the descriptor made for the first, with the
    /// raw value (a pixel format Metal refuses aborts here).
    fn set_pixel_format(&self, pixel_format: u64);
    /// sel: rotation SPI — Metal's reading of the rotation set.
    fn rotation(&self) -> u64;
    /// sel: storageMode
    fn storage_mode(&self) -> u64;
    /// sel: pixelFormat
    fn pixel_format(&self) -> u64;
    /// sel: textureType
    fn texture_type(&self) -> u64;
}

/// `MTLRenderPassDescriptor` (with its private selectors). Each typed set sends the properties
/// its `*Set` value holds, in the order the value's type states; an attachment's set goes to the
/// attachment Metal answers (`colorAttachments`, `depthAttachment`, `stencilAttachment`), and a
/// nil attachment takes nothing.
pub trait RenderPassDescriptor<P: Platform>: Handle {
    /// sel: setSamplePositions:count: — sent when the record sets them: Metal checks the count
    /// here.
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()>;
    /// The pass's own properties ([`PassSet`]).
    fn set_pass(&self, set: &PassSet<'_, P>);
    /// Colour attachment `index`'s properties ([`ColorAttachmentSet`]):
    /// `objectAtIndexedSubscript:` may raise.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn set_color_attachment(
        &self,
        on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, P>,
    ) -> RaiseResult<()>;
    /// The depth attachment's properties ([`DepthAttachmentSet`]).
    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, P>);
    /// The stencil attachment's properties ([`StencilAttachmentSet`]).
    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, P>);
    /// Colour attachment `index`'s load and store actions as Metal reads them (zeros for a nil
    /// array or attachment); `objectAtIndexedSubscript:` may raise.
    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState>;
    /// The depth attachment's load and store actions as Metal reads them (zeros for nil).
    fn depth_attachment_state(&self) -> AttachmentState;
    /// The stencil attachment's load and store actions as Metal reads them (zeros for nil).
    fn stencil_attachment_state(&self) -> AttachmentState;
}
