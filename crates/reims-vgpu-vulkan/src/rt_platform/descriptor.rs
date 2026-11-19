//! The descriptors the rt platform keeps as objects: the texture descriptor, the render pass
//! descriptor and the rasterization rate map's.
//!
//! - **The texture descriptor** keeps Metal's raw values as they are set ([`TextureDesc`]) and
//!   reads them back as Metal's descriptor reads them. A texture's creation is what turns them into
//!   rt's descriptor. Metal also ends the process at a `setPixelFormat:` of a format it makes no
//!   texture of. The rt platform keeps any value, so it is the texture's creation that refuses it.
//! - **The render pass descriptor** keeps what rt's pass has a field of, as it is set
//!   ([`PassFields`]), from Metal's defaults: each attachment's texture, held as Metal's
//!   descriptor retains it, its level and slice, its load and store actions as Metal's raw values,
//!   which its reads answer, and its clear value; and the render target's width and height. A
//!   render encoder's creation turns them into rt's descriptor ([`PassFields::rt_pass`]): an
//!   action Metal has none of, or a level or slice past 32 bits, is Metal's refusal there, as
//!   Metal asserts at the encoder's creation, and a width or height past 32 bits renders the
//!   attachments' area, as Metal's does. A colour attachment past the eighth raises at its
//!   subscript, as Metal's does.
//!   - **Kept nowhere, changing nothing rt renders**: a resolve texture, level, slice and depth
//!     plane, and the resolve filters (a resolving store action is rt's refusal at the encoder); a
//!     depth plane (rt renders into no 3D texture); store action options (rt renders one sample);
//!     `defaultRasterSampleCount` (rt renders no pass without an attachment);
//!     `defaultColorSampleCount` (SPI); and `visibilityResultType` (no visibility buffer is built).
//!   - **Unbuilt** ([`Unbuilt::RenderPass`]), counted at a set of other than Metal's default, and
//!     kept nowhere, so the pass renders without them: a visibility result buffer, a render target
//!     array length past 1, an imageblock sample length, a threadgroup memory length, a tile width
//!     or height, and sample positions. No rasterization rate map is ever made to set.
//! - **The rate map descriptor** is unbuilt ([`Unbuilt::RateMap`]): rt has no rate maps.

use crate::core::platform::descriptor::{RenderPassDescriptor, TextureDescriptor};
use crate::core::platform::rate_map::RateMapDescriptor;
use crate::core::platform::types::{ClearColor, MtlSamplePosition};
use crate::core::platform::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
    TextureDesc,
};
use crate::core::platform::{Handle, Platform};
use crate::core::raise::{OnFault, Raise, RaiseResult};
use crate::core::sync::{Mutex, MutexGuard};
use metal2vulkan_rt as rt;

use super::unbuilt::{Unbuilt, refuse};
use super::{RtObj, RtPlatform, RtTexture};

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

/// `MTLLoadActionDontCare`.
const LOAD_DONT_CARE: u64 = 0;

/// `MTLLoadActionClear`.
const LOAD_CLEAR: u64 = 2;

/// `MTLStoreActionDontCare`.
const STORE_DONT_CARE: u64 = 0;

/// `MTLStoreActionStore`.
const STORE_STORE: u64 = 1;

/// The colour attachments' subscript, which raises past the eighth.
const SUBSCRIPT: &str = "objectAtIndexedSubscript:";

/// `MTLRenderPassDescriptor`: what is set, as Metal's descriptor keeps it (the module's docs).
#[derive(Default)]
pub struct PassFields(Mutex<Pass>);

/// One attachment as set: its texture, held as Metal's descriptor retains it, its level and slice,
/// and its load and store actions as Metal's raw values.
struct Attachment {
    texture: Option<RtObj<RtTexture>>,
    level: u64,
    slice: u64,
    load_action: u64,
    store_action: u64,
}

impl Attachment {
    /// One of no texture, level and slice 0, and the actions `load_action` and `store_action`.
    fn of(load_action: u64, store_action: u64) -> Attachment {
        Attachment {
            texture: None,
            level: 0,
            slice: 0,
            load_action,
            store_action,
        }
    }

    /// Takes the texture, level, slice and actions a set sends; nil for `Some(None)`.
    fn set(
        &mut self,
        texture: Option<Option<&RtObj<RtTexture>>>,
        (level, slice): (Option<u64>, Option<u64>),
        (load_action, store_action): (Option<u64>, Option<u64>),
    ) {
        if let Some(texture) = texture {
            self.texture = texture.map(Handle::clone_ref);
        }
        self.level = level.unwrap_or(self.level);
        self.slice = slice.unwrap_or(self.slice);
        self.load_action = load_action.unwrap_or(self.load_action);
        self.store_action = store_action.unwrap_or(self.store_action);
    }

    /// Its actions, as Metal reads them.
    fn state(&self) -> AttachmentState {
        AttachmentState {
            load_action: self.load_action,
            store_action: self.store_action,
        }
    }

    /// rt's texture, level, slice and actions of it, which `what` names; `Err` names a level or
    /// slice past 32 bits, or an action Metal has none of.
    fn rt(&self, what: &str) -> Result<RtAttachment, String> {
        let word =
            |name: &str, raw: u64| u32::try_from(raw).map_err(|_| format!("{what}'s {name} {raw}"));
        let load = match self.load_action {
            0 => rt::LoadAction::DontCare,
            1 => rt::LoadAction::Load,
            2 => rt::LoadAction::Clear,
            raw => return Err(format!("{what}'s load action {raw}")),
        };
        let store = match self.store_action {
            0 => rt::StoreAction::DontCare,
            1 => rt::StoreAction::Store,
            2 => rt::StoreAction::MultisampleResolve,
            3 => rt::StoreAction::StoreAndMultisampleResolve,
            4 => rt::StoreAction::Unknown,
            5 => rt::StoreAction::CustomSampleDepthStore,
            raw => return Err(format!("{what}'s store action {raw}")),
        };
        Ok(RtAttachment {
            texture: self.texture.as_ref().map(|t| t.rt().clone()),
            level: word("level", self.level)?,
            slice: word("slice", self.slice)?,
            load,
            store,
        })
    }
}

/// What rt's attachment descriptors share, of one attachment.
struct RtAttachment {
    texture: Option<rt::Texture>,
    level: u32,
    slice: u32,
    load: rt::LoadAction,
    store: rt::StoreAction,
}

/// What a render pass descriptor keeps.
struct Pass {
    colors: [Attachment; rt::MAX_COLOR_ATTACHMENTS],
    clear_colors: [ClearColor; rt::MAX_COLOR_ATTACHMENTS],
    depth: Attachment,
    clear_depth: f64,
    stencil: Attachment,
    clear_stencil: u32,
    render_target_width: u64,
    render_target_height: u64,
}

impl Default for Pass {
    /// Metal's: colour attachments of `DontCare` loads and `Store` stores, cleared to opaque
    /// black; depth and stencil of `Clear` loads and `DontCare` stores, cleared to 1 and 0; no
    /// texture, level and slice 0; no render target size.
    fn default() -> Pass {
        let black = ClearColor {
            alpha: 1.0,
            ..ClearColor::default()
        };
        Pass {
            colors: std::array::from_fn(|_| Attachment::of(LOAD_DONT_CARE, STORE_STORE)),
            clear_colors: [black; rt::MAX_COLOR_ATTACHMENTS],
            depth: Attachment::of(LOAD_CLEAR, STORE_DONT_CARE),
            clear_depth: 1.0,
            stencil: Attachment::of(LOAD_CLEAR, STORE_DONT_CARE),
            clear_stencil: 0,
            render_target_width: 0,
            render_target_height: 0,
        }
    }
}

impl PassFields {
    /// The pass's values, to set or read.
    fn lock(&self) -> MutexGuard<'_, Pass> {
        self.0.lock()
    }

    /// rt's descriptor of what is set (the module's docs). `Err` names the first value Metal
    /// takes no pass of: an action Metal has none of, or a level or slice past 32 bits.
    pub fn rt_pass(&self) -> Result<rt::RenderPassDescriptor, String> {
        let pass = self.lock();
        let mut out = rt::RenderPassDescriptor::default();
        let colors = (out.color_attachments.iter_mut())
            .zip(&pass.colors)
            .zip(&pass.clear_colors);
        for (index, ((out, a), c)) in colors.enumerate() {
            let a = a.rt(&format!("colour attachment {index}"))?;
            *out = rt::RenderPassColorAttachmentDescriptor {
                texture: a.texture,
                level: a.level,
                slice: a.slice,
                load_action: a.load,
                store_action: a.store,
                clear_color: rt::ClearColor::new(c.red, c.green, c.blue, c.alpha),
            };
        }
        let a = pass.depth.rt("the depth attachment")?;
        out.depth_attachment = rt::RenderPassDepthAttachmentDescriptor {
            texture: a.texture,
            level: a.level,
            slice: a.slice,
            load_action: a.load,
            store_action: a.store,
            clear_depth: pass.clear_depth,
        };
        let a = pass.stencil.rt("the stencil attachment")?;
        out.stencil_attachment = rt::RenderPassStencilAttachmentDescriptor {
            texture: a.texture,
            level: a.level,
            slice: a.slice,
            load_action: a.load,
            store_action: a.store,
            clear_stencil: pass.clear_stencil,
        };
        // A size past 32 bits renders the area a size of the most 32 bits do: the attachments'.
        out.render_target_width = u32::try_from(pass.render_target_width).unwrap_or(u32::MAX);
        out.render_target_height = u32::try_from(pass.render_target_height).unwrap_or(u32::MAX);
        Ok(out)
    }
}

/// Colour attachment `index`, or Metal's raise at its subscript past the eighth.
fn color_index(index: u64) -> Result<usize, Raise> {
    match usize::try_from(index) {
        Ok(i) if i < rt::MAX_COLOR_ATTACHMENTS => Ok(i),
        _ => Err(Raise::platform(SUBSCRIPT).with_detail(format!(
            "colour attachment {index} of {}",
            rt::MAX_COLOR_ATTACHMENTS
        ))),
    }
}

impl RenderPassDescriptor<RtPlatform> for RtObj<PassFields> {
    /// Counted for any position ([`Unbuilt::RenderPass`]): rt renders one sample alone, at its
    /// centre, and keeps none.
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()> {
        if !positions.is_empty() {
            refuse(Unbuilt::RenderPass, "setSamplePositions:count:");
        }
        Ok(())
    }

    /// The render target's width and height; the rest as the module's docs say.
    fn set_pass(&self, set: &PassSet<'_, RtPlatform>) {
        let unbuilt = [
            (
                matches!(set.visibility_result_buffer, Some(Some(_))),
                "setVisibilityResultBuffer:",
            ),
            (
                set.render_target_array_length.is_some_and(|n| n > 1),
                "setRenderTargetArrayLength:",
            ),
            (
                set.imageblock_sample_length.is_some_and(|n| n != 0),
                "setImageblockSampleLength:",
            ),
            (
                set.threadgroup_memory_length.is_some_and(|n| n != 0),
                "setThreadgroupMemoryLength:",
            ),
            (set.tile_width.is_some_and(|n| n != 0), "setTileWidth:"),
            (set.tile_height.is_some_and(|n| n != 0), "setTileHeight:"),
        ];
        for (_, sel) in unbuilt.into_iter().filter(|&(sent, _)| sent) {
            refuse(Unbuilt::RenderPass, sel);
        }
        let mut pass = self.lock();
        if let Some(width) = set.render_target_width {
            pass.render_target_width = width;
        }
        if let Some(height) = set.render_target_height {
            pass.render_target_height = height;
        }
    }

    /// An index past the eight attachments raises before any set, as Metal's subscript does.
    fn set_color_attachment(
        &self,
        on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, RtPlatform>,
    ) -> RaiseResult<()> {
        let i = match color_index(index) {
            Ok(i) => i,
            Err(raise) => return on.throw(raise),
        };
        let mut pass = self.lock();
        let actions = (set.load_action, set.store_action);
        pass.colors[i].set(set.texture, (set.level, set.slice), actions);
        if let Some(c) = set.clear_color {
            pass.clear_colors[i] = c;
        }
        Ok(())
    }

    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, RtPlatform>) {
        let mut pass = self.lock();
        let actions = (set.load_action, set.store_action);
        pass.depth.set(set.texture, (set.level, set.slice), actions);
        pass.clear_depth = set.clear_depth.unwrap_or(pass.clear_depth);
    }

    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, RtPlatform>) {
        let mut pass = self.lock();
        let actions = (set.load_action, set.store_action);
        pass.stencil
            .set(set.texture, (set.level, set.slice), actions);
        pass.clear_stencil = set.clear_stencil.unwrap_or(pass.clear_stencil);
    }

    /// An index past the eight attachments raises, the caller's to catch.
    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState> {
        let i = color_index(index)?;
        Ok(self.lock().colors[i].state())
    }

    fn depth_attachment_state(&self) -> AttachmentState {
        self.lock().depth.state()
    }

    fn stencil_attachment_state(&self) -> AttachmentState {
        self.lock().stencil.state()
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
    use crate::core::raise::Origin;

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

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A pass descriptor answers Metal's defaults and the actions set, raw; rt's descriptor takes
    /// each value kept, and names an action Metal has none of or a level past 32 bits; a colour
    /// attachment past the eighth raises at the subscript; a property rt has no field of is
    /// counted where it is set to other than Metal's default.
    #[test]
    fn a_pass_keeps_what_rt_renders() {
        let pass = RtObj::new(PassFields::default());
        let state = |load_action, store_action| AttachmentState {
            load_action,
            store_action,
        };
        assert_eq!(pass.color_attachment_state(7), Ok(state(0, 1)));
        assert_eq!(pass.depth_attachment_state(), state(2, 0));
        assert_eq!(pass.stencil_attachment_state(), state(2, 0));
        let d = pass.rt_pass().expect("rt's descriptor");
        let c = &d.color_attachments[0];
        assert_eq!(c.load_action, rt::LoadAction::DontCare);
        assert_eq!(c.store_action, rt::StoreAction::Store);
        assert_eq!(c.clear_color, rt::ClearColor::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(d.depth_attachment.load_action, rt::LoadAction::Clear);
        assert_eq!(d.depth_attachment.clear_depth, 1.0);
        assert_eq!(d.stencil_attachment.store_action, rt::StoreAction::DontCare);
        assert_eq!((d.render_target_width, d.render_target_height), (0, 0));

        let on = OnFault::Return;
        let red = ColorAttachmentSet {
            slice: Some(2),
            load_action: Some(2),
            store_action: Some(0),
            clear_color: Some(ClearColor {
                red: 1.0,
                alpha: 1.0,
                ..ClearColor::default()
            }),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 3, &red), Ok(()));
        pass.set_depth_attachment(&DepthAttachmentSet {
            load_action: Some(1),
            store_action: Some(4),
            clear_depth: Some(0.5),
            ..DepthAttachmentSet::default()
        });
        pass.set_stencil_attachment(&StencilAttachmentSet {
            level: Some(1),
            clear_stencil: Some(0x1ff),
            ..StencilAttachmentSet::default()
        });
        pass.set_pass(&PassSet {
            render_target_width: Some(64),
            render_target_height: Some(1 << 40),
            ..PassSet::default()
        });
        assert_eq!(pass.color_attachment_state(3), Ok(state(2, 0)));
        assert_eq!(pass.depth_attachment_state(), state(1, 4));
        let d = pass.rt_pass().expect("rt's descriptor");
        let c = &d.color_attachments[3];
        assert_eq!((c.slice, c.level), (2, 0));
        assert_eq!(c.load_action, rt::LoadAction::Clear);
        assert_eq!(c.store_action, rt::StoreAction::DontCare);
        assert_eq!(c.clear_color, rt::ClearColor::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(d.depth_attachment.load_action, rt::LoadAction::Load);
        assert_eq!(d.depth_attachment.store_action, rt::StoreAction::Unknown);
        assert_eq!(d.depth_attachment.clear_depth, 0.5);
        let s = &d.stencil_attachment;
        assert_eq!((s.level, s.clear_stencil), (1, 0x1ff));
        assert_eq!(
            (d.render_target_width, d.render_target_height),
            (64, u32::MAX)
        );

        // Metal's refusals, named at the encoder's creation.
        let unknown = ColorAttachmentSet {
            load_action: Some(7),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 3, &unknown), Ok(()));
        assert_eq!(pass.color_attachment_state(3), Ok(state(7, 0)));
        let named = pass.rt_pass().err();
        assert_eq!(
            named.as_deref(),
            Some("colour attachment 3's load action 7")
        );
        let known = ColorAttachmentSet {
            load_action: Some(0),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 3, &known), Ok(()));
        pass.set_depth_attachment(&DepthAttachmentSet {
            level: Some(1 << 32),
            ..DepthAttachmentSet::default()
        });
        let named = pass.rt_pass().err();
        assert_eq!(
            named.as_deref(),
            Some("the depth attachment's level 4294967296")
        );

        // The ninth colour attachment raises at Metal's subscript, setting nothing.
        let raised = pass.set_color_attachment(on, 8, &red).err();
        let origin = raised.map(|r| r.origin());
        assert_eq!(origin, Some(Origin::Platform(SUBSCRIPT)));
        let read = pass.color_attachment_state(u64::MAX).err();
        assert_eq!(read.map(|r| r.origin()), Some(Origin::Platform(SUBSCRIPT)));

        // What rt has no field of: counted where it is other than Metal's default.
        let code = Unbuilt::RenderPass.code();
        let before = counted(code);
        pass.set_pass(&PassSet {
            render_target_array_length: Some(1),
            tile_width: Some(0),
            visibility_result_buffer: Some(None),
            ..PassSet::default()
        });
        assert_eq!(pass.set_sample_positions(&[]), Ok(()));
        assert_eq!(counted(code), before);
        pass.set_pass(&PassSet {
            render_target_array_length: Some(2),
            imageblock_sample_length: Some(32),
            threadgroup_memory_length: Some(1024),
            tile_width: Some(16),
            tile_height: Some(16),
            ..PassSet::default()
        });
        let positions = [MtlSamplePosition { x: 0.5, y: 0.5 }];
        assert_eq!(pass.set_sample_positions(&positions), Ok(()));
        assert_eq!(counted(code), before + 6);
    }
}
