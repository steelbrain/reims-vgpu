//! Fake: the `platform/descriptor.rs` traits over the fake models (architecture 13.2).
//!
//! The descriptors core keeps as objects: the texture descriptor and the render pass descriptor.
//! Each holds a typed model of what was set on it: the texture descriptor a [`TexDesc`], the render
//! pass a [`PassModel`] with its attachments' [`AttachmentModel`]s, which hold the textures they
//! are given, so the textures go with the pass. The setters whose position Metal observes are
//! logged (`setPixelFormat:`, which Metal checks, `setResourceOptions:`, which aliases the storage
//! mode, and `setSamplePositions:count:`, which Metal checks); the others write the model, and the
//! getters read it.

use crate::testkit::script::Policy;
use std::collections::BTreeMap;

use super::super::calllog::Arg;
use super::super::objects::{AnyHandle, Fk, Kind, TexDesc, kind};
use super::super::platform::FakePlatform;
use crate::platform::ObjId;
use crate::platform::descriptor::*;
use crate::platform::types::{ClearColor, MtlSamplePosition};
use crate::platform::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
    TextureDesc,
};
use crate::raise::{OnFault, RaiseResult};

/// A texture descriptor's model.
#[derive(Debug, Default)]
pub struct TextureDescriptorModel {
    /// What its setters set, and Metal's defaults for the rest.
    pub desc: TexDesc,
    /// What `rotation` answers in place of the rotation set
    /// ([`Fk::set_reported_rotation`]).
    pub reported_rotation: Option<u64>,
}

impl TextureDescriptor<FakePlatform> for Fk<kind::TextureDescriptor> {
    fn set_resource_options(&self, resource_options: u64) {
        self.set_logged("setResourceOptions:", vec![Arg::U64(resource_options)]);
        self.with_desc(|d| d.resource_options = resource_options);
    }

    fn set_rotation(&self, rotation: u64) {
        self.with_desc(|d| d.rotation = rotation);
    }

    fn set_usage(&self, usage: u64) {
        self.with_desc(|d| d.usage = usage);
    }

    fn set_width(&self, width: u64) {
        self.with_desc(|d| d.width = width);
    }

    fn set_height(&self, height: u64) {
        self.with_desc(|d| d.height = height);
    }

    fn set_depth(&self, depth: u64) {
        self.with_desc(|d| d.depth = depth);
    }

    fn set_texture_type(&self, texture_type: u64) {
        self.with_desc(|d| d.texture_type = texture_type);
    }

    fn set_pixel_format(&self, pixel_format: u64) {
        self.set_logged("setPixelFormat:", vec![Arg::U64(pixel_format)]);
        self.with_desc(|d| d.pixel_format = pixel_format);
    }

    fn rotation(&self) -> u64 {
        self.with_model(|m: &TextureDescriptorModel| m.reported_rotation.unwrap_or(m.desc.rotation))
    }

    fn storage_mode(&self) -> u64 {
        self.desc().storage_mode()
    }

    fn pixel_format(&self) -> u64 {
        self.desc().pixel_format
    }

    fn texture_type(&self) -> u64 {
        self.desc().texture_type
    }
}

impl Fk<kind::TextureDescriptor> {
    /// What the descriptor holds: what was set, and Metal's defaults for the rest.
    pub fn desc(&self) -> TexDesc {
        self.with_model(|m: &TextureDescriptorModel| m.desc)
    }

    /// Changes what the descriptor holds.
    fn with_desc(&self, f: impl FnOnce(&mut TexDesc)) {
        self.with_model_mut(|m: &mut TextureDescriptorModel| f(&mut m.desc));
    }

    /// Makes `rotation` answer `rotation`, whatever is set: a descriptor that does not keep the
    /// rotation it is given.
    pub fn set_reported_rotation(&self, rotation: u64) {
        self.with_model_mut(|m: &mut TextureDescriptorModel| m.reported_rotation = Some(rotation));
    }

    /// Sets `desc`: the two setters whose position Metal observes, `setPixelFormat:` then
    /// `setResourceOptions:` when the value has options, are logged in [`TextureDesc`]'s order;
    /// the other fields are written to the model.
    pub(crate) fn apply(&self, desc: &TextureDesc) {
        self.set_logged("setPixelFormat:", vec![Arg::U64(desc.pixel_format)]);
        if let Some(v) = desc.resource_options {
            self.set_logged("setResourceOptions:", vec![Arg::U64(v)]);
        }
        self.with_desc(|d| {
            d.texture_type = desc.texture_type;
            d.width = desc.width;
            d.height = desc.height;
            d.depth = desc.depth;
            d.mipmap_level_count = desc.mipmap_level_count;
            d.sample_count = desc.sample_count;
            d.array_length = desc.array_length;
            d.usage = desc.usage;
            d.pixel_format = desc.pixel_format;
            let set = |to: &mut bool, v: Option<bool>| *to = v.unwrap_or(*to);
            set(&mut d.framebuffer_only, desc.framebuffer_only);
            set(&mut d.is_drawable, desc.is_drawable);
            set(
                &mut d.allow_gpu_optimized_contents,
                desc.allow_gpu_optimized_contents,
            );
            set(&mut d.write_swizzle_enabled, desc.write_swizzle_enabled);
            d.resource_options = desc.resource_options.unwrap_or(d.resource_options);
            d.protection_options = desc.protection_options.unwrap_or(d.protection_options);
            d.swizzle = desc.swizzle.unwrap_or(d.swizzle);
            if let Some((index, force)) = desc.resource_index {
                d.resource_index = Some(index);
                d.force_resource_index = force;
            }
            d.rotation = desc.rotation.unwrap_or(d.rotation);
        });
    }
}

/// A render pass attachment's model: what its setters set (`None`: never set; a texture of
/// `Some(None)`: set to nil). It holds the textures it is given.
#[derive(Debug, Default)]
pub struct AttachmentModel {
    /// `texture`.
    pub texture: Option<Option<AnyHandle>>,
    /// `resolveTexture`.
    pub resolve_texture: Option<Option<AnyHandle>>,
    /// `level`.
    pub level: Option<u64>,
    /// `slice`.
    pub slice: Option<u64>,
    /// `depthPlane`.
    pub depth_plane: Option<u64>,
    /// `resolveLevel`.
    pub resolve_level: Option<u64>,
    /// `resolveSlice`.
    pub resolve_slice: Option<u64>,
    /// `resolveDepthPlane`.
    pub resolve_depth_plane: Option<u64>,
    /// `loadAction`.
    pub load_action: Option<u64>,
    /// `storeAction`.
    pub store_action: Option<u64>,
    /// `storeActionOptions`.
    pub store_action_options: Option<u64>,
    /// A colour attachment's `clearColor`.
    pub clear_color: Option<ClearColor>,
    /// The depth attachment's `clearDepth`.
    pub clear_depth: Option<f64>,
    /// The depth attachment's `depthResolveFilter`.
    pub depth_resolve_filter: Option<u64>,
    /// The stencil attachment's `clearStencil`.
    pub clear_stencil: Option<u32>,
    /// The stencil attachment's `stencilResolveFilter`.
    pub stencil_resolve_filter: Option<u64>,
}

impl AttachmentModel {
    /// The texture's identity, if one was set.
    pub fn texture_id(&self) -> Option<ObjId> {
        id_of(self.texture.as_ref())
    }

    /// The resolve texture's identity, if one was set.
    pub fn resolve_texture_id(&self) -> Option<ObjId> {
        id_of(self.resolve_texture.as_ref())
    }

    /// The load and store actions as Metal reads them: zero when never set.
    fn state(&self) -> AttachmentState {
        AttachmentState {
            load_action: self.load_action.unwrap_or(0),
            store_action: self.store_action.unwrap_or(0),
        }
    }
}

/// The identity of an object property's value, if a non-nil one was set.
fn id_of(v: Option<&Option<AnyHandle>>) -> Option<ObjId> {
    v.and_then(Option::as_ref).map(AnyHandle::id)
}

/// A set's object argument as a model holds it: a reference taken.
fn held<K: Kind>(v: Option<Option<&Fk<K>>>) -> Option<Option<AnyHandle>> {
    v.map(|t| t.map(Fk::erase))
}

/// Writes the fields every attachment's set shares, in the 0x1a record's order: texture, resolve
/// texture, level, slice, depth plane, resolve level, slice and depth plane, load and store
/// actions, store action options. A field the set leaves `None` keeps its value.
macro_rules! shared_attachment_props {
    ($set:expr, $a:expr) => {{
        let (s, a) = ($set, $a);
        if let Some(t) = held(s.texture) {
            a.texture = Some(t);
        }
        if let Some(t) = held(s.resolve_texture) {
            a.resolve_texture = Some(t);
        }
        a.level = s.level.or(a.level);
        a.slice = s.slice.or(a.slice);
        a.depth_plane = s.depth_plane.or(a.depth_plane);
        a.resolve_level = s.resolve_level.or(a.resolve_level);
        a.resolve_slice = s.resolve_slice.or(a.resolve_slice);
        a.resolve_depth_plane = s.resolve_depth_plane.or(a.resolve_depth_plane);
        a.load_action = s.load_action.or(a.load_action);
        a.store_action = s.store_action.or(a.store_action);
        a.store_action_options = s.store_action_options.or(a.store_action_options);
    }};
}

/// A render pass descriptor's model: what its setters set (`None`: never set; an object of
/// `Some(None)`: set to nil), and its attachments.
#[derive(Debug, Default)]
pub struct PassModel {
    /// `visibilityResultBuffer`.
    pub visibility_result_buffer: Option<Option<AnyHandle>>,
    /// `visibilityResultType`.
    pub visibility_result_type: Option<u64>,
    /// `renderTargetArrayLength`.
    pub render_target_array_length: Option<u64>,
    /// `renderTargetWidth`.
    pub render_target_width: Option<u64>,
    /// `renderTargetHeight`.
    pub render_target_height: Option<u64>,
    /// `defaultRasterSampleCount`.
    pub default_raster_sample_count: Option<u64>,
    /// `defaultColorSampleCount` (SPI).
    pub default_color_sample_count: Option<u64>,
    /// `rasterizationRateMap`.
    pub rasterization_rate_map: Option<Option<AnyHandle>>,
    /// `imageblockSampleLength`.
    pub imageblock_sample_length: Option<u64>,
    /// `threadgroupMemoryLength`.
    pub threadgroup_memory_length: Option<u64>,
    /// `tileWidth`.
    pub tile_width: Option<u64>,
    /// `tileHeight`.
    pub tile_height: Option<u64>,
    /// `setSamplePositions:count:`'s positions.
    pub sample_positions: Option<Vec<MtlSamplePosition>>,
    /// The colour attachments set, by index.
    pub colors: BTreeMap<u64, AttachmentModel>,
    /// The depth attachment.
    pub depth: AttachmentModel,
    /// The stencil attachment.
    pub stencil: AttachmentModel,
}

impl PassModel {
    /// The visibility result buffer's identity, if one was set.
    pub fn visibility_result_buffer_id(&self) -> Option<ObjId> {
        id_of(self.visibility_result_buffer.as_ref())
    }

    /// The rasterization rate map's identity, if one was set.
    pub fn rasterization_rate_map_id(&self) -> Option<ObjId> {
        id_of(self.rasterization_rate_map.as_ref())
    }
}

impl Fk<kind::RenderPassDescriptor> {
    /// Reads the pass's model with `f`.
    pub fn pass<T>(&self, f: impl FnOnce(&PassModel) -> T) -> T {
        self.with_model(f)
    }

    /// Reads colour attachment `index`'s model with `f` (a default one when never set).
    pub fn color<T>(&self, index: u64, f: impl FnOnce(&AttachmentModel) -> T) -> T {
        self.pass(|p| match p.colors.get(&index) {
            Some(a) => f(a),
            None => f(&AttachmentModel::default()),
        })
    }

    /// Reads the depth attachment's model with `f`.
    pub fn depth<T>(&self, f: impl FnOnce(&AttachmentModel) -> T) -> T {
        self.pass(|p| f(&p.depth))
    }

    /// Reads the stencil attachment's model with `f`.
    pub fn stencil<T>(&self, f: impl FnOnce(&AttachmentModel) -> T) -> T {
        self.pass(|p| f(&p.stencil))
    }
}

impl RenderPassDescriptor<FakePlatform> for Fk<kind::RenderPassDescriptor> {
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "setSamplePositions:count:",
            vec![Arg::val(positions)],
        )?;
        self.with_model_mut(|p: &mut PassModel| p.sample_positions = Some(positions.to_vec()));
        Ok(())
    }

    fn set_pass(&self, set: &PassSet<'_, FakePlatform>) {
        let s = set;
        self.with_model_mut(|p: &mut PassModel| {
            if let Some(b) = held(s.visibility_result_buffer) {
                p.visibility_result_buffer = Some(b);
            }
            p.visibility_result_type = s.visibility_result_type.or(p.visibility_result_type);
            p.render_target_array_length = s
                .render_target_array_length
                .or(p.render_target_array_length);
            p.render_target_width = s.render_target_width.or(p.render_target_width);
            p.render_target_height = s.render_target_height.or(p.render_target_height);
            p.default_raster_sample_count = s
                .default_raster_sample_count
                .or(p.default_raster_sample_count);
            p.default_color_sample_count = s
                .default_color_sample_count
                .or(p.default_color_sample_count);
            if let Some(m) = held(s.rasterization_rate_map) {
                p.rasterization_rate_map = Some(m);
            }
            p.imageblock_sample_length = s.imageblock_sample_length.or(p.imageblock_sample_length);
            p.threadgroup_memory_length =
                s.threadgroup_memory_length.or(p.threadgroup_memory_length);
            p.tile_width = s.tile_width.or(p.tile_width);
            p.tile_height = s.tile_height.or(p.tile_height);
        });
    }

    /// The shared properties, then the clear colour.
    fn set_color_attachment(
        &self,
        _on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, FakePlatform>,
    ) -> RaiseResult<()> {
        self.with_model_mut(|p: &mut PassModel| {
            let a = p.colors.entry(index).or_default();
            shared_attachment_props!(set, &mut *a);
            a.clear_color = set.clear_color.or(a.clear_color);
        });
        Ok(())
    }

    /// The shared properties, then the clear depth and the resolve filter.
    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, FakePlatform>) {
        self.with_model_mut(|p: &mut PassModel| {
            let a = &mut p.depth;
            shared_attachment_props!(set, &mut *a);
            a.clear_depth = set.clear_depth.or(a.clear_depth);
            a.depth_resolve_filter = set.depth_resolve_filter.or(a.depth_resolve_filter);
        });
    }

    /// The shared properties, then the clear stencil and the resolve filter.
    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, FakePlatform>) {
        self.with_model_mut(|p: &mut PassModel| {
            let a = &mut p.stencil;
            shared_attachment_props!(set, &mut *a);
            a.clear_stencil = set.clear_stencil.or(a.clear_stencil);
            a.stencil_resolve_filter = set.stencil_resolve_filter.or(a.stencil_resolve_filter);
        });
    }

    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState> {
        Ok(self.color(index, AttachmentModel::state))
    }

    fn depth_attachment_state(&self) -> AttachmentState {
        self.depth(AttachmentModel::state)
    }

    fn stencil_attachment_state(&self) -> AttachmentState {
        self.stencil(AttachmentModel::state)
    }
}
