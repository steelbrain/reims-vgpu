//! The hottest calls' arguments as values, read from their fields where their `Debug` text would
//! be formatted and read back ([`super::debug_value`]): each value is the one that text reads as,
//! field for field, which the tests hold for every command and argument type read here.
//!
//! - **Commands** ([`Command`]): the encoder commands the window-drag walk's stream holds most of
//!   (its recorder's counts by selector: binds, draws, fences, barriers, scissors, viewports,
//!   pipeline and sampler states). Any other command answers `None`, and is read from its text.
//! - **The pass descriptor's sets** (`m2v.setPass`, `m2v.set*Attachment`), whose `Debug` names
//!   objects by their pointers. An object property set to nil is object 0, which no object is,
//!   where the text reads nil for it and for one left unset alike: a player clears the one and
//!   keeps what the descriptor had for the other.
//! - **Objects** are their stream ids where the stream made them, else nil, counted as the text's
//!   `Unrecorded(..)` and `ObjId(..)` of an unknown pointer are.

use super::format::Value;
use crate::core::platform::blit::BlitCmd;
use crate::core::platform::compute::ComputeCmd;
use crate::core::platform::render::RenderCmd;
use crate::core::platform::types::{
    ClearColor, MtlRegion, MtlScissorRect, MtlSize, MtlViewport, NsRange,
};
use crate::core::platform::values::{
    ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use crate::core::platform::{Handle, Platform};

/// What a read needs besides the arguments: the stream id of the object at a pointer, and the
/// count of objects the stream never made.
pub struct Cx<'a> {
    ids: &'a dyn Fn(u64) -> Option<u64>,
    /// Objects named that the stream never made, read as nil.
    pub unrecorded: u64,
}

impl<'a> Cx<'a> {
    /// A read whose objects' ids `ids` gives by pointer.
    pub fn new(ids: &'a dyn Fn(u64) -> Option<u64>) -> Cx<'a> {
        Cx { ids, unrecorded: 0 }
    }

    /// The object at `ptr`: `ObjRef(id)` or `ObjId(ptr)` read, or `Unrecorded(..)`.
    fn ptr(&mut self, ptr: u64) -> Value {
        match (self.ids)(ptr) {
            Some(id) => Value::Object(id),
            None => {
                self.unrecorded += 1;
                Value::Nil
            }
        }
    }

    fn obj<H: Handle>(&mut self, h: &H) -> Value {
        self.ptr(h.id().0)
    }

    /// `Option<&H>`: `Some(x)` is `x`.
    fn opt<H: Handle>(&mut self, h: Option<&H>) -> Value {
        h.map_or(Value::Nil, |h| self.obj(h))
    }

    /// `&[Option<&H>]`: a list.
    fn objs<H: Handle>(&mut self, hs: &[Option<&H>]) -> Value {
        Value::List(hs.iter().map(|h| self.opt(*h)).collect())
    }

    /// A set's object, `Option<Option<&H>>`: `Some(Some(h))` is `h`'s object, `Some(None)`, a
    /// property set to nil, object 0, which no object is, and `None`, one left unset, nil.
    fn set_obj<H: Handle>(&mut self, h: Option<Option<&H>>) -> Value {
        match h {
            Some(Some(h)) => self.obj(h),
            Some(None) => Value::Object(0),
            None => Value::Nil,
        }
    }
}

/// An unsigned integer.
pub fn uint(v: u64) -> Value {
    Value::Uint(v)
}

/// A signed integer: unsigned where it is not negative, as its text reads.
pub fn sint(v: i64) -> Value {
    u64::try_from(v).map_or(Value::Sint(v), Value::Uint)
}

/// A double: any NaN is the one NaN, as `NaN` reads.
pub fn double(v: f64) -> Value {
    Value::F64(if v.is_nan() { f64::NAN } else { v })
}

fn opt_uint(v: Option<u64>) -> Value {
    v.map_or(Value::Nil, uint)
}

fn uints(v: &[u64]) -> Value {
    Value::List(v.iter().copied().map(uint).collect())
}

fn map<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    Value::Map(pairs.map(|(k, v)| (k.to_owned(), v)).into())
}

fn size(s: MtlSize) -> Value {
    map([
        ("width", uint(s.width)),
        ("height", uint(s.height)),
        ("depth", uint(s.depth)),
    ])
}

fn region(r: MtlRegion) -> Value {
    map([
        (
            "origin",
            map([
                ("x", uint(r.origin.x)),
                ("y", uint(r.origin.y)),
                ("z", uint(r.origin.z)),
            ]),
        ),
        ("size", size(r.size)),
    ])
}

/// `NSRange`: its location and length.
pub fn ns_range(r: NsRange) -> Value {
    map([("location", uint(r.location)), ("length", uint(r.length))])
}

fn scissor(r: MtlScissorRect) -> Value {
    map([
        ("x", uint(r.x)),
        ("y", uint(r.y)),
        ("width", uint(r.width)),
        ("height", uint(r.height)),
    ])
}

fn viewport(v: MtlViewport) -> Value {
    map([
        ("originX", double(v.origin_x)),
        ("originY", double(v.origin_y)),
        ("width", double(v.width)),
        ("height", double(v.height)),
        ("znear", double(v.znear)),
        ("zfar", double(v.zfar)),
    ])
}

fn clear_color(c: Option<ClearColor>) -> Value {
    c.map_or(Value::Nil, |c| {
        map([
            ("red", double(c.red)),
            ("green", double(c.green)),
            ("blue", double(c.blue)),
            ("alpha", double(c.alpha)),
        ])
    })
}

/// An encoder's command: its selector, and its arguments where they are read from its fields.
pub trait Command {
    /// Metal's selector.
    fn sel(&self) -> &'static str;
    /// Its arguments, its variant's fields in order; `None` for a command read from its text.
    fn typed(&self, cx: &mut Cx<'_>) -> Option<Vec<Value>>;
}

impl<P: Platform> Command for RenderCmd<'_, P> {
    fn sel(&self) -> &'static str {
        self.selector()
    }

    fn typed(&self, cx: &mut Cx<'_>) -> Option<Vec<Value>> {
        use RenderCmd as C;
        Some(match self {
            C::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
            } => vec![
                uint(*primitive_type),
                uint(*index_count),
                uint(*index_type),
                cx.obj(*index_buffer),
                uint(*index_buffer_offset),
            ],
            C::DrawIndexedPrimitivesInstances {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
            } => vec![
                uint(*primitive_type),
                uint(*index_count),
                uint(*index_type),
                cx.obj(*index_buffer),
                uint(*index_buffer_offset),
                uint(*instance_count),
            ],
            C::DrawIndexedPrimitivesBase {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => vec![
                uint(*primitive_type),
                uint(*index_count),
                uint(*index_type),
                cx.obj(*index_buffer),
                uint(*index_buffer_offset),
                uint(*instance_count),
                sint(*base_vertex),
                uint(*base_instance),
            ],
            C::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
            } => vec![
                uint(*primitive_type),
                uint(*vertex_start),
                uint(*vertex_count),
            ],
            C::DrawPrimitivesInstances {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
            } => vec![
                uint(*primitive_type),
                uint(*vertex_start),
                uint(*vertex_count),
                uint(*instance_count),
            ],
            C::DrawPrimitivesBase {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => vec![
                uint(*primitive_type),
                uint(*vertex_start),
                uint(*vertex_count),
                uint(*instance_count),
                uint(*base_instance),
            ],
            C::EndEncoding => vec![],
            C::MemoryBarrierWithScope {
                scope,
                after,
                before,
            } => vec![uint(*scope), uint(*after), uint(*before)],
            C::SetCullMode { cull_mode } => vec![uint(*cull_mode)],
            C::SetDepthStencilState {
                depth_stencil_state,
            } => vec![cx.opt(*depth_stencil_state)],
            C::SetFragmentBuffer {
                buffer,
                offset,
                index,
            }
            | C::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => vec![cx.opt(*buffer), uint(*offset), uint(*index)],
            C::SetFragmentBufferOffset { offset, index }
            | C::SetVertexBufferOffset { offset, index } => vec![uint(*offset), uint(*index)],
            C::SetFragmentBuffers {
                buffers,
                offsets,
                range,
            }
            | C::SetVertexBuffers {
                buffers,
                offsets,
                range,
            } => vec![cx.objs(buffers), uints(offsets), ns_range(*range)],
            C::SetFragmentSamplerStates { samplers, range }
            | C::SetVertexSamplerStates { samplers, range } => {
                vec![cx.objs(samplers), ns_range(*range)]
            }
            C::SetFragmentTexture { texture, index } => vec![cx.opt(*texture), uint(*index)],
            C::SetFragmentTextures { textures, range }
            | C::SetVertexTextures { textures, range } => {
                vec![cx.objs(textures), ns_range(*range)]
            }
            C::SetFrontFacingWinding {
                front_facing_winding,
            } => vec![uint(*front_facing_winding)],
            C::SetRenderPipelineState { pipeline_state } => vec![cx.opt(*pipeline_state)],
            C::SetScissorRect { rect } => vec![scissor(*rect)],
            C::SetStencilFrontReferenceValue {
                front_reference_value,
                back_reference_value,
            } => vec![
                uint(u64::from(*front_reference_value)),
                uint(u64::from(*back_reference_value)),
            ],
            C::SetTriangleFillMode { fill_mode } => vec![uint(*fill_mode)],
            C::SetViewport { viewport: v } => vec![viewport(*v)],
            C::UpdateFence { fence, stages } | C::WaitForFence { fence, stages } => {
                vec![cx.obj(*fence), uint(*stages)]
            }
            _ => return None,
        })
    }
}

impl<P: Platform> Command for ComputeCmd<'_, P> {
    fn sel(&self) -> &'static str {
        self.selector()
    }

    fn typed(&self, cx: &mut Cx<'_>) -> Option<Vec<Value>> {
        use ComputeCmd as C;
        Some(match self {
            C::DispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => vec![size(*threadgroups_per_grid), size(*threads_per_threadgroup)],
            C::DispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => vec![size(*threads_per_grid), size(*threads_per_threadgroup)],
            C::EndEncoding => vec![],
            C::MemoryBarrierWithScope { scope } => vec![uint(*scope)],
            C::SetBufferOffset { offset, index } => vec![uint(*offset), uint(*index)],
            C::SetBuffers {
                buffers,
                offsets,
                range,
            } => vec![cx.objs(buffers), uints(offsets), ns_range(*range)],
            C::SetComputePipelineState { state } => vec![cx.opt(*state)],
            C::SetSamplerStates { samplers, range } => vec![cx.objs(samplers), ns_range(*range)],
            C::SetStageInRegion { region: r } => vec![region(*r)],
            C::SetTexture { texture, index } => vec![cx.opt(*texture), uint(*index)],
            C::SetTextures { textures, range } => vec![cx.objs(textures), ns_range(*range)],
            C::SetThreadgroupMemoryLength { length, index } => vec![uint(*length), uint(*index)],
            C::UpdateFence { fence } | C::WaitForFence { fence } => vec![cx.obj(*fence)],
            _ => return None,
        })
    }
}

impl<P: Platform> Command for BlitCmd<'_, P> {
    fn sel(&self) -> &'static str {
        self.selector()
    }

    fn typed(&self, cx: &mut Cx<'_>) -> Option<Vec<Value>> {
        use BlitCmd as C;
        Some(match self {
            C::EndEncoding => vec![],
            C::UpdateFence { fence } | C::WaitForFence { fence } => vec![cx.obj(*fence)],
            _ => return None,
        })
    }
}

/// `m2v.setPass`'s set, in its `Debug`'s order: the objects, then the values.
pub fn pass_set<P: Platform>(s: &PassSet<'_, P>, cx: &mut Cx<'_>) -> Value {
    map([
        (
            "visibilityResultBuffer",
            cx.set_obj(s.visibility_result_buffer),
        ),
        ("rasterizationRateMap", cx.set_obj(s.rasterization_rate_map)),
        ("visibilityResultType", opt_uint(s.visibility_result_type)),
        (
            "renderTargetArrayLength",
            opt_uint(s.render_target_array_length),
        ),
        ("renderTargetWidth", opt_uint(s.render_target_width)),
        ("renderTargetHeight", opt_uint(s.render_target_height)),
        (
            "defaultRasterSampleCount",
            opt_uint(s.default_raster_sample_count),
        ),
        (
            "defaultColorSampleCount",
            opt_uint(s.default_color_sample_count),
        ),
        (
            "imageblockSampleLength",
            opt_uint(s.imageblock_sample_length),
        ),
        (
            "threadgroupMemoryLength",
            opt_uint(s.threadgroup_memory_length),
        ),
        ("tileWidth", opt_uint(s.tile_width)),
        ("tileHeight", opt_uint(s.tile_height)),
    ])
}

/// The fields an attachment's set shares, in its `Debug`'s order: the two textures, then the
/// values up to its own.
macro_rules! attachment_common {
    ($s:expr, $cx:expr) => {
        [
            ("texture", $cx.set_obj($s.texture)),
            ("resolveTexture", $cx.set_obj($s.resolve_texture)),
            ("level", opt_uint($s.level)),
            ("slice", opt_uint($s.slice)),
            ("depthPlane", opt_uint($s.depth_plane)),
            ("resolveLevel", opt_uint($s.resolve_level)),
            ("resolveSlice", opt_uint($s.resolve_slice)),
            ("resolveDepthPlane", opt_uint($s.resolve_depth_plane)),
            ("loadAction", opt_uint($s.load_action)),
            ("storeAction", opt_uint($s.store_action)),
            ("storeActionOptions", opt_uint($s.store_action_options)),
        ]
    };
}

fn attachment(common: [(&str, Value); 11], own: [(&str, Value); 2]) -> Value {
    Value::Map(
        common
            .into_iter()
            .chain(own)
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}

/// `m2v.setColorAttachment`'s set.
pub fn color_attachment<P: Platform>(s: &ColorAttachmentSet<'_, P>, cx: &mut Cx<'_>) -> Value {
    let common = attachment_common!(s, cx);
    Value::Map(
        common
            .into_iter()
            .chain([("clearColor", clear_color(s.clear_color))])
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}

/// `m2v.setDepthAttachment`'s set.
pub fn depth_attachment<P: Platform>(s: &DepthAttachmentSet<'_, P>, cx: &mut Cx<'_>) -> Value {
    let common = attachment_common!(s, cx);
    attachment(
        common,
        [
            ("clearDepth", s.clear_depth.map_or(Value::Nil, double)),
            ("depthResolveFilter", opt_uint(s.depth_resolve_filter)),
        ],
    )
}

/// `m2v.setStencilAttachment`'s set.
pub fn stencil_attachment<P: Platform>(s: &StencilAttachmentSet<'_, P>, cx: &mut Cx<'_>) -> Value {
    let common = attachment_common!(s, cx);
    attachment(
        common,
        [
            (
                "clearStencil",
                s.clear_stencil.map_or(Value::Nil, |v| uint(u64::from(v))),
            ),
            ("stencilResolveFilter", opt_uint(s.stencil_resolve_filter)),
        ],
    )
}
