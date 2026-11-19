//! The rt platform's render pipeline states ([`RtRenderPipeline`]): rt's, made of the display's
//! value ([`RenderPipelineDesc`]) or of core's whole descriptor ([`FullRenderPipelineDesc`]), with
//! the sampler indices their samplers take, as a compute state's (`super::pipeline`).
//!
//! - **The descriptor** is rt's of core's values: each function as it was made
//!   (`super::library`), with the functions its stage links; the vertex descriptor's layouts and
//!   attributes as the value sets them; each colour attachment with its blend and write mask; the
//!   depth and stencil formats; the counts; the private state where the guest gave it; and the
//!   tessellation state, none where the value's is Metal's default. A raw value with no value of
//!   the translator's, a count past 32 bits, or a colour attachment past the eighth, is Metal's
//!   refusal: its setters end the process at one. The creator answers the `NSError`, counted
//!   (`rt-refused-metal`). So does a function made unspecialized that declares function constants,
//!   as a compute state's kernel does. The display's value is its two functions and colour 0's
//!   format. Its label is not rt's, and neither is its resource index, which a state answers only
//!   where it supports indirect command buffers, which rt's states never do.
//! - **A layout the value does not set is none**: the guest's bundle sends a vertex descriptor's
//!   layouts of a stride other than 0, Metal's default, and an attribute of a buffer with none is
//!   Metal's assertion ("references a buffer ... that has no stride"), which the translator
//!   refuses (`rt-refused-translation`).
//! - **What rt refuses** is the creator's `NSError`, counted under its kind: a state for indirect
//!   command buffers, a sample count Vulkan has no bit of, vertex amplification, and what of the
//!   functions rt does not draw yet (`rt-refused-gap`); the translator's refusals
//!   (`rt-refused-translation`).
//! - **Its samplers' indices** are taken as a compute state's are, the vertex function's samplers
//!   first, then the fragment function's.
//! - **Its answers** are a drawing state's. `maxTotalThreadsPerThreadgroup` is 0 and
//!   `threadgroupSizeMatchesTileSize` NO, as Metal's drawing state answers (metal2vulkan's
//!   measurement on an M3 Max and an M4). The object and mesh stages' answers are 0, as Metal's
//!   for a state with neither stage. `gpuResourceID` and `resourceIndex` are 0 and
//!   `supportIndirectCommandBuffers` NO, rt's, as a compute state's. `uniqueIdentifier` is one of
//!   the compute states' count. `imageblockSampleLength` and
//!   `imageblockMemoryLengthForDimensions:` are rt's: the colour attachments' texels as Metal
//!   packs them, and that times the dimensions' width and height and the sample count. Where rt
//!   refuses one (`Gap::ImageblockLength`: a colour format with no measured place in Metal's
//!   record), the answer is 0, counted (`rt-refused-gap`).
//! - **Its slots' writes** are the vertex and fragment functions' own, as a compute state's are
//!   its kernel's (rt's interface of each, `texture_accesses`). A drawing state has no tile,
//!   object or mesh function, so it writes no slot of theirs.
//! - **Mesh render pipeline states** are unbuilt: rt has none. The creator answers its `NSError`,
//!   counted (`rt-unbuilt-mesh-pipeline`).

use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveTopologyClass,
    TessellationControlPointIndexType, TessellationFactorFormat, TessellationFactorStepFunction,
    TessellationPartitionMode, VertexFormat, VertexStepFunction, Winding,
};
use metal2vulkan::pipeline::{
    PrivateRenderState, TessellationDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor,
};
use metal2vulkan_rt as rt;

use crate::core::platform::resource::{RenderFunction, RenderPipeline};
use crate::core::platform::types::{MtlSize, NsErrorText};
use crate::core::platform::values::{
    ColorAttachmentDesc, FullMeshRenderPipelineDesc, FullRenderPipelineDesc, PrivateRenderDesc,
    RenderPipelineDesc, TessellationDesc, VertexDesc,
};
use crate::core::raise::RaiseResult;

use super::device::RtDevice;
use super::pipeline::{
    IMAGEBLOCK_MEMORY_LENGTH, Lease, Taking, dimensions, linked_functions, next_identifier,
    specialized, writes, written_slots,
};
use super::state::value;
use super::unbuilt::{RtRefusal, Unbuilt, refused_error, refused_error_as, rt_refused_imageblock};
use super::{RtObj, RtPlatform};

/// `newRenderPipelineStateWithDescriptor:error:`.
pub const WITH_DESCRIPTOR: &str = "newRenderPipelineStateWithDescriptor:error:";

/// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:`.
pub const WITH_MESH_DESCRIPTOR: &str =
    "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:";

/// `imageblockSampleLength`.
const IMAGEBLOCK_SAMPLE_LENGTH: &str = "imageblockSampleLength";

/// `MTLBufferLayoutStrideDynamic`: a layout whose stride the encoder binds the buffer with.
const STRIDE_DYNAMIC: u64 = u64::MAX;

/// `id<MTLRenderPipelineState>` on metal2vulkan-rt.
pub struct RtRenderPipeline {
    /// rt's state. Declared before `lease`, so it goes first, as a compute state's.
    rt: rt::RenderPipelineState,
    /// Its samplers' indices.
    lease: Lease,
    /// Its `uniqueIdentifier`.
    unique_identifier: u64,
    /// The texture slots the vertex function may write (`written_slots`).
    vertex_written: Box<[u32]>,
    /// The texture slots the fragment function may write.
    fragment_written: Box<[u32]>,
    /// Its colour attachments' pixel formats, by index, and its sample count, as the guest's
    /// descriptor gave them: what the boot diagnostics key its `imageblockSampleLength` asks by.
    colors: ([u32; 8], u32),
}

impl RtRenderPipeline {
    /// rt's state.
    pub fn rt(&self) -> &rt::RenderPipelineState {
        &self.rt
    }

    /// The samplers it numbers, in rt's order.
    pub fn samplers(&self) -> &[rt::NumberedSampler] {
        self.lease.samplers()
    }

    /// The index each of [`samplers`](Self::samplers) holds.
    pub fn sampler_indices(&self) -> &[u32] {
        self.lease.indices()
    }
}

/// The state of the display's value `desc`: its two functions and colour 0's format, with
/// Metal's defaults elsewhere (the module's docs).
pub fn of_value(
    device: &RtObj<RtDevice>,
    desc: &RenderPipelineDesc<'_, RtPlatform>,
) -> Result<RtObj<RtRenderPipeline>, NsErrorText> {
    let (vertex, fragment) = (desc.vertex.stage(), desc.fragment.stage());
    specialized(WITH_DESCRIPTOR, &vertex)?;
    specialized(WITH_DESCRIPTOR, &fragment)?;
    let mut d = rt::RenderPipelineDescriptor::new(rt::StageSource::Function(vertex));
    d.fragment = Some(rt::StageSource::Function(fragment));
    let color0 = ColorAttachmentDesc {
        pixel_format: desc.color0_pixel_format,
        ..ColorAttachmentDesc::default()
    };
    d.color_attachments[0] = color_attachment(&color0).map_err(refused)?;
    let mut formats = [0; 8];
    formats[0] = u32::try_from(desc.color0_pixel_format).unwrap_or(u32::MAX);
    make(device, &d, (formats, 1))
}

/// The state of core's whole descriptor `desc` (the module's docs).
pub fn of_descriptor(
    device: &RtObj<RtDevice>,
    desc: &FullRenderPipelineDesc<'_, RtPlatform>,
) -> Result<RtObj<RtRenderPipeline>, NsErrorText> {
    let mut vertex = desc.vertex.stage();
    specialized(WITH_DESCRIPTOR, &vertex)?;
    vertex.linked = linked_functions(&desc.vertex_linked);
    let fragment = match desc.fragment {
        Some(function) => {
            let mut fragment = function.stage();
            specialized(WITH_DESCRIPTOR, &fragment)?;
            fragment.linked = linked_functions(&desc.fragment_linked);
            Some(rt::StageSource::Function(fragment))
        }
        None => None,
    };
    let mut d = rt::RenderPipelineDescriptor::new(rt::StageSource::Function(vertex));
    d.fragment = fragment;
    fields(&mut d, desc).map_err(refused)?;
    let mut formats = [0; 8];
    for (index, c) in &desc.colors {
        let slot = usize::try_from(*index).ok();
        if let Some(format) = slot.and_then(|i| formats.get_mut(i)) {
            *format = u32::try_from(c.pixel_format).unwrap_or(u32::MAX);
        }
    }
    let samples = u32::try_from(desc.raster_sample_count).unwrap_or(u32::MAX);
    make(device, &d, (formats, samples))
}

/// Unbuilt ([`Unbuilt::MeshPipeline`]): the creator's `NSError`.
pub fn of_mesh_descriptor(
    desc: &FullMeshRenderPipelineDesc<'_, RtPlatform>,
) -> Result<RtObj<RtRenderPipeline>, NsErrorText> {
    let _ = desc;
    Err(refused_error(Unbuilt::MeshPipeline, WITH_MESH_DESCRIPTOR))
}

/// The `NSError` of Metal's refusal of the value named by `why`, counted.
fn refused(why: String) -> NsErrorText {
    refused_error_as(WITH_DESCRIPTOR, RtRefusal::Refused, &why)
}

/// Sets `d`'s fields past its functions from `desc`'s.
///
/// Err: the first value rt's descriptor has no value of, named.
fn fields(
    d: &mut rt::RenderPipelineDescriptor,
    desc: &FullRenderPipelineDesc<'_, RtPlatform>,
) -> Result<(), String> {
    if let Some(v) = &desc.vertex_descriptor {
        d.vertex_descriptor = Some(vertex_descriptor(v)?);
    }
    for (index, c) in &desc.colors {
        let slot = usize::try_from(*index).ok();
        let slot = slot.and_then(|i| d.color_attachments.get_mut(i));
        let Some(slot) = slot else {
            return Err(format!("colour attachment {index}"));
        };
        *slot = color_attachment(c)?;
    }
    d.depth_format = format("depth pixel format", desc.depth_pixel_format)?;
    d.stencil_format = format("stencil pixel format", desc.stencil_pixel_format)?;
    d.raster_sample_count = count("raster sample count", desc.raster_sample_count)?;
    d.alpha_to_coverage = desc.alpha_to_coverage;
    d.alpha_to_one = desc.alpha_to_one;
    d.rasterization_enabled = desc.rasterization_enabled;
    d.input_primitive_topology =
        value::<PrimitiveTopologyClass>("input primitive topology", desc.input_primitive_topology)?;
    d.max_vertex_amplification_count = count(
        "maximum vertex amplification count",
        desc.max_vertex_amplification_count,
    )?;
    d.support_indirect_command_buffers = desc.support_indirect_command_buffers;
    if let Some(p) = &desc.private_state {
        private_state(&mut d.private, p);
    }
    d.tessellation = tessellation(&desc.tessellation)?;
    d.max_vertex_call_stack_depth = count(
        "maximum vertex call stack depth",
        desc.max_vertex_call_stack_depth,
    )?;
    d.max_fragment_call_stack_depth = count(
        "maximum fragment call stack depth",
        desc.max_fragment_call_stack_depth,
    )?;
    Ok(())
}

/// `raw` as a count rt holds in 32 bits, or the error naming it as `what`.
fn count(what: &str, raw: u64) -> Result<u32, String> {
    u32::try_from(raw).map_err(|_| format!("{what} {raw}"))
}

/// The pixel format `raw` names: none for `MTLPixelFormatInvalid`.
///
/// Err: a value rt has no format of, named as `what`.
fn format(what: &str, raw: u64) -> Result<Option<PixelFormat>, String> {
    match raw {
        0 => Ok(None),
        raw => value::<PixelFormat>(what, raw).map(Some),
    }
}

/// rt's colour attachment of `c`: none for `MTLPixelFormatInvalid`.
///
/// Err: the first value rt's attachment has no value of, named.
fn color_attachment(
    c: &ColorAttachmentDesc,
) -> Result<Option<rt::ColorAttachmentDescriptor>, String> {
    let Some(pixel_format) = format("colour pixel format", c.pixel_format)? else {
        return Ok(None);
    };
    let mut a = rt::ColorAttachmentDescriptor::new(pixel_format);
    a.blending_enabled = c.blending_enabled;
    a.source_rgb_blend_factor = value::<BlendFactor>("blend factor", c.source_rgb_blend_factor)?;
    a.destination_rgb_blend_factor =
        value::<BlendFactor>("blend factor", c.destination_rgb_blend_factor)?;
    a.rgb_blend_operation = value::<BlendOperation>("blend operation", c.rgb_blend_operation)?;
    a.source_alpha_blend_factor =
        value::<BlendFactor>("blend factor", c.source_alpha_blend_factor)?;
    a.destination_alpha_blend_factor =
        value::<BlendFactor>("blend factor", c.destination_alpha_blend_factor)?;
    a.alpha_blend_operation = value::<BlendOperation>("blend operation", c.alpha_blend_operation)?;
    a.write_mask = value::<ColorWriteMask>("colour write mask", c.write_mask)?;
    Ok(Some(a))
}

/// rt's vertex descriptor of `v`: each layout and attribute the value sets, at its index (the
/// module's docs).
///
/// Err: the first value it has no value of, named.
fn vertex_descriptor(v: &VertexDesc) -> Result<rt::VertexDescriptor, String> {
    let mut out = rt::VertexDescriptor::new();
    for l in &v.layouts {
        let stride = match l.stride {
            STRIDE_DYNAMIC => None,
            stride => Some(count("stride", stride)?),
        };
        let mut layout = VertexBufferLayoutDescriptor::new(stride);
        layout.step_function = value::<VertexStepFunction>("step function", l.step_function)?;
        layout.step_rate = count("step rate", l.step_rate)?;
        out.layouts.insert(count("layout", l.buffer_index)?, layout);
    }
    for a in &v.attributes {
        let attribute = VertexAttributeDescriptor::new(
            value::<VertexFormat>("vertex format", a.format)?,
            count("offset", a.offset)?,
            count("attribute buffer", a.buffer_index)?,
        );
        let index = count("attribute", a.attribute_index)?;
        out.attributes.insert(index, attribute);
    }
    Ok(out)
}

/// The guest's private state `p` on rt's, whose other fields keep Metal's defaults.
fn private_state(private: &mut PrivateRenderState, p: &PrivateRenderDesc) {
    private.open_gl_mode_enabled = p.open_gl_mode_enabled;
    private.sample_mask = p.sample_mask;
    private.sample_coverage = p.sample_coverage;
    private.sample_coverage_invert = p.sample_coverage_invert;
    private.logic_operation_enabled = p.logic_operation_enabled;
    private.logic_operation = p.logic_operation;
    private.depth_stencil_write_disabled = p.depth_stencil_write_disabled;
}

/// rt's tessellation state of `t`: none for Metal's defaults.
///
/// Err: the first value it has no value of, named.
fn tessellation(t: &TessellationDesc) -> Result<Option<TessellationDescriptor>, String> {
    if *t == TessellationDesc::default() {
        return Ok(None);
    }
    let mut d = TessellationDescriptor::default();
    d.partition_mode =
        value::<TessellationPartitionMode>("tessellation partition mode", t.partition_mode)?;
    d.max_tessellation_factor = count("maximum tessellation factor", t.max_factor)?;
    d.factor_scale_enabled = t.factor_scale_enabled;
    d.factor_format =
        value::<TessellationFactorFormat>("tessellation factor format", t.factor_format)?;
    d.control_point_index_type = value::<TessellationControlPointIndexType>(
        "tessellation control point index type",
        t.control_point_index_type,
    )?;
    d.factor_step_function = value::<TessellationFactorStepFunction>(
        "tessellation factor step function",
        t.factor_step_function,
    )?;
    d.output_winding_order = value::<Winding>("output winding order", t.output_winding)?;
    Ok(Some(d))
}

/// rt's state of `descriptor`, its samplers' indices taken (the module's docs), of colour formats
/// and sample count `colors` ([`RtRenderPipeline::colors`]). `Err` is the creator's `NSError`,
/// counted under the refusal's kind.
fn make(
    device: &RtObj<RtDevice>,
    descriptor: &rt::RenderPipelineDescriptor,
    colors: ([u32; 8], u32),
) -> Result<RtObj<RtRenderPipeline>, NsErrorText> {
    let mut taking = Taking::new(device);
    let made = (device.rt()).new_render_pipeline_state(descriptor, |samplers| taking.ids(samplers));
    let (rt, lease) = taking.made(WITH_DESCRIPTOR, made)?;
    let vertex_written = written_slots(rt.texture_accesses(rt::Stage::Vertex));
    let fragment_written = written_slots(rt.texture_accesses(rt::Stage::Fragment));
    Ok(RtObj::new(RtRenderPipeline {
        rt,
        lease,
        unique_identifier: next_identifier(),
        vertex_written,
        fragment_written,
        colors,
    }))
}

impl RenderPipeline for RtObj<RtRenderPipeline> {
    /// 0: rt makes no state for indirect command buffers.
    fn gpu_resource_id(&self) -> u64 {
        0
    }

    /// rt's answer; 0 where rt refuses it, counted (`rt_refused_imageblock`).
    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        let asked = dimensions(imageblock_dimensions);
        let made = self.rt.imageblock_memory_length_for_dimensions(asked);
        Ok(made.unwrap_or_else(|error| {
            rt_refused_imageblock(IMAGEBLOCK_MEMORY_LENGTH, &error);
            0
        }))
    }

    /// rt's answer; 0 where rt refuses it, counted (`rt_refused_imageblock`). The ask is noted by
    /// its colour formats and sample count, with the answer (`super::diag::imageblock`).
    fn imageblock_sample_length(&self) -> u64 {
        let made = self.rt.imageblock_sample_length();
        let (formats, samples) = self.colors;
        super::diag::imageblock(formats, samples, made.as_ref().ok().copied());
        made.map_or_else(
            |error| {
                rt_refused_imageblock(IMAGEBLOCK_SAMPLE_LENGTH, &error);
                0
            },
            u64::from,
        )
    }

    /// 0, a drawing state's.
    fn max_total_threads_per_threadgroup(&self) -> u64 {
        0
    }

    /// 0: no object stage.
    fn max_total_threads_per_object_threadgroup(&self) -> u64 {
        0
    }

    /// 0: no mesh stage.
    fn max_total_threads_per_mesh_threadgroup(&self) -> u64 {
        0
    }

    /// 0: no object stage.
    fn object_thread_execution_width(&self) -> u64 {
        0
    }

    /// 0: no mesh stage.
    fn mesh_thread_execution_width(&self) -> u64 {
        0
    }

    /// 0: no mesh stage.
    fn max_total_threadgroups_per_mesh_grid(&self) -> u64 {
        0
    }

    /// 0, as [`gpu_resource_id`](Self::gpu_resource_id).
    fn resource_index(&self) -> u64 {
        0
    }

    fn support_indirect_command_buffers(&self) -> bool {
        self.rt.support_indirect_command_buffers()
    }

    /// NO, a drawing state's.
    fn threadgroup_size_matches_tile_size(&self) -> bool {
        false
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_identifier
    }

    /// As a compute state answers of its kernel, of the vertex or fragment function's slot
    /// `index`; `Some(false)` for a tile, object or mesh function's slot: a drawing state has
    /// none of them.
    fn texture_slot_writes(&self, function: RenderFunction, index: u64) -> Option<bool> {
        let written = match function {
            RenderFunction::Vertex => &self.vertex_written,
            RenderFunction::Fragment => &self.fragment_written,
            RenderFunction::Tile | RenderFunction::Object | RenderFunction::Mesh => {
                return Some(false);
            }
        };
        Some(writes(written, index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::Library as _;
    use crate::core::platform::values::{AttributeDesc, BufferLayoutDesc, LinkedFunctionsDesc};
    use crate::core::raise::OnFault;
    use crate::rt_platform::RtFunction;
    use crate::rt_platform::unbuilt::IMAGEBLOCK_LENGTH;

    /// `numbered_samplers.metal`'s functions, compiled by build.rs.
    const NUMBERED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/numbered_samplers.metallib"));

    /// `rounding.metal`'s functions, compiled by build.rs: `write_fragment` writes texture 0.
    const ROUNDING: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rounding.metallib"));

    /// A vertex function fed through a vertex descriptor: attribute 0 a `float2` position, 1 a
    /// normalized `uchar4` colour.
    const VDESC: &[u8] = include_bytes!("../../../../tests/shaders/render_vdesc.metallib");

    /// A fragment function that returns the interpolated colour.
    const COLOR: &[u8] = include_bytes!("../../../../tests/shaders/render_color.metallib");

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Each value of a whole descriptor is rt's, Metal's defaults where the value's are; a value
    /// rt has none of is named.
    #[test]
    fn descriptor_values_are_rt_s() {
        let v = VertexDesc {
            layouts: vec![BufferLayoutDesc {
                buffer_index: 0,
                stride: 12,
                step_function: 1,
                step_rate: 1,
            }],
            attributes: vec![
                AttributeDesc {
                    attribute_index: 0,
                    format: 29,
                    offset: 0,
                    buffer_index: 0,
                },
                AttributeDesc {
                    attribute_index: 1,
                    format: 9,
                    offset: 8,
                    buffer_index: 0,
                },
            ],
        };
        let out = vertex_descriptor(&v).expect("rt's vertex descriptor");
        let layout = out.layouts.get(&0).expect("buffer 0's layout");
        assert_eq!(layout.stride, Some(12));
        assert_eq!(layout.step_function, VertexStepFunction::PerVertex);
        let attribute = out.attributes.get(&1).expect("attribute 1");
        assert_eq!(attribute.format, VertexFormat::UChar4Normalized);
        assert_eq!((attribute.offset, attribute.buffer_index), (8, 0));
        let dynamic = VertexDesc {
            layouts: vec![BufferLayoutDesc {
                stride: u64::MAX,
                ..v.layouts[0]
            }],
            attributes: Vec::new(),
        };
        let out = vertex_descriptor(&dynamic).expect("rt's vertex descriptor");
        assert_eq!(out.layouts.get(&0).map(|l| l.stride), Some(None));
        let unknown = VertexDesc {
            attributes: vec![AttributeDesc {
                format: 999,
                ..v.attributes[0]
            }],
            ..v
        };
        assert_eq!(
            vertex_descriptor(&unknown).err().as_deref(),
            Some("vertex format 999")
        );

        assert_eq!(color_attachment(&ColorAttachmentDesc::default()), Ok(None));
        let blended = ColorAttachmentDesc {
            pixel_format: 80,
            blending_enabled: true,
            source_rgb_blend_factor: 4,
            destination_rgb_blend_factor: 5,
            write_mask: 0x3,
            ..ColorAttachmentDesc::default()
        };
        let a = color_attachment(&blended).expect("rt's attachment");
        let a = a.expect("an attachment of BGRA8Unorm");
        assert_eq!(a.pixel_format, PixelFormat::BGRA8Unorm);
        assert!(a.blending_enabled);
        assert_eq!(a.source_rgb_blend_factor, BlendFactor::SourceAlpha);
        assert_eq!(
            a.destination_rgb_blend_factor,
            BlendFactor::OneMinusSourceAlpha
        );
        assert_eq!(a.alpha_blend_operation, BlendOperation::Add);
        assert_eq!(a.write_mask.as_raw(), 0x3);
        let unknown = ColorAttachmentDesc {
            alpha_blend_operation: 99,
            ..blended
        };
        assert_eq!(
            color_attachment(&unknown).err().as_deref(),
            Some("blend operation 99")
        );

        assert_eq!(tessellation(&TessellationDesc::default()), Ok(None));
        let t = TessellationDesc {
            partition_mode: 2,
            max_factor: 64,
            output_winding: 1,
            ..TessellationDesc::default()
        };
        let d = tessellation(&t).expect("rt's tessellation state");
        let d = d.expect("a state other than Metal's defaults");
        assert_eq!(d.partition_mode, TessellationPartitionMode::FractionalOdd);
        assert_eq!(d.max_tessellation_factor, 64);
        assert_eq!(d.output_winding_order, Winding::CounterClockwise);
        let unknown = TessellationDesc {
            factor_format: 7,
            ..t
        };
        assert_eq!(
            tessellation(&unknown).err().as_deref(),
            Some("tessellation factor format 7")
        );

        let mut private = PrivateRenderState::new();
        let p = PrivateRenderDesc {
            open_gl_mode_enabled: true,
            sample_mask: 0x5,
            ..PrivateRenderDesc::default()
        };
        private_state(&mut private, &p);
        assert!(private.open_gl_mode_enabled);
        assert_eq!(private.sample_mask, 0x5);
        assert_eq!(private.clip_distance_enable_mask, 0xff);
        let past = count("raster sample count", 1 << 32);
        assert_eq!(
            past.err().as_deref(),
            Some("raster sample count 4294967296")
        );
    }

    /// On the GPU: the display's value and whole descriptors make rt's states, whose samplers take
    /// indices as compute states' do; what Metal refuses, what rt refuses and what is unbuilt is
    /// the creator's `NSError`, counted under its kind; a state answers a drawing state's
    /// answers, its imageblock lengths unbuilt.
    #[test]
    fn render_pipeline_states_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let function = |bytes: &[u8], name: &str| -> RtObj<RtFunction> {
            let library = device.new_library(bytes).expect("never a raise");
            let library = library.expect("a metallib").expect("a library");
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function of the table")
        };
        let metal = RtRefusal::Refused.code();

        let shaders = device.new_shader_library().expect("our shaders");
        let shaders = shaders.expect("a library");
        let shader = |name: &str| {
            let made = shaders.new_function(on, name).expect("never a raise");
            made.expect("a function of our shaders")
        };
        let (present, present_fragment) = (
            shader("displayPresentVertex"),
            shader("displayPresentFragment"),
        );
        let display = RenderPipelineDesc {
            label: Some("display"),
            vertex: &present,
            fragment: &present_fragment,
            color0_pixel_format: 80,
            resource_index: None,
        };
        let made = device.new_render_pipeline_state(&display);
        let state = made.expect("a render state").expect("a render state");
        assert_eq!(state.max_total_threads_per_threadgroup(), 0);
        assert!(!state.threadgroup_size_matches_tile_size());
        assert_eq!((state.gpu_resource_id(), state.resource_index()), (0, 0));
        assert!(!state.support_indirect_command_buffers());
        assert_eq!(state.max_total_threads_per_object_threadgroup(), 0);
        assert_eq!(state.max_total_threadgroups_per_mesh_grid(), 0);
        assert_ne!(state.unique_identifier(), 0);
        // One BGRA8Unorm attachment, one sample: a texel of 4 bytes, as Metal packs it
        // (`metal2vulkan::metal::format::imageblock_sample_length`, measured on an M3 Max).
        let refusals = counted(IMAGEBLOCK_LENGTH);
        assert_eq!(state.imageblock_sample_length(), 4);
        let length = state.imageblock_memory_length(MtlSize::new(8, 8, 1));
        assert_eq!(length.expect("never a raise"), 4 * 8 * 8);
        assert_eq!(counted(IMAGEBLOCK_LENGTH), refusals);
        let before = counted(metal);
        let unknown = RenderPipelineDesc {
            color0_pixel_format: 0xdead,
            ..display
        };
        assert!(device.new_render_pipeline_state(&unknown).is_err());
        assert_eq!(counted(metal), before + 1);

        // The fragment function's texture read takes a sampler index, which a second state of the
        // same functions holds too, until both go.
        let (numbered, read) = (
            function(NUMBERED, "numbered_vertex"),
            function(NUMBERED, "numbered_fragment_read"),
        );
        let mut desc = FullRenderPipelineDesc::new(&numbered, Some(&read));
        desc.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 70,
                ..ColorAttachmentDesc::default()
            },
        )];
        let held = device.numbering().held();
        let full = |desc: &FullRenderPipelineDesc<'_, RtPlatform>| {
            let made = device.new_render_pipeline_state_full(desc);
            made.expect("a render state").expect("a render state")
        };
        let (first, second) = (full(&desc), full(&desc));
        assert_eq!(first.samplers(), [rt::NumberedSampler::Read]);
        // The fragment function reads its slot 0, the vertex function uses none.
        for function in [
            RenderFunction::Vertex,
            RenderFunction::Fragment,
            RenderFunction::Tile,
            RenderFunction::Object,
            RenderFunction::Mesh,
        ] {
            assert_eq!(first.texture_slot_writes(function, 0), Some(false));
        }
        assert_eq!(first.sampler_indices(), second.sampler_indices());
        assert_eq!(device.numbering().held(), held + 1);
        assert_ne!(first.unique_identifier(), second.unique_identifier());
        drop((first, second));
        assert_eq!(device.numbering().held(), held);

        // A fragment function that writes its slot 0 may write it.
        let (sweep, write) = (
            function(ROUNDING, "sweep_vertex"),
            function(ROUNDING, "write_fragment"),
        );
        let writer = RenderPipelineDesc {
            label: None,
            vertex: &sweep,
            fragment: &write,
            color0_pixel_format: 80,
            resource_index: None,
        };
        let writer = device.new_render_pipeline_state(&writer);
        let writer = writer.expect("a render state").expect("a render state");
        let slot = |function, index| writer.texture_slot_writes(function, index);
        assert_eq!(slot(RenderFunction::Fragment, 0), Some(true));
        assert_eq!(slot(RenderFunction::Fragment, 1), Some(false));
        assert_eq!(slot(RenderFunction::Vertex, 0), Some(false));
        drop(writer);

        // A vertex descriptor, a blended attachment, a depth format and the guest's private state.
        let (vdesc, color) = (
            function(VDESC, "render_vdesc"),
            function(COLOR, "render_color"),
        );
        let layouts = vec![BufferLayoutDesc {
            buffer_index: 0,
            stride: 12,
            step_function: 1,
            step_rate: 1,
        }];
        let attributes = vec![
            AttributeDesc {
                attribute_index: 0,
                format: 29,
                offset: 0,
                buffer_index: 0,
            },
            AttributeDesc {
                attribute_index: 1,
                format: 9,
                offset: 8,
                buffer_index: 0,
            },
        ];
        let mut desc = FullRenderPipelineDesc::new(&vdesc, Some(&color));
        desc.vertex_descriptor = Some(VertexDesc {
            layouts: layouts.clone(),
            attributes: attributes.clone(),
        });
        desc.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 80,
                blending_enabled: true,
                source_rgb_blend_factor: 4,
                destination_rgb_blend_factor: 5,
                ..ColorAttachmentDesc::default()
            },
        )];
        desc.depth_pixel_format = 252;
        desc.private_state = Some(PrivateRenderDesc {
            open_gl_mode_enabled: true,
            ..PrivateRenderDesc::default()
        });
        let drawn = full(&desc);
        assert!(drawn.samplers().is_empty());

        // An attribute of a buffer the value sets no layout of: the translator's refusal.
        let translation = RtRefusal::Translation.code();
        let before = counted(translation);
        let mut unlaid = attributes.clone();
        unlaid[1].buffer_index = 1;
        let mut refused = FullRenderPipelineDesc::new(&vdesc, Some(&color));
        refused.vertex_descriptor = Some(VertexDesc {
            layouts: layouts.clone(),
            attributes: unlaid,
        });
        assert!(device.new_render_pipeline_state_full(&refused).is_err());
        assert_eq!(counted(translation), before + 1);

        // Metal's refusals of values: a blend factor, a ninth attachment, a count past 32 bits.
        let before = counted(metal);
        let mut refused = FullRenderPipelineDesc::new(&vdesc, Some(&color));
        refused.vertex_descriptor = desc.vertex_descriptor.clone();
        refused.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 80,
                source_alpha_blend_factor: 99,
                ..ColorAttachmentDesc::default()
            },
        )];
        assert!(device.new_render_pipeline_state_full(&refused).is_err());
        refused.colors = vec![(8, ColorAttachmentDesc::default())];
        assert!(device.new_render_pipeline_state_full(&refused).is_err());
        refused.colors = Vec::new();
        refused.raster_sample_count = 1 << 32;
        assert!(device.new_render_pipeline_state_full(&refused).is_err());
        assert_eq!(counted(metal), before + 3);

        // rt's gaps: indirect command buffers and a sample count Vulkan has no bit of.
        let gap = RtRefusal::Gap.code();
        let before = counted(gap);
        refused.raster_sample_count = 1;
        refused.support_indirect_command_buffers = true;
        assert!(device.new_render_pipeline_state_full(&refused).is_err());
        refused.support_indirect_command_buffers = false;
        refused.raster_sample_count = 3;
        assert!(device.new_render_pipeline_state_full(&refused).is_err());
        assert_eq!(counted(gap), before + 2);

        // Mesh states: unbuilt.
        let mesh = Unbuilt::MeshPipeline.code();
        let before = counted(mesh);
        let desc = FullMeshRenderPipelineDesc {
            label: None,
            object: None,
            mesh: &vdesc,
            fragment: Some(&color),
            colors: Vec::new(),
            depth_pixel_format: 0,
            stencil_pixel_format: 0,
            raster_sample_count: 1,
            alpha_to_coverage: false,
            alpha_to_one: false,
            rasterization_enabled: true,
            max_total_threads_per_object_threadgroup: 0,
            max_total_threads_per_mesh_threadgroup: 0,
            max_total_threadgroups_per_mesh_grid: 0,
            object_threadgroup_size_is_multiple_of_execution_width: false,
            mesh_threadgroup_size_is_multiple_of_execution_width: false,
            payload_memory_length: 0,
            max_vertex_amplification_count: 1,
            required_threads_per_object_threadgroup: MtlSize::new(0, 0, 0),
            required_threads_per_mesh_threadgroup: MtlSize::new(0, 0, 0),
            support_indirect_command_buffers: false,
            object_linked: LinkedFunctionsDesc::default(),
            mesh_linked: LinkedFunctionsDesc::default(),
            fragment_linked: LinkedFunctionsDesc::default(),
        };
        assert!(device.new_mesh_render_pipeline_state_full(&desc).is_err());
        assert_eq!(counted(mesh), before + 1);
    }
}
