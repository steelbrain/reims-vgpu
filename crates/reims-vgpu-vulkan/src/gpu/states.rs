//! The render family's state objects, made from core's values and the translator's descriptors:
//! sampler states, depth-stencil states, render pipeline states and mesh render pipeline states
//! (`GpuDevice`'s creators of them call here).
//!
//! - **A value's raw properties** become the runtime's types in the value's order, each named by
//!   its setter: a raw value no Metal enumerator is dies at that setter
//!   ([`crate::gpu::metal_value`]), as the runtime's types cannot hold it.
//! - **A sampler state** of equal requests is one object while it lives, as Metal's is on this
//!   host ([`crate::gpu::resource::sampler_state`]). The value's resource index, which core sets
//!   without the force flag, places the state at that ID: Metal on this host places a sampler at
//!   an index set so (`render_objects_placement_sampler`: index 7 gives `gpuResourceID` 7), and the
//!   runtime's placement is that ID (index 0 is plain allocation, as Metal's is). Plain
//!   allocation hands out the IDs Metal's does, as the runtime numbers them: Metal's
//!   device keeps a sampler ID of its own for good from its first command queue, indirect
//!   command buffer or texture of its own memory (sampler states made after a texture take 2 and
//!   on, or 3 and on with 2 reserved); and a pipeline state, as it is made, takes one for each of
//!   its functions' constexpr samplers, texture reads and texture writes, and the device's own
//!   for a render state with a fragment function made before that, unless a live holder holds an
//!   equal one. The state holds them, as do every function object of its functions and every
//!   live library of their bytes, and each goes back with its last holder. The constexpr samplers
//!   of the visible functions a stage links come after the stage's own, as Metal numbers them.
//!   Where it does not number them as Metal does, the runtime announces
//!   `linked-sampler-numbering`, which dies ([`crate::answer::DIES`]): a stage that links two or
//!   more with constexpr samplers, whose order Metal itself does not decide, and a function a
//!   table can hold whose samplers the runtime cannot read, among them. One linked function
//!   alone, of `functions` or `privateFunctions`, called or not, is numbered as Metal numbers it.
//!   The paravirtualized device's guest forms link none; the graphics wire's pipelines link the
//!   functions the guest lists (`reims-vgpu`'s decoder). This platform's libraries, functions and
//!   states hold the runtime's exactly as long as the guest's handles to them live, so its sampler
//!   IDs are Metal's (`tests::sampler_ids_follow_pipeline_holders_as_metals`).
//! - **A pipeline's refusals** that Metal answers with an `NSError` are one here
//!   ([`pipeline_error`]): what the compile finds wrong with the functions or the descriptor, a
//!   vertex function writing a point size under a line or triangle topology, a function Metal's
//!   compiler itself refuses on this device, an import no dynamic library supplies, a call through
//!   a visible-function reference no linked function resolves, and a state that supports indirect
//!   command buffers whose functions use a direct texture or sampler argument. A gap of the
//!   runtime's or the device's, where Metal makes the state (a function's SIMD-group operations
//!   in a stage the runtime does not take them in, a vertex function's on MoltenVK, which Metal
//!   makes and for which device info's key 33 answers no, [`crate::gpu::caps`]; a colour
//!   attachment of a format the runtime has and does not render to there), is the runtime's
//!   refusal as the `NSError` by default, relayed, and dies in strict mode
//!   ([`crate::answer::relay_gap`]). The rest die at the selector: an ID past Metal's limit,
//!   which ends Metal's process, an unspecialized function constant, which Metal aborts for, and
//!   the runtime's other refusals, as Metal raises at none. Such a death's line, and a relayed
//!   gap's, also names the descriptor's formats (`formats`: its colour attachments' with their
//!   write masks, its depth and stencil, its vertex attributes'), which the runtime's error does
//!   not carry. A binary archive miss cannot happen: core's descriptors list no archive.
//!
//!   Measured on the M3 Max (macOS 26.7, one case per process, plain and under `MTL_DEBUG_LAYER`):
//!   the compute creators (both selectors) and the render creator answer nil and an `NSError` of
//!   code 3 for a render state setter on an indirect render command (`set_cull_mode`,
//!   `set_depth_stencil_state`) in a kernel, vertex or fragment function, and for a fragment
//!   function's read of a program-scope `thread_index_in_quadgroup` (a read of
//!   `thread_index_in_simdgroup` or `threads_per_simdgroup` there is made); and of code 2
//!   ("Undefined symbols:") for a function whose import the library at its install name lacks, or
//!   whose install name names no file. The compute and render creators answer nil and an
//!   `NSError` of code 3 for a stage whose call through a visible-function reference no function
//!   its descriptor links under that name resolves ("unresolved visible function reference:
//!   linked_add_one", "Reason: visible function not loaded", macOS 27.0.1, plain and under
//!   `MTL_DEBUG_LAYER`): none linked, one linked to the other stage, one in a group alone, or one
//!   in `binaryFunctions` not made compiled to a binary, or made so, with no list but
//!   `binaryFunctions` holding it. The functions that draw without a setter, and those whose
//!   library exports the import, are made. The runtime refuses a fragment function's read of a
//!   program-scope `thread_index_in_simdgroup` as a gap of its own (`Lowering(StageIo)`), which
//!   dies where Metal makes the pipeline. The compute and render creators answer nil and an
//!   `NSError` of code 3 for a descriptor with `supportIndirectCommandBuffers` whose kernel,
//!   vertex or fragment function uses a `[[texture(n)]]` or `[[sampler(n)]]` argument once Metal
//!   links the pipeline ("Fragment shader cannot be used with indirect command buffers", naming
//!   the fragment function alone where both use one): Metal makes the state where a vertex
//!   function's read reaches only outputs the fragment function does not read, or there is no
//!   fragment function, and where a fragment function's reaches only a colour no attachment
//!   writes (none, colour 0 `Invalid`, or a write mask of none or of none of its format's channels,
//!   measured). The runtime refuses the same states with Metal's error, deciding the use
//!   by liveness as Metal links, and the creators answer that `NSError`. A use only a rule the
//!   runtime keeps beyond Metal keeps is unproven (`Error::Unproven`): an endless loop, stack
//!   memory, a gated-off mesh output, or a colour whose attachment's write is unmeasured
//!   (`UnmeasuredColorWrite`: a mask of some of the format's channels but not all, a format of
//!   fewer than four channels even written in full, an unspecialized mask, or a dual-source second
//!   colour under blending that reads no second source). Metal's answer is unmeasured, and the
//!   creator dies, naming the stage and the rule. The compile layer decides the use from the
//!   arguments' roles alone, so an argument whose reflection Metal's answer has not been measured
//!   for decides nothing; where it refuses to decide (the decode's or the
//!   specialization's own refusal), the runtime refuses the state, unproven, and the creator dies
//!   there, naming the refusal. The runtime decides it once, as it makes the state: a state it
//!   made is one whose use it decided, and nothing here decodes its functions again. The runtime
//!   refuses a render state with rasterization off whose vertex function returns a value with
//!   Metal's error too (`UnrasterizedVertexReturn`, code 3), and one with rasterization on whose
//!   vertex function returns nothing (`RasterizedVoidVertex`, code 3) but in OpenGL mode, where
//!   Metal makes it and it rasterizes nothing (`private_state`). With rasterization off the
//!   descriptor keeps the fragment function, which the runtime checks as Metal links it and leaves
//!   out of the Vulkan compile: it answers Metal's error (code 3) where a written colour
//!   attachment keeps the fragment function's read of a `stage_in` input, which no vertex
//!   function writes then, or, with `supportIndirectCommandBuffers`, its direct texture or
//!   sampler, and where the function writes depth with no depth attachment but in OpenGL mode; it
//!   makes the state otherwise, as Metal does (measured: `unrasterized_states_are_metals`).
//!
//!   The `NSError`'s domain is the driver's (`AGXMetalG15X_M1` on the M3 Max; another GPU family's
//!   driver names another), and its text names the intrinsic or the symbol, which the runtime's
//!   refusal does not always carry. Core reads a pipeline creator's `NSError` only for its
//!   description, which goes into a log line and the creation's error string
//!   (`resources::manager::create::pipeline`): no domain, code or text reaches the guest or the
//!   host. So the `NSError` here carries the runtime's domain and text, with Metal's code; the
//!   refusal, in place of a death, is what the guest observes.
//! - **A sampler's descriptor** is a debug line under this module's target
//!   (`REIMS_VGPU_LOG=debug`, or `warn,reims_vgpu_vulkan::gpu::states=debug`), written before its
//!   first setter ([`Sampler`]): each property of the guest's record, so a death at one setter
//!   (`setMinFilter:` of a filter the runtime has no enumerator of, say) shows the whole
//!   descriptor. The record holds no LOD bias, which only a binding's setter takes, and no label.
//!   The line is a diagnostic: it changes nothing the guest or the host sees, and with the
//!   target's debug lines off nothing formats it.
//! - **A cubic sampler** (a min or mag filter of 2, `MTLSamplerMinMagFilterCubic`) is the
//!   runtime's `Cubic`. Without a mip filter the runtime samples through it as a bicubic sampler
//!   of its address modes and border colour, whatever its other filter, and announces
//!   `bicubic-filter-arithmetic` at the dispatch or draw, which dies at its selector, naming the
//!   kind ([`crate::answer::DIES`]). With a mip filter, under mirror clamp to the edge, with a
//!   minimum or maximum reduction, or with anisotropy, the runtime refuses the dispatch or draw
//!   that samples through it, which dies at its selector, naming the refusal, where Metal
//!   samples; and so under mirrored repeat where the function also gathers through it, or samples
//!   in pixel coordinates or a 1D texture. The state itself is made, as Metal makes it. Through
//!   an argument buffer, a stage other than a fragment one tests each sampler it reaches, and
//!   the runtime fails the command buffer of a function that reached a cubic one, which dies
//!   where its failure is first read ([`crate::gpu::command`]), where Metal samples; one that
//!   never reaches a cubic sampler runs as Metal's while a cubic state is alive. A fragment
//!   stage tests none: the runtime leaves a cubic state out of its sampler tables while a render
//!   pipeline samples argument samplers, and so a state the tables cannot hold, as more states
//!   live than a pipeline's set-1 sampler arrays hold (at most 65536 elements each, fewer where
//!   the device's limits are), whether or not a function reaches it
//!   (`unlisted-argument-sampler`, of [`crate::answer::DIES`]): a function that reads it through
//!   its ID reads the neutral sampler, where Metal's functions reach every live state by its ID.
//!   The runtime announces it as the state is made, or as a command is encoded while it lives;
//!   it is relayed, and in strict mode dies there, at the call's selector.
//! - **A texture slot sampled through an argument buffer's sampler**, whatever its filter, is a
//!   function the runtime refuses where its pipeline state is made (`ArgumentBufferTextures`),
//!   which dies there, where Metal makes and runs it. A texture an argument buffer holds, sampled
//!   through a sampler it holds, samples as Metal's does.
//! - **A sampler's raw border colour** is the colour Metal samples for it
//!   (`rt::sampler_border_color`): Metal takes any value and keeps its low two bits, 3 sampling
//!   as transparent black.
//! - **A sampler's or a depth-stencil state's refusal** dies at the selector: Metal makes a
//!   state of any descriptor these take (measured: a sampler of `maxAnisotropy` 0 or 17, of a
//!   minimum LOD clamp past the maximum, or of border colour 3, and a depth-stencil state of the
//!   default descriptor, each made, no exception raised), and so does the runtime of each of
//!   those. A gap of the runtime's or the device's (a sampler address mode or comparison the
//!   device does not take) answers nil by default, relayed, and dies in strict mode
//!   ([`crate::answer::relay_gap`]).
//! - **Labels** are diagnostics: a pipeline state takes its descriptor's or its value's
//!   (`PipelineOptions::label`), which no answer to the guest reads and the runtime compiles
//!   nothing of.

use std::collections::BTreeMap;
use std::sync::Weak;

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::PrimitiveTopologyClass;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, LinkedFunctions, MeshPipelineDescriptor,
    RenderPipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::resource::RenderPipeline as _;
use reims_vgpu_core::platform::types::{MtlSize, NsErrorText};
use reims_vgpu_core::platform::values::{
    ColorAttachmentDesc, DepthStencilDesc, FullComputePipelineDesc, FullMeshRenderPipelineDesc,
    FullRenderPipelineDesc, LinkedFunctionsDesc, PrivateRenderDesc, RenderPipelineDesc,
    SamplerDesc, StageInputDesc, StencilDesc, VertexDesc,
};
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::sync::Mutex;

use crate::answer::{
    Nil, Policy, create, decide_describing, error_text, get_describing, plain, plain_at, relay_gap,
    unrepresentable,
};
use crate::gpu::device::VulkanDevice;
use crate::gpu::metal_value;
use crate::gpu::resource::{
    MeshAnswers, MeshDescriptor, MeshStageHeld, RenderDescriptor, RenderPipelineObject,
    VulkanDepthStencilState, VulkanFunction, VulkanRenderPipeline, VulkanSamplerState,
    render_pipeline_of, sampler_state,
};
use crate::gpu::timing;
use crate::platform::{Obj, VulkanPlatform};

/// The render pipeline states made with indirect command buffer support, held weakly, by the
/// runtime's `gpuResourceID` of each: an indirect command names one by the runtime's state, whose
/// input primitive topology class only this platform holds ([`points_sized`]), and the runtime's
/// state keeps its ID while it lives.
static INDIRECT_STATES: Mutex<BTreeMap<u64, Vec<Weak<RenderPipelineObject>>>> =
    Mutex::new(BTreeMap::new());

/// Whether the runtime's state `raw`, which an indirect command names, sizes its points as Metal
/// does (`crate::gpu::encode_render`'s `sized`): its vertex function writes `[[point_size]]`, or
/// each live state of the program's that is `raw` is of the point input primitive topology class.
/// `None` where it writes none and no live state of the program's is `raw` (the program released
/// it). Asked once an indirect command, it reads only the states of `raw`'s ID.
pub(crate) fn points_sized(raw: &rt::RenderPipelineState) -> Option<bool> {
    if plain(|| raw.writes_point_size()) {
        return Some(true);
    }
    let id = plain(|| raw.gpu_resource_id()).0;
    // Each is let go of past the lock: a program's last reference may be among them.
    let states: Vec<_> = plain(|| {
        let states = INDIRECT_STATES.lock();
        let held = states.get(&id).map_or(&[][..], Vec::as_slice);
        held.iter().filter_map(Obj::upgrade).collect()
    });
    let classes: Vec<_> = states
        .iter()
        .filter(|state| state.raw == *raw)
        .map(|state| state.input_primitive_topology)
        .collect();
    (!classes.is_empty()).then(|| classes.iter().all(|c| *c == PrimitiveTopologyClass::Point))
}

/// A `u64` property the runtime holds as `u32`, sent with `sel`.
fn small(sel: &'static str, what: &str, raw: u64) -> u32 {
    u32::try_from(raw).unwrap_or_else(|_| unrepresentable(sel, &format!("{what} {raw}")))
}

/// The resource ID a value's or a descriptor's resource index places a state at: Metal's index,
/// forced or not (the module docs).
fn placement(resource_index: Option<(u64, bool)>) -> rt::IdPlacement {
    resource_index.map_or(rt::IdPlacement::Any, |(index, _force)| {
        rt::IdPlacement::At(rt::ResourceId(index))
    })
}

/// A sampler descriptor's debug line ([the module docs](self)): the selector, then each property
/// of `SamplerDesc` raw, `unset` for one the value does not set: the min, mag and mip filters,
/// the address modes, the border colour, normalized coordinates, the LOD clamps and average, the
/// maximum anisotropy, the compare function, the reduction mode, `supportArgumentBuffers` and the
/// resource index.
pub struct Sampler<'a>(pub &'a SamplerDesc);

/// A property of a [`Sampler`] line: its value, or `unset`.
struct Set<T>(Option<T>);

impl<T: std::fmt::Display> std::fmt::Display for Set<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(v) => v.fmt(f),
            None => f.write_str("unset"),
        }
    }
}

impl std::fmt::Display for Sampler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.0;
        write!(
            f,
            "newSamplerStateWithDescriptor: min filter {}, mag filter {}, mip filter {}, address \
             modes s {} t {} r {}, border colour {}, normalized coordinates {}, lod min {} max {} \
             average {}, max anisotropy {}, compare function {}, reduction mode {}, \
             supportArgumentBuffers {}, resource index {}",
            Set(d.min_filter),
            Set(d.mag_filter),
            Set(d.mip_filter),
            Set(d.s_address_mode),
            Set(d.t_address_mode),
            Set(d.r_address_mode),
            Set(d.border_color),
            Set(d.normalized_coordinates),
            Set(d.lod_min_clamp),
            Set(d.lod_max_clamp),
            Set(d.lod_average),
            Set(d.max_anisotropy),
            Set(d.compare_function),
            Set(d.reduction_mode),
            Set(d.support_argument_buffers),
            Set(d.resource_index),
        )
    }
}

/// The runtime's sampler descriptor and options of `d`, each property set in the value's order,
/// after its debug line ([`Sampler`]).
fn sampler_descriptor(d: &SamplerDesc) -> (SamplerDescriptor, rt::SamplerStateOptions) {
    log::debug!("{}", Sampler(d));
    let mut s = SamplerDescriptor::new();
    if let Some(v) = d.min_filter {
        s.min_filter = metal_value("setMinFilter:", "sampler filter", v);
    }
    if let Some(v) = d.mag_filter {
        s.mag_filter = metal_value("setMagFilter:", "sampler filter", v);
    }
    if let Some(v) = d.lod_average {
        s.lod_average = v;
    }
    if let Some(v) = d.compare_function {
        s.compare_function = metal_value("setCompareFunction:", "compare function", v);
    }
    if let Some(v) = d.s_address_mode {
        s.s_address_mode = metal_value("setSAddressMode:", "address mode", v);
    }
    if let Some(v) = d.t_address_mode {
        s.t_address_mode = metal_value("setTAddressMode:", "address mode", v);
    }
    if let Some(v) = d.r_address_mode {
        s.r_address_mode = metal_value("setRAddressMode:", "address mode", v);
    }
    if let Some(v) = d.mip_filter {
        s.mip_filter = metal_value("setMipFilter:", "mip filter", v);
    }
    if let Some(v) = d.max_anisotropy {
        s.max_anisotropy = small("setMaxAnisotropy:", "an anisotropy", v);
    }
    if let Some(v) = d.normalized_coordinates {
        s.normalized_coordinates = v;
    }
    let mut options = rt::SamplerStateOptions::default();
    if let Some(v) = d.support_argument_buffers {
        options.support_argument_buffers = v;
    }
    if let Some(v) = d.reduction_mode {
        s.reduction_mode = metal_value("setReductionMode:", "reduction mode", v);
    }
    if let Some(v) = d.lod_min_clamp {
        s.lod_min_clamp = v;
    }
    if let Some(v) = d.lod_max_clamp {
        s.lod_max_clamp = v;
    }
    if let Some(v) = d.border_color {
        s.border_color = rt::sampler_border_color(v);
    }
    options.placement = placement(d.resource_index.map(|index| (index, false)));
    (s, options)
}

/// `newSamplerStateWithDescriptor:`: a refusal dies at the selector, as Metal makes a state of
/// any descriptor, and its process ends at an ID past its limit; but a gap of the runtime's or
/// the device's, which by default answers nil ([`crate::answer::relay_gap`]).
pub(crate) fn new_sampler_state(
    device: &VulkanDevice,
    desc: &SamplerDesc,
) -> RaiseResult<Option<VulkanSamplerState>> {
    let sel = "newSamplerStateWithDescriptor:";
    let (descriptor, options) = sampler_descriptor(desc);
    let made = create(sel, Policy::Dies, Nil::Never, || {
        device
            .raw
            .new_sampler_state_with_options(&descriptor, options)
    })?;
    Ok(made.map(sampler_state))
}

/// A face's stencil of `s`, every property set in field order.
fn stencil(s: &StencilDesc) -> StencilDescriptor {
    let mut out = StencilDescriptor::default();
    out.stencil_compare_function = metal_value(
        "setStencilCompareFunction:",
        "compare function",
        s.stencil_compare_function,
    );
    out.stencil_failure_operation = metal_value(
        "setStencilFailureOperation:",
        "stencil operation",
        s.stencil_failure_operation,
    );
    out.depth_failure_operation = metal_value(
        "setDepthFailureOperation:",
        "stencil operation",
        s.depth_failure_operation,
    );
    out.depth_stencil_pass_operation = metal_value(
        "setDepthStencilPassOperation:",
        "stencil operation",
        s.depth_stencil_pass_operation,
    );
    out.read_mask = s.read_mask;
    out.write_mask = s.write_mask;
    out
}

/// `newDepthStencilStateWithDescriptor:`: Metal makes a state of any descriptor, and so does the
/// runtime; a refusal dies at the selector, but a gap of the runtime's or the device's, which by
/// default answers nil ([`crate::answer::relay_gap`]).
pub(crate) fn new_depth_stencil_state(
    device: &VulkanDevice,
    desc: &DepthStencilDesc,
) -> RaiseResult<Option<VulkanDepthStencilState>> {
    let sel = "newDepthStencilStateWithDescriptor:";
    let mut d = DepthStencilDescriptor::default();
    if let Some(v) = desc.depth_compare_function {
        d.depth_compare_function = metal_value("setDepthCompareFunction:", "compare function", v);
    }
    if let Some(v) = desc.depth_write_enabled {
        d.depth_write_enabled = v;
    }
    if let Some(front) = &desc.front {
        d.front_face_stencil = stencil(front);
    }
    if let Some(back) = &desc.back {
        d.back_face_stencil = stencil(back);
    }
    let made = create(sel, Policy::Dies, Nil::Never, || {
        device.raw.new_depth_stencil_state(&d)
    })?;
    Ok(made.map(Obj::new))
}

/// The `NSError` a pipeline creator answers for the runtime's refusal `e` of a pipeline, where
/// Metal answers one (the module docs): the runtime's domain and text, and Metal's code where it
/// was measured, else 0; `None` for every other refusal, which dies.
/// - Code 0: what the compile finds wrong with the functions or the descriptor, and a vertex
///   function writing a point size under a line or triangle topology.
/// - Code 3 ([`METAL_REFUSES`]): a function the runtime refuses as Metal itself refuses it on
///   this device, by a GPU family the device lacks (`Needs::MetalGpuFamily`: a render state set
///   on an indirect render command) or by the devices Metal was measured to refuse it on
///   (`Needs::MetalRefuses`: a fragment function's read of a program-scope
///   `thread_index_in_quadgroup`); and a colour attachment the fragment function's outputs do
///   not fit (`InvalidDescriptor(ColorAttachment)`), measured: a blend of the second source whose
///   function writes no second output ("Fragment shader does not write to render target
///   color(0), index(1) that is required for blending"), and a `half4` output into an
///   `RGBA8Uint` attachment (`encode_render`'s
///   `second_source_blends_without_a_second_output_are_refused_as_metals`); and a call through a
///   visible-function reference no function the stage links resolves (`NeedsInput` of
///   `LinkedFunction`), measured ("unresolved visible function reference":
///   `tests::metal_tests::unresolved_visible_calls_are_metals`).
/// - Code 2 ([`UNDEFINED_SYMBOLS`]): a function whose import no linked dynamic library exports,
///   or whose install name loaded no library (`NeedsInput` of `DynamicSymbol` or
///   `DynamicLibrary`).
/// - Metal's error as the runtime states it, with its code ([`rt::Error::MetalError`]), which the
///   runtime answers only where plain Metal answers nil with that `NSError`: a state that
///   supports indirect command buffers whose function uses a direct texture or sampler argument
///   once Metal links the pipeline (`IndirectCommandsDirectArgument`), a render state with
///   rasterization off whose vertex function returns a value (`UnrasterizedVertexReturn`), and
///   one with rasterization on whose vertex function returns nothing outside OpenGL mode
///   (`RasterizedVoidVertex`), among them; a problem the runtime adds there is the `NSError` too.
///
/// An answer the runtime cannot prove ([`rt::Error::Unproven`]) dies, naming it (`unproven`), as
/// does every other refusal.
pub(crate) fn pipeline_error(e: &rt::Error) -> Option<NsErrorText> {
    use metal2vulkan::error::{DescriptorError, Error as Compile, Needs, RuntimeInput};
    let code = match e {
        rt::Error::Compile(Compile::InvalidDescriptor(DescriptorError::ColorAttachment {
            ..
        })) => METAL_REFUSES,
        rt::Error::Compile(Compile::InvalidInput(_) | Compile::InvalidDescriptor(_)) => 0,
        rt::Error::Invalid(i) if i.problem == rt::Problem::PointSizeTopology => 0,
        rt::Error::Unsupported(u)
            if matches!(u.needs, Needs::MetalGpuFamily(_) | Needs::MetalRefuses(_)) =>
        {
            METAL_REFUSES
        }
        rt::Error::Compile(Compile::NeedsInput(RuntimeInput::LinkedFunction { .. })) => {
            METAL_REFUSES
        }
        rt::Error::Compile(Compile::NeedsInput(
            RuntimeInput::DynamicSymbol { .. } | RuntimeInput::DynamicLibrary { .. },
        )) => UNDEFINED_SYMBOLS,
        rt::Error::MetalError(m) => m.code,
        _ => return None,
    };
    Some(NsErrorText {
        code,
        ..error_text(e)
    })
}

/// The `NSError` code of Metal's pipeline creators for a function its compiler refuses on the
/// device ("Encountered unlowered function call to air.set_cull_mode_render_command", to
/// `air.get_thread_index_in_quadgroup.i32`), at the compute and render creators (measured: the
/// module docs), of the render creator for a colour attachment the fragment function's
/// outputs do not fit, and of the compute and render creators for a call through a
/// visible-function reference nothing linked resolves ([`pipeline_error`]).
const METAL_REFUSES: i64 = 3;

/// The `NSError` code of Metal's pipeline creators for a function whose import no library it
/// links exports, the library at its install name loading or not ("Undefined symbols:"), at the
/// compute and render creators (measured: the module docs).
const UNDEFINED_SYMBOLS: i64 = 2;

/// The colour attachment of a raw pixel format, sent with `setPixelFormat:`: none for
/// `MTLPixelFormatInvalid`.
fn color_attachment(raw: u64) -> Option<ColorAttachmentDescriptor> {
    (raw != 0)
        .then(|| ColorAttachmentDescriptor::new(crate::gpu::pixel_format("setPixelFormat:", raw)))
}

/// `newRenderPipelineStateWithDescriptor:error:` of the display's value, or a window cursor's:
/// its two functions and colour 0's format, blending off, no vertex descriptor, the resource index
/// as the placement. The
/// compile's refusal of a format the functions cannot write (an integer target) is the
/// `NSError`, as Metal answers nil and an error there; no path reaches it under a handler, so
/// every other refusal dies at the send.
pub(crate) fn new_render_pipeline_state(
    device: &VulkanDevice,
    desc: &RenderPipelineDesc<'_, VulkanPlatform>,
) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
    let sel = "newRenderPipelineStateWithDescriptor:error:";
    let mut d =
        RenderPipelineDescriptor::new(StageSource::Function(desc.vertex.function_descriptor()));
    d.fragment = Some(StageSource::Function(desc.fragment.function_descriptor()));
    d.color_attachments[0] = color_attachment(desc.color0_pixel_format);
    let mut options = rt::PipelineOptions::default();
    options.placement = placement(desc.resource_index);
    options.label = desc.label.map(str::to_owned);
    timing::creation(
        sel,
        timing::RENDER_LATER,
        || match plain_at(sel, || device.raw.new_render_pipeline_state(&d, &options)) {
            Ok(state) => Ok(Some(render_pipeline_of(
                state,
                PrimitiveTopologyClass::Unspecified,
                Some(RenderDescriptor::Render(Box::new(d.clone()))),
            ))),
            Err(e) => match pipeline_error(&e) {
                Some(error) => Err(error),
                // Decided as a getter's refusal: none of them returns.
                None => get_describing(sel, Some(&render_formats(&d)), || Err(e)),
            },
        },
        || timing::render_stages(&d),
        |made| timing::answer(made, |s| timing::state(s.unique_identifier())),
    )
}

/// `newRenderPipelineStateWithDescriptor:error:` of the whole value `desc`, the graphics lane's:
/// every field the value carries is the runtime descriptor's own, the private state its
/// [`PrivateRenderState`](metal2vulkan::pipeline::PrivateRenderState) but `openGLModeEnabled`
/// ([`private_state`]). A raw value the runtime has no enumerator of dies at the send
/// ([`metal_value`]); so does the compile's refusal where Metal answers no `NSError` ([`made`]).
/// The lane calls it inside its refusal scope.
pub(crate) fn new_render_pipeline_state_full(
    device: &VulkanDevice,
    desc: &FullRenderPipelineDesc<'_, VulkanPlatform>,
) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
    let sel = "newRenderPipelineStateWithDescriptor:error:";
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(
        desc.vertex
            .function_descriptor()
            .with_linked(linked_functions(&desc.vertex_linked)),
    ));
    d.fragment = desc.fragment.map(|f| {
        StageSource::Function(
            f.function_descriptor()
                .with_linked(linked_functions(&desc.fragment_linked)),
        )
    });
    d.vertex_descriptor = desc
        .vertex_descriptor
        .as_ref()
        .map(|v| vertex_descriptor(sel, v));
    for (index, c) in &desc.colors {
        let slot = usize::try_from(*index)
            .ok()
            .and_then(|i| d.color_attachments.get_mut(i))
            .unwrap_or_else(|| unrepresentable(sel, &format!("colour attachment {index}")));
        *slot = full_color_attachment(sel, c);
    }
    d.depth_format = optional_format(sel, desc.depth_pixel_format);
    d.stencil_format = optional_format(sel, desc.stencil_pixel_format);
    d.raster_sample_count = count(sel, "raster sample count", desc.raster_sample_count);
    d.alpha_to_coverage = desc.alpha_to_coverage;
    d.alpha_to_one = desc.alpha_to_one;
    d.rasterization_enabled = desc.rasterization_enabled;
    d.input_primitive_topology = metal_value(
        sel,
        "primitive topology class",
        desc.input_primitive_topology,
    );
    d.max_vertex_amplification_count = count(
        sel,
        "maximum vertex amplification count",
        desc.max_vertex_amplification_count,
    );
    d.support_indirect_command_buffers = desc.support_indirect_command_buffers;
    if let Some(p) = &desc.private_state {
        private_state(&mut d.private, p);
    }
    let mut options = rt::PipelineOptions::default();
    options.label = desc.label.map(str::to_owned);
    timing::creation(
        sel,
        timing::RENDER_LATER,
        || {
            let state = plain_at(sel, || device.raw.new_render_pipeline_state(&d, &options));
            let class = d.input_primitive_topology;
            match made(
                sel,
                (class, RenderDescriptor::Render(Box::new(d.clone()))),
                state,
            ) {
                Ok(answer) => answer,
                Err(raise) => raise.fatal(),
            }
        },
        || timing::render_stages(&d),
        |made| timing::answer(made, |s| timing::state(s.unique_identifier())),
    )
}

/// `newRenderPipelineStateWithMeshDescriptor:options:reflection:error:` of the whole value `desc`,
/// the graphics lane's, with no options and no reflection: its object, mesh and fragment
/// functions, each with the functions its stage links, its colour attachments, its depth and
/// stencil formats, its raster sample count, alpha-to-coverage, alpha-to-one, rasterization and
/// indirect command buffer support are the runtime descriptor's own, and its label the options'.
/// What the runtime's descriptor does not hold is the state's own ([`mesh_answers`]): the
/// stages' maxima, the grid's maximum and the required threadgroup sizes, which its getters
/// answer and its draws are held to. The execution-width promises are the guest's promise to
/// Metal's compiler, which changes nothing a draw that keeps them draws: they are not carried.
/// The payload length is not carried either: the runtime sizes the payload by the object and
/// mesh functions' own payload types, the larger of the two, which is what Metal reads and
/// writes of a function that reaches its payload through its type. A length past 16384 bytes is
/// Metal's `NSError` ([`payload_refused`]). The maximum vertex amplification count is not
/// carried: the runtime draws a mesh pipeline's draws at one amplification, which a descriptor
/// of any maximum draws, and refuses a draw under a count past 1, which dies there. A raw value
/// the runtime has no enumerator of dies at the send ([`metal_value`]); so does the compile's
/// refusal where Metal answers no `NSError` ([`made`]): among them, on a device with
/// `VK_EXT_mesh_shader` but without `taskShader`, an object function
/// (`Unsupported{MeshShading}`). A device without `VK_EXT_mesh_shader` (MoltenVK among them)
/// makes the state, its object and mesh functions run as compute work the runtime draws the
/// records of (`emulated-mesh-draws`). The lane calls it inside its refusal scope.
pub(crate) fn new_mesh_render_pipeline_state_full(
    device: &VulkanDevice,
    desc: &FullMeshRenderPipelineDesc<'_, VulkanPlatform>,
) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
    let sel = "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:";
    if let Some(refusal) = wrong_mesh_stage(sel, desc) {
        return Err(refusal);
    }
    if desc.payload_memory_length > METAL_PAYLOAD_BYTES {
        return Err(payload_refused(desc.payload_memory_length));
    }
    let stage = |f: &VulkanFunction, linked| {
        StageSource::Function(
            f.function_descriptor()
                .with_linked(linked_functions(linked)),
        )
    };
    let mut d = MeshPipelineDescriptor::new(stage(desc.mesh, &desc.mesh_linked));
    d.object = desc.object.map(|f| stage(f, &desc.object_linked));
    d.fragment = desc.fragment.map(|f| stage(f, &desc.fragment_linked));
    for (index, c) in &desc.colors {
        let slot = usize::try_from(*index)
            .ok()
            .and_then(|i| d.color_attachments.get_mut(i))
            .unwrap_or_else(|| unrepresentable(sel, &format!("colour attachment {index}")));
        *slot = full_color_attachment(sel, c);
    }
    d.depth_format = optional_format(sel, desc.depth_pixel_format);
    d.stencil_format = optional_format(sel, desc.stencil_pixel_format);
    d.raster_sample_count = count(sel, "raster sample count", desc.raster_sample_count);
    d.alpha_to_coverage = desc.alpha_to_coverage;
    d.alpha_to_one = desc.alpha_to_one;
    d.rasterization_enabled = desc.rasterization_enabled;
    d.support_indirect_command_buffers = desc.support_indirect_command_buffers;
    let answers = mesh_answers(device, desc);
    let mut options = rt::PipelineOptions::default();
    options.label = desc.label.map(str::to_owned);
    timing::creation(
        sel,
        timing::RENDER_LATER,
        || {
            let state = plain_at(sel, || {
                device.raw.new_mesh_render_pipeline_state(&d, &options)
            });
            let descriptor = RenderDescriptor::Mesh(Box::new(MeshDescriptor {
                descriptor: d.clone(),
                answers,
            }));
            let class = PrimitiveTopologyClass::Unspecified;
            match made(sel, (class, descriptor), state) {
                Ok(answer) => answer,
                Err(raise) => raise.fatal(),
            }
        },
        || timing::mesh_stages(&d),
        |made| timing::answer(made, |s| timing::state(s.unique_identifier())),
    )
}

/// The refusal of a mesh render pipeline whose stage function is of another stage's type, as the
/// Metal platform refuses it before Metal sees the descriptor (`reims-vgpu-metal`'s
/// `wrong_mesh_stage`): Metal on the M3 Max (macOS 27.0.1) ends the process with SIGSEGV at a
/// vertex function as the mesh function, a mesh function as the fragment function and a fragment
/// function as the object function, and refuses an object function as the mesh function and a
/// mesh function as the object function (measured by the graphics lane's `oracle_mesh`). A
/// guest's input never ends the host, so each is refused in the compiler's domain: code 2 for a
/// vertex function as the mesh function, code 1 for the rest, the mesh function's type asked
/// first, then the object function's, then the fragment function's, so both platforms answer
/// one descriptor alike. A function whose type the runtime does not answer
/// ([`rt::Function::function_type`]'s refusal) is left to the runtime's creation.
fn wrong_mesh_stage(
    sel: &'static str,
    desc: &FullMeshRenderPipelineDesc<'_, VulkanPlatform>,
) -> Option<NsErrorText> {
    use metal2vulkan::metal::FunctionType;
    let stage = |f: &VulkanFunction| plain_at(sel, || f.function_type()).ok();
    let refusal = |code: i64, description: &str| NsErrorText {
        domain: "CompilerError".to_owned(),
        code,
        description: description.to_owned(),
    };
    let mesh = stage(desc.mesh);
    if mesh == Some(FunctionType::Vertex) {
        return Some(refusal(
            2,
            "RasterizationEnabled is false but the vertex shader's return type is not void",
        ));
    }
    if mesh.is_some_and(|t| t != FunctionType::Mesh) {
        return Some(refusal(
            1,
            "meshFunction functionType is not MTLFunctionTypeMesh.",
        ));
    }
    let not = |f: Option<&VulkanFunction>, wanted| f.and_then(stage).is_some_and(|t| t != wanted);
    if not(desc.object, FunctionType::Object) {
        return Some(refusal(
            1,
            "objectFunction functionType is not MTLFunctionTypeObject.",
        ));
    }
    if not(desc.fragment, FunctionType::Fragment) {
        return Some(refusal(
            1,
            "fragmentFunction functionType is not MTLFunctionTypeFragment.",
        ));
    }
    None
}

/// The most bytes of payload Metal makes a mesh render pipeline state of
/// (`payloadMemoryLength`): past it the M3 Max (macOS 27.0.1), the AMD 5300M and the Intel 630
/// (macOS 26.7.1) answer nil and an `NSError`, of 16385 bytes (`oracle_mesh make`'s
/// `payload-big`), and make one of 256.
const METAL_PAYLOAD_BYTES: u64 = 16384;

/// The `NSError` of a mesh render pipeline whose payload length, `length`, is past
/// [`METAL_PAYLOAD_BYTES`]: the M3 Max's code ([`METAL_REFUSES`]) and text, in the runtime's
/// domain, as the compile's refusals are ([`pipeline_error`]). The AMD and the Intel 630 answer
/// code 2 and text of their own; core reads only the description.
fn payload_refused(length: u64) -> NsErrorText {
    NsErrorText {
        domain: crate::answer::RUNTIME_DOMAIN.to_owned(),
        code: METAL_REFUSES,
        description: format!(
            "Object shader payload size ({length}) exceeds the maximum payload size allowed \
             ({METAL_PAYLOAD_BYTES})"
        ),
    }
}

/// Metal's most threads of an object or mesh threadgroup: what
/// `maxTotalThreadsPerObjectThreadgroup` and `maxTotalThreadsPerMeshThreadgroup` answer for a
/// descriptor that sets no maximum, on the M3 Max, the AMD 5300M and the Intel 630
/// (`oracle_mesh make`'s `object`). A threadgroup past
/// it draws nothing on plain Metal, and the runtime's (`render/mesh.rs`).
const METAL_THREADGROUP_THREADS: u64 = 1024;

/// What `maxTotalThreadgroupsPerMeshGrid` answers on the M3 Max (macOS 27.0.1) for a descriptor
/// that sets no maximum, or sets this (`oracle_mesh make`'s `plain` and `grid-big`): one past the
/// mesh grid an object function may launch there, and on the runtime's devices, which launch none
/// past `metal2vulkan::bind::METAL_MESH_GRID_TOTAL` threadgroups (Metal's, measured on the M3 Max).
const METAL_MESH_GRID: u64 = 1 << 20;

/// The most threads this device runs an object (`object`) or a mesh threadgroup of: Metal's
/// ([`METAL_THREADGROUP_THREADS`]), within the device's own limit for the stage, which the runtime
/// refuses a draw past (`Unsupported{MeshShading}` naming it): `maxTaskWorkGroupInvocations` or
/// `maxMeshWorkGroupInvocations` where the device runs the stages natively, else
/// `maxComputeWorkGroupInvocations`, as the runtime runs both stages as compute work where it
/// emulates them (`DeviceProfile::emulates_mesh_shading`).
pub(crate) fn stage_threads(device: &VulkanDevice, object: bool) -> u64 {
    let profile = device.raw.compile_device().profile();
    let limits = profile.limits();
    let native = !profile.emulates_mesh_shading();
    let limit = match (native, object) {
        (true, true) => limits.mesh.max_task_work_group_invocations,
        (true, false) => limits.mesh.max_mesh_work_group_invocations,
        (false, _) => limits.max_compute_work_group_invocations,
    };
    u64::from(limit).min(METAL_THREADGROUP_THREADS)
}

/// What a mesh render pipeline state of `desc` answers on `device`, and holds its draws to
/// ([`MeshAnswers`]), of the values the runtime's descriptor does not hold. The answers describe
/// the device, so they are this device's, where Metal's three measured GPUs differ (`oracle_mesh
/// make`: the M3 Max on macOS 27.0.1, the AMD 5300M and the Intel 630 on macOS 26.7.1):
/// - `maxTotalThreadsPerObjectThreadgroup` and `maxTotalThreadsPerMeshThreadgroup`: the
///   descriptor's maximum where it sets one, else Metal's 1024, within the most this device runs
///   a threadgroup of the stage with ([`stage_threads`]); the object stage's 0 without an object
///   function. The M3 and the AMD answer 1024 for a stage whose maximum is not set and the
///   descriptor's where it is (64, 32 and 100 measured), and the AMD 0 for the object stage
///   without an object function, where the M3 answers 3; the Intel 630 answers 1024 whatever is
///   set, and 0 without an object function. A maximum past 1024 is made, as the Intel 630 makes
///   it, answering 1024, where the M3 and the AMD answer nil and an `NSError`: they differ, and
///   a draw within the state's answer draws as Metal's either way.
/// - `objectThreadExecutionWidth` and `meshThreadExecutionWidth`: 32, the width of the
///   simdgroups the runtime gives object and mesh functions, as it gives a kernel
///   (`metal_properties().thread_execution_width`), which all three answer; the object stage's 0
///   without an object function, as the AMD and the Intel 630 answer (the M3 answers 32).
/// - `maxTotalThreadgroupsPerMeshGrid`: the descriptor's maximum where it sets one, else
///   [`METAL_MESH_GRID`], as the M3 answers (8 where it set 8); the AMD and the Intel 630 answer
///   `NSUIntegerMax` whatever is set. The runtime holds no maximum of the descriptor's: an object
///   function's grid past it, which Metal's answer is unmeasured for, launches as one within it.
///
/// The descriptor's maxima and required sizes are also what its draws are held to
/// ([`MeshAnswers::held`]).
fn mesh_answers(
    device: &VulkanDevice,
    desc: &FullMeshRenderPipelineDesc<'_, VulkanPlatform>,
) -> MeshAnswers {
    let object = desc.object.is_some();
    let threads = |max: u64, of_object| {
        let limit = stage_threads(device, of_object);
        if max == 0 { limit } else { max.min(limit) }
    };
    let width = u64::from(plain(|| {
        device
            .raw
            .compile_device()
            .metal_properties()
            .thread_execution_width
    }));
    let required = |s: MtlSize| (s != MtlSize::new(0, 0, 0)).then_some(s);
    MeshAnswers {
        object,
        object_threads: match object {
            true => threads(desc.max_total_threads_per_object_threadgroup, true),
            false => 0,
        },
        mesh_threads: threads(desc.max_total_threads_per_mesh_threadgroup, false),
        object_width: if object { width } else { 0 },
        mesh_width: width,
        grid: match desc.max_total_threadgroups_per_mesh_grid {
            0 => METAL_MESH_GRID,
            max => max,
        },
        held: [
            MeshStageHeld {
                max: desc.max_total_threads_per_object_threadgroup,
                required: required(desc.required_threads_per_object_threadgroup),
            },
            MeshStageHeld {
                max: desc.max_total_threads_per_mesh_threadgroup,
                required: required(desc.required_threads_per_mesh_threadgroup),
            },
        ],
    }
}

/// The runtime's linked functions of a whole value's: each function as a stage links it, with its
/// constant values, under its specialized name where it was made with one, and precompiled where
/// it was made compiled to a binary.
fn linked_functions(l: &LinkedFunctionsDesc<VulkanPlatform>) -> LinkedFunctions {
    let list = |fs: &[VulkanFunction]| fs.iter().map(|f| f.linked_function()).collect();
    let mut out = LinkedFunctions::new();
    out.functions = list(&l.functions);
    out.binary_functions = list(&l.binary_functions);
    out.groups = l
        .groups
        .iter()
        .map(|(name, fs)| (name.clone(), list(fs)))
        .collect();
    out.private_functions = list(&l.private_functions);
    out
}

/// The private state `p` (the graphics wire's, as Apple's OpenGL renderer sets it) into the
/// runtime's `private`: OpenGL mode, the sample mask and coverage, the coverage inversion, the
/// logic operation and its switch, and the depth-stencil write switch, each the guest's raw
/// value. OpenGL mode, which the renderer sets on every pipeline, changes nothing Metal draws or
/// reflects, and makes two states Metal otherwise refuses: a fragment function writing depth
/// without a depth attachment, and a rasterizing state whose vertex function returns nothing
/// (metal2vulkan's measurement on the M3 Max, macOS 27.0.1; the runtime answers both).
fn private_state(private: &mut metal2vulkan::pipeline::PrivateRenderState, p: &PrivateRenderDesc) {
    private.open_gl_mode_enabled = p.open_gl_mode_enabled;
    private.sample_mask = p.sample_mask;
    private.sample_coverage = p.sample_coverage;
    private.sample_coverage_invert = p.sample_coverage_invert;
    private.logic_operation_enabled = p.logic_operation_enabled;
    private.logic_operation = p.logic_operation;
    private.depth_stencil_write_disabled = p.depth_stencil_write_disabled;
}

/// The runtime's colour attachment of a whole value's: none for `MTLPixelFormatInvalid`.
fn full_color_attachment(
    sel: &'static str,
    c: &ColorAttachmentDesc,
) -> Option<ColorAttachmentDescriptor> {
    let mut a = color_attachment(c.pixel_format)?;
    a.blending_enabled = c.blending_enabled;
    a.source_rgb_blend_factor = metal_value(sel, "blend factor", c.source_rgb_blend_factor);
    a.destination_rgb_blend_factor =
        metal_value(sel, "blend factor", c.destination_rgb_blend_factor);
    a.rgb_blend_operation = metal_value(sel, "blend operation", c.rgb_blend_operation);
    a.source_alpha_blend_factor = metal_value(sel, "blend factor", c.source_alpha_blend_factor);
    a.destination_alpha_blend_factor =
        metal_value(sel, "blend factor", c.destination_alpha_blend_factor);
    a.alpha_blend_operation = metal_value(sel, "blend operation", c.alpha_blend_operation);
    a.write_mask = metal_value(sel, "colour write mask", c.write_mask);
    Some(a)
}

/// A depth or stencil attachment's format: none for `MTLPixelFormatInvalid`.
fn optional_format(sel: &'static str, raw: u64) -> Option<metal2vulkan::metal::PixelFormat> {
    (raw != 0).then(|| crate::gpu::pixel_format(sel, raw))
}

/// A count the runtime holds as a `u32`.
fn count(sel: &'static str, what: &str, raw: u64) -> u32 {
    u32::try_from(raw).unwrap_or_else(|_| unrepresentable(sel, &format!("{what} {raw}")))
}

/// `MTLBufferLayoutStrideDynamic` (`NSUIntegerMax`), a layout's stride the runtime holds as
/// `None`.
const DYNAMIC_STRIDE: u64 = u64::MAX;

/// A layout's stride: `None` for `MTLBufferLayoutStrideDynamic`.
fn layout_stride(sel: &'static str, raw: u64) -> Option<u32> {
    (raw != DYNAMIC_STRIDE).then(|| count(sel, "buffer layout stride", raw))
}

/// The runtime's vertex descriptor of a whole value's: each layout and attribute at its index,
/// and a layout, as Metal's array holds one, for each buffer an attribute reads that the value
/// sets none of.
fn vertex_descriptor(sel: &'static str, v: &VertexDesc) -> VertexDescriptor {
    let mut out = VertexDescriptor::new();
    for l in &v.layouts {
        let mut layout = VertexBufferLayoutDescriptor::new(layout_stride(sel, l.stride));
        layout.step_function = metal_value(sel, "vertex step function", l.step_function);
        layout.step_rate = count(sel, "vertex step rate", l.step_rate);
        out.layouts
            .insert(count(sel, "vertex buffer index", l.buffer_index), layout);
    }
    for a in &v.attributes {
        let attribute = VertexAttributeDescriptor::new(
            metal_value(sel, "vertex format", a.format),
            count(sel, "vertex attribute offset", a.offset),
            count(sel, "vertex buffer index", a.buffer_index),
        );
        out.attributes.insert(
            count(sel, "vertex attribute index", a.attribute_index),
            attribute,
        );
    }
    for buffer in out.attributes.values().map(|a| a.buffer_index) {
        out.layouts
            .entry(buffer)
            .or_insert_with(|| VertexBufferLayoutDescriptor::new(Some(0)));
    }
    out
}

/// The runtime's stage-input descriptor of a whole value's, as [`vertex_descriptor`] makes a
/// vertex descriptor's, with its index type and index buffer.
pub(crate) fn stage_input_descriptor(
    sel: &'static str,
    s: &StageInputDesc,
) -> StageInputDescriptor {
    let mut out = StageInputDescriptor::new();
    for l in &s.layouts {
        let mut layout = StageInputLayoutDescriptor::new(layout_stride(sel, l.stride));
        layout.step_function = metal_value(sel, "step function", l.step_function);
        layout.step_rate = count(sel, "step rate", l.step_rate);
        out.layouts.insert(
            count(sel, "stage-input buffer index", l.buffer_index),
            layout,
        );
    }
    for a in &s.attributes {
        let attribute = StageInputAttributeDescriptor::new(
            metal_value(sel, "attribute format", a.format),
            count(sel, "stage-input attribute offset", a.offset),
            count(sel, "stage-input buffer index", a.buffer_index),
        );
        out.attributes.insert(
            count(sel, "stage-input attribute index", a.attribute_index),
            attribute,
        );
    }
    for buffer in out.attributes.values().map(|a| a.buffer_index) {
        out.layouts
            .entry(buffer)
            .or_insert_with(|| StageInputLayoutDescriptor::new(Some(0)));
    }
    out.index_type = metal_value(sel, "index type", s.index_type);
    out.index_buffer_index = count(sel, "index buffer index", s.index_buffer_index);
    out
}

/// The runtime's compute descriptor of a whole value's, the graphics lane's, OpenCL mode
/// included: the runtime links a kernel of OpenCL mode as Metal does, behind the three buffers of
/// Metal's OpenCL dispatcher, its arguments at their indices plus three.
pub(crate) fn full_compute_descriptor(
    sel: &'static str,
    desc: &FullComputePipelineDesc<'_, VulkanPlatform>,
) -> ComputePipelineDescriptor {
    let mut d = ComputePipelineDescriptor::new(StageSource::Function(
        desc.function
            .function_descriptor()
            .with_linked(linked_functions(&desc.linked)),
    ));
    d.threadgroup_size_is_multiple_of_execution_width =
        desc.threadgroup_size_is_multiple_of_execution_width;
    d.max_total_threads_per_threadgroup =
        (desc.max_total_threads_per_threadgroup != 0).then(|| {
            count(
                sel,
                "maximum threads per threadgroup",
                desc.max_total_threads_per_threadgroup,
            )
        });
    d.max_call_stack_depth = count(sel, "maximum call stack depth", desc.max_call_stack_depth);
    d.support_indirect_command_buffers = desc.support_indirect_command_buffers;
    d.stage_input = desc
        .stage_input
        .as_ref()
        .map(|s| stage_input_descriptor(sel, s));
    d.open_cl_mode_enabled = desc.open_cl_mode;
    d
}

/// A pipeline creator's answer to core: the state, of the descriptor's input primitive topology
/// `class`, the `NSError` ([`pipeline_error`]), the runtime's refusal as the `NSError` for a gap
/// of its own or the device's, relayed by default ([`crate::answer::relay_gap`]), or a death at
/// the selector for the rest: an ID past Metal's limit, an unspecialized function constant, which
/// Metal aborts for, a gap in strict mode, an answer it cannot prove ([`unproven`]: a state that
/// supports indirect command buffers whose functions' use of a direct texture or sampler argument
/// the compile layer refused to decide among them), and its other refusals, as Metal raises at
/// none.
fn made(
    sel: &'static str,
    (class, descriptor): (PrimitiveTopologyClass, RenderDescriptor),
    made: Result<rt::RenderPipelineState, rt::Error>,
) -> RaiseResult<Result<Option<VulkanRenderPipeline>, NsErrorText>> {
    match made {
        Ok(state) => {
            let indirect = matches!(&descriptor,
                RenderDescriptor::Render(d) if d.support_indirect_command_buffers);
            let made = render_pipeline_of(state, class, Some(descriptor));
            if indirect {
                let id = plain(|| made.raw.gpu_resource_id()).0;
                plain(|| {
                    let mut states = INDIRECT_STATES.lock();
                    states.retain(|_, held| {
                        held.retain(|state| state.strong_count() > 0);
                        !held.is_empty()
                    });
                    states.entry(id).or_default().push(made.downgrade());
                });
            }
            Ok(Ok(Some(made)))
        }
        Err(e) => match pipeline_error(&e) {
            Some(error) => Ok(Err(error)),
            None => match &e {
                rt::Error::Unproven(unproven) => {
                    self::unproven(sel, unproven, &e, &formats(&descriptor))
                }
                _ if relay_gap(sel, &e, Some(&formats(&descriptor))) => Ok(Err(error_text(&e))),
                _ => decide_describing(sel, Policy::Dies, e, Some(&formats(&descriptor))),
            },
        },
    }
}

/// The death at `sel` of `unproven`, a call whose answer on Metal the runtime cannot prove (`e`,
/// [`rt::Error::Unproven`]): Metal may make what it asks for, or answer nil and an `NSError`, and
/// the runtime does neither. Its line names the call and the problem: for a state that supports
/// indirect command buffers, the stage of the function whose direct texture or sampler argument
/// only a rule the runtime keeps beyond Metal keeps, and the rule (`LinkedUseGap`); for any other
/// problem, the problem as the runtime states it. Then the runtime's error and the descriptor's
/// formats, `formats`.
fn unproven(sel: &'static str, unproven: &rt::Unproven, e: &rt::Error, formats: &str) -> ! {
    let what = match &unproven.problem {
        rt::Problem::IndirectCommandsDirectArgumentUnproven { stage, reason } => format!(
            "a state that supports indirect command buffers whose {} function uses a direct \
             texture or sampler argument only by a rule the runtime keeps beyond Metal \
             ({reason:?})",
            format!("{stage:?}").to_lowercase()
        ),
        problem => format!("{problem:?}"),
    };
    unrepresentable(
        sel,
        &format!(
            "{what}, whose answer on Metal (made, or nil and an NSError) is unmeasured, in {}: {}; \
             {formats}",
            unproven.call,
            crate::answer::refusal(e)
        ),
    )
}

/// What a refused pipeline's descriptor holds that the runtime's error does not say, for its
/// death's line ([`decide_describing`]): a render descriptor's formats ([`render_formats`]) or a
/// mesh descriptor's ([`mesh_formats`]).
pub(crate) fn formats(d: &RenderDescriptor) -> String {
    match d {
        RenderDescriptor::Render(d) => render_formats(d),
        RenderDescriptor::Mesh(m) => mesh_formats(&m.descriptor),
    }
}

/// A raw Metal value as a death's line names it: the number, then its enumerator.
fn named(raw: u32, name: &str) -> String {
    format!("{raw} ({name})")
}

/// The line of a descriptor's formats, `items`, or `none` for no items.
fn listed(items: Vec<String>, none: &str) -> String {
    let items = if items.is_empty() {
        none.to_owned()
    } else {
        items.join(", ")
    };
    format!("the descriptor's formats: {items}")
}

/// A render descriptor's formats, by index: its attachments' ([`attachment_formats`]), then its
/// vertex descriptor's attribute formats.
pub(crate) fn render_formats(d: &RenderPipelineDescriptor) -> String {
    let mut items = attachment_formats(&d.color_attachments, d.depth_format, d.stencil_format);
    match &d.vertex_descriptor {
        None => items.push("no vertex descriptor".to_owned()),
        Some(v) if v.attributes.is_empty() => items.push("no vertex attributes".to_owned()),
        Some(v) => items.extend(v.attributes.iter().map(|(i, a)| {
            format!(
                "vertex attribute {i} {}",
                named(a.format.as_raw(), a.format.name())
            )
        })),
    }
    listed(items, "")
}

/// A mesh render descriptor's formats, by index: its attachments' ([`attachment_formats`]).
pub(crate) fn mesh_formats(d: &MeshPipelineDescriptor) -> String {
    listed(
        attachment_formats(&d.color_attachments, d.depth_format, d.stencil_format),
        "",
    )
}

/// A drawing descriptor's attachments' formats, by index: each colour attachment's pixel
/// format, its blend state ([`blend_state`]) and write mask, then the depth and stencil formats.
fn attachment_formats(
    colors: &[Option<ColorAttachmentDescriptor>],
    depth: Option<metal2vulkan::metal::PixelFormat>,
    stencil: Option<metal2vulkan::metal::PixelFormat>,
) -> Vec<String> {
    let format = |f: metal2vulkan::metal::PixelFormat| named(f.as_raw(), f.name());
    let mut items: Vec<String> = (colors.iter().enumerate())
        .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
        .map(|(i, c)| {
            let (format, mask) = (format(c.pixel_format), c.write_mask.as_raw());
            let blend = blend_state(c);
            format!("colour {i} {format} {blend} write mask {mask:#x}")
        })
        .collect();
    if items.is_empty() {
        items.push("no colour attachments".to_owned());
    }
    items.push(depth.map_or("no depth".to_owned(), |f| format!("depth {}", format(f))));
    items.push(stencil.map_or("no stencil".to_owned(), |f| {
        format!("stencil {}", format(f))
    }));
    items
}

/// A colour attachment's blend state as a death's line names it: whether it blends, then its rgb
/// and alpha factors and operations, numeric and named, which it holds blending or not.
fn blend_state(c: &ColorAttachmentDescriptor) -> String {
    let factor = |f: metal2vulkan::metal::BlendFactor| named(f.as_raw(), f.name());
    let operation = |o: metal2vulkan::metal::BlendOperation| named(o.as_raw(), o.name());
    format!(
        "blending {} rgb source {} destination {} operation {} alpha source {} destination {} \
         operation {}",
        if c.blending_enabled { "on" } else { "off" },
        factor(c.source_rgb_blend_factor),
        factor(c.destination_rgb_blend_factor),
        operation(c.rgb_blend_operation),
        factor(c.source_alpha_blend_factor),
        factor(c.destination_alpha_blend_factor),
        operation(c.alpha_blend_operation),
    )
}

/// A compute descriptor's stage-in attribute formats, by index.
pub(crate) fn compute_formats(d: &ComputePipelineDescriptor) -> String {
    match &d.stage_input {
        None => listed(Vec::new(), "no stage input descriptor"),
        Some(s) => listed(
            (s.attributes.iter())
                .map(|(i, a)| {
                    format!(
                        "stage-in attribute {i} {}",
                        named(a.format.as_raw(), a.format.name())
                    )
                })
                .collect(),
            "no stage-in attributes",
        ),
    }
}

#[cfg(test)]
mod tests {
    use metal2vulkan::metal::PixelFormat;
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::{DepthStencilState, SamplerState};
    use reims_vgpu_core::platform::resource::{Library, RenderPipeline};
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::VulkanFunction;

    /// `MTLSamplerMinMagFilterLinear`.
    const LINEAR: u64 = 1;

    /// A sampler descriptor's debug line names every property of the record, raw, and `unset`
    /// for those the value does not set; a float prints as Rust prints it
    #[test]
    fn a_samplers_debug_line_is_its_whole_descriptor() {
        let d = SamplerDesc {
            min_filter: Some(2),
            mag_filter: Some(LINEAR),
            lod_average: Some(false),
            compare_function: Some(7),
            s_address_mode: Some(0),
            t_address_mode: Some(2),
            r_address_mode: Some(5),
            mip_filter: Some(0),
            max_anisotropy: Some(16),
            normalized_coordinates: Some(true),
            support_argument_buffers: Some(true),
            reduction_mode: Some(0),
            lod_min_clamp: Some(0.0),
            lod_max_clamp: Some(f32::MAX),
            border_color: Some(1),
            resource_index: None,
        };
        assert_eq!(
            Sampler(&d).to_string(),
            "newSamplerStateWithDescriptor: min filter 2, mag filter 1, mip filter 0, address \
             modes s 0 t 2 r 5, border colour 1, normalized coordinates true, lod min 0 max \
             340282350000000000000000000000000000000 average false, max anisotropy 16, compare \
             function 7, reduction mode 0, supportArgumentBuffers true, resource index unset"
        );
        assert_eq!(
            Sampler(&SamplerDesc::default()).to_string(),
            "newSamplerStateWithDescriptor: min filter unset, mag filter unset, mip filter unset, \
             address modes s unset t unset r unset, border colour unset, normalized coordinates \
             unset, lod min unset max unset average unset, max anisotropy unset, compare function \
             unset, reduction mode unset, supportArgumentBuffers unset, resource index unset"
        );
    }

    /// The suite's vertex function fed through a vertex descriptor (attribute 0 a `float2`,
    /// attribute 1 a normalized `uchar4`, buffer 0) and the fragment function that returns the
    /// colour it outputs.
    fn vertex_descriptor_functions() -> (VulkanFunction, VulkanFunction) {
        use crate::gpu::device::tests::fixture_function;
        (
            fixture_function(include_bytes!(
                "../../../../tests/shaders/render_vdesc.metallib"
            )),
            fixture_function(include_bytes!(
                "../../../../tests/shaders/render_color.metallib"
            )),
        )
    }

    /// The whole render value of `vertex` and `fragment`: colour 0 BGRA8Unorm, no vertex
    /// descriptor, Metal's defaults elsewhere.
    fn full_render<'a>(
        vertex: &'a VulkanFunction,
        fragment: Option<&'a VulkanFunction>,
    ) -> FullRenderPipelineDesc<'a, VulkanPlatform> {
        FullRenderPipelineDesc {
            label: Some("whole"),
            vertex,
            fragment,
            vertex_descriptor: None,
            colors: vec![(
                0,
                ColorAttachmentDesc {
                    pixel_format: 80,
                    ..ColorAttachmentDesc::default()
                },
            )],
            depth_pixel_format: 0,
            stencil_pixel_format: 0,
            raster_sample_count: 1,
            alpha_to_coverage: false,
            alpha_to_one: false,
            rasterization_enabled: true,
            input_primitive_topology: 0,
            max_vertex_amplification_count: 1,
            support_indirect_command_buffers: false,
            vertex_linked: LinkedFunctionsDesc::default(),
            fragment_linked: LinkedFunctionsDesc::default(),
            private_state: None,
        }
    }

    /// A vertex function that reads `[[stage_in]]` attributes makes a pipeline with the vertex
    /// descriptor the whole value carries, and with none is the compile's `NSError` for the
    /// attribute it lacks
    #[test]
    fn a_whole_render_value_carries_its_vertex_descriptor() {
        use reims_vgpu_core::platform::values::{AttributeDesc, BufferLayoutDesc};
        let (v, f) = vertex_descriptor_functions();
        let d = dev();
        let mut desc = full_render(&v, Some(&f));
        let e = d.new_render_pipeline_state_full(&desc).unwrap_err();
        assert!(
            e.description.contains("VertexAttribute { attribute: 0 }"),
            "{e:?}"
        );
        desc.vertex_descriptor = Some(VertexDesc {
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
        });
        let made = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
        assert_ne!(made.unique_identifier(), 0);
    }

    /// The runtime's descriptor of a whole render value holds every field it carries, each raw
    /// value the runtime's own: the colour attachments with their blend state and write masks,
    /// the depth and stencil formats, the sample count, the flags, the topology and the
    /// amplification count; a raw value the runtime has no enumerator of dies at the send
    #[test]
    fn a_whole_render_value_is_the_runtime_s_descriptor() {
        use metal2vulkan::metal::{
            BlendFactor, BlendOperation, ColorWriteMask, PrimitiveTopologyClass,
        };
        let (v, f) = vertex_descriptor_functions();
        let sel = "newRenderPipelineStateWithDescriptor:error:";
        let c = ColorAttachmentDesc {
            pixel_format: 115,
            blending_enabled: true,
            source_rgb_blend_factor: 4,
            destination_rgb_blend_factor: 5,
            rgb_blend_operation: 1,
            source_alpha_blend_factor: 2,
            destination_alpha_blend_factor: 7,
            alpha_blend_operation: 3,
            write_mask: 0x9,
        };
        let a = full_color_attachment(sel, &c).unwrap();
        assert_eq!(a.pixel_format, PixelFormat::RGBA16Float);
        assert!(a.blending_enabled);
        assert_eq!(
            (
                a.source_rgb_blend_factor,
                a.destination_rgb_blend_factor,
                a.source_alpha_blend_factor,
                a.destination_alpha_blend_factor
            ),
            (
                BlendFactor::try_from(4).unwrap(),
                BlendFactor::try_from(5).unwrap(),
                BlendFactor::try_from(2).unwrap(),
                BlendFactor::try_from(7).unwrap()
            )
        );
        assert_eq!(
            (a.rgb_blend_operation, a.alpha_blend_operation),
            (
                BlendOperation::try_from(1).unwrap(),
                BlendOperation::try_from(3).unwrap()
            )
        );
        assert_eq!(a.write_mask, ColorWriteMask::try_from(9).unwrap());
        assert_eq!(
            full_color_attachment(sel, &ColorAttachmentDesc::default()),
            None,
            "no format, no attachment"
        );
        let site = catch_death(|| {
            full_color_attachment(
                sel,
                &ColorAttachmentDesc {
                    source_rgb_blend_factor: 99,
                    ..c
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, sel);
        assert_eq!(optional_format(sel, 0), None);
        assert_eq!(optional_format(sel, 252), Some(PixelFormat::Depth32Float));
        let mut desc = full_render(&v, Some(&f));
        desc.input_primitive_topology = 99;
        let site = catch_death(|| d_new(&desc)).unwrap_err();
        assert_eq!(site.file, sel, "no topology class 99");
        assert_eq!(
            metal_value::<PrimitiveTopologyClass>(sel, "primitive topology class", 3),
            PrimitiveTopologyClass::Triangle
        );
    }

    /// [`GpuDevice::new_render_pipeline_state_full`] on the default device.
    fn d_new(
        desc: &FullRenderPipelineDesc<'_, VulkanPlatform>,
    ) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
        dev().new_render_pipeline_state_full(desc)
    }

    /// A kernel that reads `[[stage_in]]` attributes makes a compute pipeline with the
    /// stage-input descriptor the whole value carries, and dies at the send with none; the
    /// value's other fields are the runtime descriptor's own
    #[test]
    fn a_whole_compute_value_carries_its_stage_input() {
        use crate::gpu::device::tests::{STAGEIN, fixture_function};
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        use reims_vgpu_core::platform::values::{AttributeDesc, BufferLayoutDesc};
        let f = fixture_function(STAGEIN);
        let sel = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let mut desc = FullComputePipelineDesc {
            label: Some("stage in"),
            function: &f,
            threadgroup_size_is_multiple_of_execution_width: true,
            max_total_threads_per_threadgroup: 128,
            max_call_stack_depth: 3,
            support_indirect_command_buffers: false,
            stage_input: None,
            linked: LinkedFunctionsDesc::default(),
            open_cl_mode: false,
        };
        let site = catch_death(|| dev().new_compute_pipeline_state_full(&desc)).unwrap_err();
        assert_eq!(site.file, sel);
        desc.stage_input = Some(StageInputDesc {
            layouts: vec![BufferLayoutDesc {
                buffer_index: 1,
                stride: 4,
                step_function: 5,
                step_rate: 1,
            }],
            attributes: vec![AttributeDesc {
                attribute_index: 0,
                format: 36,
                offset: 0,
                buffer_index: 1,
            }],
            index_type: 1,
            index_buffer_index: 0,
        });
        let d = full_compute_descriptor(sel, &desc);
        assert!(!d.open_cl_mode_enabled);
        assert!(d.threadgroup_size_is_multiple_of_execution_width);
        assert_eq!(d.max_total_threads_per_threadgroup, Some(128));
        assert_eq!(d.max_call_stack_depth, 3);
        let s = d.stage_input.as_ref().unwrap();
        assert_eq!(s.layouts[&1].stride, Some(4));
        assert_eq!(s.attributes[&0].buffer_index, 1);
        let made = dev()
            .new_compute_pipeline_state_full(&desc)
            .unwrap()
            .unwrap();
        assert_eq!(made.max_total_threads_per_threadgroup(), 128);
    }

    /// `linked_caller.metal`'s kernel, vertex and fragment functions, and `linked_callee.metal`'s
    /// `linked_add_one`, which they call, made by name and, with `COMPILE_TO_BINARY`, by
    /// descriptor.
    fn linked_functions_of_fixtures() -> [VulkanFunction; 5] {
        use crate::gpu::resource::VulkanLibrary;
        let d = dev();
        let library = |bytes: &[u8]| d.new_library(bytes).unwrap().unwrap().unwrap();
        let caller = library(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linked_caller.metallib"
        )));
        let callee = library(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/linked_callee.metallib"
        )));
        let named =
            |l: &VulkanLibrary, name: &str| l.new_function(OnFault::Return, name).unwrap().unwrap();
        let binary = callee
            .new_function_with_options(OnFault::Return, "linked_add_one", &[], 1, None)
            .unwrap()
            .unwrap();
        [
            named(&caller, "linked_kernel"),
            named(&caller, "linked_vertex"),
            named(&caller, "linked_fragment"),
            named(&callee, "linked_add_one"),
            binary,
        ]
    }

    /// A whole compute value of OpenCL mode is the runtime's descriptor of OpenCL mode, and the
    /// device makes such pipelines
    #[test]
    fn a_whole_compute_value_carries_its_opencl_mode() {
        use crate::gpu::device::tests::fixture_function;
        let f = fixture_function(include_bytes!("../../../../tests/shaders/kernel.metallib"));
        let sel = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let desc = |open_cl_mode| FullComputePipelineDesc {
            label: None,
            function: &f,
            threadgroup_size_is_multiple_of_execution_width: false,
            max_total_threads_per_threadgroup: 0,
            max_call_stack_depth: 1,
            support_indirect_command_buffers: false,
            stage_input: None,
            linked: LinkedFunctionsDesc::default(),
            open_cl_mode,
        };
        assert!(full_compute_descriptor(sel, &desc(true)).open_cl_mode_enabled);
        assert!(!full_compute_descriptor(sel, &desc(false)).open_cl_mode_enabled);
        assert!(dev().makes_opencl_mode_pipelines());
    }

    /// A kernel that calls a visible function its library does not define is made with the
    /// function linked by functions, private functions, or a group with it in private functions,
    /// and is refused with Metal's code with none, with a group alone, and with binary functions
    /// alone holding it, made compiled to a binary or not, as a direct call resolves by no binary
    /// function (`metal_tests::unresolved_visible_calls_are_metals`). The value carries a binary
    /// function as one made compiled to a binary
    #[test]
    fn a_whole_compute_value_carries_its_linked_functions() {
        let [kernel, _, _, add_one, binary] = linked_functions_of_fixtures();
        let sel = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let desc = |linked| FullComputePipelineDesc {
            label: None,
            function: &kernel,
            threadgroup_size_is_multiple_of_execution_width: false,
            max_total_threads_per_threadgroup: 0,
            max_call_stack_depth: 1,
            support_indirect_command_buffers: false,
            stage_input: None,
            linked,
            open_cl_mode: false,
        };
        let one = |f: &VulkanFunction| vec![f.clone_ref()];
        let linked = |functions, binary_functions, groups, private_functions| LinkedFunctionsDesc {
            functions,
            binary_functions,
            groups,
            private_functions,
        };
        let made = |l: LinkedFunctionsDesc<VulkanPlatform>| {
            catch_death(|| dev().new_compute_pipeline_state_full(&desc(l)))
        };
        for l in [
            linked(one(&add_one), vec![], vec![], vec![]),
            linked(vec![], vec![], vec![], one(&add_one)),
            linked(
                vec![],
                vec![],
                vec![("g".to_owned(), one(&add_one))],
                one(&add_one),
            ),
        ] {
            let shown = format!("{l:?}");
            let state = made(l);
            assert!(matches!(state, Ok(Ok(Some(_)))), "{shown}: {state:?}");
        }
        let d = full_compute_descriptor(
            sel,
            &desc(linked(one(&add_one), one(&binary), vec![], vec![])),
        );
        let StageSource::Function(f) = &d.kernel else {
            panic!("{:?}", d.kernel);
        };
        assert_eq!(f.linked.functions.len(), 1);
        assert!(!f.linked.functions[0].precompiled);
        assert!(f.linked.binary_functions[0].precompiled);
        for refused in [
            LinkedFunctionsDesc::default(),
            linked(
                vec![],
                vec![],
                vec![("g".to_owned(), one(&add_one))],
                vec![],
            ),
            linked(vec![], one(&binary), vec![], vec![]),
            linked(vec![], one(&add_one), vec![], vec![]),
        ] {
            let shown = format!("{refused:?}");
            let state = made(refused);
            assert!(
                matches!(&state, Ok(Err(e)) if e.code == METAL_REFUSES),
                "{shown}: {state:?}"
            );
        }
    }

    /// `linked_caller.metal`'s and `linked_callee.metal`'s metallibs, compiled by build.rs.
    pub(super) const LINKED_CALLER: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/linked_caller.metallib"));
    pub(super) const LINKED_CALLEE: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/linked_callee.metallib"));

    /// The float constant of `index` whose value is `v`.
    fn float_constant(index: u32, v: f32) -> reims_vgpu_core::platform::resource::FunctionConstant {
        let mut value = [0; reims_vgpu_core::platform::resource::FunctionConstant::MAX_BYTES];
        value[..4].copy_from_slice(&v.to_le_bytes());
        reims_vgpu_core::platform::resource::FunctionConstant {
            index,
            data_type: 3,
            value,
        }
    }

    /// `linked_callee.metal`'s `linked_gradient` of `callee` with `linked_gradient_scale` 0.5,
    /// made by descriptor under `specialized` where given: the function RenderBox links under
    /// `custom_fn`. `None` where the library makes none.
    pub(super) fn gradient<P: reims_vgpu_core::platform::Platform>(
        callee: &P::Library,
        specialized: Option<&str>,
    ) -> Option<P::Function> {
        callee
            .new_function_with_options(
                OnFault::Return,
                "linked_gradient",
                &[float_constant(1, 0.5)],
                0,
                specialized,
            )
            .unwrap()
    }

    /// What `linked_caller.metal`'s `linked_effect_fragment` of `caller`, with
    /// `linked_effect_alpha` 0.75 and `linked` linked to its stage (RenderBox's custom effect),
    /// writes for each pixel of a 4x4 target on `p`'s device `d`, by its bits: `None` where the
    /// pipeline is refused.
    pub(super) fn custom_effect<P: reims_vgpu_core::platform::Platform>(
        p: &P,
        d: &P::Device,
        caller: &P::Library,
        linked: &P::Function,
    ) -> Option<Vec<[u32; 4]>> {
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::types::NsRange;
        use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};
        let _pool = reims_vgpu_core::rt::Pool::here();
        let vertex = caller
            .new_function(OnFault::Die, "linked_effect_vertex")
            .unwrap()
            .unwrap();
        let fragment = caller
            .new_function_with_options(
                OnFault::Return,
                "linked_effect_fragment",
                &[float_constant(0, 0.75)],
                0,
                None,
            )
            .unwrap()
            .unwrap();
        let state = d
            .new_render_pipeline_state_full(&FullRenderPipelineDesc {
                label: Some("custom effect"),
                vertex: &vertex,
                fragment: Some(&fragment),
                vertex_descriptor: None,
                colors: vec![(
                    0,
                    ColorAttachmentDesc {
                        pixel_format: 80,
                        ..ColorAttachmentDesc::default()
                    },
                )],
                depth_pixel_format: 0,
                stencil_pixel_format: 0,
                raster_sample_count: 1,
                alpha_to_coverage: false,
                alpha_to_one: false,
                rasterization_enabled: true,
                input_primitive_topology: 0,
                max_vertex_amplification_count: 1,
                support_indirect_command_buffers: false,
                vertex_linked: LinkedFunctionsDesc::default(),
                fragment_linked: LinkedFunctionsDesc {
                    functions: vec![linked.clone_ref()],
                    ..LinkedFunctionsDesc::default()
                },
                private_state: None,
            })
            .ok()??;
        let out = d
            .new_buffer_of_length(OnFault::Die, 16 * 16, 0)
            .unwrap()
            .unwrap();
        let target = d
            .new_texture(
                OnFault::Die,
                &p.texture_descriptor(&TextureDesc {
                    texture_type: 2,
                    pixel_format: 80,
                    width: 4,
                    height: 4,
                    usage: 4,
                    resource_options: Some(0x20),
                    ..TextureDesc::default()
                }),
            )
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&target)),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let buffers = [Some(&out)];
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        // SAFETY: a Shared buffer of 16 float4s, read after the command buffer completed.
        let floats = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<f32>(), 64) };
        Some(
            floats
                .chunks(4)
                .map(|c| [0, 1, 2, 3].map(|i| c[i].to_bits()))
                .collect(),
        )
    }

    /// RenderBox's custom effect: a function made with a specialized name answers it as its
    /// name, and a fragment function that calls a visible function of that name links it, its
    /// own constants and the linked function's each read; each pixel reads the linked
    /// `linked_gradient` at its centre with `linked_gradient_scale` 0.5, and alpha
    /// `linked_effect_alpha` 0.75. The same function made under its own name, or under another,
    /// leaves the call unresolved, and the pipeline refused
    #[test]
    fn a_function_with_a_specialized_name_links_under_it() {
        use crate::gpu::resource::VulkanLibrary;
        let (p, d) = (crate::platform::tests::platform(), dev());
        let library = |bytes| -> VulkanLibrary { d.new_library(bytes).unwrap().unwrap().unwrap() };
        let (caller, callee) = (library(LINKED_CALLER), library(LINKED_CALLEE));
        let custom = gradient::<VulkanPlatform>(&callee, Some("custom_fn")).unwrap();
        assert_eq!(custom.name(), "custom_fn");
        let drawn = custom_effect(&p, &d, &caller, &custom).expect("the pipeline was refused");
        let want: Vec<[u32; 4]> = (0u8..16)
            .map(|i| {
                let (x, y) = (f32::from(i % 4) + 0.5, f32::from(i / 4) + 0.5);
                [x * 0.5, y * 0.25, 0.5, 0.75].map(f32::to_bits)
            })
            .collect();
        assert_eq!(drawn, want);
        for specialized in [None, Some("other_fn")] {
            let linked = gradient::<VulkanPlatform>(&callee, specialized).unwrap();
            let name = specialized.unwrap_or("linked_gradient");
            assert_eq!(linked.name(), name);
            let refused = catch_death(|| custom_effect(&p, &d, &caller, &linked));
            assert!(matches!(refused, Ok(None)), "{name}: {refused:?}");
        }
    }

    /// A vertex function that calls a visible function its library does not define is made with
    /// the function linked to the vertex stage, and refused with Metal's code with it linked to
    /// the fragment stage
    #[test]
    fn a_whole_render_value_carries_its_stages_linked_functions() {
        let [_, vertex, fragment, add_one, _] = linked_functions_of_fixtures();
        let made = |vertex_linked, fragment_linked| {
            let mut desc = full_render(&vertex, Some(&fragment));
            desc.vertex_linked = vertex_linked;
            desc.fragment_linked = fragment_linked;
            catch_death(|| dev().new_render_pipeline_state_full(&desc))
        };
        let functions = |f: &VulkanFunction| LinkedFunctionsDesc {
            functions: vec![f.clone_ref()],
            ..LinkedFunctionsDesc::default()
        };
        let state = made(functions(&add_one), LinkedFunctionsDesc::default());
        assert!(matches!(state, Ok(Ok(Some(_)))), "{state:?}");
        let state = made(LinkedFunctionsDesc::default(), functions(&add_one));
        assert!(
            matches!(&state, Ok(Err(e)) if e.code == METAL_REFUSES),
            "{state:?}"
        );
    }

    /// The suite's vertex and fragment functions, which use no texture or sampler: a state that
    /// supports indirect command buffers takes them, where Metal and the runtime
    /// refuse one of the display's, whose fragment function samples its texture.
    fn indirect_functions() -> (VulkanFunction, VulkanFunction) {
        use crate::gpu::device::tests::fixture_function;
        (
            fixture_function(include_bytes!("../../../../tests/shaders/vert.metallib")),
            fixture_function(include_bytes!("../../../../tests/shaders/frag.metallib")),
        )
    }

    /// A state of indirect command buffer support sizes its points, as an indirect command that
    /// names its runtime state asks, where it is of the point input primitive topology class
    /// and not otherwise; once the program released it, and for a state made without that
    /// support, nothing answers
    #[test]
    fn indirect_states_answer_their_classes_while_they_live() {
        let d = dev();
        let functions = indirect_functions();
        // `MTLPrimitiveTopologyClassPoint` and `Triangle`.
        let made = |class, indirect| {
            d.new_render_pipeline_state_full(&FullRenderPipelineDesc {
                colors: crate::gpu::device::tests::colours_of(&[&[(1, 80)]]),
                input_primitive_topology: class,
                support_indirect_command_buffers: indirect,
                ..FullRenderPipelineDesc::new(&functions.0, Some(&functions.1))
            })
            .unwrap()
            .unwrap()
        };
        let points = made(1, true);
        let triangles = made(3, true);
        let direct = made(1, false);
        assert_eq!(points_sized(&points.raw), Some(true));
        assert_eq!(points_sized(&triangles.raw), Some(false));
        assert_eq!(points_sized(&direct.raw), None);
        let raw = triangles.raw.clone();
        drop(triangles);
        assert_eq!(points_sized(&raw), None);
        assert_eq!(points_sized(&points.raw), Some(true));
    }

    /// Equal requests answer one object while it lives, as Metal's do; another request another
    #[test]
    fn equal_sampler_requests_are_one_object() {
        let d = dev();
        let linear = SamplerDesc {
            mag_filter: Some(LINEAR),
            ..SamplerDesc::default()
        };
        let a = d.new_sampler_state(&linear).unwrap().unwrap();
        let b = d.new_sampler_state(&linear).unwrap().unwrap();
        assert_eq!(a.id(), b.id());
        assert_eq!(a.gpu_resource_id(), b.gpu_resource_id());
        let nearest = d
            .new_sampler_state(&SamplerDesc::default())
            .unwrap()
            .unwrap();
        assert_ne!(nearest.id(), a.id());
        assert_ne!(nearest.gpu_resource_id(), a.gpu_resource_id());
    }

    /// Among many live samplers, and past a sweep of the handles of ones gone, each equal request
    /// still answers the one object while it lives
    #[test]
    fn equal_sampler_requests_are_one_object_among_many() {
        let d = dev();
        let desc = |i: u16| SamplerDesc {
            lod_max_clamp: Some(1000.0 - f32::from(i)),
            ..SamplerDesc::default()
        };
        let held: Vec<_> = (0..200)
            .map(|i| d.new_sampler_state(&desc(i)).unwrap().unwrap())
            .collect();
        for i in 200..400 {
            drop(d.new_sampler_state(&desc(i)).unwrap().unwrap());
        }
        for (i, h) in (0..).zip(&held) {
            let again = d.new_sampler_state(&desc(i)).unwrap().unwrap();
            assert_eq!(again.id(), h.id(), "{i}");
            assert_eq!(again.gpu_resource_id(), h.gpu_resource_id(), "{i}");
        }
    }

    /// A value's resource index places the sampler at that ID, and its resource index is its ID
    #[test]
    fn a_sampler_takes_the_id_of_its_resource_index() {
        let d = dev();
        let s = d
            .new_sampler_state(&SamplerDesc {
                resource_index: Some(7),
                ..SamplerDesc::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!((s.gpu_resource_id(), s.resource_index()), (7, 7));
        let any = d
            .new_sampler_state(&SamplerDesc {
                min_filter: Some(LINEAR),
                ..SamplerDesc::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(any.resource_index(), any.gpu_resource_id());
        assert_ne!(any.gpu_resource_id(), 7);
    }

    /// Descriptors Metal's validation stops, of which Metal makes a state, make one: an anisotropy
    /// of 0, a minimum LOD clamp past the maximum, and a border colour no enumerator is. One of
    /// an anisotropy of 17, which Metal samples with 32 and the device with its largest, 16,
    /// dies at its creation (`capped-anisotropy`, [`crate::answer::DIES`]); any other raw value
    /// no enumerator is dies at its setter
    #[test]
    fn sampler_values_metal_takes_and_bad_values() {
        let d = dev();
        for desc in [
            SamplerDesc {
                max_anisotropy: Some(0),
                ..SamplerDesc::default()
            },
            SamplerDesc {
                lod_min_clamp: Some(4.0),
                lod_max_clamp: Some(2.0),
                ..SamplerDesc::default()
            },
            SamplerDesc {
                border_color: Some(3),
                ..SamplerDesc::default()
            },
            SamplerDesc {
                border_color: Some(7),
                ..SamplerDesc::default()
            },
        ] {
            assert!(d.new_sampler_state(&desc).unwrap().is_some(), "{desc:?}");
        }
        let (sel, line) = death_line(|| {
            d.new_sampler_state(&SamplerDesc {
                max_anisotropy: Some(17),
                ..SamplerDesc::default()
            })
        });
        assert_eq!(sel, "newSamplerStateWithDescriptor:", "{line}");
        assert!(line.contains("capped-anisotropy"), "{line}");
        let site = catch_death(|| {
            d.new_sampler_state(&SamplerDesc {
                min_filter: Some(9),
                ..SamplerDesc::default()
            })
        })
        .unwrap_err();
        assert_eq!(site.file, "setMinFilter:");
    }

    /// Runs `f`, which dies, and answers its site's selector and the line the death carried.
    fn death_line<T>(f: impl FnOnce() -> T) -> (&'static str, String) {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(f).err().expect("it dies");
        assert_eq!(site.kind, reims_vgpu_core::raise::death::DeathKind::Abort);
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        (site.file, line)
    }

    /// Any depth-stencil value is a state; below Apple10 its resource ID is plain Metal's, 0, and
    /// its resource index raises, as Metal's do on the M3 Max (the Metal platform's
    /// `depth_stencil_state_getters_raise`)
    #[test]
    fn depth_stencil_states_are_made_and_their_ids_raise() {
        let d = dev();
        let face = StencilDesc {
            stencil_compare_function: 3,
            stencil_failure_operation: 1,
            depth_failure_operation: 2,
            depth_stencil_pass_operation: 7,
            read_mask: 0xf0,
            write_mask: 0x0f,
        };
        let s = d
            .new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(1),
                depth_write_enabled: Some(true),
                front: Some(face),
                back: None,
            })
            .unwrap()
            .unwrap();
        assert!(matches!(s.gpu_resource_id(), Ok(0)));
        let raised = s.resource_index().unwrap_err();
        assert_eq!(raised.site().file, "resourceIndex");
        let site = catch_death(|| {
            d.new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(8),
                ..DepthStencilDesc::default()
            })
        })
        .unwrap_err();
        assert_eq!(site.file, "setDepthCompareFunction:");
    }

    /// The runtime's stop of a state whose attachment is in a pixel format the device's Metal
    /// describes none of (`AttachmentNotAFormat`: a colour attachment of 630 to 650, a
    /// depth or stencil attachment of `Depth24Unorm_Stencil8` or `X24_Stencil8`), where Metal's
    /// descriptor validation asserts, is no `NSError`: it dies at the creator. No creator here
    /// reaches it, as a descriptor holds such a format raw and its state dies first
    /// (`gpu::device`'s `raw_pipeline_formats_die_where_states_are_made`)
    #[test]
    fn an_attachment_of_no_format_is_no_nserror() {
        let format = |raw| crate::gpu::runtime_pixel_format(raw).unwrap();
        for (attachment, raw) in [
            (rt::Attachment::Color(0), 630),
            (rt::Attachment::Depth, 255),
            (rt::Attachment::Stencil, 262),
        ] {
            let problem = rt::Problem::AttachmentNotAFormat {
                attachment,
                format: format(raw),
            };
            let e = rt::Error::Invalid(rt::Invalid::new("new_render_pipeline_state", problem));
            assert!(pipeline_error(&e).is_none(), "{e:?}");
        }
    }

    /// A refusal the runtime cannot prove Metal's answer for (`Error::Unproven`) dies at the
    /// creator, its line naming the call, the stage and the rule for an indirect-command
    /// argument's, of every rule, and the problem as the runtime states it for any other; a
    /// `MetalError` of any problem is the `NSError` of its code
    #[test]
    fn unproven_answers_die_naming_the_stage_and_the_rule() {
        use metal2vulkan::pipeline::Stage;
        use metal2vulkan::reflection::LinkedUseGap as G;
        let sel = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
        let call = "new_render_pipeline_state";
        let descriptor = || {
            let d = RenderPipelineDescriptor::new(cpu_stage());
            (
                PrimitiveTopologyClass::Unspecified,
                RenderDescriptor::Render(Box::new(d)),
            )
        };
        let line = |problem: rt::Problem| {
            *crate::answer::LAST_LINE.lock() = None;
            let e = rt::Error::Unproven(rt::Unproven::new(call, problem));
            let site = catch_death(|| {
                let _ = made(sel, descriptor(), Err(e));
            })
            .unwrap_err();
            assert_eq!(site.file, sel);
            crate::answer::LAST_LINE.lock().take().unwrap_or_default()
        };
        let gaps = [
            G::EndlessLoop,
            G::StackMemory,
            G::GatedOffMeshBuiltin,
            G::UnmeasuredColorWrite,
        ];
        for (stage, name) in [(Stage::Vertex, "vertex"), (Stage::Mesh, "mesh")] {
            for reason in gaps {
                let problem = rt::Problem::IndirectCommandsDirectArgumentUnproven { stage, reason };
                let line = line(problem);
                let named = format!(
                    "whose {name} function uses a direct texture or sampler argument only by a \
                     rule the runtime keeps beyond Metal ({reason:?})"
                );
                assert!(line.contains(&named), "{line}");
                assert!(
                    line.contains(&format!("is unmeasured, in {call}")),
                    "{line}"
                );
            }
        }
        let line = line(rt::Problem::UnrasterizedVertexReturn);
        let named = "the Vulkan runtime cannot hold UnrasterizedVertexReturn, whose answer";
        assert!(line.starts_with(named), "{line}");
        let e = rt::Error::MetalError(rt::MetalError::new(
            call,
            rt::Problem::UnrasterizedVertexReturn,
            "domain",
            3,
            "text",
        ));
        let answer = made(sel, descriptor(), Err(e)).unwrap().unwrap_err();
        assert_eq!(answer.code, 3);
    }

    /// A stage of the platform's own vertex function, from its metallib on the CPU: the lines of
    /// a descriptor's formats read no stage.
    fn cpu_stage() -> StageSource {
        use metal2vulkan::library::Library;
        use metal2vulkan::pipeline::FunctionDescriptor;
        use reims_vgpu_core::display::pipeline::VERTEX_FUNCTION;
        let library = Library::from_metallib(crate::gpu::shaders::METALLIB).unwrap();
        StageSource::Function(FunctionDescriptor::new(
            library.function(VERTEX_FUNCTION).unwrap(),
        ))
    }

    /// A refused pipeline's death line names its descriptor's formats, numeric and named: a
    /// render descriptor's colour attachments with their blend states and write masks, its depth
    /// and stencil and its vertex attributes (the XR10 state 26.6.2's compositor makes), and a
    /// compute descriptor's stage-in attributes
    #[test]
    fn a_refused_pipeline_names_its_descriptors_formats() {
        use metal2vulkan::metal::{
            AttributeFormat, BlendFactor, BlendOperation, ColorWriteMask, VertexFormat,
        };
        use metal2vulkan::pipeline::{
            StageInputAttributeDescriptor, StageInputDescriptor, VertexAttributeDescriptor,
            VertexDescriptor,
        };
        use metal2vulkan_runtime::{Error, RuntimeGap};

        use crate::answer::refusal_describing;

        let attachment = |format, mask| {
            let mut c = ColorAttachmentDescriptor::new(format);
            c.write_mask = mask;
            c
        };
        let mut d = RenderPipelineDescriptor::new(cpu_stage());
        d.fragment = Some(cpu_stage());
        d.color_attachments[0] = Some(attachment(PixelFormat::RGB10A82PXR10, ColorWriteMask::ALL));
        {
            let c = d.color_attachments[0].as_mut().unwrap();
            c.blending_enabled = true;
            c.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            c.destination_alpha_blend_factor = BlendFactor::OneMinusSource1Alpha;
            c.alpha_blend_operation = BlendOperation::Max;
        }
        d.color_attachments[1] = Some(attachment(PixelFormat::RG16Float, ColorWriteMask::NONE));
        d.color_attachments[1].as_mut().unwrap().rgb_blend_operation = BlendOperation::Subtract;
        d.color_attachments[2] = Some(attachment(PixelFormat::RGB10A82PXR10, ColorWriteMask::NONE));
        d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        let xr10 = "550 (MTLPixelFormatRGB10A8_2P_XR10)";
        let packed = "260 (MTLPixelFormatDepth32Float_Stencil8)";
        let (one, zero, add) = (
            "1 (MTLBlendFactorOne)",
            "0 (MTLBlendFactorZero)",
            "0 (MTLBlendOperationAdd)",
        );
        let unblended = format!(
            "blending off rgb source {one} destination {zero} operation {add} alpha source {one} \
             destination {zero} operation {add}"
        );
        assert_eq!(
            refusal_describing(
                &Error::NotImplemented(RuntimeGap::Textures),
                Some(&formats(&RenderDescriptor::Render(Box::new(d.clone()))))
            ),
            format!(
                "{}; the descriptor's formats: colour 0 {xr10} blending on rgb source 4 \
                 (MTLBlendFactorSourceAlpha) destination 5 (MTLBlendFactorOneMinusSourceAlpha) \
                 operation {add} alpha source {one} destination 18 \
                 (MTLBlendFactorOneMinusSource1Alpha) operation 4 (MTLBlendOperationMax) write \
                 mask 0xf, colour 1 65 (MTLPixelFormatRG16Float) blending off rgb source {one} \
                 destination {zero} operation 1 (MTLBlendOperationSubtract) alpha source {one} \
                 destination {zero} operation {add} write mask 0x0, colour 2 {xr10} {unblended} \
                 write mask 0x0, depth {packed}, stencil {packed}, no vertex descriptor",
                crate::answer::refusal(&Error::NotImplemented(RuntimeGap::Textures)),
            )
        );
        let mut v = VertexDescriptor::new();
        v.attributes.insert(
            3,
            VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
        );
        let mut d = RenderPipelineDescriptor::new(cpu_stage());
        d.vertex_descriptor = Some(v);
        assert_eq!(
            render_formats(&d),
            "the descriptor's formats: no colour attachments, no depth, no stencil, vertex \
             attribute 3 29 (MTLVertexFormatFloat2)"
        );
        d.vertex_descriptor = Some(VertexDescriptor::new());
        assert!(render_formats(&d).ends_with(", no vertex attributes"));

        let mut c = ComputePipelineDescriptor::new(cpu_stage());
        assert_eq!(
            compute_formats(&c),
            "the descriptor's formats: no stage input descriptor"
        );
        let mut s = StageInputDescriptor::new();
        s.attributes.insert(
            0,
            StageInputAttributeDescriptor::new(AttributeFormat::Float2, 0, 1),
        );
        c.stage_input = Some(s);
        assert_eq!(
            compute_formats(&c),
            "the descriptor's formats: stage-in attribute 0 29 (MTLAttributeFormatFloat2)"
        );
        assert_eq!(
            refusal_describing(&Error::DeviceLost, None),
            crate::answer::refusal(&Error::DeviceLost)
        );
    }

    /// The graphics wire's private render state fills the runtime's, each field the guest's
    /// value; the rest stay Metal's defaults
    #[test]
    fn the_private_render_state_fills_the_runtimes() {
        let mut private = metal2vulkan::pipeline::PrivateRenderState::new();
        private_state(
            &mut private,
            &PrivateRenderDesc {
                open_gl_mode_enabled: true,
                sample_mask: 3,
                sample_coverage: 0.5,
                sample_coverage_invert: true,
                logic_operation_enabled: true,
                logic_operation: 10,
                depth_stencil_write_disabled: true,
            },
        );
        let mut want = metal2vulkan::pipeline::PrivateRenderState::new();
        want.open_gl_mode_enabled = true;
        want.sample_mask = 3;
        want.sample_coverage = 0.5;
        want.sample_coverage_invert = true;
        want.logic_operation_enabled = true;
        want.logic_operation = 10;
        want.depth_stencil_write_disabled = true;
        assert_eq!(private, want);
    }

    /// A rasterizing render state whose vertex function returns nothing is Metal's
    /// (metal2vulkan's measurement on the M3 Max, macOS 27.0.1, and the Metal platform's
    /// `opengl_mode_makes_a_rasterizing_void_vertex_state`): the `NSError` of code 3
    /// (`RasterizedVoidVertex`) outside OpenGL mode, with or without a fragment function, and
    /// made in it
    #[test]
    fn a_rasterizing_void_vertex_state_is_made_in_opengl_mode_alone() {
        use crate::gpu::device::tests::fixture_function;
        let d = dev();
        let vertex = fixture_function(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/unrasterized.metallib"
        )));
        let (_, fragment) = indirect_functions();
        let made = |fragment: Option<&VulkanFunction>, private: Option<PrivateRenderDesc>| {
            d.new_render_pipeline_state_full(&FullRenderPipelineDesc {
                label: None,
                vertex: &vertex,
                fragment,
                vertex_descriptor: None,
                colors: vec![(
                    0,
                    ColorAttachmentDesc {
                        pixel_format: 70,
                        ..ColorAttachmentDesc::default()
                    },
                )],
                depth_pixel_format: 0,
                stencil_pixel_format: 0,
                raster_sample_count: 1,
                alpha_to_coverage: false,
                alpha_to_one: false,
                rasterization_enabled: true,
                input_primitive_topology: 0,
                max_vertex_amplification_count: 1,
                support_indirect_command_buffers: false,
                vertex_linked: LinkedFunctionsDesc::default(),
                fragment_linked: LinkedFunctionsDesc::default(),
                private_state: private,
            })
            .map(|state| state.is_some())
            .map_err(|e| e.code)
        };
        let gl = PrivateRenderDesc {
            open_gl_mode_enabled: true,
            ..PrivateRenderDesc::default()
        };
        for fragment in [Some(&fragment), None] {
            let has = fragment.is_some();
            assert_eq!(made(fragment, None), Err(3), "fragment {has}");
            let plain = Some(PrivateRenderDesc::default());
            assert_eq!(
                made(fragment, plain),
                Err(3),
                "fragment {has}, the mode off"
            );
            assert_eq!(made(fragment, Some(gl)), Ok(true), "fragment {has}");
        }
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
