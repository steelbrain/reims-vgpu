//! Narrow raw `msg_send` for Metal APIs missing from metal-0.33 — and for the
//! ones it has that cannot report a failed allocation.
//!
//! # Why an allocator metal-0.33 already exposes is re-spelled here
//!
//! `Device::new_texture` is `msg_send![self, newTextureWithDescriptor: d]` with
//! the return typed as `metal::Texture`, and `foreign_types` 0.5 declares that
//! as `struct Texture(NonNull<MTLTexture>)`. `newTextureWithDescriptor:` returns
//! **nil when the allocation fails**, which is what a Metal device does when its
//! VRAM is full — so the failing case writes a null pointer into a `NonNull`
//! field. That is an invalid value, and therefore undefined behaviour, rather
//! than a `Texture` the caller could test. The same holds for every
//! `new_buffer*`.
//!
//! It is the same class as [`super::mtl_enum`]'s: a value that has no legal
//! representation in the Rust type must be checked *before* it becomes one, and
//! the check cannot be moved after the conversion. So the pointer is taken raw,
//! tested, and only then wrapped — exactly what [`new_texture_view_swizzled`]
//! below has always done, because that one API is missing from metal-0.33 and so
//! had to be hand-written. The allocators in this section exist to give the ones
//! metal-0.33 *does* expose the same treatment; nothing about the swizzled view
//! made it special except that writing it out forced the question.
//!
//! `None` is a real answer here, not a defensive one: it is the device saying it
//! has no memory left, which every caller turns into a typed refusal. That is
//! what a GPU does with an allocation it cannot serve.

use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    ArgumentEncoder, BlitCommandEncoderRef, Buffer, BufferRef, CommandBufferRef, CommandQueueRef,
    ComputeCommandEncoderRef, ComputePipelineState, DeviceRef, FunctionRef, IndirectCommandBuffer,
    IndirectCommandBufferDescriptorRef, IndirectCommandBufferRef, IndirectComputeCommandRef,
    IndirectRenderCommandRef, MTLDispatchType, MTLIndexType, MTLPixelFormat, MTLPrimitiveType,
    MTLRegion, MTLResourceOptions, MTLSize, MTLTextureType, NSInteger, NSRange, NSUInteger,
    RenderCommandEncoderRef, RenderPassDescriptorRef, RenderPipelineDescriptorRef, Texture,
    TextureDescriptorRef, TextureRef,
};
use objc::runtime::{Object, BOOL, NO, YES};
use objc::{msg_send, sel, sel_impl};

// SDK MTLTessellation* enums (not fully exposed by metal-0.33).
pub const MTL_TESSELLATION_PARTITION_INTEGER: NSUInteger = 1;
pub const MTL_TESSELLATION_FACTOR_STEP_CONSTANT: NSUInteger = 0;
pub const MTL_TESSELLATION_FACTOR_FORMAT_HALF: NSUInteger = 0;
pub const MTL_WINDING_CLOCKWISE: NSUInteger = 0;
pub const MTL_TESSELLATION_CONTROL_POINT_INDEX_NONE: NSUInteger = 0;
pub const MTL_TESSELLATION_CONTROL_POINT_INDEX_UINT16: NSUInteger = 1;

/// Configure tessellation fields on `MTLRenderPipelineDescriptor` for ICB
/// `drawPatches` / `drawIndexedPatches` (metal-0.33 leaves these as TODOs).
pub fn configure_tessellation_pipeline(
    desc: &RenderPipelineDescriptorRef,
    max_factor: NSUInteger,
    control_point_index_type: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![desc, setMaxTessellationFactor: max_factor];
        let _: () = msg_send![
            desc,
            setTessellationFactorFormat: MTL_TESSELLATION_FACTOR_FORMAT_HALF
        ];
        let _: () = msg_send![
            desc,
            setTessellationPartitionMode: MTL_TESSELLATION_PARTITION_INTEGER
        ];
        let _: () = msg_send![
            desc,
            setTessellationFactorStepFunction: MTL_TESSELLATION_FACTOR_STEP_CONSTANT
        ];
        let _: () = msg_send![
            desc,
            setTessellationOutputWindingOrder: MTL_WINDING_CLOCKWISE
        ];
        let _: () = msg_send![
            desc,
            setTessellationControlPointIndexType: control_point_index_type
        ];
        let _: () = msg_send![desc, setTessellationFactorScaleEnabled: NO];
    }
}

/// ICB `drawPatches:…` with optional (null) patch index buffer.
/// metal-rs requires `&BufferRef`; SDK allows `nullable` for patchIndexBuffer.
#[allow(clippy::too_many_arguments)]
pub fn icb_draw_patches(
    cmd: &IndirectRenderCommandRef,
    number_of_patch_control_points: NSUInteger,
    patch_start: NSUInteger,
    patch_count: NSUInteger,
    patch_index_buffer: Option<&BufferRef>,
    patch_index_buffer_offset: NSUInteger,
    instance_count: NSUInteger,
    base_instance: NSUInteger,
    tessellation_factor_buffer: &BufferRef,
    tessellation_factor_buffer_offset: NSUInteger,
    tessellation_factor_buffer_instance_stride: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            drawPatches: number_of_patch_control_points
            patchStart: patch_start
            patchCount: patch_count
            patchIndexBuffer: patch_index_buffer
            patchIndexBufferOffset: patch_index_buffer_offset
            instanceCount: instance_count
            baseInstance: base_instance
            tessellationFactorBuffer: tessellation_factor_buffer
            tessellationFactorBufferOffset: tessellation_factor_buffer_offset
            tessellationFactorBufferInstanceStride: tessellation_factor_buffer_instance_stride
        ];
    }
}

/// ICB `drawIndexedPatches:…` with optional (null) patch index buffer.
#[allow(clippy::too_many_arguments)]
pub fn icb_draw_indexed_patches(
    cmd: &IndirectRenderCommandRef,
    number_of_patch_control_points: NSUInteger,
    patch_start: NSUInteger,
    patch_count: NSUInteger,
    patch_index_buffer: Option<&BufferRef>,
    patch_index_buffer_offset: NSUInteger,
    control_point_index_buffer: &BufferRef,
    control_point_index_buffer_offset: NSUInteger,
    instance_count: NSUInteger,
    base_instance: NSUInteger,
    tessellation_factor_buffer: &BufferRef,
    tessellation_factor_buffer_offset: NSUInteger,
    tessellation_factor_buffer_instance_stride: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            drawIndexedPatches: number_of_patch_control_points
            patchStart: patch_start
            patchCount: patch_count
            patchIndexBuffer: patch_index_buffer
            patchIndexBufferOffset: patch_index_buffer_offset
            controlPointIndexBuffer: control_point_index_buffer
            controlPointIndexBufferOffset: control_point_index_buffer_offset
            instanceCount: instance_count
            baseInstance: base_instance
            tessellationFactorBuffer: tessellation_factor_buffer
            tessellationFactorBufferOffset: tessellation_factor_buffer_offset
            tessellationFactorBufferInstanceStride: tessellation_factor_buffer_instance_stride
        ];
    }
}

/// `setSupportIndirectCommandBuffers:` on `MTLMeshRenderPipelineDescriptor`
/// (metal-0.33 exposes this only on classic `MTLRenderPipelineDescriptor`).
pub fn mesh_pipeline_set_support_indirect_command_buffers(
    desc: &metal::MeshRenderPipelineDescriptorRef,
    support: bool,
) {
    unsafe {
        let v: BOOL = if support { YES } else { NO };
        let _: () = msg_send![desc, setSupportIndirectCommandBuffers: v];
    }
}

/// Mesh / object bind counts on `MTLIndirectCommandBufferDescriptor` (macOS 14+).
pub fn set_max_mesh_buffer_bind_count(
    desc: &metal::IndirectCommandBufferDescriptorRef,
    count: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![desc, setMaxMeshBufferBindCount: count];
    }
}

pub fn set_max_object_buffer_bind_count(
    desc: &metal::IndirectCommandBufferDescriptorRef,
    count: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![desc, setMaxObjectBufferBindCount: count];
    }
}

/// ICB `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`
/// (metal-0.33 has no IndirectRenderCommand mesh entry points).
pub fn icb_draw_mesh_threads(
    cmd: &IndirectRenderCommandRef,
    threads_per_grid: MTLSize,
    threads_per_object_threadgroup: MTLSize,
    threads_per_mesh_threadgroup: MTLSize,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            drawMeshThreads: threads_per_grid
            threadsPerObjectThreadgroup: threads_per_object_threadgroup
            threadsPerMeshThreadgroup: threads_per_mesh_threadgroup
        ];
    }
}

/// ICB `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
pub fn icb_draw_mesh_threadgroups(
    cmd: &IndirectRenderCommandRef,
    threadgroups_per_grid: MTLSize,
    threads_per_object_threadgroup: MTLSize,
    threads_per_mesh_threadgroup: MTLSize,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            drawMeshThreadgroups: threadgroups_per_grid
            threadsPerObjectThreadgroup: threads_per_object_threadgroup
            threadsPerMeshThreadgroup: threads_per_mesh_threadgroup
        ];
    }
}

/// ICB `setMeshBuffer:offset:atIndex:` (mesh stage binds).
pub fn icb_set_mesh_buffer(
    cmd: &IndirectRenderCommandRef,
    buffer: Option<&BufferRef>,
    offset: NSUInteger,
    index: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            setMeshBuffer: buffer
            offset: offset
            atIndex: index
        ];
    }
}

/// ICB `setObjectBuffer:offset:atIndex:` (object stage binds).
pub fn icb_set_object_buffer(
    cmd: &IndirectRenderCommandRef,
    buffer: Option<&BufferRef>,
    offset: NSUInteger,
    index: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            setObjectBuffer: buffer
            offset: offset
            atIndex: index
        ];
    }
}

/// SDK `MTLFunctionType` for a macOS 13+ mesh shader function; metal-0.33
/// omits it.
pub const MTL_FUNCTION_TYPE_MESH: NSUInteger = 7;
/// SDK `MTLFunctionType` for a macOS 13+ object shader function.
pub const MTL_FUNCTION_TYPE_OBJECT: NSUInteger = 8;

/// `functionType` on `MTLFunction` (raw NSUInteger; metal-0.33 omits Mesh/Object).
pub fn function_type(function: &FunctionRef) -> NSUInteger {
    unsafe { msg_send![function, functionType] }
}

/// ICB `setObjectThreadgroupMemoryLength:atIndex:`.
pub fn icb_set_object_threadgroup_memory_length(
    cmd: &IndirectRenderCommandRef,
    length: NSUInteger,
    index: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            setObjectThreadgroupMemoryLength: length
            atIndex: index
        ];
    }
}

/// `setMaxObjectThreadgroupMemoryBindCount:` on ICB descriptor.
pub fn set_max_object_threadgroup_memory_bind_count(
    desc: &metal::IndirectCommandBufferDescriptorRef,
    count: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![desc, setMaxObjectThreadgroupMemoryBindCount: count];
    }
}

/// MTLTextureSwizzleChannels (C struct).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MtlTextureSwizzleChannels {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

pub const SWIZZLE_ZERO: u8 = 0;
pub const SWIZZLE_ONE: u8 = 1;
pub const SWIZZLE_RED: u8 = 2;
pub const SWIZZLE_GREEN: u8 = 3;
pub const SWIZZLE_BLUE: u8 = 4;
pub const SWIZZLE_ALPHA: u8 = 5;

pub fn swizzle_selector(selector: u8) -> Option<u8> {
    match selector {
        SWIZZLE_ZERO | SWIZZLE_ONE | SWIZZLE_RED | SWIZZLE_GREEN | SWIZZLE_BLUE | SWIZZLE_ALPHA => {
            Some(selector)
        }
        _ => None,
    }
}

pub fn texture_swizzle_channels(swizzle: [u8; 4]) -> Option<MtlTextureSwizzleChannels> {
    Some(MtlTextureSwizzleChannels {
        red: swizzle_selector(swizzle[0])?,
        green: swizzle_selector(swizzle[1])?,
        blue: swizzle_selector(swizzle[2])?,
        alpha: swizzle_selector(swizzle[3])?,
    })
}

/// `newTextureWithDescriptor:`, with the nil an exhausted device returns.
///
/// The checked replacement for `metal::Device::new_texture`. See this module's
/// own doc for why that one cannot be used: its return type cannot hold the
/// failure.
pub fn new_texture(device: &DeviceRef, descriptor: &TextureDescriptorRef) -> Option<Texture> {
    unsafe {
        let ptr: *mut Object = msg_send![device, newTextureWithDescriptor: descriptor];
        (!ptr.is_null()).then(|| Texture::from_ptr(ptr as *mut _))
    }
}

/// `newBufferWithLength:options:`, with the nil an exhausted device returns.
pub fn new_buffer(
    device: &DeviceRef,
    length: NSUInteger,
    options: MTLResourceOptions,
) -> Option<Buffer> {
    unsafe {
        let ptr: *mut Object = msg_send![device, newBufferWithLength: length options: options];
        (!ptr.is_null()).then(|| Buffer::from_ptr(ptr as *mut _))
    }
}

/// `newBufferWithBytes:length:options:`, with the nil an exhausted device
/// returns.
///
/// # Safety
///
/// `bytes` must point to at least `length` readable bytes for the duration of
/// the call. Metal copies them before returning, so the caller owes nothing
/// afterwards — this is the copying constructor, not the no-copy one.
pub unsafe fn new_buffer_with_data(
    device: &DeviceRef,
    bytes: *const std::ffi::c_void,
    length: NSUInteger,
    options: MTLResourceOptions,
) -> Option<Buffer> {
    unsafe {
        let ptr: *mut Object =
            msg_send![device, newBufferWithBytes: bytes length: length options: options];
        (!ptr.is_null()).then(|| Buffer::from_ptr(ptr as *mut _))
    }
}

/// `newBufferWithBytesNoCopy:length:options:deallocator:`, with the nil an
/// exhausted device returns.
///
/// The deallocator is always nil: this device keeps every no-copy buffer's bytes
/// alive for the command buffer's lifetime itself, which is the contract
/// [`super::runtime::new_buffer_from_host`] states for its caller.
///
/// # Safety
///
/// `bytes` must point to `length` readable bytes that stay alive, unmoved, for
/// as long as the returned buffer is in use by the GPU. Metal does **not** copy
/// them. `bytes` and `length` must both be page-aligned or Metal returns nil,
/// which this reports as `None` rather than as a `Buffer` that is not one.
///
/// Aliasing guest RAM through this call is permitted: it is the Metal-direct
/// spelling of the host-pointer import, and MoltenVK implements
/// `VK_EXT_external_memory_host` over exactly this message. What bounds such a
/// caller is [`crate::runtime::guest_ram`]'s type pair, not this function —
/// which is why the safety contract above is the whole of what this one
/// promises, and callers passing guest bytes owe that module's rules on top.
pub unsafe fn new_buffer_no_copy(
    device: &DeviceRef,
    bytes: *mut std::ffi::c_void,
    length: NSUInteger,
    options: MTLResourceOptions,
) -> Option<Buffer> {
    unsafe {
        let ptr: *mut Object = msg_send![
            device,
            newBufferWithBytesNoCopy: bytes
            length: length
            options: options
            deallocator: std::ptr::null::<Object>()
        ];
        (!ptr.is_null()).then(|| Buffer::from_ptr(ptr as *mut _))
    }
}

/// `[MTLCommandQueue commandBuffer]`, with the nil it can answer.
///
/// **Borrowed, exactly as metal-0.33 returns it.** The object is autoreleased,
/// so ownership is taken by the caller's `.to_owned()` (which retains); handing
/// back an owned `CommandBuffer` from `from_ptr` here would take ownership
/// without retaining and over-release at drop.
///
/// A nil is the queue declining to issue another command buffer — a resource
/// limit, and one of the few Metal refusals that is genuinely about pressure
/// rather than about the request.
pub fn new_command_buffer(queue: &CommandQueueRef) -> Option<&CommandBufferRef> {
    unsafe {
        let ptr: *mut Object = msg_send![queue, commandBuffer];
        (!ptr.is_null()).then(|| CommandBufferRef::from_ptr(ptr as *mut _))
    }
}

/// `[MTLCommandBuffer renderCommandEncoderWithDescriptor:]`, with the nil an
/// invalid pass descriptor answers.
///
/// Borrowed for the same reason as [`new_command_buffer`]. The lifetime ties the
/// encoder to the command buffer that vended it, which is what metal-0.33's
/// signature does and what Metal's ownership actually is.
pub fn new_render_command_encoder<'a>(
    command_buffer: &'a CommandBufferRef,
    descriptor: &RenderPassDescriptorRef,
) -> Option<&'a RenderCommandEncoderRef> {
    unsafe {
        let ptr: *mut Object =
            msg_send![command_buffer, renderCommandEncoderWithDescriptor: descriptor];
        (!ptr.is_null()).then(|| RenderCommandEncoderRef::from_ptr(ptr as *mut _))
    }
}

/// `[MTLCommandBuffer blitCommandEncoder]`, with its nil.
pub fn new_blit_command_encoder(
    command_buffer: &CommandBufferRef,
) -> Option<&BlitCommandEncoderRef> {
    unsafe {
        let ptr: *mut Object = msg_send![command_buffer, blitCommandEncoder];
        (!ptr.is_null()).then(|| BlitCommandEncoderRef::from_ptr(ptr as *mut _))
    }
}

/// `[MTLCommandBuffer computeCommandEncoderWithDispatchType:]`, with its nil.
pub fn new_compute_command_encoder_with_dispatch_type(
    command_buffer: &CommandBufferRef,
    dispatch_type: MTLDispatchType,
) -> Option<&ComputeCommandEncoderRef> {
    unsafe {
        let ptr: *mut Object =
            msg_send![command_buffer, computeCommandEncoderWithDispatchType: dispatch_type];
        (!ptr.is_null()).then(|| ComputeCommandEncoderRef::from_ptr(ptr as *mut _))
    }
}

/// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`, with the
/// nil an exhausted device returns.
///
/// This one is sized by the guest's own `maxCommandCount`, so its nil is
/// squarely the out-of-memory case.
pub fn new_indirect_command_buffer(
    device: &DeviceRef,
    descriptor: &IndirectCommandBufferDescriptorRef,
    max_command_count: NSUInteger,
    options: MTLResourceOptions,
) -> Option<IndirectCommandBuffer> {
    unsafe {
        let ptr: *mut Object = msg_send![
            device,
            newIndirectCommandBufferWithDescriptor: descriptor
            maxCommandCount: max_command_count
            options: options
        ];
        (!ptr.is_null()).then(|| IndirectCommandBuffer::from_ptr(ptr as *mut _))
    }
}

/// `[MTLFunction newArgumentEncoderWithBufferIndex:]`, with its nil.
pub fn new_argument_encoder(
    function: &FunctionRef,
    buffer_index: NSUInteger,
) -> Option<ArgumentEncoder> {
    unsafe {
        let ptr: *mut Object = msg_send![function, newArgumentEncoderWithBufferIndex: buffer_index];
        (!ptr.is_null()).then(|| ArgumentEncoder::from_ptr(ptr as *mut _))
    }
}

pub fn new_texture_view_swizzled(
    texture: &TextureRef,
    pixel_format: MTLPixelFormat,
    swizzle: MtlTextureSwizzleChannels,
) -> Option<Texture> {
    unsafe {
        let levels = NSRange::new(0, 1);
        let slices = NSRange::new(0, 1);
        let ptr: *mut Object = msg_send![texture,
            newTextureViewWithPixelFormat: pixel_format
            textureType: MTLTextureType::D2
            levels: levels
            slices: slices
            swizzle: swizzle
        ];
        if ptr.is_null() {
            None
        } else {
            Some(Texture::from_ptr(ptr as *mut _))
        }
    }
}

pub fn set_buffer_with_attribute_stride(
    encoder: &ComputeCommandEncoderRef,
    buffer: &BufferRef,
    offset: NSUInteger,
    attribute_stride: NSUInteger,
    index: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![encoder,
            setBuffer: buffer
            offset: offset
            attributeStride: attribute_stride
            atIndex: index
        ];
    }
}

pub fn set_stage_in_region(encoder: &ComputeCommandEncoderRef, region: MTLRegion) {
    unsafe {
        let _: () = msg_send![encoder, setStageInRegion: region];
    }
}

pub fn set_stage_in_region_indirect(
    encoder: &ComputeCommandEncoderRef,
    buffer: &BufferRef,
    offset: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![encoder,
            setStageInRegionWithIndirectBuffer: buffer
            indirectBufferOffset: offset
        ];
    }
}

pub fn set_imageblock_width_height(
    encoder: &ComputeCommandEncoderRef,
    width: NSUInteger,
    height: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![encoder,
            setImageblockWidth: width
            height: height
        ];
    }
}

// ---------------------------------------------------------------------------
// Compute encoder control-flow + ICB SPI
//
// These selectors are present on Apple Silicon MTLComputeCommandEncoder
// (AGX*FamilyComputeContext) but are not in the public Metal.framework headers
// that metal-0.33 wraps. Verified by runtime respondsToSelector + smoke encode.
// Wire contract: Reims VGPU compute 0xdc..0xe2 / 0xe4..0xe5 (compute-surface-manifest).
//
// Condition `comparison` is the Reims VGPU/MetalSerializer enum passed through as-is
// (NOT MTLCompareFunction): host probe shows Equal=0 (buffer==reference),
// Less=1, Always=7 among others. Product must not remap wire values.
// ---------------------------------------------------------------------------

/// `encodeStartDoWhile` (wire 0xdc) — empty start of do-while; condition is on end.
pub fn encode_start_do_while(encoder: &ComputeCommandEncoderRef) {
    unsafe {
        let _: () = msg_send![encoder, encodeStartDoWhile];
    }
}

/// `encodeEndDoWhile:offset:comparison:referenceValue:` (wire 0xdd). Returns Metal BOOL.
pub fn encode_end_do_while(
    encoder: &ComputeCommandEncoderRef,
    buffer: &BufferRef,
    offset: NSUInteger,
    comparison: NSUInteger,
    reference_value: u32,
) -> bool {
    unsafe {
        let ok: BOOL = msg_send![encoder,
            encodeEndDoWhile: buffer
            offset: offset
            comparison: comparison
            referenceValue: reference_value
        ];
        ok == YES
    }
}

/// `encodeStartWhile:offset:comparison:referenceValue:` (wire 0xde).
pub fn encode_start_while(
    encoder: &ComputeCommandEncoderRef,
    buffer: &BufferRef,
    offset: NSUInteger,
    comparison: NSUInteger,
    reference_value: u32,
) {
    unsafe {
        let _: () = msg_send![encoder,
            encodeStartWhile: buffer
            offset: offset
            comparison: comparison
            referenceValue: reference_value
        ];
    }
}

/// `encodeEndWhile` (wire 0xdf).
pub fn encode_end_while(encoder: &ComputeCommandEncoderRef) -> bool {
    unsafe {
        let ok: BOOL = msg_send![encoder, encodeEndWhile];
        ok == YES
    }
}

/// `encodeStartIf:offset:comparison:referenceValue:` (wire 0xe0).
pub fn encode_start_if(
    encoder: &ComputeCommandEncoderRef,
    buffer: &BufferRef,
    offset: NSUInteger,
    comparison: NSUInteger,
    reference_value: u32,
) {
    unsafe {
        let _: () = msg_send![encoder,
            encodeStartIf: buffer
            offset: offset
            comparison: comparison
            referenceValue: reference_value
        ];
    }
}

/// `encodeStartElse` (wire 0xe1).
pub fn encode_start_else(encoder: &ComputeCommandEncoderRef) {
    unsafe {
        let _: () = msg_send![encoder, encodeStartElse];
    }
}

/// `encodeEndIf` (wire 0xe2).
pub fn encode_end_if(encoder: &ComputeCommandEncoderRef) -> bool {
    unsafe {
        let ok: BOOL = msg_send![encoder, encodeEndIf];
        ok == YES
    }
}

/// `executeCommandsInBuffer:withRange:` on a compute encoder (wire 0xe4).
pub fn execute_commands_in_buffer(
    encoder: &ComputeCommandEncoderRef,
    icb: &IndirectCommandBufferRef,
    location: NSUInteger,
    length: NSUInteger,
) {
    unsafe {
        let range = NSRange { location, length };
        let _: () = msg_send![encoder,
            executeCommandsInBuffer: icb
            withRange: range
        ];
    }
}

/// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` (wire 0xe5).
pub fn execute_commands_in_buffer_indirect(
    encoder: &ComputeCommandEncoderRef,
    icb: &IndirectCommandBufferRef,
    indirect: &BufferRef,
    offset: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![encoder,
            executeCommandsInBuffer: icb
            indirectBuffer: indirect
            indirectBufferOffset: offset
        ];
    }
}

/// A Metal compute-pipeline reflection call that returned no pipeline state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalPipelineDecline {
    pub detail: String,
}

impl crate::observe::Decline for MetalPipelineDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self { .. } => "metal_compute_reflection_pipeline_create",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "detail",
            self.detail.split_whitespace().collect::<Vec<_>>().join("_"),
        )]
    }
}

crate::observe::decline_display!(MetalPipelineDecline);

impl std::error::Error for MetalPipelineDecline {}

/// `newComputePipelineStateWithFunction:options:reflection:error:` for BindingInfo reflection.
pub fn new_compute_pso_with_function_reflection(
    device: &DeviceRef,
    function: &FunctionRef,
    options: NSUInteger,
) -> Result<(ComputePipelineState, *mut Object), MetalPipelineDecline> {
    use std::ptr;
    unsafe {
        let mut err: *mut Object = ptr::null_mut();
        let mut reflection: *mut Object = ptr::null_mut();
        let pso: *mut Object = msg_send![device,
            newComputePipelineStateWithFunction: function
            options: options
            reflection: &mut reflection
            error: &mut err
        ];
        if pso.is_null() {
            let msg = if !err.is_null() {
                let desc: *mut Object = msg_send![err, localizedDescription];
                let cstr: *const i8 = msg_send![desc, UTF8String];
                if cstr.is_null() {
                    "(no detail)".to_string()
                } else {
                    std::ffi::CStr::from_ptr(cstr)
                        .to_string_lossy()
                        .into_owned()
                }
            } else {
                "(no detail)".to_string()
            };
            return Err(MetalPipelineDecline { detail: msg });
        }
        if !reflection.is_null() {
            let _: *mut Object = msg_send![reflection, retain];
        }
        Ok((ComputePipelineState::from_ptr(pso as *mut _), reflection))
    }
}

/// Iterate `bindings` array on a reflection object (MTLBinding protocol).
pub struct BindingInfo {
    pub used: bool,
    pub type_: NSUInteger,
    pub access: NSUInteger,
    pub index: NSUInteger,
    pub array_length: NSUInteger,
}

pub const BINDING_TYPE_BUFFER: NSUInteger = 0;
pub const BINDING_TYPE_TEXTURE: NSUInteger = 2;
pub const BINDING_TYPE_SAMPLER: NSUInteger = 3;
pub const BINDING_ACCESS_READ_ONLY: NSUInteger = 0;
pub const BINDING_ACCESS_READ_WRITE: NSUInteger = 1;
pub const BINDING_ACCESS_WRITE_ONLY: NSUInteger = 2;

/// `MTLDataType` values used in AB struct reflection (SDK).
pub const MTL_DATA_TYPE_TEXTURE: NSUInteger = 58;
pub const MTL_DATA_TYPE_SAMPLER: NSUInteger = 59;

/// One texture field inside a kernel argument-buffer struct.
#[derive(Clone, Debug)]
pub struct AbTextureMember {
    pub argument_index: NSUInteger,
    /// `BINDING_ACCESS_*` from the texture reference type.
    pub access: NSUInteger,
}

/// One sampler field inside a kernel argument-buffer struct.
#[derive(Clone, Debug)]
pub struct AbSamplerMember {
    pub argument_index: NSUInteger,
}

/// Layout of a kernel buffer that is an argument buffer holding textures/samplers.
#[derive(Clone, Debug)]
pub struct AbBufferLayout {
    pub buffer_index: NSUInteger,
    pub textures: Vec<AbTextureMember>,
    pub samplers: Vec<AbSamplerMember>,
}

/// Reflect argument-buffer texture/sampler members from a compute function's
/// BindingInfo (pipeline option `1`). Returns the first buffer binding that
/// contains texture or sampler struct members (the ICB-capable texture path).
pub fn reflect_argument_buffer_layout(
    device: &DeviceRef,
    function: &FunctionRef,
) -> Result<Option<AbBufferLayout>, MetalPipelineDecline> {
    // MTLPipelineOptionBindingInfo = 1
    const OPT_BINDING_INFO: NSUInteger = 1;
    let (_pso, reflection) =
        new_compute_pso_with_function_reflection(device, function, OPT_BINDING_INFO)?;
    let layout = unsafe { walk_ab_layout(reflection) };
    unsafe {
        if !reflection.is_null() {
            let _: () = msg_send![reflection, release];
        }
    }
    Ok(layout)
}

unsafe fn walk_ab_layout(reflection: *mut Object) -> Option<AbBufferLayout> {
    if reflection.is_null() {
        return None;
    }
    let bindings: *mut Object = msg_send![reflection, bindings];
    if bindings.is_null() {
        return None;
    }
    let count: NSUInteger = msg_send![bindings, count];
    for i in 0..count {
        let b: *mut Object = msg_send![bindings, objectAtIndex: i];
        if b.is_null() {
            continue;
        }
        let type_: NSUInteger = msg_send![b, type];
        if type_ != BINDING_TYPE_BUFFER {
            continue;
        }
        let index: NSUInteger = msg_send![b, index];
        // MTLBufferBinding.bufferStructType
        let struct_type: *mut Object = msg_send![b, bufferStructType];
        if struct_type.is_null() {
            continue;
        }
        let members: *mut Object = msg_send![struct_type, members];
        if members.is_null() {
            continue;
        }
        let mcount: NSUInteger = msg_send![members, count];
        let mut textures = Vec::new();
        let mut samplers = Vec::new();
        for j in 0..mcount {
            let m: *mut Object = msg_send![members, objectAtIndex: j];
            if m.is_null() {
                continue;
            }
            let data_type: NSUInteger = msg_send![m, dataType];
            let arg_index: NSUInteger = msg_send![m, argumentIndex];
            if data_type == MTL_DATA_TYPE_TEXTURE {
                // textureReferenceType → access
                let tr: *mut Object = msg_send![m, textureReferenceType];
                let access: NSUInteger = if !tr.is_null() {
                    msg_send![tr, access]
                } else {
                    BINDING_ACCESS_READ_WRITE
                };
                textures.push(AbTextureMember {
                    argument_index: arg_index,
                    access,
                });
            } else if data_type == MTL_DATA_TYPE_SAMPLER {
                samplers.push(AbSamplerMember {
                    argument_index: arg_index,
                });
            }
        }
        if !textures.is_empty() || !samplers.is_empty() {
            textures.sort_by_key(|t| t.argument_index);
            samplers.sort_by_key(|s| s.argument_index);
            return Some(AbBufferLayout {
                buffer_index: index,
                textures,
                samplers,
            });
        }
    }
    None
}

pub fn reflection_bindings(reflection: *mut Object) -> Vec<BindingInfo> {
    if reflection.is_null() {
        return Vec::new();
    }
    unsafe {
        let bindings: *mut Object = msg_send![reflection, bindings];
        if bindings.is_null() {
            return Vec::new();
        }
        let count: NSUInteger = msg_send![bindings, count];
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let b: *mut Object = msg_send![bindings, objectAtIndex: i];
            if b.is_null() {
                continue;
            }
            let used: BOOL = msg_send![b, isUsed];
            let type_: NSUInteger = msg_send![b, type];
            let access: NSUInteger = msg_send![b, access];
            let index: NSUInteger = msg_send![b, index];
            // arrayLength exists on MTLTextureBinding; for others default 1.
            let array_length: NSUInteger = if type_ == BINDING_TYPE_TEXTURE {
                let al: NSUInteger = msg_send![b, arrayLength];
                if al == 0 {
                    1
                } else {
                    al
                }
            } else {
                1
            };
            out.push(BindingInfo {
                used: used == YES,
                type_,
                access,
                index,
                array_length,
            });
        }
        out
    }
}

/// Which sampler slots one stage of a built PSO actually samples, as a bit per
/// slot, from Metal's own pipeline reflection.
///
/// `Ok(0)` where there is no reflection to read — no argument info was
/// requested, or the stage has no bindings — which is a stage that samples
/// nothing and not a failure. `Err` only where the reflection names a slot
/// outside the argument table, which is the alarm described at the check.
pub fn render_reflection_sampler_mask(
    reflection: *mut Object,
    vertex: bool,
) -> Result<u32, MetalSamplerMaskOverflow> {
    if reflection.is_null() {
        return Ok(0);
    }
    unsafe {
        let bindings: *mut Object = if vertex {
            msg_send![reflection, vertexBindings]
        } else {
            msg_send![reflection, fragmentBindings]
        };
        if bindings.is_null() {
            return Ok(0);
        }
        let count: NSUInteger = msg_send![bindings, count];
        let mut mask = 0u32;
        for i in 0..count {
            let b: *mut Object = msg_send![bindings, objectAtIndex: i];
            if b.is_null() {
                continue;
            }
            let used: BOOL = msg_send![b, isUsed];
            if used == NO {
                continue;
            }
            let type_: NSUInteger = msg_send![b, type];
            if type_ != BINDING_TYPE_SAMPLER {
                continue;
            }
            let index: NSUInteger = msg_send![b, index];
            let index = index as usize;
            if !crate::backend::metal::util::valid_sampler_index(index) {
                // A healthy zero: Metal's sampler argument table is what this
                // bound *is*, so its own reflection should never name a slot
                // outside it. A firing means this backend's idea of the table
                // has parted from the driver's.
                //
                // Refused rather than skipped. Dropping the bit built a mask
                // that says the shader does not sample that slot, so the slot
                // never receives its default sampler and the shader reads an
                // undefined one — a wrong frame with nothing to explain it. The
                // sibling `bind_compute_samplers` has always refused on this
                // same bound, one file away, and there is no reason the two
                // stages should answer differently.
                crate::observe::Emit::decline(
                    "metal_render_reflection",
                    &MetalSamplerMaskOverflow { index, vertex },
                )
                .fail_once(index as u64);
                return Err(MetalSamplerMaskOverflow { index, vertex });
            }
            mask |= 1u32 << index;
        }
        Ok(mask)
    }
}

/// Metal's own pipeline reflection named a used sampler at a slot the sampler
/// argument table does not have.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalSamplerMaskOverflow {
    /// The slot the reflection reported.
    pub index: usize,
    /// Which stage's binding list it came from.
    pub vertex: bool,
}

impl crate::observe::Decline for MetalSamplerMaskOverflow {
    fn slug(&self) -> &'static str {
        "metal_reflection_sampler_index_past_table"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("index", self.index.to_string()),
            (
                "stage",
                if self.vertex { "vertex" } else { "fragment" }.to_string(),
            ),
        ]
    }
}

crate::observe::decline_display!(MetalSamplerMaskOverflow);

/// Helper: MTLSize constructor.
pub fn mtl_size(x: u64, y: u64, z: u64) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: z,
    }
}

/// `setMaxKernelThreadgroupMemoryBindCount:` on `MTLIndirectCommandBufferDescriptor`
/// (macOS 14+; not exposed by metal-0.33).
pub fn set_max_kernel_threadgroup_memory_bind_count(
    desc: &metal::IndirectCommandBufferDescriptorRef,
    count: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![desc, setMaxKernelThreadgroupMemoryBindCount: count];
    }
}

/// `setCommandTypes:` with raw SDK bits.
///
/// metal-0.33's `MTLIndirectCommandType` bitflags omit mesh (1<<7 / 1<<8) and
/// mis-shift ConcurrentDispatch; `from_bits_truncate` drops unknown bits. Pass
/// the wire/SDK `u64` through so mesh ICB create works.
pub fn icb_descriptor_set_command_types(
    desc: &metal::IndirectCommandBufferDescriptorRef,
    command_types: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![desc, setCommandTypes: command_types];
    }
}

/// `setKernelBuffer:offset:attributeStride:atIndex:` on `MTLIndirectComputeCommand`
/// (not exposed by metal-0.33).
pub fn icb_set_kernel_buffer_attribute_stride(
    cmd: &IndirectComputeCommandRef,
    buffer: Option<&BufferRef>,
    offset: NSUInteger,
    attribute_stride: NSUInteger,
    index: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            setKernelBuffer: buffer
            offset: offset
            attributeStride: attribute_stride
            atIndex: index
        ];
    }
}

/// `setVertexBuffer:offset:attributeStride:atIndex:` on `MTLIndirectRenderCommand`
/// (not exposed by metal-0.33).
pub fn icb_set_vertex_buffer_attribute_stride(
    cmd: &IndirectRenderCommandRef,
    buffer: Option<&BufferRef>,
    offset: NSUInteger,
    attribute_stride: NSUInteger,
    index: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            setVertexBuffer: buffer
            offset: offset
            attributeStride: attribute_stride
            atIndex: index
        ];
    }
}

/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`
/// on `MTLIndirectRenderCommand`.
///
/// SDK types `baseVertex` as **`NSInteger`** (signed). metal-0.33 incorrectly
/// types it as `NSUInteger`, which cannot represent negative wire values.
/// Pass signed `base_vertex` bit-identical to the guest u64 store.
#[allow(clippy::too_many_arguments)]
pub fn icb_draw_indexed_primitives(
    cmd: &IndirectRenderCommandRef,
    primitive_type: MTLPrimitiveType,
    index_count: NSUInteger,
    index_type: MTLIndexType,
    index_buffer: &BufferRef,
    index_buffer_offset: NSUInteger,
    instance_count: NSUInteger,
    base_vertex: NSInteger,
    base_instance: NSUInteger,
) {
    unsafe {
        let _: () = msg_send![
            cmd,
            drawIndexedPrimitives: primitive_type
            indexCount: index_count
            indexType: index_type
            indexBuffer: index_buffer
            indexBufferOffset: index_buffer_offset
            instanceCount: instance_count
            baseVertex: base_vertex
            baseInstance: base_instance
        ];
    }
}

pub fn command_buffer_error_description(command_buffer: &metal::CommandBufferRef) -> String {
    unsafe {
        let err: *mut Object = msg_send![command_buffer, error];
        if err.is_null() {
            return "(no detail)".to_string();
        }
        let desc: *mut Object = msg_send![err, localizedDescription];
        if desc.is_null() {
            return "(no detail)".to_string();
        }
        let cstr: *const i8 = msg_send![desc, UTF8String];
        if cstr.is_null() {
            "(no detail)".to_string()
        } else {
            std::ffi::CStr::from_ptr(cstr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Decline as _;

    #[test]
    fn metal_pipeline_decline_is_registered_shape_and_log_safe() {
        let decline = MetalPipelineDecline {
            detail: "driver detail with spaces".into(),
        };
        assert_eq!(decline.slug(), "metal_compute_reflection_pipeline_create");
        assert_eq!(
            decline.fields(),
            vec![("detail", "driver_detail_with_spaces".into())]
        );
    }

    #[test]
    fn argument_buffer_reflection_preserves_pipeline_failure_in_its_api() {
        let _reflect: fn(
            &DeviceRef,
            &FunctionRef,
        ) -> Result<Option<AbBufferLayout>, MetalPipelineDecline> = reflect_argument_buffer_layout;
    }
}
