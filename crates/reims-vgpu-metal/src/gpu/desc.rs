//! Metal's descriptors: real Metal objects, made and filled here from core's values.
//!
//! - **A value is sent whole.** A descriptor core fills in and hands to a creator comes as a
//!   `platform::values` value, and one `apply_*` function per value makes a fresh Metal
//!   descriptor hold it ([`apply_buffer`], [`apply_heap`], [`apply_sampler`],
//!   [`apply_depth_stencil`], [`apply_icb`], [`apply_compute_pass`], [`apply_render_pipeline`],
//!   [`apply_texture`]). Its setters go straight to Metal, in the order written at the function's
//!   top, which is the order the value's type states. That order is behaviour: properties that
//!   alias one another (a texture or heap descriptor's `resourceOptions` and `storageMode`,
//!   `cpuCacheMode` and `hazardTrackingMode`) keep the last one set, and some setters fold what
//!   they are given.
//! - **The descriptors core keeps are typed handles.** A texture descriptor and a render pass
//!   descriptor answer core's `TextureDescriptor` and `RenderPassDescriptor` traits, each method
//!   one setter or getter (a render pass takes each record's `*Set` whole); the pipeline
//!   descriptors Metal makes from serialized data take only their resource index.
//! - Setters and getters are `Fatal` sends: a raise inside one dies at the call ([`msg::get`]).
//!   The attachment arrays' subscripts (`On` where the display's frame sets a colour attachment
//!   with no handler and the decoder under one, `Caught` for the decoder's attachment state)
//!   and `setSamplePositions:count:` (`Caught`: only the decoder sends it) answer on the thrown
//!   channel the traits carry ([`msg::send`]); a display's pipeline's subscript is a `Fatal`
//!   send, as its setters are.
//! - A raw guest value goes to Metal as it is: an enum or option set is its newtype over the
//!   same word, whatever Metal makes of it.
//! - A sub-object (an attachment, a stencil descriptor, an attachment array) comes back `+0`,
//!   and its handle retains it ([`Obj::from_retained`]).

use std::ffi::c_void;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSDictionary, NSString};
use objc2_metal::*;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::types::MtlSamplePosition;
use reims_vgpu_core::platform::values::{
    AttachmentState, BufferDeallocator, BufferDesc, ColorAttachmentDesc, ColorAttachmentSet,
    ComputePassDesc, DepthAttachmentSet, DepthStencilDesc, FullComputePipelineDesc,
    FullRenderPipelineDesc, HeapDesc, IcbDesc, LinkedFunctionsDesc, OptionSet, PassSet,
    RenderPipelineDesc, SamplerDesc, StageInputDesc, StencilAttachmentSet, StencilDesc,
    TextureDesc, VertexDesc,
};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use crate::gpu::array;
use crate::gpu::resource::MetalFunction;
use crate::msg::encode::ToMtl;
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::classes::MTLBufferDescriptor;
use crate::spi::methods::{
    AttachmentArraySpi, ForceResourceIndexSpi, HeapDescriptorSpi, IcbDescriptorSpi,
    RenderPassDescriptorSpi, ResourceIndexDescriptorSpi, TextureDescriptorSpi,
};

/// A sub-object getter's `+0` result as a handle, retained.
fn part<T: objc2::Message>(
    sel: &'static str,
    call: impl FnOnce() -> objc2::rc::Retained<T>,
) -> Option<Obj<T>> {
    Some(Obj::from_retained(msg::get(sel, call)))
}

/// An attachment array's `objectAtIndexedSubscript:`: its `+0` attachment, retained, or nil. It
/// raises past the end; `policy` is the caller's own.
fn subscript<A: AttachmentArraySpi>(
    policy: Policy,
    array: &Obj<A>,
    index: u64,
) -> RaiseResult<Option<Obj<A::Item>>> {
    let array = array.get();
    msg::send("objectAtIndexedSubscript:", policy, || {
        array.objectAtIndexedSubscript_or_nil(index)
    })
    .map(|a| a.map(Obj::from_retained))
}

/// `MTLTextureDescriptor *`.
pub type MetalTextureDescriptor = Obj<MTLTextureDescriptor>;

impl reims_vgpu_core::platform::descriptor::TextureDescriptor<MetalPlatform>
    for MetalTextureDescriptor
{
    fn set_resource_options(&self, resource_options: u64) {
        texture_resource_options(self, resource_options);
    }
    fn set_rotation(&self, rotation: u64) {
        let d = self.get();
        msg::get("setRotation:", || d.setRotation(rotation));
    }
    fn set_usage(&self, usage: u64) {
        let (d, v) = (self.get(), MTLTextureUsage(usage.to_usize()));
        msg::get("setUsage:", || d.setUsage(v));
    }
    // SAFETY (the size setters): objc2-metal marks them unsafe because Metal does not check
    // them here; the texture's creator does.
    fn set_width(&self, width: u64) {
        let (d, n) = (self.get(), width.to_usize());
        msg::get("setWidth:", || unsafe { d.setWidth(n) });
    }
    fn set_height(&self, height: u64) {
        let (d, n) = (self.get(), height.to_usize());
        msg::get("setHeight:", || unsafe { d.setHeight(n) });
    }
    fn set_depth(&self, depth: u64) {
        let (d, n) = (self.get(), depth.to_usize());
        msg::get("setDepth:", || unsafe { d.setDepth(n) });
    }
    fn set_texture_type(&self, texture_type: u64) {
        let (d, v) = (self.get(), MTLTextureType(texture_type.to_usize()));
        msg::get("setTextureType:", || d.setTextureType(v));
    }
    fn set_pixel_format(&self, pixel_format: u64) {
        texture_pixel_format(self, pixel_format);
    }
    fn rotation(&self) -> u64 {
        let d = self.get();
        msg::get("rotation", || d.rotation())
    }
    fn storage_mode(&self) -> u64 {
        let d = self.get();
        msg::get("storageMode", || d.storageMode()).0.to_u64()
    }
    fn pixel_format(&self) -> u64 {
        let d = self.get();
        msg::get("pixelFormat", || d.pixelFormat()).0.to_u64()
    }
    fn texture_type(&self) -> u64 {
        let d = self.get();
        msg::get("textureType", || d.textureType()).0.to_u64()
    }
}

/// The private `MTLBufferDescriptor *`.
pub type MetalBufferDescriptor = Obj<MTLBufferDescriptor>;

/// `MTLHeapDescriptor *`.
pub type MetalHeapDescriptor = Obj<MTLHeapDescriptor>;

/// `MTLSamplerDescriptor *`.
pub type MetalSamplerDescriptor = Obj<MTLSamplerDescriptor>;

/// `MTLDepthStencilDescriptor *`.
pub type MetalDepthStencilDescriptor = Obj<MTLDepthStencilDescriptor>;

/// `MTLStencilDescriptor *`.
pub type MetalStencilDescriptor = Obj<MTLStencilDescriptor>;

/// `MTLRenderPassDescriptor *`.
pub type MetalRenderPassDescriptor = Obj<MTLRenderPassDescriptor>;

impl reims_vgpu_core::platform::descriptor::RenderPassDescriptor<MetalPlatform>
    for MetalRenderPassDescriptor
{
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()> {
        let d = self.get();
        let (pointer, count) = (array(positions), positions.len());
        // SAFETY: `count` positions at the pointer, which the call borrows; core's
        // `MtlSamplePosition` is `MTLSamplePosition`'s layout (`{?=ff}`).
        msg::send("setSamplePositions:count:", Policy::Caught, || unsafe {
            d.setSamplePositions_count(pointer.as_ptr(), count)
        })
    }

    // Typed, sent natively.
    fn set_pass(&self, set: &PassSet<'_, MetalPlatform>) {
        apply_pass(set, self);
    }
    /// A raise from `objectAtIndexedSubscript:` comes before any set.
    fn set_color_attachment(
        &self,
        on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, MetalPlatform>,
    ) -> RaiseResult<()> {
        if let Some(a) = color_attachment_at(Policy::On(on), self, index)? {
            apply_color_attachment(set, &a);
        }
        Ok(())
    }
    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, MetalPlatform>) {
        apply_depth_attachment(set, &depth_attachment_of(self));
    }
    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, MetalPlatform>) {
        apply_stencil_attachment(set, &stencil_attachment_of(self));
    }
    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState> {
        Ok(match color_attachment_at(Policy::Caught, self, index)? {
            Some(a) => {
                let a = a.get();
                AttachmentState {
                    load_action: msg::get("loadAction", || a.loadAction()).0.to_u64(),
                    store_action: msg::get("storeAction", || a.storeAction()).0.to_u64(),
                }
            }
            None => AttachmentState::default(),
        })
    }
    fn depth_attachment_state(&self) -> AttachmentState {
        let a = depth_attachment_of(self);
        let a = a.get();
        AttachmentState {
            load_action: msg::get("loadAction", || a.loadAction()).0.to_u64(),
            store_action: msg::get("storeAction", || a.storeAction()).0.to_u64(),
        }
    }
    fn stencil_attachment_state(&self) -> AttachmentState {
        let a = stencil_attachment_of(self);
        let a = a.get();
        AttachmentState {
            load_action: msg::get("loadAction", || a.loadAction()).0.to_u64(),
            store_action: msg::get("storeAction", || a.storeAction()).0.to_u64(),
        }
    }
}

/// Colour attachment `index` of a render pass: `colorAttachments`, then
/// `objectAtIndexedSubscript:` (it raises past the end, as `policy`, the caller's, says); each
/// `+0` result is retained.
fn color_attachment_at(
    policy: Policy,
    d: &MetalRenderPassDescriptor,
    index: u64,
) -> RaiseResult<Option<MetalRenderPassColorAttachment>> {
    let o = d.get();
    let array = Obj::from_retained(msg::get("colorAttachments", || o.colorAttachments()));
    subscript(policy, &array, index)
}

/// A render pass's `depthAttachment` (`+0`, retained).
fn depth_attachment_of(d: &MetalRenderPassDescriptor) -> MetalRenderPassDepthAttachment {
    let o = d.get();
    Obj::from_retained(msg::get("depthAttachment", || o.depthAttachment()))
}

/// A render pass's `stencilAttachment` (`+0`, retained).
fn stencil_attachment_of(d: &MetalRenderPassDescriptor) -> MetalRenderPassStencilAttachment {
    let o = d.get();
    Obj::from_retained(msg::get("stencilAttachment", || o.stencilAttachment()))
}

/// `MTLRenderPassColorAttachmentDescriptor *`.
pub type MetalRenderPassColorAttachment = Obj<MTLRenderPassColorAttachmentDescriptor>;

/// `MTLRenderPassDepthAttachmentDescriptor *`.
pub type MetalRenderPassDepthAttachment = Obj<MTLRenderPassDepthAttachmentDescriptor>;

/// `MTLRenderPassStencilAttachmentDescriptor *`.
pub type MetalRenderPassStencilAttachment = Obj<MTLRenderPassStencilAttachmentDescriptor>;

/// `MTLComputePassDescriptor *`.
pub type MetalComputePassDescriptor = Obj<MTLComputePassDescriptor>;

/// `MTLRenderPipelineDescriptor *`.
pub type MetalRenderPipelineDescriptor = Obj<MTLRenderPipelineDescriptor>;

impl reims_vgpu_core::platform::descriptor::RenderPipelineDescriptor<MetalPlatform>
    for MetalRenderPipelineDescriptor
{
    fn set_resource_index(&self, resource_index: u64, force: bool) {
        pipeline_resource_index(self.get(), resource_index, force);
    }
}

/// `MTLComputePipelineDescriptor *`.
pub type MetalComputePipelineDescriptor = Obj<MTLComputePipelineDescriptor>;

impl reims_vgpu_core::platform::descriptor::ComputePipelineDescriptor<MetalPlatform>
    for MetalComputePipelineDescriptor
{
    fn set_resource_index(&self, resource_index: u64, force: bool) {
        let d = self.get();
        msg::get("setResourceIndex:", || {
            ResourceIndexDescriptorSpi::setResourceIndex(d, resource_index)
        });
        msg::get("setForceResourceIndex:", || {
            ForceResourceIndexSpi::setForceResourceIndex(d, force)
        });
    }
}

/// `MTLTileRenderPipelineDescriptor *`.
pub type MetalTileRenderPipelineDescriptor = Obj<MTLTileRenderPipelineDescriptor>;

impl reims_vgpu_core::platform::descriptor::TileRenderPipelineDescriptor<MetalPlatform>
    for MetalTileRenderPipelineDescriptor
{
}

/// `MTLIndirectCommandBufferDescriptor *`.
pub type MetalIcbDescriptor = Obj<MTLIndirectCommandBufferDescriptor>;

// ---- Values.
//
// Each function below makes Metal's descriptor hold a value of `platform::values`: it sends the
// value's setters straight to the descriptor, in the order written at its top, which is the
// order the value's type states.

/// `setResourceOptions:` on a buffer descriptor (an aliasing setter).
fn buffer_resource_options(d: &MetalBufferDescriptor, resource_options: u64) {
    let o = d.get();
    msg::get("setResourceOptions:", || {
        o.setResourceOptions(resource_options)
    });
}

/// `setDeallocator:` of a block that runs `deallocator` at most once. Metal copies the block and
/// calls it once, with the buffer's contents and length, after the buffer it made is released,
/// on a thread of its own: the block runs marked as ours there, so a panic in `deallocator`
/// aborts before it unwinds into Metal (6.5).
fn set_buffer_deallocator(d: &MetalBufferDescriptor, deallocator: BufferDeallocator) {
    let once = Mutex::new(Some(deallocator));
    let block = RcBlock::new(move |pointer: *mut c_void, length: u64| {
        let _ours = reims_vgpu_core::rt::enter_ours();
        if let Some(d) = once.lock().take() {
            d(HostPtr::from_ptr(pointer), length);
        }
    });
    let (d, block) = (d.get(), &*block);
    msg::get("setDeallocator:", || d.setDeallocator(Some(block)));
}

/// Makes `d` hold `desc`. Order: `setLength:`; the options in list order (`ResourceOptionsOr`
/// reads `resourceOptions` first; a hazard-tracking mode is not sent); `setPinnedGPUAddress:`;
/// `setContents:`; `setNoCopy:YES`; `setDeallocator:`.
pub fn apply_buffer(desc: BufferDesc<'_>, d: &MetalBufferDescriptor) {
    let o = d.get();
    msg::get("setLength:", || o.setLength(desc.length));
    for option in &desc.options {
        match *option {
            OptionSet::ResourceOptions(resource_options) => {
                buffer_resource_options(d, resource_options);
            }
            OptionSet::ResourceOptionsOr(bits) => {
                let options = msg::get("resourceOptions", || o.resourceOptions());
                buffer_resource_options(d, options | bits);
            }
            OptionSet::StorageMode(storage_mode) => {
                msg::get("setStorageMode:", || o.setStorageMode(storage_mode));
            }
            OptionSet::HazardTrackingMode(_) => {}
        }
    }
    if let Some(address) = desc.pinned_gpu_address {
        msg::get("setPinnedGPUAddress:", || o.setPinnedGPUAddress(address));
    }
    if let Some(contents) = desc.contents {
        let pointer = contents.ptr().as_ptr::<c_void>();
        // SAFETY: the descriptor stores the pointer; `HostRange::new`'s contract covers the bytes
        // for the buffer made from it.
        msg::get("setContents:", || unsafe { o.setContents(pointer) });
    }
    if desc.no_copy {
        msg::get("setNoCopy:", || o.setNoCopy(true));
    }
    if let Some(deallocator) = desc.deallocator {
        set_buffer_deallocator(d, deallocator);
    }
}

/// One storage property on a heap descriptor: each is an aliasing setter.
fn heap_option(option: OptionSet, d: &MetalHeapDescriptor) {
    let o = d.get();
    match option {
        OptionSet::ResourceOptions(resource_options) => {
            heap_resource_options(d, resource_options);
        }
        OptionSet::ResourceOptionsOr(bits) => {
            let options = msg::get("resourceOptions", || o.resourceOptions())
                .0
                .to_u64();
            heap_resource_options(d, options | bits);
        }
        OptionSet::StorageMode(storage_mode) => {
            let v = MTLStorageMode(storage_mode.to_usize());
            msg::get("setStorageMode:", || o.setStorageMode(v));
        }
        OptionSet::HazardTrackingMode(hazard_tracking_mode) => {
            let v = MTLHazardTrackingMode(hazard_tracking_mode.to_usize());
            msg::get("setHazardTrackingMode:", || o.setHazardTrackingMode(v));
        }
    }
}

/// `setResourceOptions:` on a heap descriptor (an aliasing setter).
fn heap_resource_options(d: &MetalHeapDescriptor, resource_options: u64) {
    let (o, v) = (d.get(), MTLResourceOptions(resource_options.to_usize()));
    msg::get("setResourceOptions:", || o.setResourceOptions(v));
}

/// Makes `d` hold `desc`. Order: `setSize:`; `setType:`; the options up to the first
/// hazard-tracking mode; `setAddressRanges:`; the rest of the options; `setPinnedGPUAddress:`.
pub fn apply_heap(desc: &HeapDesc<'_, MetalPlatform>, d: &MetalHeapDescriptor) {
    let o = d.get();
    let size = desc.size.to_usize();
    msg::get("setSize:", || o.setSize(size));
    // `NSInteger` is `i64` on arm64: lossless.
    let heap_type = MTLHeapType(desc.heap_type as isize);
    msg::get("setType:", || o.setType(heap_type));
    let split = desc
        .options
        .iter()
        .position(|o| matches!(o, OptionSet::HazardTrackingMode(_)))
        .unwrap_or(desc.options.len());
    let (before, after) = desc.options.split_at(split);
    for option in before {
        heap_option(*option, d);
    }
    if let Some(ranges) = desc.address_ranges {
        let ranges = ranges.get();
        msg::get("setAddressRanges:", || o.setAddressRanges(Some(ranges)));
    }
    for option in after {
        heap_option(*option, d);
    }
    if let Some(address) = desc.pinned_gpu_address {
        msg::get("setPinnedGPUAddress:", || o.setPinnedGPUAddress(address));
    }
}

/// The descriptor's `resourceOptions`, as Metal reports them.
pub fn heap_descriptor_resource_options(d: &MetalHeapDescriptor) -> u64 {
    let o = d.get();
    msg::get("resourceOptions", || o.resourceOptions())
        .0
        .to_u64()
}

/// Makes `d` hold `desc`, each property sent when set. Order: min filter, mag filter, LOD
/// average, compare function, S, T and R address modes, mip filter, max anisotropy, normalized
/// coordinates, argument buffer support, reduction mode, LOD min and max clamps, border colour,
/// resource index.
pub fn apply_sampler(desc: &SamplerDesc, d: &MetalSamplerDescriptor) {
    let o = d.get();
    if let Some(v) = desc.min_filter {
        let v = MTLSamplerMinMagFilter(v.to_usize());
        msg::get("setMinFilter:", || o.setMinFilter(v));
    }
    if let Some(v) = desc.mag_filter {
        let v = MTLSamplerMinMagFilter(v.to_usize());
        msg::get("setMagFilter:", || o.setMagFilter(v));
    }
    if let Some(v) = desc.lod_average {
        msg::get("setLodAverage:", || o.setLodAverage(v));
    }
    if let Some(v) = desc.compare_function {
        let v = MTLCompareFunction(v.to_usize());
        msg::get("setCompareFunction:", || o.setCompareFunction(v));
    }
    if let Some(v) = desc.s_address_mode {
        let v = MTLSamplerAddressMode(v.to_usize());
        msg::get("setSAddressMode:", || o.setSAddressMode(v));
    }
    if let Some(v) = desc.t_address_mode {
        let v = MTLSamplerAddressMode(v.to_usize());
        msg::get("setTAddressMode:", || o.setTAddressMode(v));
    }
    if let Some(v) = desc.r_address_mode {
        let v = MTLSamplerAddressMode(v.to_usize());
        msg::get("setRAddressMode:", || o.setRAddressMode(v));
    }
    if let Some(v) = desc.mip_filter {
        let v = MTLSamplerMipFilter(v.to_usize());
        msg::get("setMipFilter:", || o.setMipFilter(v));
    }
    if let Some(v) = desc.max_anisotropy {
        let n = v.to_usize();
        msg::get("setMaxAnisotropy:", || o.setMaxAnisotropy(n));
    }
    if let Some(v) = desc.normalized_coordinates {
        msg::get("setNormalizedCoordinates:", || {
            o.setNormalizedCoordinates(v)
        });
    }
    if let Some(v) = desc.support_argument_buffers {
        msg::get("setSupportArgumentBuffers:", || {
            o.setSupportArgumentBuffers(v)
        });
    }
    if let Some(v) = desc.reduction_mode {
        let v = MTLSamplerReductionMode(v.to_usize());
        msg::get("setReductionMode:", || o.setReductionMode(v));
    }
    if let Some(v) = desc.lod_min_clamp {
        msg::get("setLodMinClamp:", || o.setLodMinClamp(v));
    }
    if let Some(v) = desc.lod_max_clamp {
        msg::get("setLodMaxClamp:", || o.setLodMaxClamp(v));
    }
    if let Some(v) = desc.border_color {
        let v = MTLSamplerBorderColor(v.to_usize());
        msg::get("setBorderColor:", || o.setBorderColor(v));
    }
    if let Some(v) = desc.resource_index {
        msg::get("setResourceIndex:", || {
            ResourceIndexDescriptorSpi::setResourceIndex(o, v)
        });
    }
}

/// Makes a face's stencil descriptor hold `s`. Order: compare function, stencil failure, depth
/// failure and pass operations, read mask, write mask.
fn apply_stencil(s: &StencilDesc, d: &MetalStencilDescriptor) {
    let o = d.get();
    let v = MTLCompareFunction(s.stencil_compare_function.to_usize());
    msg::get("setStencilCompareFunction:", || {
        o.setStencilCompareFunction(v)
    });
    let v = MTLStencilOperation(s.stencil_failure_operation.to_usize());
    msg::get("setStencilFailureOperation:", || {
        o.setStencilFailureOperation(v)
    });
    let v = MTLStencilOperation(s.depth_failure_operation.to_usize());
    msg::get("setDepthFailureOperation:", || {
        o.setDepthFailureOperation(v)
    });
    let v = MTLStencilOperation(s.depth_stencil_pass_operation.to_usize());
    msg::get("setDepthStencilPassOperation:", || {
        o.setDepthStencilPassOperation(v)
    });
    msg::get("setReadMask:", || o.setReadMask(s.read_mask));
    msg::get("setWriteMask:", || o.setWriteMask(s.write_mask));
}

/// Makes `d` hold `desc`. Order: depth compare function, depth write, then the front face's
/// stencil (`frontFaceStencil`, then its six properties), then the back face's; a face not set
/// is not asked for.
pub fn apply_depth_stencil(desc: &DepthStencilDesc, d: &MetalDepthStencilDescriptor) {
    let o = d.get();
    if let Some(v) = desc.depth_compare_function {
        let v = MTLCompareFunction(v.to_usize());
        msg::get("setDepthCompareFunction:", || o.setDepthCompareFunction(v));
    }
    if let Some(v) = desc.depth_write_enabled {
        msg::get("setDepthWriteEnabled:", || o.setDepthWriteEnabled(v));
    }
    if let Some(front) = &desc.front
        && let Some(face) = part("frontFaceStencil", || o.frontFaceStencil())
    {
        apply_stencil(front, &face);
    }
    if let Some(back) = &desc.back
        && let Some(face) = part("backFaceStencil", || o.backFaceStencil())
    {
        apply_stencil(back, &face);
    }
}

/// Makes `d` hold `desc`. Order: command types; the vertex, fragment, kernel, object and mesh
/// buffer bind counts; the kernel and object threadgroup memory bind counts; inherit pipeline
/// state and buffers; ray tracing and dynamic attribute stride support; the scissor rect and
/// viewport counts; inherit stencil reference values, depth test bounds, scissor rects,
/// viewports and blend colour; allow override render states; inherit depth-stencil state, depth
/// bias, depth clip mode, cull mode, front-facing winding and triangle fill mode; then the
/// resource index, when set.
pub fn apply_icb(desc: &IcbDesc, d: &MetalIcbDescriptor) {
    let o = d.get();
    let v = MTLIndirectCommandType(desc.command_types.to_usize());
    msg::get("setCommandTypes:", || o.setCommandTypes(v));
    let n = desc.max_vertex_buffer_bind_count.to_usize();
    msg::get("setMaxVertexBufferBindCount:", || {
        o.setMaxVertexBufferBindCount(n)
    });
    let n = desc.max_fragment_buffer_bind_count.to_usize();
    msg::get("setMaxFragmentBufferBindCount:", || {
        o.setMaxFragmentBufferBindCount(n)
    });
    let n = desc.max_kernel_buffer_bind_count.to_usize();
    msg::get("setMaxKernelBufferBindCount:", || {
        o.setMaxKernelBufferBindCount(n)
    });
    // SAFETY (the next four): objc2-metal marks these bind counts unsafe because Metal checks
    // them only when it makes the buffer.
    let n = desc.max_object_buffer_bind_count.to_usize();
    msg::get("setMaxObjectBufferBindCount:", || unsafe {
        o.setMaxObjectBufferBindCount(n)
    });
    let n = desc.max_mesh_buffer_bind_count.to_usize();
    msg::get("setMaxMeshBufferBindCount:", || unsafe {
        o.setMaxMeshBufferBindCount(n)
    });
    let n = desc.max_kernel_threadgroup_memory_bind_count.to_usize();
    msg::get("setMaxKernelThreadgroupMemoryBindCount:", || unsafe {
        o.setMaxKernelThreadgroupMemoryBindCount(n)
    });
    let n = desc.max_object_threadgroup_memory_bind_count.to_usize();
    msg::get("setMaxObjectThreadgroupMemoryBindCount:", || unsafe {
        o.setMaxObjectThreadgroupMemoryBindCount(n)
    });
    msg::get("setInheritPipelineState:", || {
        o.setInheritPipelineState(desc.inherit_pipeline_state)
    });
    msg::get("setInheritBuffers:", || {
        o.setInheritBuffers(desc.inherit_buffers)
    });
    msg::get("setSupportRayTracing:", || {
        o.setSupportRayTracing(desc.support_ray_tracing)
    });
    msg::get("setSupportDynamicAttributeStride:", || {
        o.setSupportDynamicAttributeStride(desc.support_dynamic_attribute_stride)
    });
    msg::get("setMaxScissorRectCount:", || {
        o.setMaxScissorRectCount(desc.max_scissor_rect_count)
    });
    msg::get("setMaxViewportCount:", || {
        o.setMaxViewportCount(desc.max_viewport_count)
    });
    msg::get("setInheritStencilReferenceValues:", || {
        o.setInheritStencilReferenceValues(desc.inherit_stencil_reference_values)
    });
    msg::get("setInheritDepthTestBounds:", || {
        o.setInheritDepthTestBounds(desc.inherit_depth_test_bounds)
    });
    msg::get("setInheritScissorRects:", || {
        o.setInheritScissorRects(desc.inherit_scissor_rects)
    });
    msg::get("setInheritViewports:", || {
        o.setInheritViewports(desc.inherit_viewports)
    });
    msg::get("setInheritBlendColor:", || {
        o.setInheritBlendColor(desc.inherit_blend_color)
    });
    msg::get("setAllowOverrideRenderStates:", || {
        o.setAllowOverrideRenderStates(desc.allow_override_render_states)
    });
    msg::get("setInheritDepthStencilState:", || {
        o.setInheritDepthStencilState(desc.inherit_depth_stencil_state)
    });
    msg::get("setInheritDepthBias:", || {
        o.setInheritDepthBias(desc.inherit_depth_bias)
    });
    msg::get("setInheritDepthClipMode:", || {
        o.setInheritDepthClipMode(desc.inherit_depth_clip_mode)
    });
    msg::get("setInheritCullMode:", || {
        o.setInheritCullMode(desc.inherit_cull_mode)
    });
    msg::get("setInheritFrontFacingWinding:", || {
        o.setInheritFrontFacingWinding(desc.inherit_front_facing_winding)
    });
    msg::get("setInheritTriangleFillMode:", || {
        o.setInheritTriangleFillMode(desc.inherit_triangle_fill_mode)
    });
    if let Some(v) = desc.resource_index {
        msg::get("setResourceIndex:", || {
            ResourceIndexDescriptorSpi::setResourceIndex(o, v)
        });
    }
}

/// Makes `d` hold `desc`: its dispatch type, when set.
pub fn apply_compute_pass(desc: &ComputePassDesc, d: &MetalComputePassDescriptor) {
    let o = d.get();
    if let Some(v) = desc.dispatch_type {
        let v = MTLDispatchType(v.to_usize());
        msg::get("setDispatchType:", || o.setDispatchType(v));
    }
}

/// Makes `d` hold `desc`. Order: label, vertex function, fragment function, then colour
/// attachment 0's pixel format (`colorAttachments`, then `objectAtIndexedSubscript:`, whose raise
/// dies at the send: no path reaches it under a handler) and, with `blend_over`, its blending,
/// then the resource index and its force flag.
pub fn apply_render_pipeline(
    desc: &RenderPipelineDesc<'_, MetalPlatform>,
    d: &MetalRenderPipelineDescriptor,
) {
    let o = d.get();
    if let Some(label) = desc.label {
        let label = msg::new_string(label);
        msg::get("setLabel:", || o.setLabel(Some(&label)));
    }
    let function = desc.vertex.get();
    msg::get("setVertexFunction:", || o.setVertexFunction(Some(function)));
    let function = desc.fragment.get();
    msg::get("setFragmentFunction:", || {
        o.setFragmentFunction(Some(function))
    });
    if let Some(array) = part("colorAttachments", || o.colorAttachments())
        && let Some(attachment) = msg::get("objectAtIndexedSubscript:", || {
            array.get().objectAtIndexedSubscript_or_nil(0)
        })
        .map(Obj::from_retained)
    {
        let (a, v) = (
            attachment.get(),
            MTLPixelFormat(desc.color0_pixel_format.to_usize()),
        );
        msg::get("setPixelFormat:", || a.setPixelFormat(v));
        if desc.blend_over {
            blend_over(a);
        }
    }
    if let Some((index, force)) = desc.resource_index {
        pipeline_resource_index(o, index, force);
    }
}

/// Has `a` blend each fragment, its colour premultiplied, over the target by its alpha
/// (`RenderPipelineDesc::blend_over`).
fn blend_over(a: &MTLRenderPipelineColorAttachmentDescriptor) {
    msg::get("setBlendingEnabled:", || a.setBlendingEnabled(true));
    msg::get("setSourceRGBBlendFactor:", || {
        a.setSourceRGBBlendFactor(MTLBlendFactor::One);
    });
    msg::get("setDestinationRGBBlendFactor:", || {
        a.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    });
    msg::get("setSourceAlphaBlendFactor:", || {
        a.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    });
    msg::get("setDestinationAlphaBlendFactor:", || {
        a.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    });
}

/// The highest index of a vertex or stage-input descriptor's layout and attribute arrays: Metal
/// raises past it, and the graphics lane refuses a larger one before it reaches here.
const LAST_BUFFER_INDEX: u64 = 30;

/// Applies the whole render pipeline value `desc` to the fresh descriptor `d`, in the value's
/// order ([`FullRenderPipelineDesc`]). Each raw value goes to Metal unvalidated; Metal checks it.
pub fn apply_full_render_pipeline(
    desc: &FullRenderPipelineDesc<'_, MetalPlatform>,
    d: &MetalRenderPipelineDescriptor,
) {
    let o = d.get();
    if let Some(label) = desc.label {
        let label = msg::new_string(label);
        msg::get("setLabel:", || o.setLabel(Some(&label)));
    }
    let function = desc.vertex.get();
    msg::get("setVertexFunction:", || o.setVertexFunction(Some(function)));
    if let Some(function) = desc.fragment {
        let function = function.get();
        msg::get("setFragmentFunction:", || {
            o.setFragmentFunction(Some(function))
        });
    }
    if let Some(v) = &desc.vertex_descriptor {
        let vd = Obj::from_retained(msg::get(
            "vertexDescriptor",
            MTLVertexDescriptor::vertexDescriptor,
        ));
        apply_vertex_descriptor(v, vd.get());
        msg::get("setVertexDescriptor:", || {
            o.setVertexDescriptor(Some(vd.get()))
        });
    }
    if let Some(array) = part("colorAttachments", || o.colorAttachments()) {
        for (index, c) in &desc.colors {
            if let Some(a) = msg::get("objectAtIndexedSubscript:", || {
                array.get().objectAtIndexedSubscript_or_nil(*index)
            })
            .map(Obj::from_retained)
            {
                apply_pipeline_color_attachment(c, a.get());
            }
        }
    }
    msg::get("setDepthAttachmentPixelFormat:", || {
        o.setDepthAttachmentPixelFormat(MTLPixelFormat(desc.depth_pixel_format.to_usize()));
    });
    msg::get("setStencilAttachmentPixelFormat:", || {
        o.setStencilAttachmentPixelFormat(MTLPixelFormat(desc.stencil_pixel_format.to_usize()));
    });
    msg::get("setRasterSampleCount:", || {
        o.setRasterSampleCount(desc.raster_sample_count.to_usize());
    });
    msg::get("setAlphaToCoverageEnabled:", || {
        o.setAlphaToCoverageEnabled(desc.alpha_to_coverage);
    });
    msg::get("setAlphaToOneEnabled:", || {
        o.setAlphaToOneEnabled(desc.alpha_to_one)
    });
    msg::get("setRasterizationEnabled:", || {
        o.setRasterizationEnabled(desc.rasterization_enabled);
    });
    // SAFETY: raw guest values go to Metal unvalidated; Metal checks them itself.
    msg::get("setInputPrimitiveTopology:", || unsafe {
        o.setInputPrimitiveTopology(MTLPrimitiveTopologyClass(
            desc.input_primitive_topology.to_usize(),
        ));
    });
    // SAFETY: as above.
    msg::get("setMaxVertexAmplificationCount:", || unsafe {
        o.setMaxVertexAmplificationCount(desc.max_vertex_amplification_count.to_usize());
    });
    msg::get("setSupportIndirectCommandBuffers:", || {
        o.setSupportIndirectCommandBuffers(desc.support_indirect_command_buffers);
    });
    if let Some(linked) = linked_functions(&desc.vertex_linked) {
        msg::get("setVertexLinkedFunctions:", || {
            o.setVertexLinkedFunctions(Some(&linked))
        });
    }
    if let Some(linked) = linked_functions(&desc.fragment_linked) {
        msg::get("setFragmentLinkedFunctions:", || {
            o.setFragmentLinkedFunctions(Some(&linked))
        });
    }
}

/// `l` as Metal's `MTLLinkedFunctions`, each list set where it holds a function, in the
/// properties' order (functions, binary functions, groups, private functions); `None` where it
/// links nothing, which leaves the stage's as Metal's descriptor has them.
fn linked_functions(
    l: &LinkedFunctionsDesc<MetalPlatform>,
) -> Option<Retained<MTLLinkedFunctions>> {
    if l.is_empty() {
        return None;
    }
    let array = |fs: &[MetalFunction]| {
        let fs: Vec<&ProtocolObject<dyn MTLFunction>> = fs.iter().map(|f| f.get()).collect();
        msg::value("arrayWithObjects:count:", || NSArray::from_slice(&fs))
    };
    let o = msg::get("linkedFunctions", MTLLinkedFunctions::linkedFunctions);
    if !l.functions.is_empty() {
        let functions = array(&l.functions);
        msg::get("setFunctions:", || o.setFunctions(Some(&functions)));
    }
    if !l.binary_functions.is_empty() {
        let functions = array(&l.binary_functions);
        msg::get("setBinaryFunctions:", || {
            o.setBinaryFunctions(Some(&functions))
        });
    }
    if !l.groups.is_empty() {
        let names: Vec<Retained<NSString>> = l
            .groups
            .iter()
            .map(|(name, _)| msg::new_string(name))
            .collect();
        let lists: Vec<_> = l.groups.iter().map(|(_, fs)| array(fs)).collect();
        let keys: Vec<&NSString> = names.iter().map(|n| &**n).collect();
        let values: Vec<&NSArray<ProtocolObject<dyn MTLFunction>>> =
            lists.iter().map(|a| &**a).collect();
        let groups = msg::value("dictionaryWithObjects:forKeys:count:", || {
            NSDictionary::from_slices(&keys, &values)
        });
        msg::get("setGroups:", || o.setGroups(Some(&groups)));
    }
    if !l.private_functions.is_empty() {
        let functions = array(&l.private_functions);
        msg::get("setPrivateFunctions:", || {
            o.setPrivateFunctions(Some(&functions))
        });
    }
    Some(o)
}

/// One colour attachment's every property, in `MTLRenderPipelineColorAttachmentDescriptor`'s
/// order.
fn apply_pipeline_color_attachment(
    c: &ColorAttachmentDesc,
    a: &MTLRenderPipelineColorAttachmentDescriptor,
) {
    msg::get("setPixelFormat:", || {
        a.setPixelFormat(MTLPixelFormat(c.pixel_format.to_usize()))
    });
    msg::get("setBlendingEnabled:", || {
        a.setBlendingEnabled(c.blending_enabled)
    });
    msg::get("setSourceRGBBlendFactor:", || {
        a.setSourceRGBBlendFactor(MTLBlendFactor(c.source_rgb_blend_factor.to_usize()));
    });
    msg::get("setDestinationRGBBlendFactor:", || {
        a.setDestinationRGBBlendFactor(MTLBlendFactor(c.destination_rgb_blend_factor.to_usize()));
    });
    msg::get("setRgbBlendOperation:", || {
        a.setRgbBlendOperation(MTLBlendOperation(c.rgb_blend_operation.to_usize()));
    });
    msg::get("setSourceAlphaBlendFactor:", || {
        a.setSourceAlphaBlendFactor(MTLBlendFactor(c.source_alpha_blend_factor.to_usize()));
    });
    msg::get("setDestinationAlphaBlendFactor:", || {
        a.setDestinationAlphaBlendFactor(MTLBlendFactor(
            c.destination_alpha_blend_factor.to_usize(),
        ));
    });
    msg::get("setAlphaBlendOperation:", || {
        a.setAlphaBlendOperation(MTLBlendOperation(c.alpha_blend_operation.to_usize()));
    });
    msg::get("setWriteMask:", || {
        a.setWriteMask(MTLColorWriteMask(c.write_mask.to_usize()))
    });
}

/// A vertex descriptor's layouts, then its attributes, onto `vd`.
fn apply_vertex_descriptor(v: &VertexDesc, vd: &MTLVertexDescriptor) {
    let layouts = Obj::from_retained(msg::get("layouts", || vd.layouts()));
    for l in v
        .layouts
        .iter()
        .filter(|l| l.buffer_index <= LAST_BUFFER_INDEX)
    {
        // SAFETY: the index is within the array (`LAST_BUFFER_INDEX`); raw guest values go to
        // Metal unvalidated, and Metal checks them itself.
        let layout = Obj::from_retained(msg::get("objectAtIndexedSubscript:", || unsafe {
            layouts
                .get()
                .objectAtIndexedSubscript(l.buffer_index.to_usize())
        }));
        let layout = layout.get();
        // SAFETY: as above.
        msg::get("setStride:", || unsafe {
            layout.setStride(l.stride.to_usize())
        });
        msg::get("setStepFunction:", || {
            layout.setStepFunction(MTLVertexStepFunction(l.step_function.to_usize()));
        });
        // SAFETY: as above.
        msg::get("setStepRate:", || unsafe {
            layout.setStepRate(l.step_rate.to_usize())
        });
    }
    let attributes = Obj::from_retained(msg::get("attributes", || vd.attributes()));
    for a in v
        .attributes
        .iter()
        .filter(|a| a.attribute_index <= LAST_BUFFER_INDEX)
    {
        // SAFETY: as above.
        let attribute = Obj::from_retained(msg::get("objectAtIndexedSubscript:", || unsafe {
            attributes
                .get()
                .objectAtIndexedSubscript(a.attribute_index.to_usize())
        }));
        let attribute = attribute.get();
        msg::get("setFormat:", || {
            attribute.setFormat(MTLVertexFormat(a.format.to_usize()))
        });
        // SAFETY: as above.
        msg::get("setOffset:", || unsafe {
            attribute.setOffset(a.offset.to_usize())
        });
        // SAFETY: as above.
        msg::get("setBufferIndex:", || unsafe {
            attribute.setBufferIndex(a.buffer_index.to_usize());
        });
    }
}

/// Applies the whole compute pipeline value `desc` to the fresh descriptor `d`, in the value's
/// order ([`FullComputePipelineDesc`]). Each raw value goes to Metal unvalidated; Metal checks it.
pub fn apply_full_compute_pipeline(
    desc: &FullComputePipelineDesc<'_, MetalPlatform>,
    d: &MetalComputePipelineDescriptor,
) {
    let o = d.get();
    if let Some(label) = desc.label {
        let label = msg::new_string(label);
        msg::get("setLabel:", || o.setLabel(Some(&label)));
    }
    let function = desc.function.get();
    msg::get("setComputeFunction:", || {
        o.setComputeFunction(Some(function))
    });
    // SAFETY: a plain flag.
    msg::get(
        "setThreadGroupSizeIsMultipleOfThreadExecutionWidth:",
        || unsafe {
            o.setThreadGroupSizeIsMultipleOfThreadExecutionWidth(
                desc.threadgroup_size_is_multiple_of_execution_width,
            );
        },
    );
    if desc.max_total_threads_per_threadgroup != 0 {
        msg::get("setMaxTotalThreadsPerThreadgroup:", || {
            o.setMaxTotalThreadsPerThreadgroup(desc.max_total_threads_per_threadgroup.to_usize());
        });
    }
    msg::get("setMaxCallStackDepth:", || {
        o.setMaxCallStackDepth(desc.max_call_stack_depth.to_usize());
    });
    msg::get("setSupportIndirectCommandBuffers:", || {
        o.setSupportIndirectCommandBuffers(desc.support_indirect_command_buffers);
    });
    if let Some(s) = &desc.stage_input {
        let sd = Obj::from_retained(msg::get(
            "stageInputOutputDescriptor",
            MTLStageInputOutputDescriptor::stageInputOutputDescriptor,
        ));
        apply_stage_input(s, sd.get());
        msg::get("setStageInputDescriptor:", || {
            o.setStageInputDescriptor(Some(sd.get()))
        });
    }
    if let Some(linked) = linked_functions(&desc.linked) {
        msg::get("setLinkedFunctions:", || {
            o.setLinkedFunctions(Some(&linked))
        });
    }
}

/// A stage-input descriptor's layouts, attributes, then index type and index buffer, onto `sd`.
fn apply_stage_input(s: &StageInputDesc, sd: &MTLStageInputOutputDescriptor) {
    let layouts = Obj::from_retained(msg::get("layouts", || sd.layouts()));
    for l in s
        .layouts
        .iter()
        .filter(|l| l.buffer_index <= LAST_BUFFER_INDEX)
    {
        // SAFETY: the index is within the array (`LAST_BUFFER_INDEX`); raw guest values go to
        // Metal unvalidated, and Metal checks them itself.
        let layout = Obj::from_retained(msg::get("objectAtIndexedSubscript:", || unsafe {
            layouts
                .get()
                .objectAtIndexedSubscript(l.buffer_index.to_usize())
        }));
        let layout = layout.get();
        msg::get("setStride:", || layout.setStride(l.stride.to_usize()));
        msg::get("setStepFunction:", || {
            layout.setStepFunction(MTLStepFunction(l.step_function.to_usize()));
        });
        msg::get("setStepRate:", || {
            layout.setStepRate(l.step_rate.to_usize())
        });
    }
    let attributes = Obj::from_retained(msg::get("attributes", || sd.attributes()));
    for a in s
        .attributes
        .iter()
        .filter(|a| a.attribute_index <= LAST_BUFFER_INDEX)
    {
        // SAFETY: as above.
        let attribute = Obj::from_retained(msg::get("objectAtIndexedSubscript:", || unsafe {
            attributes
                .get()
                .objectAtIndexedSubscript(a.attribute_index.to_usize())
        }));
        let attribute = attribute.get();
        msg::get("setFormat:", || {
            attribute.setFormat(MTLAttributeFormat(a.format.to_usize()))
        });
        msg::get("setOffset:", || attribute.setOffset(a.offset.to_usize()));
        // SAFETY: as above.
        msg::get("setBufferIndex:", || unsafe {
            attribute.setBufferIndex(a.buffer_index.to_usize());
        });
    }
    msg::get("setIndexType:", || {
        sd.setIndexType(MTLIndexType(s.index_type.to_usize()))
    });
    // SAFETY: as above.
    msg::get("setIndexBufferIndex:", || unsafe {
        sd.setIndexBufferIndex(s.index_buffer_index.to_usize());
    });
}

/// `setResourceIndex:` then `setForceResourceIndex:` on a render pipeline descriptor.
fn pipeline_resource_index(o: &MTLRenderPipelineDescriptor, index: u64, force: bool) {
    msg::get("setResourceIndex:", || {
        ResourceIndexDescriptorSpi::setResourceIndex(o, index)
    });
    msg::get("setForceResourceIndex:", || {
        ForceResourceIndexSpi::setForceResourceIndex(o, force)
    });
}

/// A texture descriptor holding `desc`: `+new`, then [`apply_texture`].
pub fn texture_descriptor(desc: &TextureDesc) -> MetalTextureDescriptor {
    let d = crate::platform::class_new(MTLTextureDescriptor::new);
    apply_texture(desc, &d);
    d
}

/// `setPixelFormat:` with a raw value, which Metal checks here (an ordered setter).
fn texture_pixel_format(d: &MetalTextureDescriptor, pixel_format: u64) {
    let (o, v) = (d.get(), MTLPixelFormat(pixel_format.to_usize()));
    msg::get("setPixelFormat:", || o.setPixelFormat(v));
}

/// `setResourceOptions:` on a texture descriptor (an aliasing setter).
fn texture_resource_options(d: &MetalTextureDescriptor, resource_options: u64) {
    let (o, v) = (d.get(), MTLResourceOptions(resource_options.to_usize()));
    msg::get("setResourceOptions:", || o.setResourceOptions(v));
}

/// Makes `d` hold `desc`, the first nine fields always and the rest when set. Order: texture
/// type, framebuffer-only, drawable, GPU-optimized contents, width, height, depth, mipmap level
/// count, sample count, array length, usage, pixel format, resource options, protection
/// options, write swizzle, swizzle, resource index and its force flag (`Serialized::build`'s, in
/// `resources/manager/texture.rs`), then rotation.
pub fn apply_texture(desc: &TextureDesc, d: &MetalTextureDescriptor) {
    let o = d.get();
    let v = MTLTextureType(desc.texture_type.to_usize());
    msg::get("setTextureType:", || o.setTextureType(v));
    if let Some(v) = desc.framebuffer_only {
        msg::get("setFramebufferOnly:", || o.setFramebufferOnly(v));
    }
    if let Some(v) = desc.is_drawable {
        msg::get("setIsDrawable:", || o.setIsDrawable(v));
    }
    if let Some(v) = desc.allow_gpu_optimized_contents {
        msg::get("setAllowGPUOptimizedContents:", || {
            o.setAllowGPUOptimizedContents(v)
        });
    }
    // SAFETY (the size setters): objc2-metal marks them unsafe because Metal does not check
    // them here; the texture's creator does.
    let n = desc.width.to_usize();
    msg::get("setWidth:", || unsafe { o.setWidth(n) });
    let n = desc.height.to_usize();
    msg::get("setHeight:", || unsafe { o.setHeight(n) });
    let n = desc.depth.to_usize();
    msg::get("setDepth:", || unsafe { o.setDepth(n) });
    let n = desc.mipmap_level_count.to_usize();
    msg::get("setMipmapLevelCount:", || unsafe {
        o.setMipmapLevelCount(n)
    });
    let n = desc.sample_count.to_usize();
    msg::get("setSampleCount:", || unsafe { o.setSampleCount(n) });
    let n = desc.array_length.to_usize();
    msg::get("setArrayLength:", || unsafe { o.setArrayLength(n) });
    let v = MTLTextureUsage(desc.usage.to_usize());
    msg::get("setUsage:", || o.setUsage(v));
    texture_pixel_format(d, desc.pixel_format);
    if let Some(v) = desc.resource_options {
        texture_resource_options(d, v);
    }
    if let Some(v) = desc.protection_options {
        msg::get("setProtectionOptions:", || o.setProtectionOptions(v));
    }
    if let Some(v) = desc.write_swizzle_enabled {
        msg::get("setWriteSwizzleEnabled:", || o.setWriteSwizzleEnabled(v));
    }
    if let Some(v) = desc.swizzle {
        let v = v.mtl();
        msg::get("setSwizzle:", || o.setSwizzle(v));
    }
    if let Some((index, force)) = desc.resource_index {
        msg::get("setResourceIndex:", || {
            TextureDescriptorSpi::setResourceIndex(o, index)
        });
        msg::get("setForceResourceIndex:", || {
            TextureDescriptorSpi::setForceResourceIndex(o, force)
        });
    }
    if let Some(v) = desc.rotation {
        msg::get("setRotation:", || o.setRotation(v));
    }
}

/// Makes a render pass descriptor hold `set`, each property sent when set. Order: visibility
/// result buffer and type, render target array length, width and height, default raster and
/// colour sample counts, rasterization rate map, imageblock sample length, threadgroup memory
/// length, tile width and height. The type's setter is macOS 26's: only a type other than
/// `Reset`, every descriptor's own, is sent, and only where the descriptor answers the setter,
/// so a host before macOS 26 resets each pass's results, as it does whatever the type.
fn apply_pass(set: &PassSet<'_, MetalPlatform>, d: &MetalRenderPassDescriptor) {
    let o = d.get();
    if let Some(buffer) = set.visibility_result_buffer {
        let buffer = buffer.map(Obj::get);
        msg::get("setVisibilityResultBuffer:", || {
            o.setVisibilityResultBuffer(buffer)
        });
    }
    if let Some(t) = set.visibility_result_type.filter(|&t| t != 0)
        && msg::get("respondsToSelector:", || {
            objc2_foundation::NSObjectProtocol::respondsToSelector(
                o,
                objc2::sel!(setVisibilityResultType:),
            )
        })
    {
        let t = MTLVisibilityResultType(isize::try_from(t).unwrap_or(isize::MAX));
        msg::get("setVisibilityResultType:", || o.setVisibilityResultType(t));
    }
    if let Some(v) = set.render_target_array_length {
        let n = v.to_usize();
        msg::get("setRenderTargetArrayLength:", || {
            o.setRenderTargetArrayLength(n)
        });
    }
    if let Some(v) = set.render_target_width {
        let n = v.to_usize();
        msg::get("setRenderTargetWidth:", || o.setRenderTargetWidth(n));
    }
    if let Some(v) = set.render_target_height {
        let n = v.to_usize();
        msg::get("setRenderTargetHeight:", || o.setRenderTargetHeight(n));
    }
    if let Some(v) = set.default_raster_sample_count {
        let n = v.to_usize();
        msg::get("setDefaultRasterSampleCount:", || {
            o.setDefaultRasterSampleCount(n)
        });
    }
    if let Some(v) = set.default_color_sample_count {
        msg::get("setDefaultColorSampleCount:", || {
            o.setDefaultColorSampleCount(v)
        });
    }
    if let Some(map) = set.rasterization_rate_map {
        let map = map.map(Obj::get);
        msg::get("setRasterizationRateMap:", || {
            o.setRasterizationRateMap(map)
        });
    }
    if let Some(v) = set.imageblock_sample_length {
        let n = v.to_usize();
        msg::get("setImageblockSampleLength:", || {
            o.setImageblockSampleLength(n)
        });
    }
    if let Some(v) = set.threadgroup_memory_length {
        let n = v.to_usize();
        msg::get("setThreadgroupMemoryLength:", || {
            o.setThreadgroupMemoryLength(n)
        });
    }
    if let Some(v) = set.tile_width {
        let n = v.to_usize();
        msg::get("setTileWidth:", || o.setTileWidth(n));
    }
    if let Some(v) = set.tile_height {
        let n = v.to_usize();
        msg::get("setTileHeight:", || o.setTileHeight(n));
    }
}

/// One property every render pass attachment shares (`MTLRenderPassAttachmentDescriptor`), set
/// on `$d` with the selector `$sel`.
macro_rules! attachment_prop {
    ($d:ident, $sel:literal, texture, $v:expr) => {{
        let texture = $v.map(Obj::get);
        msg::get($sel, || $d.setTexture(texture))
    }};
    ($d:ident, $sel:literal, resolve_texture, $v:expr) => {{
        let texture = $v.map(Obj::get);
        msg::get($sel, || $d.setResolveTexture(texture))
    }};
    ($d:ident, $sel:literal, load_action, $v:expr) => {{
        let v = MTLLoadAction($v.to_usize());
        msg::get($sel, || $d.setLoadAction(v))
    }};
    ($d:ident, $sel:literal, store_action, $v:expr) => {{
        let v = MTLStoreAction($v.to_usize());
        msg::get($sel, || $d.setStoreAction(v))
    }};
    ($d:ident, $sel:literal, store_action_options, $v:expr) => {{
        let v = MTLStoreActionOptions($v.to_usize());
        msg::get($sel, || $d.setStoreActionOptions(v))
    }};
    ($d:ident, $sel:literal, $setter:ident, $v:expr) => {{
        let n = $v.to_usize();
        msg::get($sel, || $d.$setter(n))
    }};
}

/// Declares the function that makes one kind of render pass attachment hold its `*Set`: the
/// properties every attachment shares, then its own. Order: texture, resolve texture, level,
/// slice, depth plane, resolve level, slice and depth plane, load and store actions, store
/// action options, then the attachment's own.
macro_rules! apply_attachment {
    (
        $(#[$m:meta])* $fn:ident($set:ident, $handle:ty) |$s:ident, $o:ident| { $($own:tt)* }
    ) => {
        $(#[$m])*
        fn $fn($s: &$set<'_, MetalPlatform>, a: &$handle) {
            let $o = a.get();
            if let Some(v) = $s.texture {
                attachment_prop!($o, "setTexture:", texture, v);
            }
            if let Some(v) = $s.resolve_texture {
                attachment_prop!($o, "setResolveTexture:", resolve_texture, v);
            }
            if let Some(v) = $s.level {
                attachment_prop!($o, "setLevel:", setLevel, v);
            }
            if let Some(v) = $s.slice {
                attachment_prop!($o, "setSlice:", setSlice, v);
            }
            if let Some(v) = $s.depth_plane {
                attachment_prop!($o, "setDepthPlane:", setDepthPlane, v);
            }
            if let Some(v) = $s.resolve_level {
                attachment_prop!($o, "setResolveLevel:", setResolveLevel, v);
            }
            if let Some(v) = $s.resolve_slice {
                attachment_prop!($o, "setResolveSlice:", setResolveSlice, v);
            }
            if let Some(v) = $s.resolve_depth_plane {
                attachment_prop!($o, "setResolveDepthPlane:", setResolveDepthPlane, v);
            }
            if let Some(v) = $s.load_action {
                attachment_prop!($o, "setLoadAction:", load_action, v);
            }
            if let Some(v) = $s.store_action {
                attachment_prop!($o, "setStoreAction:", store_action, v);
            }
            if let Some(v) = $s.store_action_options {
                attachment_prop!($o, "setStoreActionOptions:", store_action_options, v);
            }
            $($own)*
        }
    };
}

apply_attachment!(
    /// A colour attachment's set; its own: the clear colour.
    apply_color_attachment(ColorAttachmentSet, MetalRenderPassColorAttachment)
        | set,
    o | {
        if let Some(v) = set.clear_color {
            let v = v.mtl();
            msg::get("setClearColor:", || o.setClearColor(v));
        }
    }
);

apply_attachment!(
    /// The depth attachment's set; its own: the clear depth, then the resolve filter.
    apply_depth_attachment(DepthAttachmentSet, MetalRenderPassDepthAttachment)
        | set,
    o | {
        if let Some(v) = set.clear_depth {
            msg::get("setClearDepth:", || o.setClearDepth(v));
        }
        if let Some(v) = set.depth_resolve_filter {
            let v = MTLMultisampleDepthResolveFilter(v.to_usize());
            msg::get("setDepthResolveFilter:", || o.setDepthResolveFilter(v));
        }
    }
);

apply_attachment!(
    /// The stencil attachment's set; its own: the clear stencil, then the resolve filter.
    apply_stencil_attachment(StencilAttachmentSet, MetalRenderPassStencilAttachment)
        | set,
    o | {
        if let Some(v) = set.clear_stencil {
            msg::get("setClearStencil:", || o.setClearStencil(v));
        }
        if let Some(v) = set.stencil_resolve_filter {
            let v = MTLMultisampleStencilResolveFilter(v.to_usize());
            msg::get("setStencilResolveFilter:", || o.setStencilResolveFilter(v));
        }
    }
);

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use objc2_metal::{
        MTLComputePassDescriptor, MTLComputePipelineDescriptor, MTLDepthStencilDescriptor,
        MTLHeapDescriptor, MTLIndirectCommandBufferDescriptor, MTLRenderPassDescriptor,
        MTLRenderPipelineDescriptor, MTLSamplerDescriptor, MTLTextureDescriptor,
    };

    use objc2::runtime::AnyObject;
    use reims_vgpu_core::platform::descriptor::{
        ComputePipelineDescriptor as _, RenderPassDescriptor as _, RenderPipelineDescriptor as _,
        TextureDescriptor as _,
    };
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::{ClearColor, Swizzle};
    use reims_vgpu_core::platform::{Handle as _, HostRange, Platform as _};
    use reims_vgpu_core::sync::Mutex;

    use super::*;
    use crate::gpu::resource::testkit::*;
    use crate::platform::tests::platform;

    /// The object `getter` answers on `o` (`+0`), for reading a sub-descriptor back.
    fn part_of<'a>(o: &'a AnyObject, getter: &str) -> &'a AnyObject {
        let p = raw::<*mut AnyObject>(o, getter);
        assert!(!p.is_null(), "{getter}");
        // SAFETY: a non-nil object `o` owns and keeps while `o` lives.
        unsafe { &*p }
    }

    /// A buffer value reaches the private descriptor: each property with a getter reads back
    #[test]
    fn buffer_value_reaches_the_descriptor() {
        let x: MetalBufferDescriptor = Obj::from_retained(MTLBufferDescriptor::new());
        apply_buffer(
            BufferDesc {
                length: 2,
                options: vec![OptionSet::StorageMode(2)],
                pinned_gpu_address: Some(2),
                contents: None,
                no_copy: true,
                deallocator: None,
            },
            &x,
        );
        let o = obj(&x);
        readback(o, "setLength:", 2u64);
        readback(o, "setNoCopy:", true);
        readback(o, "setPinnedGPUAddress:", 2u64);
        readback(o, "setStorageMode:", 2u64);
        assert_eq!(raw_u64(o, "resourceOptions"), 0x20);
        // The options apply in order, the last one winning; an or reads the options first.
        apply_buffer(
            BufferDesc {
                length: 2,
                options: vec![
                    OptionSet::StorageMode(2),
                    OptionSet::ResourceOptions(0x1),
                    OptionSet::ResourceOptionsOr(0x10),
                ],
                pinned_gpu_address: None,
                contents: None,
                no_copy: false,
                deallocator: None,
            },
            &x,
        );
        assert_eq!(raw_u64(o, "resourceOptions"), 0x11);
    }

    /// A heap value reaches the descriptor, and the descriptor's options read back as Metal holds
    /// them
    #[test]
    fn heap_value_reaches_the_descriptor() {
        let k = kit();
        let x: MetalHeapDescriptor = Obj::from_retained(MTLHeapDescriptor::new());
        apply_heap(
            &HeapDesc {
                size: 65536,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(2)],
                address_ranges: Some(&k.ranges),
                pinned_gpu_address: Some(2),
            },
            &x,
        );
        let o = obj(&x);
        readback(o, "setSize:", 65536u64);
        readback(o, "setStorageMode:", 2u64);
        readback(o, "setHazardTrackingMode:", 2u64);
        readback(o, "setPinnedGPUAddress:", 2u64);
        assert_eq!(
            heap_descriptor_resource_options(&x),
            raw_u64(o, "resourceOptions")
        );
    }

    /// A sampler value reaches the descriptor, every property read back
    #[test]
    fn sampler_value_reaches_the_descriptor() {
        let x: MetalSamplerDescriptor = Obj::from_retained(MTLSamplerDescriptor::new());
        apply_sampler(
            &SamplerDesc {
                min_filter: Some(1),
                mag_filter: Some(1),
                lod_average: Some(true),
                compare_function: Some(3),
                s_address_mode: Some(2),
                t_address_mode: Some(2),
                r_address_mode: Some(2),
                mip_filter: Some(2),
                max_anisotropy: Some(2),
                normalized_coordinates: Some(true),
                support_argument_buffers: Some(true),
                reduction_mode: Some(1),
                lod_min_clamp: Some(1.5),
                lod_max_clamp: Some(2.5),
                border_color: Some(1),
                resource_index: Some(2),
            },
            &x,
        );
        let o = obj(&x);
        for (setter, v) in [
            ("setMinFilter:", 1u64),
            ("setMagFilter:", 1),
            ("setCompareFunction:", 3),
            ("setSAddressMode:", 2),
            ("setTAddressMode:", 2),
            ("setRAddressMode:", 2),
            ("setMipFilter:", 2),
            ("setMaxAnisotropy:", 2),
            ("setReductionMode:", 1),
            ("setBorderColor:", 1),
            ("setResourceIndex:", 2),
        ] {
            readback(o, setter, v);
        }
        readback(o, "setLodAverage:", true);
        readback(o, "setNormalizedCoordinates:", true);
        readback(o, "setSupportArgumentBuffers:", true);
        readback(o, "setLodMinClamp:", 1.5f32);
        readback(o, "setLodMaxClamp:", 2.5f32);
    }

    /// A depth-stencil value reaches the descriptor and both faces' stencil descriptors
    #[test]
    fn depth_stencil_value_reaches_the_descriptor() {
        let x: MetalDepthStencilDescriptor = Obj::from_retained(MTLDepthStencilDescriptor::new());
        let face = |n: u64, mask: u32| StencilDesc {
            stencil_compare_function: n + 1,
            stencil_failure_operation: n,
            depth_failure_operation: n,
            depth_stencil_pass_operation: n,
            read_mask: mask,
            write_mask: mask >> 1,
        };
        apply_depth_stencil(
            &DepthStencilDesc {
                depth_compare_function: Some(3),
                depth_write_enabled: Some(true),
                front: Some(face(2, 0x7f)),
                back: Some(face(3, 0x3f)),
            },
            &x,
        );
        let o = obj(&x);
        readback(o, "setDepthCompareFunction:", 3u64);
        readback(o, "setDepthWriteEnabled:", true);
        for (getter, n, mask) in [
            ("frontFaceStencil", 2u64, 0x7fu32),
            ("backFaceStencil", 3, 0x3f),
        ] {
            let s = part_of(o, getter);
            readback(s, "setStencilCompareFunction:", n + 1);
            readback(s, "setStencilFailureOperation:", n);
            readback(s, "setDepthFailureOperation:", n);
            readback(s, "setDepthStencilPassOperation:", n);
            readback(s, "setReadMask:", mask);
            readback(s, "setWriteMask:", mask >> 1);
        }
    }

    /// An ICB value reaches the descriptor, every property with a getter read back
    #[test]
    fn icb_value_reaches_the_descriptor() {
        let x: MetalIcbDescriptor = Obj::from_retained(MTLIndirectCommandBufferDescriptor::new());
        apply_icb(
            &IcbDesc {
                command_types: 1,
                max_vertex_buffer_bind_count: 2,
                max_fragment_buffer_bind_count: 2,
                max_kernel_buffer_bind_count: 2,
                max_object_buffer_bind_count: 2,
                max_mesh_buffer_bind_count: 2,
                max_kernel_threadgroup_memory_bind_count: 2,
                max_object_threadgroup_memory_bind_count: 2,
                inherit_pipeline_state: true,
                inherit_buffers: true,
                support_ray_tracing: true,
                support_dynamic_attribute_stride: true,
                max_scissor_rect_count: 2,
                max_viewport_count: 2,
                inherit_stencil_reference_values: true,
                inherit_depth_test_bounds: true,
                inherit_scissor_rects: true,
                inherit_viewports: true,
                inherit_blend_color: true,
                allow_override_render_states: 0,
                inherit_depth_stencil_state: true,
                inherit_depth_bias: true,
                inherit_depth_clip_mode: true,
                inherit_cull_mode: true,
                inherit_front_facing_winding: true,
                inherit_triangle_fill_mode: true,
                resource_index: Some(2),
            },
            &x,
        );
        let o = obj(&x);
        for (setter, v) in [
            ("setCommandTypes:", 1u64),
            ("setMaxVertexBufferBindCount:", 2),
            ("setMaxFragmentBufferBindCount:", 2),
            ("setMaxKernelBufferBindCount:", 2),
            ("setMaxObjectBufferBindCount:", 2),
            ("setMaxMeshBufferBindCount:", 2),
            ("setMaxKernelThreadgroupMemoryBindCount:", 2),
            ("setMaxObjectThreadgroupMemoryBindCount:", 2),
            ("setMaxScissorRectCount:", 2),
            ("setMaxViewportCount:", 2),
            ("setResourceIndex:", 2),
        ] {
            readback(o, setter, v);
        }
        for setter in [
            "setInheritPipelineState:",
            "setInheritBuffers:",
            "setSupportRayTracing:",
            "setSupportDynamicAttributeStride:",
            "setInheritStencilReferenceValues:",
            "setInheritDepthTestBounds:",
            "setInheritScissorRects:",
            "setInheritViewports:",
            "setInheritBlendColor:",
            "setInheritDepthStencilState:",
            "setInheritDepthBias:",
            "setInheritDepthClipMode:",
            "setInheritCullMode:",
            "setInheritFrontFacingWinding:",
            "setInheritTriangleFillMode:",
        ] {
            readback(o, setter, true);
        }
    }

    /// A compute pass value reaches the descriptor
    #[test]
    fn compute_pass_value_reaches_the_descriptor() {
        let x: MetalComputePassDescriptor = Obj::from_retained(MTLComputePassDescriptor::new());
        apply_compute_pass(
            &ComputePassDesc {
                dispatch_type: Some(1),
            },
            &x,
        );
        assert!(readback(obj(&x), "setDispatchType:", 1u64));
    }

    /// A render pipeline value reaches the descriptor and its first colour attachment
    #[test]
    fn render_pipeline_value_reaches_the_descriptor() {
        let k = kit();
        let x: MetalRenderPipelineDescriptor =
            Obj::from_retained(MTLRenderPipelineDescriptor::new());
        apply_render_pipeline(
            &RenderPipelineDesc {
                label: Some("label"),
                vertex: &k.function,
                fragment: &k.function,
                color0_pixel_format: 80,
                blend_over: false,
                resource_index: Some((2, true)),
            },
            &x,
        );
        let o = obj(&x);
        readback(o, "setResourceIndex:", 2u64);
        readback(o, "setForceResourceIndex:", true);
        assert_eq!(
            raw::<*mut AnyObject>(o, "vertexFunction").cast_const(),
            std::ptr::from_ref(obj(&k.function))
        );
        assert_eq!(
            raw::<*mut AnyObject>(o, "fragmentFunction").cast_const(),
            std::ptr::from_ref(obj(&k.function))
        );
        let a = x.get().colorAttachments();
        // SAFETY: index 0 of the colour attachments.
        let c = unsafe { a.objectAtIndexedSubscript(0) };
        assert_eq!(c.pixelFormat(), MTLPixelFormat(80));
        assert!(!c.isBlendingEnabled());
    }

    /// `blend_over` has colour attachment 0 blend the fragment, premultiplied, over by its alpha
    #[test]
    fn a_blended_pipeline_value_blends_over_by_alpha() {
        let k = kit();
        let x: MetalRenderPipelineDescriptor =
            Obj::from_retained(MTLRenderPipelineDescriptor::new());
        apply_render_pipeline(
            &RenderPipelineDesc {
                label: None,
                vertex: &k.function,
                fragment: &k.function,
                color0_pixel_format: 80,
                blend_over: true,
                resource_index: None,
            },
            &x,
        );
        let a = x.get().colorAttachments();
        // SAFETY: index 0 of the colour attachments.
        let c = unsafe { a.objectAtIndexedSubscript(0) };
        assert_eq!(c.pixelFormat(), MTLPixelFormat(80));
        assert!(c.isBlendingEnabled());
        assert_eq!(c.sourceRGBBlendFactor(), MTLBlendFactor::One);
        assert_eq!(
            c.destinationRGBBlendFactor(),
            MTLBlendFactor::OneMinusSourceAlpha
        );
        assert_eq!(c.sourceAlphaBlendFactor(), MTLBlendFactor::One);
        assert_eq!(
            c.destinationAlphaBlendFactor(),
            MTLBlendFactor::OneMinusSourceAlpha
        );
    }

    /// The whole render value reaches the descriptor: every colour attachment it sets with its
    /// blend state and write mask, the vertex descriptor's layouts and attributes, the depth and
    /// stencil formats, the sample count, the flags, the topology, the amplification count and
    /// each stage's linked functions in their lists; with no fragment function the descriptor's
    /// is nil
    #[test]
    fn a_whole_render_value_reaches_the_descriptor() {
        use reims_vgpu_core::platform::values::{AttributeDesc, BufferLayoutDesc};
        let k = kit();
        let x: MetalRenderPipelineDescriptor =
            Obj::from_retained(MTLRenderPipelineDescriptor::new());
        let blended = ColorAttachmentDesc {
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
        apply_full_render_pipeline(
            &FullRenderPipelineDesc {
                label: Some("whole"),
                vertex: &k.function,
                fragment: None,
                vertex_descriptor: Some(VertexDesc {
                    layouts: vec![BufferLayoutDesc {
                        buffer_index: 3,
                        stride: 16,
                        step_function: 2,
                        step_rate: 4,
                    }],
                    attributes: vec![AttributeDesc {
                        attribute_index: 5,
                        format: 31,
                        offset: 4,
                        buffer_index: 3,
                    }],
                }),
                colors: vec![(2, blended)],
                depth_pixel_format: 252,
                stencil_pixel_format: 253,
                raster_sample_count: 4,
                alpha_to_coverage: true,
                alpha_to_one: true,
                rasterization_enabled: false,
                input_primitive_topology: 3,
                max_vertex_amplification_count: 2,
                support_indirect_command_buffers: true,
                vertex_linked: LinkedFunctionsDesc {
                    functions: vec![k.function.clone_ref()],
                    groups: vec![(
                        "g".to_owned(),
                        vec![k.function.clone_ref(), k.function.clone_ref()],
                    )],
                    ..LinkedFunctionsDesc::default()
                },
                fragment_linked: LinkedFunctionsDesc {
                    private_functions: vec![k.function.clone_ref()],
                    ..LinkedFunctionsDesc::default()
                },
            },
            &x,
        );
        let o = x.get();
        assert!(o.fragmentFunction().is_none());
        let (v, f) = (o.vertexLinkedFunctions(), o.fragmentLinkedFunctions());
        let count = |a: Option<Retained<NSArray<ProtocolObject<dyn MTLFunction>>>>| {
            a.map_or(0, |a| a.count())
        };
        assert_eq!(count(v.functions()), 1);
        assert_eq!(count(v.privateFunctions()), 0);
        let groups = v.groups().unwrap();
        assert_eq!(groups.count(), 1);
        assert_eq!(count(groups.objectForKey(&NSString::from_str("g"))), 2);
        assert_eq!(count(f.privateFunctions()), 1);
        assert_eq!(count(f.functions()), 0);
        // SAFETY: index 2 of the colour attachments.
        let c = unsafe { o.colorAttachments().objectAtIndexedSubscript(2) };
        assert_eq!(c.pixelFormat(), MTLPixelFormat(115));
        assert!(c.isBlendingEnabled());
        assert_eq!(c.sourceRGBBlendFactor(), MTLBlendFactor(4));
        assert_eq!(c.destinationRGBBlendFactor(), MTLBlendFactor(5));
        assert_eq!(c.rgbBlendOperation(), MTLBlendOperation(1));
        assert_eq!(c.sourceAlphaBlendFactor(), MTLBlendFactor(2));
        assert_eq!(c.destinationAlphaBlendFactor(), MTLBlendFactor(7));
        assert_eq!(c.alphaBlendOperation(), MTLBlendOperation(3));
        assert_eq!(c.writeMask(), MTLColorWriteMask(9));
        let vd = o.vertexDescriptor().unwrap();
        // SAFETY: indices within the arrays.
        let (l, a) = unsafe {
            (
                vd.layouts().objectAtIndexedSubscript(3),
                vd.attributes().objectAtIndexedSubscript(5),
            )
        };
        assert_eq!(
            (l.stride(), l.stepFunction(), l.stepRate()),
            (16, MTLVertexStepFunction(2), 4)
        );
        assert_eq!(
            (a.format(), a.offset(), a.bufferIndex()),
            (MTLVertexFormat(31), 4, 3)
        );
        assert_eq!(o.depthAttachmentPixelFormat(), MTLPixelFormat(252));
        assert_eq!(o.stencilAttachmentPixelFormat(), MTLPixelFormat(253));
        assert_eq!(o.rasterSampleCount(), 4);
        assert!(o.isAlphaToCoverageEnabled() && o.isAlphaToOneEnabled());
        assert!(!o.isRasterizationEnabled());
        assert_eq!(o.inputPrimitiveTopology(), MTLPrimitiveTopologyClass(3));
        assert_eq!(o.maxVertexAmplificationCount(), 2);
        assert!(o.supportIndirectCommandBuffers());
    }

    /// The whole compute value reaches the descriptor: the threadgroup-size multiple, the maximum
    /// threads per threadgroup, the maximum call stack depth, indirect command buffer support, the
    /// stage-input descriptor's layouts, attributes, index type and index buffer, and the linked
    /// functions in their lists
    #[test]
    fn a_whole_compute_value_reaches_the_descriptor() {
        use reims_vgpu_core::platform::values::{AttributeDesc, BufferLayoutDesc};
        let k = kit();
        let x: MetalComputePipelineDescriptor =
            Obj::from_retained(MTLComputePipelineDescriptor::new());
        apply_full_compute_pipeline(
            &FullComputePipelineDesc {
                label: Some("whole"),
                function: &k.function,
                threadgroup_size_is_multiple_of_execution_width: true,
                max_total_threads_per_threadgroup: 128,
                max_call_stack_depth: 3,
                support_indirect_command_buffers: true,
                stage_input: Some(StageInputDesc {
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
                    index_buffer_index: 2,
                }),
                linked: LinkedFunctionsDesc {
                    binary_functions: vec![k.function.clone_ref()],
                    ..LinkedFunctionsDesc::default()
                },
            },
            &x,
        );
        let o = x.get();
        assert!(o.threadGroupSizeIsMultipleOfThreadExecutionWidth());
        assert_eq!(o.maxTotalThreadsPerThreadgroup(), 128);
        assert_eq!(o.maxCallStackDepth(), 3);
        assert!(o.supportIndirectCommandBuffers());
        let sd = o.stageInputDescriptor().unwrap();
        // SAFETY: indices within the arrays.
        let (l, a) = unsafe {
            (
                sd.layouts().objectAtIndexedSubscript(1),
                sd.attributes().objectAtIndexedSubscript(0),
            )
        };
        assert_eq!(
            (l.stride(), l.stepFunction(), l.stepRate()),
            (4, MTLStepFunction(5), 1)
        );
        assert_eq!(
            (a.format(), a.offset(), a.bufferIndex()),
            (MTLAttributeFormat(36), 0, 1)
        );
        assert_eq!(
            (sd.indexType(), sd.indexBufferIndex()),
            (MTLIndexType(1), 2)
        );
        let linked = o.linkedFunctions().unwrap();
        assert_eq!(linked.binaryFunctions().map(|a| a.count()), Some(1));
        assert!(linked.functions().is_none_or(|a| a.count() == 0));
    }

    /// setContents: and setDeallocator: (SPI) on the private buffer descriptor: the buffer Metal
    /// makes uses the caller's memory, and Metal calls the deallocator once, with the contents
    /// and length, when the buffer is released
    #[test]
    fn no_copy_contents_and_deallocator() {
        let k = kit();
        let page = 16384usize;
        let mut mem = vec![0u8; page * 2];
        let base = mem.as_mut_ptr();
        let aligned = base.wrapping_add(base.align_offset(page));
        // SAFETY: `aligned..aligned+page` lies in `mem`, which outlives the buffer (dropped
        // below, before `mem`), as a guest mapping outlives its buffer.
        let contents = unsafe { HostRange::new(HostPtr::from_ptr(aligned), page.to_u64()) };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let b = k
            .device
            .new_buffer(BufferDesc {
                length: page.to_u64(),
                options: vec![],
                pinned_gpu_address: None,
                contents: Some(contents),
                no_copy: true,
                deallocator: Some(Box::new(move |p, len| s2.lock().push((p, len)))),
            })
            .unwrap()
            .unwrap();
        assert_eq!(b.contents(), HostPtr::from_ptr(aligned));
        drop(b);
        for _ in 0..500 {
            if !seen.lock().is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(
            *seen.lock(),
            vec![(HostPtr::from_ptr(aligned), page.to_u64())]
        );
        drop(mem);
    }

    /// Metal calls a descriptor's deallocator after the buffer's release, on a thread of its own:
    /// the deallocator runs marked as ours there, so a panic in it aborts before it unwinds into
    /// Metal's frames
    #[test]
    fn a_descriptor_deallocator_runs_marked_as_ours() {
        use reims_vgpu_core::rt::is_our_thread;
        let k = kit();
        let page = 16384usize;
        let mut mem = vec![0u8; page * 2];
        let base = mem.as_mut_ptr();
        let aligned = base.wrapping_add(base.align_offset(page));
        // SAFETY: `aligned..aligned+page` lies in `mem`, which outlives the buffer (its
        // deallocator has run before `mem` is dropped).
        let contents = unsafe { HostRange::new(HostPtr::from_ptr(aligned), page.to_u64()) };
        let ours = Arc::new(Mutex::new(None));
        let o = ours.clone();
        let b = k
            .device
            .new_buffer(BufferDesc {
                length: page.to_u64(),
                options: vec![],
                pinned_gpu_address: None,
                contents: Some(contents),
                no_copy: true,
                deallocator: Some(Box::new(move |_, _| *o.lock() = Some(is_our_thread()))),
            })
            .unwrap()
            .unwrap();
        drop(b);
        for _ in 0..500 {
            if ours.lock().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(*ours.lock(), Some(true));
        assert!(!is_our_thread());
        drop(mem);
    }

    /// The sets core can pass nil to send nil (a rate map, a visibility buffer, a texture), and
    /// the getters read nil back
    #[test]
    fn nil_sets_send_nil() {
        let k = kit();
        let d: MetalRenderPassDescriptor = Obj::from_retained(MTLRenderPassDescriptor::new());
        let o = obj(&d);
        d.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&k.rate_map)),
            visibility_result_buffer: Some(Some(&k.buffer)),
            ..PassSet::default()
        });
        assert!(!raw::<*mut AnyObject>(o, "rasterizationRateMap").is_null());
        d.set_pass(&PassSet {
            rasterization_rate_map: Some(None),
            visibility_result_buffer: Some(None),
            ..PassSet::default()
        });
        assert!(raw::<*mut AnyObject>(o, "rasterizationRateMap").is_null());
        assert!(raw::<*mut AnyObject>(o, "visibilityResultBuffer").is_null());
        let color = |d: &MetalRenderPassDescriptor| {
            color_attachment_at(Policy::Caught, d, 0).unwrap().unwrap()
        };
        d.set_color_attachment(
            OnFault::Return,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&k.texture)),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        assert!(!raw::<*mut AnyObject>(obj(&color(&d)), "texture").is_null());
        d.set_color_attachment(
            OnFault::Return,
            0,
            &ColorAttachmentSet {
                texture: Some(None),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        assert!(raw::<*mut AnyObject>(obj(&color(&d)), "texture").is_null());
    }

    /// A texture descriptor built from a value holds it, the SPI properties included; the typed
    /// setters and getters read and write Metal's own values, a later plane's included
    #[test]
    fn texture_descriptor_from_a_value() {
        let desc = TextureDesc {
            texture_type: 3,
            pixel_format: 80,
            width: 64,
            height: 32,
            depth: 1,
            mipmap_level_count: 2,
            sample_count: 1,
            array_length: 4,
            usage: 5,
            resource_options: Some(0x20),
            swizzle: Some(Swizzle {
                red: 5,
                green: 4,
                blue: 3,
                alpha: 2,
            }),
            resource_index: Some((7, true)),
            protection_options: Some(2),
            framebuffer_only: Some(true),
            is_drawable: Some(true),
            write_swizzle_enabled: Some(true),
            allow_gpu_optimized_contents: Some(false),
            rotation: Some(3),
        };
        let x = platform().texture_descriptor(&desc);
        let o = obj(&x);
        for (sel, v) in [
            ("textureType", 3),
            ("pixelFormat", 80),
            ("width", 64),
            ("height", 32),
            ("depth", 1),
            ("mipmapLevelCount", 2),
            ("sampleCount", 1),
            ("arrayLength", 4),
            ("usage", 5),
            ("resourceOptions", 0x20),
            ("protectionOptions", 2),
            ("resourceIndex", 7),
            ("rotation", 3),
        ] {
            assert_eq!(raw_u64(o, sel), v, "{sel}");
        }
        for (sel, v) in [
            ("forceResourceIndex", true),
            ("framebufferOnly", true),
            ("isDrawable", true),
            ("writeSwizzleEnabled", true),
            ("allowGPUOptimizedContents", false),
        ] {
            assert_eq!(raw_bool(o, sel), v, "{sel}");
        }
        let swizzle: MTLTextureSwizzleChannels = raw(o, "swizzle");
        assert_eq!(
            [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha],
            [5, 4, 3, 2].map(MTLTextureSwizzle)
        );
        assert_eq!(
            (x.pixel_format(), x.texture_type(), x.rotation()),
            (80, 3, 3)
        );
        assert_eq!(x.storage_mode(), 2);
        x.set_resource_options(0x10);
        assert_eq!(x.storage_mode(), 1);
        x.set_rotation(1);
        x.set_usage(1);
        assert_eq!((x.rotation(), raw_u64(o, "usage")), (1, 1));
        // A later plane's size, type and format on the same descriptor.
        x.set_width(16);
        x.set_height(8);
        x.set_depth(2);
        x.set_texture_type(7);
        x.set_pixel_format(81);
        for (sel, v) in [
            ("width", 16),
            ("height", 8),
            ("depth", 2),
            ("textureType", 7),
            ("pixelFormat", 81),
        ] {
            assert_eq!(raw_u64(o, sel), v, "{sel}");
        }
        // What is not set keeps Metal's own value.
        let plain = platform().texture_descriptor(&TextureDesc::default());
        let fresh: MetalTextureDescriptor = Obj::from_retained(MTLTextureDescriptor::new());
        for sel in [
            "protectionOptions",
            "resourceOptions",
            "rotation",
            "resourceIndex",
        ] {
            assert_eq!(
                raw_u64(obj(&plain), sel),
                raw_u64(obj(&fresh), sel),
                "{sel}"
            );
        }
    }

    /// The typed sets of a render pass reach the pass and its attachments, and the states read
    /// Metal's load and store actions back
    #[test]
    fn render_pass_typed_sets() {
        let k = kit();
        let d: MetalRenderPassDescriptor = Obj::from_retained(MTLRenderPassDescriptor::new());
        d.set_pass(&PassSet {
            visibility_result_buffer: Some(Some(&k.buffer)),
            visibility_result_type: Some(1),
            render_target_array_length: Some(2),
            render_target_width: Some(64),
            render_target_height: Some(32),
            default_raster_sample_count: Some(4),
            default_color_sample_count: Some(4),
            rasterization_rate_map: Some(Some(&k.rate_map)),
            imageblock_sample_length: Some(8),
            threadgroup_memory_length: Some(16),
            tile_width: Some(16),
            tile_height: Some(16),
        });
        let o = obj(&d);
        for (sel, v) in [
            ("visibilityResultType", 1),
            ("renderTargetArrayLength", 2),
            ("renderTargetWidth", 64),
            ("renderTargetHeight", 32),
            ("defaultRasterSampleCount", 4),
            ("imageblockSampleLength", 8),
            ("threadgroupMemoryLength", 16),
            ("tileWidth", 16),
            ("tileHeight", 16),
        ] {
            assert_eq!(raw_u64(o, sel), v, "{sel}");
        }
        assert_eq!(
            raw::<*mut AnyObject>(o, "visibilityResultBuffer").cast_const(),
            std::ptr::from_ref(obj(&k.buffer))
        );
        d.set_pass(&PassSet {
            visibility_result_buffer: Some(None),
            ..PassSet::default()
        });
        assert!(raw::<*mut AnyObject>(o, "visibilityResultBuffer").is_null());
        assert_eq!(raw_u64(o, "renderTargetWidth"), 64);

        d.set_color_attachment(
            OnFault::Return,
            1,
            &ColorAttachmentSet {
                texture: Some(Some(&k.texture)),
                level: Some(0),
                load_action: Some(2),
                store_action: Some(1),
                clear_color: Some(ClearColor {
                    red: 0.5,
                    green: 0.25,
                    blue: 0.0,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        let a = color_attachment_at(Policy::Caught, &d, 1).unwrap().unwrap();
        assert!(!raw::<*mut AnyObject>(obj(&a), "texture").is_null());
        let clear: MTLClearColor = raw(obj(&a), "clearColor");
        assert_eq!((clear.red, clear.green), (0.5, 0.25));
        assert_eq!(
            d.color_attachment_state(1).unwrap(),
            AttachmentState {
                load_action: 2,
                store_action: 1
            }
        );
        d.set_depth_attachment(&DepthAttachmentSet {
            load_action: Some(2),
            store_action: Some(0),
            clear_depth: Some(0.5),
            depth_resolve_filter: Some(1),
            ..DepthAttachmentSet::default()
        });
        assert_eq!(
            d.depth_attachment_state(),
            AttachmentState {
                load_action: 2,
                store_action: 0
            }
        );
        let depth = depth_attachment_of(&d);
        assert_eq!(raw::<f64>(obj(&depth), "clearDepth"), 0.5);
        assert_eq!(raw_u64(obj(&depth), "depthResolveFilter"), 1);
        d.set_stencil_attachment(&StencilAttachmentSet {
            load_action: Some(1),
            store_action: Some(1),
            clear_stencil: Some(7),
            stencil_resolve_filter: Some(1),
            ..StencilAttachmentSet::default()
        });
        assert_eq!(
            d.stencil_attachment_state(),
            AttachmentState {
                load_action: 1,
                store_action: 1
            }
        );
        let stencil = stencil_attachment_of(&d);
        assert_eq!(raw::<u32>(obj(&stencil), "clearStencil"), 7);
    }

    /// The pipeline descriptors force their resource index
    #[test]
    fn pipeline_descriptors_force_their_index() {
        let r: MetalRenderPipelineDescriptor =
            Obj::from_retained(MTLRenderPipelineDescriptor::new());
        r.set_resource_index(9, true);
        assert_eq!(raw_u64(obj(&r), "resourceIndex"), 9);
        assert!(raw_bool(obj(&r), "forceResourceIndex"));
        let c: MetalComputePipelineDescriptor =
            Obj::from_retained(MTLComputePipelineDescriptor::new());
        c.set_resource_index(5, true);
        assert_eq!(raw_u64(obj(&c), "resourceIndex"), 5);
        assert!(raw_bool(obj(&c), "forceResourceIndex"));
    }

    /// setSamplePositions:count: with 4 positions sets them; aborts: Metal's own assertion for a
    /// count it does not support (1 here: "not a supported sample count for custom positions"): a
    /// process death at the send
    #[test]
    fn sample_positions() {
        use std::os::unix::process::ExitStatusExt as _;
        let d: MetalRenderPassDescriptor = Obj::from_retained(MTLRenderPassDescriptor::new());
        {
            d.set_sample_positions(&[MtlSamplePosition { x: 0.25, y: 0.75 }; 4])
                .unwrap();
            d.set_sample_positions(&[]).unwrap();
            let st =
                crate::gpu::device::tests::in_child("gpu::desc::tests::sample_positions", || {
                    let d: MetalRenderPassDescriptor =
                        Obj::from_retained(MTLRenderPassDescriptor::new());
                    let _ = d.set_sample_positions(&[MtlSamplePosition { x: 0.5, y: 0.5 }]);
                });
            assert_eq!(st.and_then(|s| s.signal()), Some(6));
        }
    }
}
