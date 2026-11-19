//! [`Unmade`]: the type of each kind of object the rt platform makes none of yet.
//!
//! A kind's creators are refused while its piece is unbuilt ([`super::unbuilt`]), so no object of
//! the kind exists: its type has no values, and each of its methods is the empty match of one,
//! never reached. A piece that makes the kind gives it a type of its own in place of this one.

use std::marker::PhantomData;

use crate::core::host::HostPtr;
use crate::core::platform::blit::BlitCmd;
use crate::core::platform::caps::DeviceCaps;
use crate::core::platform::caps::IdentityCaps;
use crate::core::platform::command::CompletedHandler;
use crate::core::platform::command::PartCut;
use crate::core::platform::command::ScheduledHandler;
use crate::core::platform::compute::ComputeCmd;
use crate::core::platform::device::NoCopyDeallocator;
use crate::core::platform::event::EventRef;
use crate::core::platform::event::Notification;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::icb::DispatchThreadgroupsArgs;
use crate::core::platform::icb::DispatchThreadsArgs;
use crate::core::platform::icb::DrawArgs;
use crate::core::platform::icb::DrawIndexedArgs;
use crate::core::platform::icb::DrawIndexedPatchesArgs;
use crate::core::platform::icb::DrawPatchesArgs;
use crate::core::platform::icb::IcbComputeCmd;
use crate::core::platform::icb::IcbRenderCmd;
use crate::core::platform::icb::ImageBlockArgs;
use crate::core::platform::icb::TessellationFactorArgs;
use crate::core::platform::render::RenderCmd;
use crate::core::platform::resource::FunctionConstant;
use crate::core::platform::types::ArgumentLayout;
use crate::core::platform::types::CommandBufferStatus;
use crate::core::platform::types::MtlCoordinate2D;
use crate::core::platform::types::MtlRegion;
use crate::core::platform::types::MtlSize;
use crate::core::platform::types::NsErrorText;
use crate::core::platform::types::NsRange;
use crate::core::platform::types::PixelFormatInfo;
use crate::core::platform::types::SizeAndAlign;
use crate::core::platform::types::Swizzle;
use crate::core::platform::types::TextureLevelInfo;
use crate::core::platform::types::TextureLevelQuery;
use crate::core::platform::values::ArgumentDesc;
use crate::core::platform::values::BufferDesc;
use crate::core::platform::values::ComputePassDesc;
use crate::core::platform::values::DepthStencilDesc;
use crate::core::platform::values::FullComputePipelineDesc;
use crate::core::platform::values::FullMeshRenderPipelineDesc;
use crate::core::platform::values::FullRenderPipelineDesc;
use crate::core::platform::values::HeapDesc;
use crate::core::platform::values::HeapMade;
use crate::core::platform::values::IcbDesc;
use crate::core::platform::values::RenderPipelineDesc;
use crate::core::platform::values::SamplerDesc;
use crate::core::platform::values::SurfacePlane;
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::platform::{
    blit, command, compute, device, event, function_table, icb, rate_map, render, resource,
};
use crate::core::raise::OnFault;
use crate::core::raise::RaiseResult;
use std::ops::Range;
use std::sync::Arc;

/// No value: the field that makes [`Unmade`] a type without values.
enum Never {}

/// An object of the kind `K` ([`kinds`]), of which the platform makes none yet: no value of this
/// type exists.
pub struct Unmade<K>(Never, PhantomData<fn() -> K>);

impl<K> Unmade<K> {
    /// Never returns: there is no such object.
    pub fn unreachable(&self) -> ! {
        match self.0 {}
    }
}

impl<K: 'static> Handle for Unmade<K> {
    fn clone_ref(&self) -> Self {
        self.unreachable()
    }
    fn id(&self) -> ObjId {
        self.unreachable()
    }
}

/// The kinds of object [`Unmade`] stands for, one type each, so each associated type of the
/// platform is a type of its own.
pub mod kinds {
    /// `id<MTLDevice>`.
    pub enum Device {}
    /// `id<MTLCommandQueue>`.
    pub enum CommandQueue {}
    /// `id<MTLCommandBuffer>`.
    pub enum CommandBuffer {}
    /// `id<MTLBlitCommandEncoder>`.
    pub enum BlitEncoder {}
    /// `id<MTLComputeCommandEncoder>`.
    pub enum ComputeEncoder {}
    /// `id<MTLRenderCommandEncoder>`.
    pub enum RenderEncoder {}
    /// `id<MTLBuffer>`.
    pub enum Buffer {}
    /// `id<MTLTexture>`.
    pub enum Texture {}
    /// `id<MTLHeap>`.
    pub enum Heap {}
    /// `id<MTLSamplerState>`.
    pub enum SamplerState {}
    /// `id<MTLDepthStencilState>`.
    pub enum DepthStencilState {}
    /// `id<MTLFence>`.
    pub enum Fence {}
    /// `id<MTLEvent>`.
    pub enum Event {}
    /// `id<MTLSharedEvent>`.
    pub enum SharedEvent {}
    /// `MTLSharedTextureHandle *`.
    pub enum SharedTextureHandle {}
    /// `id<MTLLibrary>`.
    pub enum Library {}
    /// `id<MTLFunction>`.
    pub enum Function {}
    /// `id<MTLComputePipelineState>`.
    pub enum ComputePipeline {}
    /// `id<MTLFunctionHandle>`.
    pub enum FunctionHandle {}
    /// `id<MTLVisibleFunctionTable>`.
    pub enum VisibleFunctionTable {}
    /// `id<MTLDynamicLibrary>`.
    pub enum DynamicLibrary {}
    /// `id<MTLRenderPipelineState>`.
    pub enum RenderPipeline {}
    /// `id<MTLRasterizationRateMap>`.
    pub enum RateMap {}
    /// `id<MTLIndirectCommandBuffer>`.
    pub enum Icb {}
    /// `id<MTLIndirectRenderCommand>`.
    pub enum IndirectRenderCommand {}
    /// `id<MTLIndirectComputeCommand>`.
    pub enum IndirectComputeCommand {}
    /// `MTLRasterizationRateLayerDescriptor *`.
    pub enum RateMapLayerDescriptor {}
}

impl<P: Platform> device::GpuDevice<P> for Unmade<kinds::Device> {
    fn allow_foreign_libraries(&self) {
        self.unreachable()
    }
    fn argument_layout(&self, _arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        self.unreachable()
    }
    fn function_argument_layout(
        &self,
        _function: &P::Function,
        _buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        self.unreachable()
    }
    fn heap_texture_size_and_align(
        &self,
        _desc: &P::TextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        self.unreachable()
    }
    fn new_buffer_over_host_memory(
        &self,
        _on: OnFault,
        _bytes: HostRange<'_>,
        _options: u64,
        _deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<P::Buffer>> {
        self.unreachable()
    }
    fn new_buffer_of_length(
        &self,
        _on: OnFault,
        _length: u64,
        _options: u64,
    ) -> RaiseResult<Option<P::Buffer>> {
        self.unreachable()
    }
    fn new_command_queue(&self) -> Option<P::CommandQueue> {
        self.unreachable()
    }
    fn new_command_queue_unshared(&self) -> Option<P::CommandQueue> {
        self.unreachable()
    }
    fn copies_guest_ram(&self) -> bool {
        self.unreachable()
    }
    fn new_compute_pipeline_state_for_function(
        &self,
        _compute_function: Option<&P::Function>,
    ) -> Result<Option<P::ComputePipeline>, NsErrorText> {
        self.unreachable()
    }
    fn new_shader_library(&self) -> Result<Option<P::Library>, NsErrorText> {
        self.unreachable()
    }
    fn new_event(&self, _on: OnFault) -> RaiseResult<Option<P::Event>> {
        self.unreachable()
    }
    fn new_fence(&self) -> RaiseResult<Option<P::Fence>> {
        self.unreachable()
    }
    fn new_library(&self, _data: &[u8]) -> RaiseResult<Result<Option<P::Library>, NsErrorText>> {
        self.unreachable()
    }
    fn new_dynamic_library(
        &self,
        _library: &P::Library,
    ) -> RaiseResult<Result<Option<P::DynamicLibrary>, NsErrorText>> {
        self.unreachable()
    }
    fn new_rate_map(&self, _descriptor: &P::RateMapDescriptor) -> RaiseResult<Option<P::RateMap>> {
        self.unreachable()
    }
    fn new_shared_event(&self) -> Option<P::SharedEvent> {
        self.unreachable()
    }
    fn new_shared_texture(
        &self,
        _descriptor: &P::TextureDescriptor,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn open_shared_texture(
        &self,
        _handle: Option<&P::SharedTextureHandle>,
        _resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_texture(
        &self,
        _on: OnFault,
        _descriptor: &P::TextureDescriptor,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn reserve_resource_indices(&self, _resource_type: u64, _indices: &[u64]) -> RaiseResult<()> {
        self.unreachable()
    }
    fn pixel_format_info(&self, _format: u64) -> PixelFormatInfo {
        self.unreachable()
    }
    fn texture_level_info(&self, _q: &TextureLevelQuery) -> TextureLevelInfo {
        self.unreachable()
    }
    fn caps(&self) -> Arc<DeviceCaps> {
        self.unreachable()
    }
    fn identity_caps(&self) -> IdentityCaps {
        self.unreachable()
    }
    fn takes_private_render_state(&self) -> bool {
        self.unreachable()
    }
    fn own_memory_budget(&self) -> u64 {
        self.unreachable()
    }
    fn memory_heaps_total(&self) -> u64 {
        self.unreachable()
    }
    fn makes_opencl_mode_pipelines(&self) -> bool {
        self.unreachable()
    }
    fn new_buffer(&self, _desc: BufferDesc<'_>) -> RaiseResult<Option<P::Buffer>> {
        self.unreachable()
    }
    fn new_heap(&self, _desc: &HeapDesc) -> RaiseResult<HeapMade<P>> {
        self.unreachable()
    }
    fn new_heap_over_host_memory(
        &self,
        _bytes: HostRange<'_>,
        _desc: &HeapDesc,
    ) -> RaiseResult<HeapMade<P>> {
        self.unreachable()
    }
    fn new_sampler_state(&self, _desc: &SamplerDesc) -> RaiseResult<Option<P::SamplerState>> {
        self.unreachable()
    }
    fn new_depth_stencil_state(
        &self,
        _desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<P::DepthStencilState>> {
        self.unreachable()
    }
    fn new_indirect_command_buffer(
        &self,
        _desc: &IcbDesc,
        _max_count: u64,
        _options: u64,
    ) -> RaiseResult<Option<P::Icb>> {
        self.unreachable()
    }
    fn new_render_pipeline_state(
        &self,
        _desc: &RenderPipelineDesc<'_, P>,
    ) -> Result<Option<P::RenderPipeline>, NsErrorText> {
        self.unreachable()
    }
    fn new_render_pipeline_state_full(
        &self,
        _desc: &FullRenderPipelineDesc<'_, P>,
    ) -> Result<Option<P::RenderPipeline>, NsErrorText> {
        self.unreachable()
    }
    fn new_mesh_render_pipeline_state_full(
        &self,
        _desc: &FullMeshRenderPipelineDesc<'_, P>,
    ) -> Result<Option<P::RenderPipeline>, NsErrorText> {
        self.unreachable()
    }
    fn new_compute_pipeline_state_full(
        &self,
        _desc: &FullComputePipelineDesc<'_, P>,
    ) -> Result<Option<P::ComputePipeline>, NsErrorText> {
        self.unreachable()
    }
}

impl<P: Platform> command::CommandQueue<P> for Unmade<kinds::CommandQueue> {
    fn command_buffer(&self, _on: OnFault) -> RaiseResult<Option<P::CommandBuffer>> {
        self.unreachable()
    }
    fn set_label(&self, _label: &str) {
        self.unreachable()
    }
}

impl<P: Platform> command::CommandBuffer<P> for Unmade<kinds::CommandBuffer> {
    fn add_completed_handler(&self, _on: OnFault, _block: CompletedHandler) -> RaiseResult<()> {
        self.unreachable()
    }
    fn add_scheduled_handler(&self, _on: OnFault, _block: ScheduledHandler) -> RaiseResult<()> {
        self.unreachable()
    }
    fn add_submission_handler(&self, _on: OnFault, _block: ScheduledHandler) -> RaiseResult<()> {
        self.unreachable()
    }
    fn add_completion_seen_handler(
        &self,
        _on: OnFault,
        _block: CompletedHandler,
    ) -> RaiseResult<()> {
        self.unreachable()
    }
    fn blit_encoder(&self, _on: OnFault) -> RaiseResult<Option<P::BlitEncoder>> {
        self.unreachable()
    }
    fn commit(&self, _on: OnFault) -> RaiseResult<()> {
        self.unreachable()
    }
    fn compute_encoder(&self, _on: OnFault) -> RaiseResult<Option<P::ComputeEncoder>> {
        self.unreachable()
    }
    fn device(&self) -> Option<P::Device> {
        self.unreachable()
    }
    fn encode_signal(&self, _event: Option<EventRef<'_, P>>, _value: u64) -> RaiseResult<()> {
        self.unreachable()
    }
    fn encode_signal_when_scheduled(
        &self,
        _signal_event_scheduled: Option<&P::SharedEvent>,
        _value: u64,
    ) -> RaiseResult<()> {
        self.unreachable()
    }
    fn encode_wait(&self, _event: Option<EventRef<'_, P>>, _value: u64) -> RaiseResult<()> {
        self.unreachable()
    }
    fn error(&self) -> Option<NsErrorText> {
        self.unreachable()
    }
    fn render_encoder(
        &self,
        _on: OnFault,
        _render_pass_descriptor: &P::RenderPassDescriptor,
    ) -> RaiseResult<Option<P::RenderEncoder>> {
        self.unreachable()
    }
    fn set_protection_options(&self, _on: OnFault, _options: u64) -> RaiseResult<()> {
        self.unreachable()
    }
    fn status(&self) -> CommandBufferStatus {
        self.unreachable()
    }
    fn wait_until_completed(&self, _on: OnFault) -> RaiseResult<()> {
        self.unreachable()
    }
    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        self.unreachable()
    }
    fn compute_encoder_for_pass(
        &self,
        _desc: &ComputePassDesc,
    ) -> RaiseResult<Option<P::ComputeEncoder>> {
        self.unreachable()
    }
    fn part(&self, _on: OnFault) -> RaiseResult<Option<P::CommandBuffer>> {
        self.unreachable()
    }
    fn commit_part(&self, _on: OnFault) -> RaiseResult<PartCut> {
        self.unreachable()
    }
    fn fold(&self, _later: P::CommandBuffer) {
        self.unreachable()
    }
}

impl<P: Platform> blit::BlitEncoder<P> for Unmade<kinds::BlitEncoder> {
    fn encode(&self, _on: OnFault, _cmd: BlitCmd<'_, P>) -> RaiseResult<()> {
        self.unreachable()
    }
    fn copy_from_guest_ram(
        &self,
        _on: OnFault,
        _runs: &[Range<u64>],
        _destination: &P::Buffer,
        _destination_offset: u64,
    ) -> RaiseResult<bool> {
        self.unreachable()
    }
    fn copy_from_guest_ram_at(
        &self,
        _on: OnFault,
        _runs: &[(u64, Range<u64>)],
        _destination: &P::Buffer,
    ) -> RaiseResult<bool> {
        self.unreachable()
    }
    fn copy_to_guest_ram(
        &self,
        _on: OnFault,
        _source: &P::Buffer,
        _runs: &[(u64, Range<u64>)],
    ) -> RaiseResult<bool> {
        self.unreachable()
    }
}

impl<P: Platform> compute::ComputeEncoder<P> for Unmade<kinds::ComputeEncoder> {
    fn encode(&self, _on: OnFault, _cmd: ComputeCmd<'_, P>) -> RaiseResult<()> {
        self.unreachable()
    }
    fn encode_end_do_while(
        &self,
        _buffer: Option<&P::Buffer>,
        _offset: u64,
        _comparison: u64,
        _reference_value: u32,
    ) -> RaiseResult<bool> {
        self.unreachable()
    }
    fn encode_end_if(&self) -> RaiseResult<bool> {
        self.unreachable()
    }
    fn encode_end_while(&self) -> RaiseResult<bool> {
        self.unreachable()
    }
}

impl<P: Platform> render::RenderEncoder<P> for Unmade<kinds::RenderEncoder> {
    fn encode(&self, _on: OnFault, _cmd: RenderCmd<'_, P>) -> RaiseResult<()> {
        self.unreachable()
    }
    fn tile_height(&self) -> u64 {
        self.unreachable()
    }
    fn tile_width(&self) -> u64 {
        self.unreachable()
    }
}

impl<P: Platform> resource::GpuResource<P> for Unmade<kinds::Buffer> {
    fn device(&self) -> Option<P::Device> {
        self.unreachable()
    }
    fn resource_options(&self) -> u64 {
        self.unreachable()
    }
    fn set_label(&self, _label: &str) -> RaiseResult<()> {
        self.unreachable()
    }
    fn set_purgeable_state(&self, _state: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn storage_mode(&self) -> u64 {
        self.unreachable()
    }
}

impl<P: Platform> resource::GpuResource<P> for Unmade<kinds::Texture> {
    fn device(&self) -> Option<P::Device> {
        self.unreachable()
    }
    fn resource_options(&self) -> u64 {
        self.unreachable()
    }
    fn set_label(&self, _label: &str) -> RaiseResult<()> {
        self.unreachable()
    }
    fn set_purgeable_state(&self, _state: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn storage_mode(&self) -> u64 {
        self.unreachable()
    }
}

impl<P: Platform> resource::Buffer<P> for Unmade<kinds::Buffer> {
    fn contents(&self) -> HostPtr {
        self.unreachable()
    }
    fn did_modify_range(&self, _on: OnFault, _range: NsRange) -> RaiseResult<()> {
        self.unreachable()
    }
    fn gpu_address(&self) -> u64 {
        self.unreachable()
    }
    fn length(&self) -> u64 {
        self.unreachable()
    }
    fn new_linear_texture(
        &self,
        _descriptor: &P::TextureDescriptor,
        _offset: u64,
        _bytes_per_row: u64,
        _bytes_per_image: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_texture(
        &self,
        _on: OnFault,
        _descriptor: &P::TextureDescriptor,
        _offset: u64,
        _bytes_per_row: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_tiled_texture(
        &self,
        _descriptor: &P::TextureDescriptor,
        _offset: u64,
        _bytes_per_row: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_texture_over_surface_plane(
        &self,
        _on: OnFault,
        _descriptor: &P::TextureDescriptor,
        _plane: SurfacePlane,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
}

impl<P: Platform> resource::Texture<P> for Unmade<kinds::Texture> {
    fn array_length(&self) -> u64 {
        self.unreachable()
    }
    fn depth(&self) -> u64 {
        self.unreachable()
    }
    fn did_modify_data(&self) {
        self.unreachable()
    }
    fn gpu_resource_id(&self) -> u64 {
        self.unreachable()
    }
    fn height(&self) -> u64 {
        self.unreachable()
    }
    fn is_shareable(&self) -> bool {
        self.unreachable()
    }
    fn mipmap_level_count(&self) -> u64 {
        self.unreachable()
    }
    fn new_shared_texture_handle(&self) -> RaiseResult<Option<P::SharedTextureHandle>> {
        self.unreachable()
    }
    fn new_texture_view(&self, _pixel_format: u64) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_texture_view_at_index(
        &self,
        _pixel_format: u64,
        _resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_texture_subview(
        &self,
        _pixel_format: u64,
        _texture_type: u64,
        _level_range: NsRange,
        _slice_range: NsRange,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_swizzled_texture_subview(
        &self,
        _pixel_format: u64,
        _texture_type: u64,
        _level_range: NsRange,
        _slice_range: NsRange,
        _swizzle: Swizzle,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_swizzled_texture_subview_at_index(
        &self,
        _pixel_format: u64,
        _texture_type: u64,
        _levels: NsRange,
        _slices: NsRange,
        _swizzle: Swizzle,
        _resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn new_texture_subview_at_index(
        &self,
        _pixel_format: u64,
        _texture_type: u64,
        _levels: NsRange,
        _slices: NsRange,
        _resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn pixel_format(&self) -> u64 {
        self.unreachable()
    }
    fn protection_options(&self) -> u64 {
        self.unreachable()
    }
    fn replace_region(
        &self,
        _region: MtlRegion,
        _level: u64,
        _slice: u64,
        _pixel_bytes: HostRange<'_>,
        _bytes_per_row: u64,
        _bytes_per_image: u64,
    ) -> RaiseResult<()> {
        self.unreachable()
    }
    fn resource_index(&self) -> u64 {
        self.unreachable()
    }
    fn sample_count(&self) -> u64 {
        self.unreachable()
    }
    fn texture_type(&self) -> u64 {
        self.unreachable()
    }
    fn unique_identifier(&self) -> u64 {
        self.unreachable()
    }
    fn usage(&self) -> u64 {
        self.unreachable()
    }
    fn width(&self) -> u64 {
        self.unreachable()
    }
}

impl<P: Platform> resource::Heap<P> for Unmade<kinds::Heap> {
    fn gpu_address(&self) -> u64 {
        self.unreachable()
    }
    fn new_buffer_of_length(
        &self,
        _length: u64,
        _options: u64,
        _offset: u64,
    ) -> RaiseResult<Option<P::Buffer>> {
        self.unreachable()
    }
    fn new_texture(
        &self,
        _descriptor: &P::TextureDescriptor,
        _offset: u64,
    ) -> RaiseResult<Option<P::Texture>> {
        self.unreachable()
    }
    fn resource_options(&self) -> u64 {
        self.unreachable()
    }
    fn set_purgeable_state(&self, _state: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn storage_mode(&self) -> u64 {
        self.unreachable()
    }
}

impl resource::SamplerState for Unmade<kinds::SamplerState> {
    fn gpu_resource_id(&self) -> u64 {
        self.unreachable()
    }
    fn resource_index(&self) -> u64 {
        self.unreachable()
    }
}

impl resource::DepthStencilState for Unmade<kinds::DepthStencilState> {
    fn gpu_resource_id(&self) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn resource_index(&self) -> RaiseResult<u64> {
        self.unreachable()
    }
}

impl<P: Platform> resource::ComputePipeline<P> for Unmade<kinds::ComputePipeline> {
    fn gpu_resource_id(&self) -> u64 {
        self.unreachable()
    }
    fn function_handle(&self, _function: &P::Function) -> RaiseResult<Option<P::FunctionHandle>> {
        self.unreachable()
    }
    fn new_visible_function_table(
        &self,
        _function_count: u64,
    ) -> RaiseResult<Option<P::VisibleFunctionTable>> {
        self.unreachable()
    }
    fn imageblock_memory_length(&self, _imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.unreachable()
    }
    fn resource_index(&self) -> u64 {
        self.unreachable()
    }
    fn static_threadgroup_memory_length(&self) -> u64 {
        self.unreachable()
    }
    fn support_indirect_command_buffers(&self) -> bool {
        self.unreachable()
    }
    fn thread_execution_width(&self) -> u64 {
        self.unreachable()
    }
    fn unique_identifier(&self) -> u64 {
        self.unreachable()
    }
}

impl resource::RenderPipeline for Unmade<kinds::RenderPipeline> {
    fn gpu_resource_id(&self) -> u64 {
        self.unreachable()
    }
    fn imageblock_memory_length(&self, _imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn imageblock_sample_length(&self) -> u64 {
        self.unreachable()
    }
    fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.unreachable()
    }
    fn max_total_threads_per_object_threadgroup(&self) -> u64 {
        self.unreachable()
    }
    fn max_total_threads_per_mesh_threadgroup(&self) -> u64 {
        self.unreachable()
    }
    fn object_thread_execution_width(&self) -> u64 {
        self.unreachable()
    }
    fn mesh_thread_execution_width(&self) -> u64 {
        self.unreachable()
    }
    fn max_total_threadgroups_per_mesh_grid(&self) -> u64 {
        self.unreachable()
    }
    fn resource_index(&self) -> u64 {
        self.unreachable()
    }
    fn support_indirect_command_buffers(&self) -> bool {
        self.unreachable()
    }
    fn threadgroup_size_matches_tile_size(&self) -> bool {
        self.unreachable()
    }
    fn unique_identifier(&self) -> u64 {
        self.unreachable()
    }
}

impl<P: Platform> resource::Library<P> for Unmade<kinds::Library> {
    fn function_names(&self) -> Vec<String> {
        self.unreachable()
    }
    fn new_function(&self, _on: OnFault, _function_name: &str) -> RaiseResult<Option<P::Function>> {
        self.unreachable()
    }
    fn new_function_with_constants(
        &self,
        _on: OnFault,
        _function_name: &str,
        _constants: &[FunctionConstant],
    ) -> RaiseResult<Option<P::Function>> {
        self.unreachable()
    }
    fn new_function_with_options(
        &self,
        _on: OnFault,
        _function_name: &str,
        _constants: &[FunctionConstant],
        _options: u64,
        _specialized_name: Option<&str>,
    ) -> RaiseResult<Option<P::Function>> {
        self.unreachable()
    }
}

impl<P: Platform> event::SharedEvent<P> for Unmade<kinds::SharedEvent> {
    fn set_signaled_value(&self, _signaled_value: u64) -> RaiseResult<()> {
        self.unreachable()
    }
    fn signaled_value(&self) -> u64 {
        self.unreachable()
    }
    fn wait_until_signaled_value(&self, _value: u64, _milliseconds: u64) -> bool {
        self.unreachable()
    }
    fn notify_listener(
        &self,
        _listener: &P::SharedEventListener,
        _value: u64,
        _block: Notification,
    ) -> RaiseResult<()> {
        self.unreachable()
    }
}

impl<P: Platform> function_table::VisibleFunctionTable<P> for Unmade<kinds::VisibleFunctionTable> {
    fn gpu_resource_id(&self) -> u64 {
        self.unreachable()
    }
    fn allocated_size(&self) -> u64 {
        self.unreachable()
    }
    fn resource_options(&self) -> u64 {
        self.unreachable()
    }
    fn set_functions(
        &self,
        _functions: &[Option<&P::FunctionHandle>],
        _range: NsRange,
    ) -> RaiseResult<()> {
        self.unreachable()
    }
}

impl<P: Platform> icb::IndirectCommandBuffer<P> for Unmade<kinds::Icb> {
    fn gpu_resource_id(&self) -> u64 {
        self.unreachable()
    }
    fn compute_command(
        &self,
        _command_index: u64,
    ) -> RaiseResult<Option<P::IndirectComputeCommand<'_>>> {
        self.unreachable()
    }
    fn render_command(
        &self,
        _command_index: u64,
    ) -> RaiseResult<Option<P::IndirectRenderCommand<'_>>> {
        self.unreachable()
    }
    fn reset_with_range(&self, _location: u64, _length: u64) -> RaiseResult<()> {
        self.unreachable()
    }
    fn resource_index(&self) -> u64 {
        self.unreachable()
    }
    fn size(&self) -> u64 {
        self.unreachable()
    }
    fn unique_identifier(&self) -> u64 {
        self.unreachable()
    }
}

impl<P: Platform> icb::IndirectRenderCommand<P> for Unmade<kinds::IndirectRenderCommand> {
    fn encode(&self, _cmd: IcbRenderCmd<'_, P>) -> RaiseResult<()> {
        self.unreachable()
    }
    fn command_type(&self) -> u64 {
        self.unreachable()
    }
    fn fragment_buffer_address(&self, _index: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn pipeline_state_unique_identifier(&self) -> u64 {
        self.unreachable()
    }
    fn vertex_attribute_stride(&self, _index: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn vertex_buffer_address(&self, _index: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn has_barrier(&self) -> bool {
        self.unreachable()
    }
    fn draw_args(&self) -> Option<DrawArgs> {
        self.unreachable()
    }
    fn draw_indexed_args(&self) -> Option<DrawIndexedArgs> {
        self.unreachable()
    }
    fn draw_patches_args(&self) -> Option<DrawPatchesArgs> {
        self.unreachable()
    }
    fn draw_indexed_patches_args(&self) -> Option<DrawIndexedPatchesArgs> {
        self.unreachable()
    }
    fn tessellation_factor_args(&self) -> Option<TessellationFactorArgs> {
        self.unreachable()
    }
}

impl<P: Platform> icb::IndirectComputeCommand<P> for Unmade<kinds::IndirectComputeCommand> {
    fn encode(&self, _cmd: IcbComputeCmd<'_, P>) -> RaiseResult<()> {
        self.unreachable()
    }
    fn command_type(&self) -> u64 {
        self.unreachable()
    }
    fn kernel_attribute_stride(&self, _index: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn kernel_buffer_address(&self, _index: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn pipeline_state_unique_identifier(&self) -> u64 {
        self.unreachable()
    }
    fn stage_in_region(&self) -> MtlRegion {
        self.unreachable()
    }
    fn threadgroup_memory_length(&self, _index: u64) -> RaiseResult<u64> {
        self.unreachable()
    }
    fn has_barrier(&self) -> bool {
        self.unreachable()
    }
    fn dispatch_threadgroups_args(&self) -> Option<DispatchThreadgroupsArgs> {
        self.unreachable()
    }
    fn dispatch_threads_args(&self) -> Option<DispatchThreadsArgs> {
        self.unreachable()
    }
    fn image_block_args(&self) -> Option<ImageBlockArgs> {
        self.unreachable()
    }
}

impl<P: Platform> rate_map::RateMap<P> for Unmade<kinds::RateMap> {
    fn copy_parameter_data(&self, _buffer: &P::Buffer, _offset: u64) -> RaiseResult<()> {
        self.unreachable()
    }
    fn layer_count(&self) -> u64 {
        self.unreachable()
    }
    fn physical_to_screen(
        &self,
        _physical_coordinates: MtlCoordinate2D,
        _layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        self.unreachable()
    }
    fn screen_to_physical(
        &self,
        _screen_coordinates: MtlCoordinate2D,
        _layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        self.unreachable()
    }
    fn mutability(&self) -> u64 {
        self.unreachable()
    }
    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        self.unreachable()
    }
    fn physical_granularity(&self) -> MtlSize {
        self.unreachable()
    }
    fn physical_size(&self, _layer_index: u64) -> RaiseResult<MtlSize> {
        self.unreachable()
    }
    fn screen_size(&self) -> MtlSize {
        self.unreachable()
    }
    fn reset(&self, _descriptor: &P::RateMapDescriptor) {
        self.unreachable()
    }
}

impl rate_map::RateMapLayerDescriptor for Unmade<kinds::RateMapLayerDescriptor> {
    fn horizontal_sample_storage(&self) -> HostPtr {
        self.unreachable()
    }
    fn sample_count(&self) -> MtlSize {
        self.unreachable()
    }
    fn vertical_sample_storage(&self) -> HostPtr {
        self.unreachable()
    }
}
