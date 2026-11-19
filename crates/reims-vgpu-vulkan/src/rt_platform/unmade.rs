//! [`Unmade`]: the type of each kind of object the rt platform makes none of yet.
//!
//! A kind's creators are refused while its piece is unbuilt ([`super::unbuilt`]), so no object of
//! the kind exists: its type has no values, and each of its methods is the empty match of one,
//! never reached. A piece that makes the kind gives it a type of its own in place of this one.

use std::marker::PhantomData;

use crate::core::host::HostPtr;
use crate::core::platform::compute::ComputeCmd;
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
use crate::core::platform::types::MtlCoordinate2D;
use crate::core::platform::types::MtlRegion;
use crate::core::platform::types::MtlSize;
use crate::core::platform::types::NsRange;
use crate::core::platform::types::SizeAndAlign;
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::platform::{compute, function_table, icb, rate_map, render, resource};
use crate::core::raise::OnFault;
use crate::core::raise::RaiseResult;

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
    /// `id<MTLComputeCommandEncoder>`.
    pub enum ComputeEncoder {}
    /// `id<MTLRenderCommandEncoder>`.
    pub enum RenderEncoder {}
    /// `id<MTLSamplerState>`.
    pub enum SamplerState {}
    /// `id<MTLDepthStencilState>`.
    pub enum DepthStencilState {}
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
