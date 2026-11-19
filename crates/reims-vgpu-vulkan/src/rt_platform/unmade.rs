//! [`Unmade`]: the type of each kind of object the rt platform makes none of yet.
//!
//! A kind's creators are refused while its piece is unbuilt ([`super::unbuilt`]), so no object of
//! the kind exists: its type has no values, and each of its methods is the empty match of one,
//! never reached. A piece that makes the kind gives it a type of its own in place of this one.

use std::marker::PhantomData;

use crate::core::host::HostPtr;
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
use crate::core::platform::types::MtlCoordinate2D;
use crate::core::platform::types::MtlRegion;
use crate::core::platform::types::MtlSize;
use crate::core::platform::types::NsRange;
use crate::core::platform::types::SizeAndAlign;
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::platform::{function_table, icb, rate_map};
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
    /// `MTLSharedTextureHandle *`.
    pub enum SharedTextureHandle {}
    /// `id<MTLFunctionHandle>`.
    pub enum FunctionHandle {}
    /// `id<MTLVisibleFunctionTable>`.
    pub enum VisibleFunctionTable {}
    /// `id<MTLDynamicLibrary>`.
    pub enum DynamicLibrary {}
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
