//! Typed declarations of the Metal methods objc2-metal does not declare, as extension traits of
//! the objects that answer them:
//! - the private selectors (`MTLDeviceSPI`, `MTLRenderCommandEncoderSPI`, the descriptors'
//!   resource-index setters, …), typed from the runtime's encodings;
//! - the public selectors whose objc2-metal declaration cannot carry what core sends: a nil the
//!   header marks nonnull (the guest's nil goes to Metal as it is), a pointer that may be NULL,
//!   or an `error:` out-parameter Metal may leave nil (objc2 would invent an error there, and
//!   core tells the two apart).
//!
//! Each method is one `msg_send!`. `tests/spi_encodings.rs` checks every method's types against
//! the runtime, and that the reference host implements it. Three of these selectors are declared
//! in no header, the SDK's or Metal's private ones, so their types are the runtime's alone:
//! `encodeSignalEventScheduled:value:`, and `dispatchThreadsPerTile:inRegion:` with its
//! `withRenderTargetArrayIndex:` form (as is `setDeallocator:` of the private buffer descriptor,
//! in `classes`). The traits are `unsafe` to implement: an implementor is an object whose class
//! answers every selector of the trait with these types.
//!
//! A re-declared public selector keeps objc2-metal's name, so a call names its trait
//! (`RenderEncoderSpi::setRenderPipelineState(encoder, state)`).

#![allow(non_snake_case, missing_docs)]

use std::ffi::c_void;

use block2::DynBlock;
use dispatch2::DispatchData;
use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{Message, msg_send};
use objc2_foundation::{NSArray, NSError, NSRange, NSString};
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLArgumentEncoder, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDepthStencilState, MTLDevice, MTLEvent, MTLFunction, MTLHeap,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectComputeCommand,
    MTLIndirectRenderCommand, MTLLibrary, MTLMeshRenderPipelineDescriptor, MTLOrigin,
    MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMap,
    MTLRasterizationRateMapDescriptor, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassColorAttachmentDescriptor, MTLRenderPassColorAttachmentDescriptorArray,
    MTLRenderPassDescriptor, MTLRenderPipelineColorAttachmentDescriptor,
    MTLRenderPipelineColorAttachmentDescriptorArray, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResource, MTLSamplerDescriptor, MTLSamplerState, MTLSharedEvent,
    MTLSharedEventHandle, MTLSharedTextureHandle, MTLSize, MTLTexture, MTLTextureDescriptor,
    MTLTextureSwizzleChannels, MTLTileRenderPipelineDescriptor,
};

use crate::spi::classes::{
    ImageBlockArguments, IndirectDispatchThreadgroupsArguments, IndirectDispatchThreadsArguments,
    IndirectDrawArguments, IndirectDrawIndexedArguments, IndirectDrawIndexedPatchesArguments,
    IndirectDrawPatchesArguments, IndirectTessellationFactorArguments, MTLBufferDescriptor,
    MTLResourceAddressRangeArray,
};

/// The deallocator block a no-copy buffer takes: the pointer and the length, as Metal passes
/// them.
pub type Deallocator = DynBlock<dyn Fn(*mut c_void, u64)>;

// -------------------------------------------------------------------------------------------------
// The device.
// -------------------------------------------------------------------------------------------------

/// `MTLDeviceSPI`, the device's creators whose arguments objc2-metal cannot carry, and `-name`
/// with a nil result carried.
///
/// # Safety
/// The implementor is an `id<MTLDevice>`.
pub unsafe trait DeviceSpi: Message {
    fn allowLibrariesFromOtherPlatforms(&self) {
        unsafe { msg_send![self, allowLibrariesFromOtherPlatforms] }
    }
    fn deviceLinearTextureAlignmentBytes(&self) -> u64 {
        unsafe { msg_send![self, deviceLinearTextureAlignmentBytes] }
    }
    fn doubleFPConfig(&self) -> u64 {
        unsafe { msg_send![self, doubleFPConfig] }
    }
    fn featureProfile(&self) -> u64 {
        unsafe { msg_send![self, featureProfile] }
    }
    /// `unsigned int`, as the runtime answers (`I@:`).
    fn gpuCoreCount(&self) -> u32 {
        unsafe { msg_send![self, gpuCoreCount] }
    }
    fn halfFPConfig(&self) -> u64 {
        unsafe { msg_send![self, halfFPConfig] }
    }
    fn iosurfaceReadOnlyTextureAlignmentBytes(&self) -> u64 {
        unsafe { msg_send![self, iosurfaceReadOnlyTextureAlignmentBytes] }
    }
    fn iosurfaceTextureAlignmentBytes(&self) -> u64 {
        unsafe { msg_send![self, iosurfaceTextureAlignmentBytes] }
    }
    fn isFramebufferReadSupported(&self) -> bool {
        unsafe { msg_send![self, isFramebufferReadSupported] }
    }
    fn isLargeMRTSupported(&self) -> bool {
        unsafe { msg_send![self, isLargeMRTSupported] }
    }
    fn isRGB10A2GammaSupported(&self) -> bool {
        unsafe { msg_send![self, isRGB10A2GammaSupported] }
    }
    fn linearTextureAlignmentBytes(&self) -> u64 {
        unsafe { msg_send![self, linearTextureAlignmentBytes] }
    }
    fn maxComputeBuffers(&self) -> u64 {
        unsafe { msg_send![self, maxComputeBuffers] }
    }
    fn maxComputeLocalMemorySizes(&self) -> u64 {
        unsafe { msg_send![self, maxComputeLocalMemorySizes] }
    }
    fn maxComputeSamplers(&self) -> u64 {
        unsafe { msg_send![self, maxComputeSamplers] }
    }
    fn maxComputeTextures(&self) -> u64 {
        unsafe { msg_send![self, maxComputeTextures] }
    }
    fn maxComputeThreadgroupMemory(&self) -> u64 {
        unsafe { msg_send![self, maxComputeThreadgroupMemory] }
    }
    fn maxComputeThreadgroupMemoryAlignmentBytes(&self) -> u64 {
        unsafe { msg_send![self, maxComputeThreadgroupMemoryAlignmentBytes] }
    }
    fn maxFragmentBuffers(&self) -> u64 {
        unsafe { msg_send![self, maxFragmentBuffers] }
    }
    fn maxFragmentSamplers(&self) -> u64 {
        unsafe { msg_send![self, maxFragmentSamplers] }
    }
    fn maxFragmentTextures(&self) -> u64 {
        unsafe { msg_send![self, maxFragmentTextures] }
    }
    fn maxPredicatedNestingDepth(&self) -> u64 {
        unsafe { msg_send![self, maxPredicatedNestingDepth] }
    }
    fn maxRasterizationRateLayerCount(&self) -> u64 {
        unsafe { msg_send![self, maxRasterizationRateLayerCount] }
    }
    fn maxTextureHeight2D(&self) -> u64 {
        unsafe { msg_send![self, maxTextureHeight2D] }
    }
    fn maxTextureLayers(&self) -> u64 {
        unsafe { msg_send![self, maxTextureLayers] }
    }
    fn maxTextureWidth2D(&self) -> u64 {
        unsafe { msg_send![self, maxTextureWidth2D] }
    }
    fn maxTileBuffers(&self) -> u64 {
        unsafe { msg_send![self, maxTileBuffers] }
    }
    fn maxTileSamplers(&self) -> u64 {
        unsafe { msg_send![self, maxTileSamplers] }
    }
    fn maxTileTextures(&self) -> u64 {
        unsafe { msg_send![self, maxTileTextures] }
    }
    fn maxTotalComputeThreadsPerThreadgroup(&self) -> u64 {
        unsafe { msg_send![self, maxTotalComputeThreadsPerThreadgroup] }
    }
    fn maxVertexAmplificationCount(&self) -> u64 {
        unsafe { msg_send![self, maxVertexAmplificationCount] }
    }
    fn maxVertexBuffers(&self) -> u64 {
        unsafe { msg_send![self, maxVertexBuffers] }
    }
    fn maxVertexSamplers(&self) -> u64 {
        unsafe { msg_send![self, maxVertexSamplers] }
    }
    fn maxVertexTextures(&self) -> u64 {
        unsafe { msg_send![self, maxVertexTextures] }
    }
    fn maxViewportCount(&self) -> u64 {
        unsafe { msg_send![self, maxViewportCount] }
    }
    fn singleFPConfig(&self) -> u64 {
        unsafe { msg_send![self, singleFPConfig] }
    }
    fn supportsBufferWithAddressRanges(&self) -> bool {
        unsafe { msg_send![self, supportsBufferWithAddressRanges] }
    }
    fn supportsBufferWithIOSurface(&self) -> bool {
        unsafe { msg_send![self, supportsBufferWithIOSurface] }
    }
    fn supportsCommandBufferJump(&self) -> bool {
        unsafe { msg_send![self, supportsCommandBufferJump] }
    }
    fn supportsComputeCompressedTextureWrite(&self) -> bool {
        unsafe { msg_send![self, supportsComputeCompressedTextureWrite] }
    }
    fn supportsConditionalLoadStore(&self) -> bool {
        unsafe { msg_send![self, supportsConditionalLoadStore] }
    }
    fn supportsDynamicAttributeStride(&self) -> bool {
        unsafe { msg_send![self, supportsDynamicAttributeStride] }
    }
    fn supportsExtendedXR10Formats(&self) -> bool {
        unsafe { msg_send![self, supportsExtendedXR10Formats] }
    }
    fn supportsFloat16BCubicFiltering(&self) -> bool {
        unsafe { msg_send![self, supportsFloat16BCubicFiltering] }
    }
    fn supportsHeapWithAddressRanges(&self) -> bool {
        unsafe { msg_send![self, supportsHeapWithAddressRanges] }
    }
    fn supportsImageBlocks(&self) -> bool {
        unsafe { msg_send![self, supportsImageBlocks] }
    }
    fn supportsMemoryOrderAtomics(&self) -> bool {
        unsafe { msg_send![self, supportsMemoryOrderAtomics] }
    }
    fn supportsNativeHardwareFP16(&self) -> bool {
        unsafe { msg_send![self, supportsNativeHardwareFP16] }
    }
    fn supportsNonSquareTileShaders(&self) -> bool {
        unsafe { msg_send![self, supportsNonSquareTileShaders] }
    }
    /// The macOS 26 SDK's getter, which objc2-metal does not declare: sent only where the device
    /// responds to it.
    fn supportsPlacementSparse(&self) -> bool {
        unsafe { msg_send![self, supportsPlacementSparse] }
    }
    fn supportsPrimitiveType(&self, primitive_type: u64) -> bool {
        unsafe { msg_send![self, supportsPrimitiveType: primitive_type] }
    }
    fn supportsProgrammableSamplePositions(&self) -> bool {
        unsafe { msg_send![self, supportsProgrammableSamplePositions] }
    }
    fn supportsResourceDetachBacking(&self) -> bool {
        unsafe { msg_send![self, supportsResourceDetachBacking] }
    }
    fn supportsSharedTextureHandles(&self) -> bool {
        unsafe { msg_send![self, supportsSharedTextureHandles] }
    }
    fn supportsSIMDReduction(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDReduction] }
    }
    fn supportsSIMDShuffleAndFill(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDShuffleAndFill] }
    }
    fn supportsTexture2DMultisampleArray(&self) -> bool {
        unsafe { msg_send![self, supportsTexture2DMultisampleArray] }
    }
    fn supportsTextureWriteRoundingMode(&self, mode: i64) -> bool {
        unsafe { msg_send![self, supportsTextureWriteRoundingMode: mode] }
    }
    fn supportsTileShaders(&self) -> bool {
        unsafe { msg_send![self, supportsTileShaders] }
    }

    /// `-name` with a nil result carried (objc2-metal's is nonnull): a host's device may answer
    /// nil.
    fn name_or_nil(&self) -> Option<Retained<NSString>> {
        unsafe { msg_send![self, name] }
    }

    fn newBufferWithDescriptor(
        &self,
        descriptor: &MTLBufferDescriptor,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        unsafe { msg_send![self, newBufferWithDescriptor: descriptor] }
    }
    /// The buffer retains the surface; a nil surface is sent as it is.
    fn newBufferWithIOSurface(
        &self,
        surface: Option<&IOSurfaceRef>,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        unsafe { msg_send![self, newBufferWithIOSurface: surface] }
    }
    /// `bytes` may be any address the caller holds a lease on (or none, for Metal to refuse).
    ///
    /// # Safety
    /// `bytes` is `length` bytes mapped until the deallocator runs, or for the buffer's life.
    unsafe fn newBufferWithBytesNoCopy_length_options_deallocator(
        &self,
        bytes: *mut c_void,
        length: u64,
        options: u64,
        deallocator: Option<&Deallocator>,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        unsafe {
            msg_send![self, newBufferWithBytesNoCopy: bytes, length: length, options: options, deallocator: deallocator]
        }
    }
    fn newComputePipelineDescriptorWithSerializedData_deserializationContext(
        &self,
        data: &DispatchData,
        context: &AnyObject,
    ) -> Option<Retained<MTLComputePipelineDescriptor>> {
        unsafe {
            msg_send![self, newComputePipelineDescriptorWithSerializedData: data, deserializationContext: context]
        }
    }
    fn newRenderPipelineDescriptorWithSerializedData_deserializationContext(
        &self,
        data: &DispatchData,
        context: &AnyObject,
    ) -> Option<Retained<MTLRenderPipelineDescriptor>> {
        unsafe {
            msg_send![self, newRenderPipelineDescriptorWithSerializedData: data, deserializationContext: context]
        }
    }
    fn newTileRenderPipelineDescriptorWithSerializedData_deserializationContext(
        &self,
        data: &DispatchData,
        context: &AnyObject,
    ) -> Option<Retained<MTLTileRenderPipelineDescriptor>> {
        unsafe {
            msg_send![self, newTileRenderPipelineDescriptorWithSerializedData: data, deserializationContext: context]
        }
    }
    fn newSharedTextureWithHandle_withResourceIndex(
        &self,
        handle: Option<&MTLSharedTextureHandle>,
        resource_index: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        unsafe {
            msg_send![self, newSharedTextureWithHandle: handle, withResourceIndex: resource_index]
        }
    }
    /// # Safety
    /// `indices` is `count` readable `u64`s.
    unsafe fn reserveResourceIndicesForResourceType_indices_indexCount(
        &self,
        resource_type: u64,
        indices: *const u64,
        count: u64,
    ) {
        unsafe {
            msg_send![self, reserveResourceIndicesForResourceType: resource_type, indices: indices, indexCount: count]
        }
    }

    // Public selectors whose arguments objc2-metal cannot carry.

    /// A nil descriptor is sent as it is; reflection is nil.
    ///
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newComputePipelineStateWithDescriptor_options_reflection_error(
        &self,
        descriptor: Option<&MTLComputePipelineDescriptor>,
        options: u64,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let reflection: *mut *mut AnyObject = std::ptr::null_mut();
        unsafe {
            msg_send![self, newComputePipelineStateWithDescriptor: descriptor, options: options, reflection: reflection, error: error]
        }
    }
    /// A nil function is sent as it is.
    ///
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newComputePipelineStateWithFunction_error(
        &self,
        function: Option<&ProtocolObject<dyn MTLFunction>>,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        unsafe { msg_send![self, newComputePipelineStateWithFunction: function, error: error] }
    }
    /// The public selector with the `error:` out-parameter as a pointer, so a nil library with a
    /// nil error is answered as such.
    ///
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newLibraryWithSource_options_error(
        &self,
        source: &NSString,
        options: Option<&MTLCompileOptions>,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
        unsafe { msg_send![self, newLibraryWithSource: source, options: options, error: error] }
    }
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newLibraryWithData_error(
        &self,
        data: &DispatchData,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
        unsafe { msg_send![self, newLibraryWithData: data, error: error] }
    }
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newRenderPipelineStateWithDescriptor_error(
        &self,
        descriptor: &MTLRenderPipelineDescriptor,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        unsafe { msg_send![self, newRenderPipelineStateWithDescriptor: descriptor, error: error] }
    }
    /// A nil descriptor is sent as it is; reflection is nil.
    ///
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newRenderPipelineStateWithDescriptor_options_reflection_error(
        &self,
        descriptor: Option<&MTLRenderPipelineDescriptor>,
        options: u64,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let reflection: *mut *mut AnyObject = std::ptr::null_mut();
        unsafe {
            msg_send![self, newRenderPipelineStateWithDescriptor: descriptor, options: options, reflection: reflection, error: error]
        }
    }
    /// Reflection is nil.
    ///
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
        &self,
        descriptor: &MTLMeshRenderPipelineDescriptor,
        options: u64,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let reflection: *mut *mut AnyObject = std::ptr::null_mut();
        unsafe {
            msg_send![self, newRenderPipelineStateWithMeshDescriptor: descriptor, options: options, reflection: reflection, error: error]
        }
    }
    /// A nil descriptor is sent as it is; reflection is nil.
    ///
    /// # Safety
    /// `error` is a writable out-parameter.
    unsafe fn newRenderPipelineStateWithTileDescriptor_options_reflection_error(
        &self,
        descriptor: Option<&MTLTileRenderPipelineDescriptor>,
        options: u64,
        error: *mut *mut NSError,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let reflection: *mut *mut AnyObject = std::ptr::null_mut();
        unsafe {
            msg_send![self, newRenderPipelineStateWithTileDescriptor: descriptor, options: options, reflection: reflection, error: error]
        }
    }
    /// A nil handle is sent as it is.
    fn newSharedEventWithHandle(
        &self,
        handle: Option<&MTLSharedEventHandle>,
    ) -> Option<Retained<ProtocolObject<dyn MTLSharedEvent>>> {
        unsafe { msg_send![self, newSharedEventWithHandle: handle] }
    }
    /// A nil surface is sent as it is (Metal's own validation ends the process then).
    fn newTextureWithDescriptor_iosurface_plane(
        &self,
        descriptor: &MTLTextureDescriptor,
        surface: Option<&IOSurfaceRef>,
        plane: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        unsafe {
            msg_send![self, newTextureWithDescriptor: descriptor, iosurface: surface, plane: plane]
        }
    }
}

// SAFETY: an `id<MTLDevice>`.
unsafe impl DeviceSpi for ProtocolObject<dyn MTLDevice> {}

/// `-[MTLLibrary functionNames]` with a nil result carried (objc2-metal's is nonnull).
///
/// # Safety
/// The implementor is an `id<MTLLibrary>`.
pub unsafe trait LibrarySpi: Message {
    fn functionNames_or_nil(&self) -> Option<Retained<NSArray<NSString>>> {
        unsafe { msg_send![self, functionNames] }
    }
}

// SAFETY: an `id<MTLLibrary>`.
unsafe impl LibrarySpi for ProtocolObject<dyn MTLLibrary> {}

/// `-[MTLFunction newArgumentEncoderWithBufferIndex:reflection:]` with a nil result carried
/// (objc2-metal's is nonnull).
///
/// # Safety
/// The implementor is an `id<MTLFunction>`.
pub unsafe trait FunctionSpi: Message {
    /// The encoder of the argument buffer the function takes at `buffer_index`, or nil, and the
    /// argument Metal reflects for it in `reflection` (an autoreleased object, or nil). The
    /// selector is deprecated for `newArgumentEncoderWithBufferBinding:` of a pipeline's binding;
    /// it is the one that reflects the argument buffer of a function alone.
    #[allow(deprecated)]
    fn newArgumentEncoderWithBufferIndex_reflection_or_nil(
        &self,
        buffer_index: u64,
        reflection: &mut Option<Retained<objc2_metal::MTLArgument>>,
    ) -> Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>> {
        unsafe {
            msg_send![self, newArgumentEncoderWithBufferIndex: buffer_index, reflection: reflection]
        }
    }
}

// SAFETY: an `id<MTLFunction>`.
unsafe impl FunctionSpi for ProtocolObject<dyn MTLFunction> {}

/// The attachment arrays' `objectAtIndexedSubscript:` with a nil result carried (objc2-metal's
/// is nonnull). It raises past the end.
///
/// # Safety
/// The implementor is an attachment array whose elements are `Item`s.
pub unsafe trait AttachmentArraySpi: Message {
    /// The attachment's class.
    type Item: Message;
    fn objectAtIndexedSubscript_or_nil(&self, index: u64) -> Option<Retained<Self::Item>> {
        unsafe { msg_send![self, objectAtIndexedSubscript: index] }
    }
}

// SAFETY: its elements are render pass colour attachments.
unsafe impl AttachmentArraySpi for MTLRenderPassColorAttachmentDescriptorArray {
    type Item = MTLRenderPassColorAttachmentDescriptor;
}

// SAFETY: its elements are render pipeline colour attachments.
unsafe impl AttachmentArraySpi for MTLRenderPipelineColorAttachmentDescriptorArray {
    type Item = MTLRenderPipelineColorAttachmentDescriptor;
}

// -------------------------------------------------------------------------------------------------
// The command buffer and the encoders.
// -------------------------------------------------------------------------------------------------

/// `MTLCommandBufferSPI`, and the event encoders with a nil event.
///
/// # Safety
/// The implementor is an `id<MTLCommandBuffer>`.
pub unsafe trait CommandBufferSpi: Message {
    fn encodeSignalEventScheduled_value(
        &self,
        event: Option<&ProtocolObject<dyn MTLSharedEvent>>,
        value: u64,
    ) {
        unsafe { msg_send![self, encodeSignalEventScheduled: event, value: value] }
    }
    fn setProtectionOptions(&self, options: u64) {
        unsafe { msg_send![self, setProtectionOptions: options] }
    }
    /// A nil event is sent as it is.
    fn encodeSignalEvent_value(&self, event: Option<&ProtocolObject<dyn MTLEvent>>, value: u64) {
        unsafe { msg_send![self, encodeSignalEvent: event, value: value] }
    }
    /// A nil event is sent as it is.
    fn encodeWaitForEvent_value(&self, event: Option<&ProtocolObject<dyn MTLEvent>>, value: u64) {
        unsafe { msg_send![self, encodeWaitForEvent: event, value: value] }
    }
}

// SAFETY: an `id<MTLCommandBuffer>`.
unsafe impl CommandBufferSpi for ProtocolObject<dyn MTLCommandBuffer> {}

/// `-[MTLSharedEvent newSharedEventHandle]` with a nil result carried (objc2-metal's is
/// nonnull).
///
/// # Safety
/// The implementor is an `id<MTLSharedEvent>`.
pub unsafe trait SharedEventSpi: Message {
    fn newSharedEventHandle_or_nil(&self) -> Option<Retained<MTLSharedEventHandle>> {
        unsafe { msg_send![self, newSharedEventHandle] }
    }
}

// SAFETY: an `id<MTLSharedEvent>`.
unsafe impl SharedEventSpi for ProtocolObject<dyn MTLSharedEvent> {}

/// `MTLBlitCommandEncoderSPI`, and the copies with a nil source or destination.
///
/// # Safety
/// The implementor is an `id<MTLBlitCommandEncoder>`.
pub unsafe trait BlitEncoderSpi: Message {
    #[expect(clippy::too_many_arguments, reason = "one argument per selector part")]
    fn copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
        &self,
        source: Option<&ProtocolObject<dyn MTLTexture>>,
        source_slice: u64,
        source_level: u64,
        source_origin: MTLOrigin,
        source_size: MTLSize,
        destination: Option<&ProtocolObject<dyn MTLTexture>>,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: MTLOrigin,
        options: u64,
    ) {
        unsafe {
            msg_send![self, copyFromTexture: source, sourceSlice: source_slice, sourceLevel: source_level, sourceOrigin: source_origin, sourceSize: source_size, toTexture: destination, destinationSlice: destination_slice, destinationLevel: destination_level, destinationOrigin: destination_origin, options: options]
        }
    }
    fn fillBuffer_range_pattern4(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        range: NSRange,
        pattern: u32,
    ) {
        unsafe { msg_send![self, fillBuffer: buffer, range: range, pattern4: pattern] }
    }
    /// # Safety
    /// `bytes` is `length` readable bytes for the call.
    unsafe fn fillTexture_level_slice_region_bytes_length(
        &self,
        texture: &ProtocolObject<dyn MTLTexture>,
        level: u64,
        slice: u64,
        region: MTLRegion,
        bytes: *const c_void,
        length: u64,
    ) {
        unsafe {
            msg_send![self, fillTexture: texture, level: level, slice: slice, region: region, bytes: bytes, length: length]
        }
    }
    fn fillTexture_level_slice_region_color_pixelFormat(
        &self,
        texture: &ProtocolObject<dyn MTLTexture>,
        level: u64,
        slice: u64,
        region: MTLRegion,
        color: MTLClearColor,
        pixel_format: u64,
    ) {
        unsafe {
            msg_send![self, fillTexture: texture, level: level, slice: slice, region: region, color: color, pixelFormat: pixel_format]
        }
    }
    fn invalidateCompressedTexture(&self, texture: &ProtocolObject<dyn MTLTexture>) {
        unsafe { msg_send![self, invalidateCompressedTexture: texture] }
    }
    fn invalidateCompressedTexture_slice_level(
        &self,
        texture: &ProtocolObject<dyn MTLTexture>,
        slice: u64,
        level: u64,
    ) {
        unsafe { msg_send![self, invalidateCompressedTexture: texture, slice: slice, level: level] }
    }

    // Public selectors with a nil the header marks nonnull.

    #[expect(clippy::too_many_arguments, reason = "one argument per selector part")]
    fn copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
        &self,
        source: Option<&ProtocolObject<dyn MTLBuffer>>,
        source_offset: u64,
        source_bytes_per_row: u64,
        source_bytes_per_image: u64,
        source_size: MTLSize,
        destination: Option<&ProtocolObject<dyn MTLTexture>>,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: MTLOrigin,
        options: u64,
    ) {
        unsafe {
            msg_send![self, copyFromBuffer: source, sourceOffset: source_offset, sourceBytesPerRow: source_bytes_per_row, sourceBytesPerImage: source_bytes_per_image, sourceSize: source_size, toTexture: destination, destinationSlice: destination_slice, destinationLevel: destination_level, destinationOrigin: destination_origin, options: options]
        }
    }
    fn copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
        &self,
        source: Option<&ProtocolObject<dyn MTLBuffer>>,
        source_offset: u64,
        destination: Option<&ProtocolObject<dyn MTLBuffer>>,
        destination_offset: u64,
        size: u64,
    ) {
        unsafe {
            msg_send![self, copyFromBuffer: source, sourceOffset: source_offset, toBuffer: destination, destinationOffset: destination_offset, size: size]
        }
    }
    #[expect(clippy::too_many_arguments, reason = "one argument per selector part")]
    fn copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
        &self,
        source: Option<&ProtocolObject<dyn MTLTexture>>,
        source_slice: u64,
        source_level: u64,
        source_origin: MTLOrigin,
        source_size: MTLSize,
        destination: Option<&ProtocolObject<dyn MTLBuffer>>,
        destination_offset: u64,
        destination_bytes_per_row: u64,
        destination_bytes_per_image: u64,
        options: u64,
    ) {
        unsafe {
            msg_send![self, copyFromTexture: source, sourceSlice: source_slice, sourceLevel: source_level, sourceOrigin: source_origin, sourceSize: source_size, toBuffer: destination, destinationOffset: destination_offset, destinationBytesPerRow: destination_bytes_per_row, destinationBytesPerImage: destination_bytes_per_image, options: options]
        }
    }
    fn generateMipmapsForTexture(&self, texture: Option<&ProtocolObject<dyn MTLTexture>>) {
        unsafe { msg_send![self, generateMipmapsForTexture: texture] }
    }
    fn synchronizeResource(&self, resource: Option<&ProtocolObject<dyn MTLResource>>) {
        unsafe { msg_send![self, synchronizeResource: resource] }
    }
}

// SAFETY: an `id<MTLBlitCommandEncoder>`.
unsafe impl BlitEncoderSpi for ProtocolObject<dyn MTLBlitCommandEncoder> {}

/// `MTLComputeCommandEncoderSPI`, and the calls with a nil the header marks nonnull.
///
/// # Safety
/// The implementor is an `id<MTLComputeCommandEncoder>`.
pub unsafe trait ComputeEncoderSpi: Message {
    fn dispatchThreadsWithIndirectBuffer_indirectBufferOffset(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
    ) {
        unsafe {
            msg_send![self, dispatchThreadsWithIndirectBuffer: buffer, indirectBufferOffset: offset]
        }
    }
    fn encodeEndDoWhile_offset_comparison_referenceValue(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> bool {
        unsafe {
            msg_send![self, encodeEndDoWhile: buffer, offset: offset, comparison: comparison, referenceValue: reference_value]
        }
    }
    fn encodeEndIf(&self) -> bool {
        unsafe { msg_send![self, encodeEndIf] }
    }
    fn encodeEndWhile(&self) -> bool {
        unsafe { msg_send![self, encodeEndWhile] }
    }
    fn encodeStartDoWhile(&self) {
        unsafe { msg_send![self, encodeStartDoWhile] }
    }
    fn encodeStartElse(&self) {
        unsafe { msg_send![self, encodeStartElse] }
    }
    fn encodeStartIf_offset_comparison_referenceValue(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) {
        unsafe {
            msg_send![self, encodeStartIf: buffer, offset: offset, comparison: comparison, referenceValue: reference_value]
        }
    }
    fn encodeStartWhile_offset_comparison_referenceValue(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) {
        unsafe {
            msg_send![self, encodeStartWhile: buffer, offset: offset, comparison: comparison, referenceValue: reference_value]
        }
    }
    fn insertCompressedTextureReinterpretationFlush(&self) {
        unsafe { msg_send![self, insertCompressedTextureReinterpretationFlush] }
    }

    // Public selectors with a nil the header marks nonnull.

    fn dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        threads_per_threadgroup: MTLSize,
    ) {
        unsafe {
            msg_send![self, dispatchThreadgroupsWithIndirectBuffer: buffer, indirectBufferOffset: offset, threadsPerThreadgroup: threads_per_threadgroup]
        }
    }
    fn setComputePipelineState(&self, state: Option<&ProtocolObject<dyn MTLComputePipelineState>>) {
        unsafe { msg_send![self, setComputePipelineState: state] }
    }
}

// SAFETY: an `id<MTLComputeCommandEncoder>`.
unsafe impl ComputeEncoderSpi for ProtocolObject<dyn MTLComputeCommandEncoder> {}

/// `MTLRenderCommandEncoderSPI`, and the calls with a nil the header marks nonnull.
///
/// # Safety
/// The implementor is an `id<MTLRenderCommandEncoder>`.
pub unsafe trait RenderEncoderSpi: Message {
    fn dispatchThreadsPerTile_inRegion(&self, threads_per_tile: MTLSize, region: MTLRegion) {
        unsafe { msg_send![self, dispatchThreadsPerTile: threads_per_tile, inRegion: region] }
    }
    fn dispatchThreadsPerTile_inRegion_withRenderTargetArrayIndex(
        &self,
        threads_per_tile: MTLSize,
        region: MTLRegion,
        render_target_array_index: u32,
    ) {
        unsafe {
            msg_send![self, dispatchThreadsPerTile: threads_per_tile, inRegion: region, withRenderTargetArrayIndex: render_target_array_index]
        }
    }
    fn setAlphaTestReferenceValue(&self, value: f32) {
        unsafe { msg_send![self, setAlphaTestReferenceValue: value] }
    }
    fn setColorResolveTexture_slice_depthPlane_level_yInvert_atIndex(
        &self,
        texture: Option<&ProtocolObject<dyn MTLTexture>>,
        slice: u64,
        depth_plane: u64,
        level: u64,
        y_invert: bool,
        index: u64,
    ) {
        unsafe {
            msg_send![self, setColorResolveTexture: texture, slice: slice, depthPlane: depth_plane, level: level, yInvert: y_invert, atIndex: index]
        }
    }
    fn setDepthCleared(&self) {
        unsafe { msg_send![self, setDepthCleared] }
    }
    fn setDepthClipModeSPI(&self, mode: u64) {
        unsafe { msg_send![self, setDepthClipModeSPI: mode] }
    }
    fn setDepthResolveTexture_slice_depthPlane_level_yInvert(
        &self,
        texture: Option<&ProtocolObject<dyn MTLTexture>>,
        slice: u64,
        depth_plane: u64,
        level: u64,
        y_invert: bool,
    ) {
        unsafe {
            msg_send![self, setDepthResolveTexture: texture, slice: slice, depthPlane: depth_plane, level: level, yInvert: y_invert]
        }
    }
    fn setLineWidth(&self, width: f32) {
        unsafe { msg_send![self, setLineWidth: width] }
    }
    fn setPointSize(&self, size: f32) {
        unsafe { msg_send![self, setPointSize: size] }
    }
    fn setPrimitiveRestartEnabled_index(&self, enabled: bool, index: u64) {
        unsafe { msg_send![self, setPrimitiveRestartEnabled: enabled, index: index] }
    }
    fn setProvokingVertexMode(&self, mode: u64) {
        unsafe { msg_send![self, setProvokingVertexMode: mode] }
    }
    fn setStencilCleared(&self) {
        unsafe { msg_send![self, setStencilCleared] }
    }
    fn setStencilResolveTexture_slice_depthPlane_level_yInvert(
        &self,
        texture: Option<&ProtocolObject<dyn MTLTexture>>,
        slice: u64,
        depth_plane: u64,
        level: u64,
        y_invert: bool,
    ) {
        unsafe {
            msg_send![self, setStencilResolveTexture: texture, slice: slice, depthPlane: depth_plane, level: level, yInvert: y_invert]
        }
    }
    fn setVertexAmplificationMode_value(&self, mode: u64, value: u64) {
        unsafe { msg_send![self, setVertexAmplificationMode: mode, value: value] }
    }
    fn setViewportTransformEnabled(&self, enabled: bool) {
        unsafe { msg_send![self, setViewportTransformEnabled: enabled] }
    }

    // Public selectors with a nil the header marks nonnull.

    fn setRenderPipelineState(&self, state: Option<&ProtocolObject<dyn MTLRenderPipelineState>>) {
        unsafe { msg_send![self, setRenderPipelineState: state] }
    }
}

// SAFETY: an `id<MTLRenderCommandEncoder>`.
unsafe impl RenderEncoderSpi for ProtocolObject<dyn MTLRenderCommandEncoder> {}

// -------------------------------------------------------------------------------------------------
// Resources and state objects.
// -------------------------------------------------------------------------------------------------

/// `MTLBufferSPI`, and the getters whose results objc2-metal types too narrowly.
///
/// # Safety
/// The implementor is an `id<MTLBuffer>`.
pub unsafe trait BufferSpi: Message {
    /// Metal's `BOOL`.
    fn detachBacking(&self) -> bool {
        unsafe { msg_send![self, detachBacking] }
    }
    fn newLinearTextureWithDescriptor_offset_bytesPerRow_bytesPerImage(
        &self,
        descriptor: &MTLTextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        unsafe {
            msg_send![self, newLinearTextureWithDescriptor: descriptor, offset: offset, bytesPerRow: bytes_per_row, bytesPerImage: bytes_per_image]
        }
    }
    fn replaceBackingWithRanges_readOnly(
        &self,
        ranges: &MTLResourceAddressRangeArray,
        read_only: bool,
    ) -> bool {
        unsafe { msg_send![self, replaceBackingWithRanges: ranges, readOnly: read_only] }
    }
    /// NULL for a private buffer (objc2-metal declares it nonnull).
    fn contents(&self) -> *mut c_void {
        unsafe { msg_send![self, contents] }
    }
    /// The buffer's surface at `+0`, or NULL.
    fn iosurface(&self) -> *mut IOSurfaceRef {
        unsafe { msg_send![self, iosurface] }
    }
}

// SAFETY: an `id<MTLBuffer>`.
unsafe impl BufferSpi for ProtocolObject<dyn MTLBuffer> {}

/// `MTLTextureSPI`, and the getters and uploads objc2-metal types too narrowly.
///
/// # Safety
/// The implementor is an `id<MTLTexture>`.
pub unsafe trait TextureSpi: Message {
    fn newTextureViewWithPixelFormat_resourceIndex(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        unsafe {
            msg_send![self, newTextureViewWithPixelFormat: pixel_format, resourceIndex: resource_index]
        }
    }
    fn newTextureViewWithPixelFormat_textureType_levels_slices_resourceIndex(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NSRange,
        slices: NSRange,
        resource_index: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        unsafe {
            msg_send![self, newTextureViewWithPixelFormat: pixel_format, textureType: texture_type, levels: levels, slices: slices, resourceIndex: resource_index]
        }
    }
    fn newTextureViewWithPixelFormat_textureType_levels_slices_swizzle_resourceIndex(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NSRange,
        slices: NSRange,
        swizzle: MTLTextureSwizzleChannels,
        resource_index: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        unsafe {
            msg_send![self, newTextureViewWithPixelFormat: pixel_format, textureType: texture_type, levels: levels, slices: slices, swizzle: swizzle, resourceIndex: resource_index]
        }
    }
    fn protectionOptions(&self) -> u64 {
        unsafe { msg_send![self, protectionOptions] }
    }
    fn resourceIndex(&self) -> u64 {
        unsafe { msg_send![self, resourceIndex] }
    }
    fn uniqueIdentifier(&self) -> u64 {
        unsafe { msg_send![self, uniqueIdentifier] }
    }
    /// The texture's surface at `+0`, or NULL (objc2-metal retains it).
    fn iosurface(&self) -> *mut IOSurfaceRef {
        unsafe { msg_send![self, iosurface] }
    }
    /// # Safety
    /// `bytes` covers the region's bytes for the call.
    unsafe fn replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
        &self,
        region: MTLRegion,
        level: u64,
        slice: u64,
        bytes: *const c_void,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) {
        unsafe {
            msg_send![self, replaceRegion: region, mipmapLevel: level, slice: slice, withBytes: bytes, bytesPerRow: bytes_per_row, bytesPerImage: bytes_per_image]
        }
    }
}

// SAFETY: an `id<MTLTexture>`.
unsafe impl TextureSpi for ProtocolObject<dyn MTLTexture> {}

/// `MTLHeapSPI`.
///
/// # Safety
/// The implementor is an `id<MTLHeap>`.
pub unsafe trait HeapSpi: Message {
    /// Metal's `BOOL`.
    fn detachBacking(&self) -> bool {
        unsafe { msg_send![self, detachBacking] }
    }
    fn gpuAddress(&self) -> u64 {
        unsafe { msg_send![self, gpuAddress] }
    }
    fn replaceBackingWithRanges_readOnly(
        &self,
        ranges: &MTLResourceAddressRangeArray,
        read_only: bool,
    ) -> bool {
        unsafe { msg_send![self, replaceBackingWithRanges: ranges, readOnly: read_only] }
    }
}

// SAFETY: an `id<MTLHeap>`.
unsafe impl HeapSpi for ProtocolObject<dyn MTLHeap> {}

/// The resource index every state object and indirect command buffer answers (SPI).
///
/// # Safety
/// The implementor answers `resourceIndex` (`Q@:`).
pub unsafe trait ResourceIndexSpi: Message {
    fn resourceIndex(&self) -> u64 {
        unsafe { msg_send![self, resourceIndex] }
    }
}

// SAFETY: each answers `resourceIndex` (`Q@:`; `tests/spi_encodings.rs`).
unsafe impl ResourceIndexSpi for ProtocolObject<dyn MTLSamplerState> {}
// SAFETY: as above; the reference host's stub raises.
unsafe impl ResourceIndexSpi for ProtocolObject<dyn MTLDepthStencilState> {}
// SAFETY: as above.
unsafe impl ResourceIndexSpi for ProtocolObject<dyn MTLComputePipelineState> {}
// SAFETY: as above.
unsafe impl ResourceIndexSpi for ProtocolObject<dyn MTLRenderPipelineState> {}
// SAFETY: as above.
unsafe impl ResourceIndexSpi for ProtocolObject<dyn MTLIndirectCommandBuffer> {}

/// The unique identifier pipelines and indirect command buffers answer (SPI).
///
/// # Safety
/// The implementor answers `uniqueIdentifier` (`Q@:`).
pub unsafe trait UniqueIdentifierSpi: Message {
    fn uniqueIdentifier(&self) -> u64 {
        unsafe { msg_send![self, uniqueIdentifier] }
    }
}

// SAFETY: each answers `uniqueIdentifier` (`Q@:`; `tests/spi_encodings.rs`).
unsafe impl UniqueIdentifierSpi for ProtocolObject<dyn MTLComputePipelineState> {}
// SAFETY: as above.
unsafe impl UniqueIdentifierSpi for ProtocolObject<dyn MTLRenderPipelineState> {}
// SAFETY: as above.
unsafe impl UniqueIdentifierSpi for ProtocolObject<dyn MTLIndirectCommandBuffer> {}

/// `MTLRasterizationRateMapSPI`.
///
/// # Safety
/// The implementor is an `id<MTLRasterizationRateMap>`.
pub unsafe trait RateMapSpi: Message {
    fn mutability(&self) -> u64 {
        unsafe { msg_send![self, mutability] }
    }
    fn resetUsingDescriptor(&self, descriptor: &MTLRasterizationRateMapDescriptor) {
        unsafe { msg_send![self, resetUsingDescriptor: descriptor] }
    }
}

// SAFETY: an `id<MTLRasterizationRateMap>`.
unsafe impl RateMapSpi for ProtocolObject<dyn MTLRasterizationRateMap> {}

// -------------------------------------------------------------------------------------------------
// Indirect commands.
// -------------------------------------------------------------------------------------------------

/// The indirect command buffer's command getters with a nil result carried (objc2-metal's are
/// nonnull).
///
/// # Safety
/// The implementor is an `id<MTLIndirectCommandBuffer>`.
pub unsafe trait IcbSpi: Message {
    fn indirectComputeCommandAtIndex_or_nil(
        &self,
        index: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLIndirectComputeCommand>>> {
        unsafe { msg_send![self, indirectComputeCommandAtIndex: index] }
    }
    fn indirectRenderCommandAtIndex_or_nil(
        &self,
        index: u64,
    ) -> Option<Retained<ProtocolObject<dyn MTLIndirectRenderCommand>>> {
        unsafe { msg_send![self, indirectRenderCommandAtIndex: index] }
    }
}

// SAFETY: an `id<MTLIndirectCommandBuffer>`.
unsafe impl IcbSpi for ProtocolObject<dyn MTLIndirectCommandBuffer> {}

/// `MTLIndirectRenderCommandSPI`, and the commands with a nil state or buffer.
///
/// # Safety
/// The implementor is an `id<MTLIndirectRenderCommand>`.
pub unsafe trait IndirectRenderCommandSpi: Message {
    fn drawArguments(&self) -> Option<Retained<IndirectDrawArguments>> {
        unsafe { msg_send![self, drawArguments] }
    }
    fn drawIndexedArguments(&self) -> Option<Retained<IndirectDrawIndexedArguments>> {
        unsafe { msg_send![self, drawIndexedArguments] }
    }
    fn drawIndexedPatchesArguments(&self) -> Option<Retained<IndirectDrawIndexedPatchesArguments>> {
        unsafe { msg_send![self, drawIndexedPatchesArguments] }
    }
    fn drawPatchesArguments(&self) -> Option<Retained<IndirectDrawPatchesArguments>> {
        unsafe { msg_send![self, drawPatchesArguments] }
    }
    fn getCommandType(&self) -> u64 {
        unsafe { msg_send![self, getCommandType] }
    }
    fn getFragmentBufferAddressAtIndex(&self, index: u64) -> u64 {
        unsafe { msg_send![self, getFragmentBufferAddressAtIndex: index] }
    }
    fn getPipelineStateUniqueIdentifier(&self) -> u64 {
        unsafe { msg_send![self, getPipelineStateUniqueIdentifier] }
    }
    fn getTessellationFactorArguments(
        &self,
    ) -> Option<Retained<IndirectTessellationFactorArguments>> {
        unsafe { msg_send![self, getTessellationFactorArguments] }
    }
    fn getVertexAttributeStrideAtIndex(&self, index: u64) -> u64 {
        unsafe { msg_send![self, getVertexAttributeStrideAtIndex: index] }
    }
    fn getVertexBufferAddressAtIndex(&self, index: u64) -> u64 {
        unsafe { msg_send![self, getVertexBufferAddressAtIndex: index] }
    }
    fn hasBarrier(&self) -> bool {
        unsafe { msg_send![self, hasBarrier] }
    }

    // Public selectors with a nil the header marks nonnull.

    fn setRenderPipelineState(&self, state: Option<&ProtocolObject<dyn MTLRenderPipelineState>>) {
        unsafe { msg_send![self, setRenderPipelineState: state] }
    }
    #[expect(clippy::too_many_arguments, reason = "one argument per selector part")]
    fn drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
        &self,
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        control_point_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        instance_stride: u64,
    ) {
        unsafe {
            msg_send![self, drawIndexedPatches: number_of_patch_control_points, patchStart: patch_start, patchCount: patch_count, patchIndexBuffer: patch_index_buffer, patchIndexBufferOffset: patch_index_buffer_offset, controlPointIndexBuffer: control_point_index_buffer, controlPointIndexBufferOffset: control_point_index_buffer_offset, instanceCount: instance_count, baseInstance: base_instance, tessellationFactorBuffer: buffer, tessellationFactorBufferOffset: offset, tessellationFactorBufferInstanceStride: instance_stride]
        }
    }
    #[expect(clippy::too_many_arguments, reason = "one argument per selector part")]
    fn drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
        &self,
        primitive_type: u64,
        index_count: u64,
        index_type: u64,
        index_buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        index_buffer_offset: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    ) {
        unsafe {
            msg_send![self, drawIndexedPrimitives: primitive_type, indexCount: index_count, indexType: index_type, indexBuffer: index_buffer, indexBufferOffset: index_buffer_offset, instanceCount: instance_count, baseVertex: base_vertex, baseInstance: base_instance]
        }
    }
    #[expect(clippy::too_many_arguments, reason = "one argument per selector part")]
    fn drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
        &self,
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        patch_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        instance_stride: u64,
    ) {
        unsafe {
            msg_send![self, drawPatches: number_of_patch_control_points, patchStart: patch_start, patchCount: patch_count, patchIndexBuffer: patch_index_buffer, patchIndexBufferOffset: patch_index_buffer_offset, instanceCount: instance_count, baseInstance: base_instance, tessellationFactorBuffer: buffer, tessellationFactorBufferOffset: offset, tessellationFactorBufferInstanceStride: instance_stride]
        }
    }
    fn setFragmentBuffer_offset_atIndex(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        index: u64,
    ) {
        unsafe { msg_send![self, setFragmentBuffer: buffer, offset: offset, atIndex: index] }
    }
    fn setVertexBuffer_offset_atIndex(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        index: u64,
    ) {
        unsafe { msg_send![self, setVertexBuffer: buffer, offset: offset, atIndex: index] }
    }
    fn setVertexBuffer_offset_attributeStride_atIndex(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        stride: u64,
        index: u64,
    ) {
        unsafe {
            msg_send![self, setVertexBuffer: buffer, offset: offset, attributeStride: stride, atIndex: index]
        }
    }
}

// SAFETY: an `id<MTLIndirectRenderCommand>`.
unsafe impl IndirectRenderCommandSpi for ProtocolObject<dyn MTLIndirectRenderCommand> {}

/// `MTLIndirectComputeCommandSPI`, and the commands with a nil state or buffer.
///
/// # Safety
/// The implementor is an `id<MTLIndirectComputeCommand>`.
pub unsafe trait IndirectComputeCommandSpi: Message {
    fn dispatchThreadgroupsArguments(
        &self,
    ) -> Option<Retained<IndirectDispatchThreadgroupsArguments>> {
        unsafe { msg_send![self, dispatchThreadgroupsArguments] }
    }
    fn dispatchThreadsArguments(&self) -> Option<Retained<IndirectDispatchThreadsArguments>> {
        unsafe { msg_send![self, dispatchThreadsArguments] }
    }
    fn getCommandType(&self) -> u64 {
        unsafe { msg_send![self, getCommandType] }
    }
    fn getImageBlockSize(&self) -> Option<Retained<ImageBlockArguments>> {
        unsafe { msg_send![self, getImageBlockSize] }
    }
    fn getKernelAttributeStrideAtIndex(&self, index: u64) -> u64 {
        unsafe { msg_send![self, getKernelAttributeStrideAtIndex: index] }
    }
    fn getKernelBufferAddressAtIndex(&self, index: u64) -> u64 {
        unsafe { msg_send![self, getKernelBufferAddressAtIndex: index] }
    }
    fn getPipelineStateUniqueIdentifier(&self) -> u64 {
        unsafe { msg_send![self, getPipelineStateUniqueIdentifier] }
    }
    fn getStageInRegion(&self) -> MTLRegion {
        unsafe { msg_send![self, getStageInRegion] }
    }
    fn getThreadgroupMemoryLengthAtIndex(&self, index: u64) -> u64 {
        unsafe { msg_send![self, getThreadgroupMemoryLengthAtIndex: index] }
    }
    fn hasBarrier(&self) -> bool {
        unsafe { msg_send![self, hasBarrier] }
    }

    // Public selectors with a nil the header marks nonnull.

    fn setComputePipelineState(&self, state: Option<&ProtocolObject<dyn MTLComputePipelineState>>) {
        unsafe { msg_send![self, setComputePipelineState: state] }
    }
    fn setKernelBuffer_offset_atIndex(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        index: u64,
    ) {
        unsafe { msg_send![self, setKernelBuffer: buffer, offset: offset, atIndex: index] }
    }
    fn setKernelBuffer_offset_attributeStride_atIndex(
        &self,
        buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        stride: u64,
        index: u64,
    ) {
        unsafe {
            msg_send![self, setKernelBuffer: buffer, offset: offset, attributeStride: stride, atIndex: index]
        }
    }
}

// SAFETY: an `id<MTLIndirectComputeCommand>`.
unsafe impl IndirectComputeCommandSpi for ProtocolObject<dyn MTLIndirectComputeCommand> {}

// -------------------------------------------------------------------------------------------------
// Descriptors.
// -------------------------------------------------------------------------------------------------

/// `MTLTextureDescriptorSPI`.
///
/// # Safety
/// The implementor is an `MTLTextureDescriptor`.
pub unsafe trait TextureDescriptorSpi: Message {
    fn rotation(&self) -> u64 {
        unsafe { msg_send![self, rotation] }
    }
    fn setForceResourceIndex(&self, force: bool) {
        unsafe { msg_send![self, setForceResourceIndex: force] }
    }
    fn setFramebufferOnly(&self, framebuffer_only: bool) {
        unsafe { msg_send![self, setFramebufferOnly: framebuffer_only] }
    }
    fn setIsDrawable(&self, is_drawable: bool) {
        unsafe { msg_send![self, setIsDrawable: is_drawable] }
    }
    fn setProtectionOptions(&self, options: u64) {
        unsafe { msg_send![self, setProtectionOptions: options] }
    }
    fn setResourceIndex(&self, index: u64) {
        unsafe { msg_send![self, setResourceIndex: index] }
    }
    fn setRotation(&self, rotation: u64) {
        unsafe { msg_send![self, setRotation: rotation] }
    }
    fn setWriteSwizzleEnabled(&self, enabled: bool) {
        unsafe { msg_send![self, setWriteSwizzleEnabled: enabled] }
    }
}

// SAFETY: an `MTLTextureDescriptor`.
unsafe impl TextureDescriptorSpi for MTLTextureDescriptor {}

/// The resource-index setters of the sampler and pipeline descriptors (SPI).
///
/// # Safety
/// The implementor answers `setResourceIndex:` (`v@:Q`).
pub unsafe trait ResourceIndexDescriptorSpi: Message {
    fn setResourceIndex(&self, index: u64) {
        unsafe { msg_send![self, setResourceIndex: index] }
    }
}

// SAFETY: each answers `setResourceIndex:` (`tests/spi_encodings.rs`).
unsafe impl ResourceIndexDescriptorSpi for MTLSamplerDescriptor {}
// SAFETY: as above.
unsafe impl ResourceIndexDescriptorSpi for MTLRenderPipelineDescriptor {}
// SAFETY: as above.
unsafe impl ResourceIndexDescriptorSpi for MTLComputePipelineDescriptor {}
// SAFETY: as above.
unsafe impl ResourceIndexDescriptorSpi for MTLIndirectCommandBufferDescriptor {}

/// The force-resource-index setters of the pipeline descriptors (SPI).
///
/// # Safety
/// The implementor answers `setForceResourceIndex:` (`v@:B`).
pub unsafe trait ForceResourceIndexSpi: Message {
    fn setForceResourceIndex(&self, force: bool) {
        unsafe { msg_send![self, setForceResourceIndex: force] }
    }
}

// SAFETY: each answers `setForceResourceIndex:` (`tests/spi_encodings.rs`).
unsafe impl ForceResourceIndexSpi for MTLRenderPipelineDescriptor {}
// SAFETY: as above.
unsafe impl ForceResourceIndexSpi for MTLComputePipelineDescriptor {}

/// `MTLHeapDescriptorSPI`.
///
/// # Safety
/// The implementor is an `MTLHeapDescriptor`.
pub unsafe trait HeapDescriptorSpi: Message {
    fn setAddressRanges(&self, ranges: Option<&MTLResourceAddressRangeArray>) {
        unsafe { msg_send![self, setAddressRanges: ranges] }
    }
    fn setPinnedGPUAddress(&self, address: u64) {
        unsafe { msg_send![self, setPinnedGPUAddress: address] }
    }
}

// SAFETY: an `MTLHeapDescriptor`.
unsafe impl HeapDescriptorSpi for objc2_metal::MTLHeapDescriptor {}

/// `MTLRenderPassDescriptorSPI`.
///
/// # Safety
/// The implementor is an `MTLRenderPassDescriptor`.
pub unsafe trait RenderPassDescriptorSpi: Message {
    fn setDefaultColorSampleCount(&self, count: u64) {
        unsafe { msg_send![self, setDefaultColorSampleCount: count] }
    }
}

// SAFETY: an `MTLRenderPassDescriptor`.
unsafe impl RenderPassDescriptorSpi for MTLRenderPassDescriptor {}

/// The private render state of `MTLRenderPipelineDescriptor` (SPI) that Apple's OpenGL renderer
/// sets, where the descriptor answers each setter ([`RENDER_PIPELINE_PRIVATE_SETTERS`]).
///
/// # Safety
/// The implementor is an `MTLRenderPipelineDescriptor`.
pub unsafe trait RenderPipelineDescriptorSpi: Message {
    fn setOpenGLModeEnabled(&self, enabled: bool) {
        unsafe { msg_send![self, setOpenGLModeEnabled: enabled] }
    }
    fn setSampleMask(&self, mask: u64) {
        unsafe { msg_send![self, setSampleMask: mask] }
    }
    fn setSampleCoverage(&self, coverage: f32) {
        unsafe { msg_send![self, setSampleCoverage: coverage] }
    }
    fn setSampleCoverageInvert(&self, invert: bool) {
        unsafe { msg_send![self, setSampleCoverageInvert: invert] }
    }
    fn setLogicOperationEnabled(&self, enabled: bool) {
        unsafe { msg_send![self, setLogicOperationEnabled: enabled] }
    }
    fn setLogicOperation(&self, operation: u64) {
        unsafe { msg_send![self, setLogicOperation: operation] }
    }
    fn setDepthStencilWriteDisabled(&self, disabled: bool) {
        unsafe { msg_send![self, setDepthStencilWriteDisabled: disabled] }
    }
}

/// The setters of [`RenderPipelineDescriptorSpi`], each of which a descriptor must answer for
/// the platform to take the private render state.
pub const RENDER_PIPELINE_PRIVATE_SETTERS: [&std::ffi::CStr; 7] = [
    c"setOpenGLModeEnabled:",
    c"setSampleMask:",
    c"setSampleCoverage:",
    c"setSampleCoverageInvert:",
    c"setLogicOperationEnabled:",
    c"setLogicOperation:",
    c"setDepthStencilWriteDisabled:",
];

// SAFETY: an `MTLRenderPipelineDescriptor`.
unsafe impl RenderPipelineDescriptorSpi for MTLRenderPipelineDescriptor {}

/// `MTLIndirectCommandBufferDescriptorSPI`.
///
/// # Safety
/// The implementor is an `MTLIndirectCommandBufferDescriptor`.
pub unsafe trait IcbDescriptorSpi: Message {
    fn setAllowOverrideRenderStates(&self, allow: i64) {
        unsafe { msg_send![self, setAllowOverrideRenderStates: allow] }
    }
    fn setInheritBlendColor(&self, inherit: bool) {
        unsafe { msg_send![self, setInheritBlendColor: inherit] }
    }
    fn setInheritDepthTestBounds(&self, inherit: bool) {
        unsafe { msg_send![self, setInheritDepthTestBounds: inherit] }
    }
    fn setInheritScissorRects(&self, inherit: bool) {
        unsafe { msg_send![self, setInheritScissorRects: inherit] }
    }
    fn setInheritStencilReferenceValues(&self, inherit: bool) {
        unsafe { msg_send![self, setInheritStencilReferenceValues: inherit] }
    }
    fn setInheritViewports(&self, inherit: bool) {
        unsafe { msg_send![self, setInheritViewports: inherit] }
    }
    fn setMaxScissorRectCount(&self, count: u64) {
        unsafe { msg_send![self, setMaxScissorRectCount: count] }
    }
    fn setMaxViewportCount(&self, count: u64) {
        unsafe { msg_send![self, setMaxViewportCount: count] }
    }
}

// SAFETY: an `MTLIndirectCommandBufferDescriptor`.
unsafe impl IcbDescriptorSpi for MTLIndirectCommandBufferDescriptor {}

/// `-[MTLRasterizationRateLayerDescriptor initWithSampleCount:]`, which answers nil for a count
/// Metal refuses (objc2-metal declares it nonnull), and the layer's getters objc2-metal lacks or
/// declares nonnull.
///
/// # Safety
/// The implementor is `MTLRasterizationRateLayerDescriptor`.
pub unsafe trait RateLayerDescriptorSpi: Message + Sized {
    fn initWithSampleCount_or_nil(this: Allocated<Self>, count: MTLSize) -> Option<Retained<Self>> {
        unsafe { msg_send![this, initWithSampleCount: count] }
    }
    /// The layer's sample count (objc2-metal does not declare the getter).
    fn sampleCount(&self) -> MTLSize {
        unsafe { msg_send![self, sampleCount] }
    }
    /// The horizontal sample storage, NULL carried (objc2-metal's is nonnull).
    fn horizontalSampleStorage_or_null(&self) -> *mut f32 {
        unsafe { msg_send![self, horizontalSampleStorage] }
    }
    /// The vertical sample storage, NULL carried (objc2-metal's is nonnull).
    fn verticalSampleStorage_or_null(&self) -> *mut f32 {
        unsafe { msg_send![self, verticalSampleStorage] }
    }
}

// SAFETY: `MTLRasterizationRateLayerDescriptor`.
unsafe impl RateLayerDescriptorSpi for MTLRasterizationRateLayerDescriptor {}

/// `MTLRasterizationRateMapDescriptorSPI`.
///
/// # Safety
/// The implementor is an `MTLRasterizationRateMapDescriptor`.
pub unsafe trait RateMapDescriptorSpi: Message {
    fn setMinFactor(&self, factor: f32) {
        unsafe { msg_send![self, setMinFactor: factor] }
    }
    fn setMutability(&self, mutability: u64) {
        unsafe { msg_send![self, setMutability: mutability] }
    }
}

// SAFETY: an `MTLRasterizationRateMapDescriptor`.
unsafe impl RateMapDescriptorSpi for MTLRasterizationRateMapDescriptor {}
