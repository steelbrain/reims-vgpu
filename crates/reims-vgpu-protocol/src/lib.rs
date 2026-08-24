//! Semantic boundary for the paravirtualized GPU protocol.
//!
//! [`reims_vgpu_wire`] owns byte-accurate views. This crate is the first layer
//! allowed to assign meaning to their values. Raw tags are consumed here and
//! semantic state uses the resulting types.

#![no_std]

extern crate alloc;

pub mod blit;
pub mod dispatch;
pub mod geometry;
pub mod identity;
pub mod iosurface;
pub mod mapper_request;
pub mod metal_pixel;
pub mod pass_action;
pub mod pipeline;
pub mod pixel;
pub mod resource;
pub mod resource_state;
pub mod stamp;
pub mod submission;
pub mod texture;
pub mod vertex;
pub mod vertex_step;

pub use blit::{
    BlitCommand, BlitCopyKind, BlitFillSource, BlitKind, BlitPoint, BlitRefKind, BlitSize,
};
pub use dispatch::*;
pub use geometry::{
    align_up_u64, checked_add_u64, checked_mul_u64, mip_extent, size_fits_u32, tight_image_bytes,
    tight_image_layout, tight_layered_image_bytes, Extent3,
};
pub use identity::{
    BackingGeneration, ByteLength, ByteOffset, ContentVersion, GuestPhysicalAddress,
    GuestVirtualAddress, MapperResolvedSurfaceId, MapperSurfaceRef, MappingId, ObjectTableRef,
    PlaneIndex, PreparedShaderId, ResourceId, ResourceNamespaceId, SerializerRef, StorageId,
    SubmissionId, SurfaceBackingId, SurfaceId, TaskId, TextureRotation,
};
pub use iosurface::*;
pub use mapper_request::*;
pub use pass_action::*;
pub use pipeline::{
    blend_factor, blend_operation, blend_state, compare_function, cull_mode, decode_index_type,
    depth_clip_mode, fill_mode, front_face_ccw, index_type, primitive_topology,
    sampler_address_mode, sampler_border_color, sampler_filter, sampler_mip_filter,
    stencil_operation, visibility_result_mode, BlendFactor, BlendOp, BlendStateResource, CullMode,
    DepthClipMode, FillMode, IndexType, IndexTypeDecodeError, PipelineStateDecodeError,
    PrimitiveTopology, SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction,
    SamplerFilter, SamplerMipFilter, StencilOp, VisibilityResultMode,
};
pub use pixel::{
    apply_swizzle_rgba8, swizzle_identity, swizzle_is_identity, swizzle_plan, ImageFormat,
    SampledImageFormat, StorageImageFormat, SwizzlePlan, SwizzleSource, TexelLayout,
    TransferFunction,
};
pub use resource::{
    decode_object_list_entry, decode_surface_backing_descriptor, BufferDescriptor,
    BufferTextureDescriptor, ColorWriteMask, ComputePipelineObject, ComputeStageInputAttribute,
    ComputeStageInputDescriptor, ComputeStageInputLayout, DepthStencilDescriptor, DepthStencilFace,
    DepthStencilObject, EventObject, FenceObject, FunctionDescriptor, FunctionObject,
    IOSurfacePlaneViewDecodeState, IOSurfacePlaneViewDescriptor, IOSurfacePlaneViewRecordKind,
    IOSurfacePlaneViewResourceDescriptor, IcbCommandLayout, IcbCommandMemory, IcbUnappliedFlag,
    IndirectCommandBufferDescriptor, LinearTextureDescriptor, ObjectKind, ObjectListDecodeError,
    ObjectListEntry, RenderPipelineDescriptor, RenderPipelineObject, ResourceDecodeError,
    ResourceDescriptor, SamplerDescriptor, SamplerObject, SurfaceBackingDescriptor,
    SurfaceBackingPlane, TextureLevelLayout, TextureViewDescriptor, TextureViewForm,
    VertexAttribute, CUBE_FACES, MAX_COLOR_ATTACHMENTS, MTL_COLOR_WRITE_MASK_ALL,
    MTL_COLOR_WRITE_MASK_ALPHA, MTL_COLOR_WRITE_MASK_BLUE, MTL_COLOR_WRITE_MASK_GREEN,
    MTL_COLOR_WRITE_MASK_NONE, MTL_COLOR_WRITE_MASK_RED, OBJECT_LIST_ENTRY_LEN,
};
pub use resource_state::ResourceValidityOps;
pub use stamp::{StampWait, STAMP_INDEX_MASK, STAMP_SLOT_LEN};
pub use submission::{
    HeapObject, IndirectCommandBufferObject, ResourceObject, ResourceValidity, SegmentBoundary,
    SegmentKind, SubmissionIdentity, SubmissionResourceUse,
};
pub use texture::{
    decode_heap_texture_descriptor, decode_mapper_iosurface_texture_view,
    texture_declaration_from_narrow, texture_declaration_from_wide, GuestWriteAnnouncement,
    HeapTextureDescriptor, MapperIOSurfaceTextureDecodeError, MapperIOSurfaceTextureView,
    StorageMode, TextureDeclaration,
};
pub use vertex::{decode_vertex_attribute_format, VertexAttributeFormat, VertexFormatDecodeError};
pub use vertex_step::{
    decode_vertex_step_function, step_rate_in_contract, VertexStepDecodeError, VertexStepFunction,
    MTL_VERTEX_STEP_FUNCTION_CONSTANT, MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
    MTL_VERTEX_STEP_FUNCTION_PER_PATCH, MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT,
    MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
};
