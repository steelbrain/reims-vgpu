//! The render-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_RENDER`](super::RGFX_OP_GROUP_RENDER)):
//! one tag and one body struct per `MTLRenderCommandEncoder` selector the guest
//! serialises. The host's render decoder (`reims-vgpu`'s `graphics::decode::render`)
//! reads the body, resolves its object references and sends the matching
//! `RenderCmd` (docs/graphics-kext/design.md, "Stage 3 wire protocol").
//!
//! A body struct is the op's bytes **after** its [`RgfxOpHeader`](super::RgfxOpHeader),
//! not the whole op (see the [module doc](super)). Object references are `u64`
//! `obj_id` (0 = nil). Floating-point arguments travel as the bits of their
//! `f64`/`f32`, so the schema stays all-integer and comparable, as the sampler's
//! LOD clamps do in [`crate::wire`].
//!
//! This first slice carries the reference subset the host decoder implements —
//! enough to prove the pattern end to end; the rest of the ~70 render selectors
//! extend this module, one tag and one body struct each, their tags staying in
//! the `0x1xxx` range.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Op `SET_RENDER_PIPELINE_STATE` ([`RgfxOpSetRenderPipelineState`]):
/// `setRenderPipelineState:`.
pub const RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE: u16 = 0x1000;
/// Op `SET_VERTEX_BUFFER` ([`RgfxOpSetVertexBuffer`]):
/// `setVertexBuffer:offset:atIndex:`.
pub const RGFX_OP_RENDER_SET_VERTEX_BUFFER: u16 = 0x1001;
/// Op `SET_FRAGMENT_BUFFER` ([`RgfxOpSetFragmentBuffer`]):
/// `setFragmentBuffer:offset:atIndex:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_BUFFER: u16 = 0x1002;
/// Op `SET_FRAGMENT_TEXTURE` ([`RgfxOpSetFragmentTexture`]):
/// `setFragmentTexture:atIndex:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE: u16 = 0x1003;
/// Op `SET_VIEWPORT` ([`RgfxOpSetViewport`]): `setViewport:`.
pub const RGFX_OP_RENDER_SET_VIEWPORT: u16 = 0x1004;
/// Op `SET_SCISSOR_RECT` ([`RgfxOpSetScissorRect`]): `setScissorRect:`.
pub const RGFX_OP_RENDER_SET_SCISSOR_RECT: u16 = 0x1005;
/// Op `DRAW_PRIMITIVES` ([`RgfxOpDrawPrimitives`]):
/// `drawPrimitives:vertexStart:vertexCount:`.
pub const RGFX_OP_RENDER_DRAW_PRIMITIVES: u16 = 0x1006;
/// Op `DRAW_INDEXED_PRIMITIVES` ([`RgfxOpDrawIndexedPrimitives`]):
/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES: u16 = 0x1007;

// The rest of the render-encoder ops, in `RenderCmd`'s order. Their tags continue
// from the reference subset's, staying in the render group (`0x1xxx`).

/// Op `DRAW_PRIMITIVES_INSTANCES` ([`RgfxOpDrawPrimitivesInstances`]):
/// `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
pub const RGFX_OP_RENDER_DRAW_PRIMITIVES_INSTANCES: u16 = 0x1008;
/// Op `DRAW_PRIMITIVES_BASE` ([`RgfxOpDrawPrimitivesBase`]):
/// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
pub const RGFX_OP_RENDER_DRAW_PRIMITIVES_BASE: u16 = 0x1009;
/// Op `DRAW_PRIMITIVES_INDIRECT` ([`RgfxOpDrawPrimitivesIndirect`]):
/// `drawPrimitives:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_RENDER_DRAW_PRIMITIVES_INDIRECT: u16 = 0x100a;
/// Op `DRAW_INDEXED_PRIMITIVES_INSTANCES` ([`RgfxOpDrawIndexedPrimitivesInstances`]):
/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES: u16 = 0x100b;
/// Op `DRAW_INDEXED_PRIMITIVES_BASE` ([`RgfxOpDrawIndexedPrimitivesBase`]):
/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_BASE: u16 = 0x100c;
/// Op `DRAW_INDEXED_PRIMITIVES_INDIRECT` ([`RgfxOpDrawIndexedPrimitivesIndirect`]):
/// `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INDIRECT: u16 = 0x100d;
/// Op `DRAW_PATCHES` ([`RgfxOpDrawPatches`]):
/// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:`.
pub const RGFX_OP_RENDER_DRAW_PATCHES: u16 = 0x100e;
/// Op `DRAW_PATCHES_INDIRECT` ([`RgfxOpDrawPatchesIndirect`]):
/// `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_RENDER_DRAW_PATCHES_INDIRECT: u16 = 0x100f;
/// Op `DRAW_INDEXED_PATCHES` ([`RgfxOpDrawIndexedPatches`]):
/// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PATCHES: u16 = 0x1010;
/// Op `DRAW_INDEXED_PATCHES_INDIRECT` ([`RgfxOpDrawIndexedPatchesIndirect`]):
/// `drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PATCHES_INDIRECT: u16 = 0x1011;
/// Op `DISPATCH_THREADS_PER_TILE` ([`RgfxOpDispatchThreadsPerTile`]):
/// `dispatchThreadsPerTile:`.
pub const RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE: u16 = 0x1012;
/// Op `DISPATCH_THREADS_PER_TILE_IN_REGION` ([`RgfxOpDispatchThreadsPerTileInRegion`]):
/// `dispatchThreadsPerTile:inRegion:` (SPI).
pub const RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION: u16 = 0x1013;
/// Op `DISPATCH_THREADS_PER_TILE_IN_REGION_INDEX`
/// ([`RgfxOpDispatchThreadsPerTileInRegionIndex`]):
/// `dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:` (SPI).
pub const RGFX_OP_RENDER_DISPATCH_THREADS_PER_TILE_IN_REGION_INDEX: u16 = 0x1014;
/// Op `EXECUTE_COMMANDS_IN_BUFFER` ([`RgfxOpExecuteCommandsInBuffer`]):
/// `executeCommandsInBuffer:withRange:`.
pub const RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER: u16 = 0x1015;
/// Op `EXECUTE_COMMANDS_INDIRECT` ([`RgfxOpExecuteCommandsIndirect`]):
/// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT: u16 = 0x1016;
/// Op `MEMORY_BARRIER_WITH_SCOPE` ([`RgfxOpMemoryBarrierWithScope`]):
/// `memoryBarrierWithScope:afterStages:beforeStages:`.
pub const RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE: u16 = 0x1017;
/// Op `UPDATE_FENCE` ([`RgfxOpUpdateFence`]): `updateFence:afterStages:`.
pub const RGFX_OP_RENDER_UPDATE_FENCE: u16 = 0x1018;
/// Op `WAIT_FOR_FENCE` ([`RgfxOpWaitForFence`]): `waitForFence:beforeStages:`.
pub const RGFX_OP_RENDER_WAIT_FOR_FENCE: u16 = 0x1019;
/// Op `SET_CULL_MODE` ([`RgfxOpSetCullMode`]): `setCullMode:`.
pub const RGFX_OP_RENDER_SET_CULL_MODE: u16 = 0x101a;
/// Op `SET_FRONT_FACING_WINDING` ([`RgfxOpSetFrontFacingWinding`]):
/// `setFrontFacingWinding:`.
pub const RGFX_OP_RENDER_SET_FRONT_FACING_WINDING: u16 = 0x101b;
/// Op `SET_TRIANGLE_FILL_MODE` ([`RgfxOpSetTriangleFillMode`]): `setTriangleFillMode:`.
pub const RGFX_OP_RENDER_SET_TRIANGLE_FILL_MODE: u16 = 0x101c;
/// Op `SET_DEPTH_CLIP_MODE_SPI` ([`RgfxOpSetDepthClipModeSpi`]): `setDepthClipModeSPI:`
/// (SPI).
pub const RGFX_OP_RENDER_SET_DEPTH_CLIP_MODE_SPI: u16 = 0x101d;
/// Op `SET_PROVOKING_VERTEX_MODE` ([`RgfxOpSetProvokingVertexMode`]):
/// `setProvokingVertexMode:` (SPI).
pub const RGFX_OP_RENDER_SET_PROVOKING_VERTEX_MODE: u16 = 0x101e;
/// Op `SET_DEPTH_STORE_ACTION` ([`RgfxOpSetDepthStoreAction`]): `setDepthStoreAction:`.
pub const RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION: u16 = 0x101f;
/// Op `SET_STENCIL_STORE_ACTION` ([`RgfxOpSetStencilStoreAction`]):
/// `setStencilStoreAction:`.
pub const RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION: u16 = 0x1020;
/// Op `SET_STENCIL_STORE_ACTION_OPTIONS` ([`RgfxOpSetStencilStoreActionOptions`]):
/// `setStencilStoreActionOptions:`.
pub const RGFX_OP_RENDER_SET_STENCIL_STORE_ACTION_OPTIONS: u16 = 0x1021;
/// Op `SET_COLOR_STORE_ACTION` ([`RgfxOpSetColorStoreAction`]):
/// `setColorStoreAction:atIndex:`.
pub const RGFX_OP_RENDER_SET_COLOR_STORE_ACTION: u16 = 0x1022;
/// Op `SET_COLOR_STORE_ACTION_OPTIONS` ([`RgfxOpSetColorStoreActionOptions`]):
/// `setColorStoreActionOptions:atIndex:`.
pub const RGFX_OP_RENDER_SET_COLOR_STORE_ACTION_OPTIONS: u16 = 0x1023;
/// Op `SET_VISIBILITY_RESULT_MODE` ([`RgfxOpSetVisibilityResultMode`]):
/// `setVisibilityResultMode:offset:`.
pub const RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE: u16 = 0x1024;
/// Op `SET_VERTEX_AMPLIFICATION_MODE` ([`RgfxOpSetVertexAmplificationMode`]):
/// `setVertexAmplificationMode:value:` (SPI).
pub const RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_MODE: u16 = 0x1025;
/// Op `SET_VERTEX_BUFFER_OFFSET` ([`RgfxOpSetVertexBufferOffset`]):
/// `setVertexBufferOffset:atIndex:`.
pub const RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET: u16 = 0x1026;
/// Op `SET_FRAGMENT_BUFFER_OFFSET` ([`RgfxOpSetFragmentBufferOffset`]):
/// `setFragmentBufferOffset:atIndex:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_BUFFER_OFFSET: u16 = 0x1027;
/// Op `SET_TILE_BUFFER_OFFSET` ([`RgfxOpSetTileBufferOffset`]):
/// `setTileBufferOffset:atIndex:`.
pub const RGFX_OP_RENDER_SET_TILE_BUFFER_OFFSET: u16 = 0x1028;
/// Op `SET_VERTEX_BUFFER_OFFSET_ATTRIBUTE_STRIDE`
/// ([`RgfxOpSetVertexBufferOffsetAttributeStride`]):
/// `setVertexBufferOffset:attributeStride:atIndex:`.
pub const RGFX_OP_RENDER_SET_VERTEX_BUFFER_OFFSET_ATTRIBUTE_STRIDE: u16 = 0x1029;
/// Op `SET_THREADGROUP_MEMORY_LENGTH` ([`RgfxOpSetThreadgroupMemoryLength`]):
/// `setThreadgroupMemoryLength:offset:atIndex:`.
pub const RGFX_OP_RENDER_SET_THREADGROUP_MEMORY_LENGTH: u16 = 0x102a;
/// Op `SET_ALPHA_TEST_REFERENCE_VALUE` ([`RgfxOpSetAlphaTestReferenceValue`]):
/// `setAlphaTestReferenceValue:` (SPI).
pub const RGFX_OP_RENDER_SET_ALPHA_TEST_REFERENCE_VALUE: u16 = 0x102b;
/// Op `SET_LINE_WIDTH` ([`RgfxOpSetLineWidth`]): `setLineWidth:` (SPI).
pub const RGFX_OP_RENDER_SET_LINE_WIDTH: u16 = 0x102c;
/// Op `SET_POINT_SIZE` ([`RgfxOpSetPointSize`]): `setPointSize:` (SPI).
pub const RGFX_OP_RENDER_SET_POINT_SIZE: u16 = 0x102d;
/// Op `SET_TESSELLATION_FACTOR_SCALE` ([`RgfxOpSetTessellationFactorScale`]):
/// `setTessellationFactorScale:`.
pub const RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_SCALE: u16 = 0x102e;
/// Op `SET_BLEND_COLOR_RED` ([`RgfxOpSetBlendColorRed`]):
/// `setBlendColorRed:green:blue:alpha:`.
pub const RGFX_OP_RENDER_SET_BLEND_COLOR_RED: u16 = 0x102f;
/// Op `SET_DEPTH_BIAS` ([`RgfxOpSetDepthBias`]): `setDepthBias:slopeScale:clamp:`.
pub const RGFX_OP_RENDER_SET_DEPTH_BIAS: u16 = 0x1030;
/// Op `SET_VIEWPORT_TRANSFORM_ENABLED` ([`RgfxOpSetViewportTransformEnabled`]):
/// `setViewportTransformEnabled:` (SPI).
pub const RGFX_OP_RENDER_SET_VIEWPORT_TRANSFORM_ENABLED: u16 = 0x1031;
/// Op `SET_PRIMITIVE_RESTART_ENABLED` ([`RgfxOpSetPrimitiveRestartEnabled`]):
/// `setPrimitiveRestartEnabled:index:` (SPI).
pub const RGFX_OP_RENDER_SET_PRIMITIVE_RESTART_ENABLED: u16 = 0x1032;
/// Op `SET_STENCIL_FRONT_REFERENCE_VALUE` ([`RgfxOpSetStencilFrontReferenceValue`]):
/// `setStencilFrontReferenceValue:backReferenceValue:`.
pub const RGFX_OP_RENDER_SET_STENCIL_FRONT_REFERENCE_VALUE: u16 = 0x1033;
/// Op `SET_DEPTH_CLEARED` ([`RgfxOpSetDepthCleared`]): `setDepthCleared` (SPI, no body).
pub const RGFX_OP_RENDER_SET_DEPTH_CLEARED: u16 = 0x1034;
/// Op `SET_STENCIL_CLEARED` ([`RgfxOpSetStencilCleared`]): `setStencilCleared` (SPI, no
/// body).
pub const RGFX_OP_RENDER_SET_STENCIL_CLEARED: u16 = 0x1035;
/// Op `SET_DEPTH_STENCIL_STATE` ([`RgfxOpSetDepthStencilState`]): `setDepthStencilState:`.
pub const RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE: u16 = 0x1036;
/// Op `SET_COLOR_RESOLVE_TEXTURE` ([`RgfxOpSetColorResolveTexture`]):
/// `setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:` (SPI).
pub const RGFX_OP_RENDER_SET_COLOR_RESOLVE_TEXTURE: u16 = 0x1037;
/// Op `SET_DEPTH_RESOLVE_TEXTURE` ([`RgfxOpSetDepthResolveTexture`]):
/// `setDepthResolveTexture:slice:depthPlane:level:yInvert:` (SPI).
pub const RGFX_OP_RENDER_SET_DEPTH_RESOLVE_TEXTURE: u16 = 0x1038;
/// Op `SET_STENCIL_RESOLVE_TEXTURE` ([`RgfxOpSetStencilResolveTexture`]):
/// `setStencilResolveTexture:slice:depthPlane:level:yInvert:` (SPI).
pub const RGFX_OP_RENDER_SET_STENCIL_RESOLVE_TEXTURE: u16 = 0x1039;
/// Op `SET_TESSELLATION_FACTOR_BUFFER` ([`RgfxOpSetTessellationFactorBuffer`]):
/// `setTessellationFactorBuffer:offset:instanceStride:`.
pub const RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_BUFFER: u16 = 0x103a;
/// Op `SET_VERTEX_BUFFERS` ([`RgfxOpSetVertexBuffers`]):
/// `setVertexBuffers:offsets:withRange:`.
pub const RGFX_OP_RENDER_SET_VERTEX_BUFFERS: u16 = 0x103b;
/// Op `SET_FRAGMENT_BUFFERS` ([`RgfxOpSetFragmentBuffers`]):
/// `setFragmentBuffers:offsets:withRange:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_BUFFERS: u16 = 0x103c;
/// Op `SET_TILE_BUFFERS` ([`RgfxOpSetTileBuffers`]): `setTileBuffers:offsets:withRange:`.
pub const RGFX_OP_RENDER_SET_TILE_BUFFERS: u16 = 0x103d;
/// Op `SET_VERTEX_BUFFERS_ATTRIBUTE_STRIDES` ([`RgfxOpSetVertexBuffersAttributeStrides`]):
/// `setVertexBuffers:offsets:attributeStrides:withRange:`.
pub const RGFX_OP_RENDER_SET_VERTEX_BUFFERS_ATTRIBUTE_STRIDES: u16 = 0x103e;
/// Op `SET_VERTEX_TEXTURES` ([`RgfxOpSetVertexTextures`]): `setVertexTextures:withRange:`.
pub const RGFX_OP_RENDER_SET_VERTEX_TEXTURES: u16 = 0x103f;
/// Op `SET_FRAGMENT_TEXTURES` ([`RgfxOpSetFragmentTextures`]):
/// `setFragmentTextures:withRange:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_TEXTURES: u16 = 0x1040;
/// Op `SET_TILE_TEXTURES` ([`RgfxOpSetTileTextures`]): `setTileTextures:withRange:`.
pub const RGFX_OP_RENDER_SET_TILE_TEXTURES: u16 = 0x1041;
/// Op `SET_VERTEX_SAMPLER_STATES` ([`RgfxOpSetVertexSamplerStates`]):
/// `setVertexSamplerStates:withRange:`.
pub const RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES: u16 = 0x1042;
/// Op `SET_FRAGMENT_SAMPLER_STATES` ([`RgfxOpSetFragmentSamplerStates`]):
/// `setFragmentSamplerStates:withRange:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES: u16 = 0x1043;
/// Op `SET_TILE_SAMPLER_STATES` ([`RgfxOpSetTileSamplerStates`]):
/// `setTileSamplerStates:withRange:`.
pub const RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES: u16 = 0x1044;
/// Op `SET_VERTEX_SAMPLER_STATES_LOD_CLAMPS` ([`RgfxOpSetVertexSamplerStatesLodClamps`]):
/// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_RENDER_SET_VERTEX_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x1045;
/// Op `SET_FRAGMENT_SAMPLER_STATES_LOD_CLAMPS`
/// ([`RgfxOpSetFragmentSamplerStatesLodClamps`]):
/// `setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x1046;
/// Op `SET_TILE_SAMPLER_STATES_LOD_CLAMPS` ([`RgfxOpSetTileSamplerStatesLodClamps`]):
/// `setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_RENDER_SET_TILE_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x1047;
/// Op `SET_SCISSOR_RECTS` ([`RgfxOpSetScissorRects`]): `setScissorRects:count:`.
pub const RGFX_OP_RENDER_SET_SCISSOR_RECTS: u16 = 0x1048;
/// Op `SET_VIEWPORTS` ([`RgfxOpSetViewports`]): `setViewports:count:`.
pub const RGFX_OP_RENDER_SET_VIEWPORTS: u16 = 0x1049;
/// Op `SET_VERTEX_AMPLIFICATION_COUNT` ([`RgfxOpSetVertexAmplificationCount`]):
/// `setVertexAmplificationCount:viewMappings:`.
pub const RGFX_OP_RENDER_SET_VERTEX_AMPLIFICATION_COUNT: u16 = 0x104a;
/// Op `USE_HEAPS` ([`RgfxOpUseHeaps`]): `useHeaps:count:`.
pub const RGFX_OP_RENDER_USE_HEAPS: u16 = 0x104b;
/// Op `USE_HEAPS_STAGES` ([`RgfxOpUseHeapsStages`]): `useHeaps:count:stages:`.
pub const RGFX_OP_RENDER_USE_HEAPS_STAGES: u16 = 0x104c;
/// Op `USE_RESOURCES` ([`RgfxOpUseResources`]): `useResources:count:usage:`.
pub const RGFX_OP_RENDER_USE_RESOURCES: u16 = 0x104d;
/// Op `USE_RESOURCES_STAGES` ([`RgfxOpUseResourcesStages`]):
/// `useResources:count:usage:stages:`.
pub const RGFX_OP_RENDER_USE_RESOURCES_STAGES: u16 = 0x104e;
/// Op `SET_DEPTH_STORE_ACTION_OPTIONS` ([`RgfxOpSetDepthStoreActionOptions`]):
/// `setDepthStoreActionOptions:`.
pub const RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS: u16 = 0x104f;
/// Op `RESET_STATE` (no body), where the host offers
/// [`RGFX_FEATURE_CB_PARTS`](crate::wire::RGFX_FEATURE_CB_PARTS): the open render encoder's state
/// becomes a new encoder's for its pass, its bindings kept. No host offers the feature yet, and a
/// host without it faults the buffer at the op, as at any tag it does not know.
pub const RGFX_OP_RENDER_RESET_STATE: u16 = 0x1050;

// The object and mesh stages' bindings and the mesh draws, where the guest took
// [`RGFX_FEATURE_MESH_PIPELINES`](crate::wire::RGFX_FEATURE_MESH_PIPELINES): a host without the
// feature faults the buffer at each, as at any tag it does not know.

/// Op `SET_OBJECT_BUFFERS` ([`RgfxOpSetObjectBuffers`]): `setObjectBuffers:offsets:withRange:`,
/// and `setObjectBuffer:offset:atIndex:` as a range of one.
pub const RGFX_OP_RENDER_SET_OBJECT_BUFFERS: u16 = 0x1051;
/// Op `SET_MESH_BUFFERS` ([`RgfxOpSetMeshBuffers`]): `setMeshBuffers:offsets:withRange:`, and
/// `setMeshBuffer:offset:atIndex:` as a range of one.
pub const RGFX_OP_RENDER_SET_MESH_BUFFERS: u16 = 0x1052;
/// Op `SET_OBJECT_BUFFER_OFFSET` ([`RgfxOpSetObjectBufferOffset`]):
/// `setObjectBufferOffset:atIndex:`.
pub const RGFX_OP_RENDER_SET_OBJECT_BUFFER_OFFSET: u16 = 0x1053;
/// Op `SET_MESH_BUFFER_OFFSET` ([`RgfxOpSetMeshBufferOffset`]): `setMeshBufferOffset:atIndex:`.
pub const RGFX_OP_RENDER_SET_MESH_BUFFER_OFFSET: u16 = 0x1054;
/// Op `SET_OBJECT_TEXTURES` ([`RgfxOpSetObjectTextures`]): `setObjectTextures:withRange:`.
pub const RGFX_OP_RENDER_SET_OBJECT_TEXTURES: u16 = 0x1055;
/// Op `SET_MESH_TEXTURES` ([`RgfxOpSetMeshTextures`]): `setMeshTextures:withRange:`.
pub const RGFX_OP_RENDER_SET_MESH_TEXTURES: u16 = 0x1056;
/// Op `SET_OBJECT_SAMPLER_STATES` ([`RgfxOpSetObjectSamplerStates`]):
/// `setObjectSamplerStates:withRange:`.
pub const RGFX_OP_RENDER_SET_OBJECT_SAMPLER_STATES: u16 = 0x1057;
/// Op `SET_MESH_SAMPLER_STATES` ([`RgfxOpSetMeshSamplerStates`]):
/// `setMeshSamplerStates:withRange:`.
pub const RGFX_OP_RENDER_SET_MESH_SAMPLER_STATES: u16 = 0x1058;
/// Op `SET_OBJECT_SAMPLER_STATES_LOD_CLAMPS` ([`RgfxOpSetObjectSamplerStatesLodClamps`]):
/// `setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_RENDER_SET_OBJECT_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x1059;
/// Op `SET_MESH_SAMPLER_STATES_LOD_CLAMPS` ([`RgfxOpSetMeshSamplerStatesLodClamps`]):
/// `setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_RENDER_SET_MESH_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x105a;
/// Op `SET_OBJECT_THREADGROUP_MEMORY_LENGTH` ([`RgfxOpSetObjectThreadgroupMemoryLength`]):
/// `setObjectThreadgroupMemoryLength:atIndex:`.
pub const RGFX_OP_RENDER_SET_OBJECT_THREADGROUP_MEMORY_LENGTH: u16 = 0x105b;
/// Op `DRAW_MESH_THREADGROUPS` ([`RgfxOpDrawMeshThreadgroups`]):
/// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
pub const RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS: u16 = 0x105c;
/// Op `DRAW_MESH_THREADS` ([`RgfxOpDrawMeshThreads`]):
/// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
pub const RGFX_OP_RENDER_DRAW_MESH_THREADS: u16 = 0x105d;
/// Op `DRAW_MESH_THREADGROUPS_INDIRECT` ([`RgfxOpDrawMeshThreadgroupsIndirect`]):
/// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
pub const RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT: u16 = 0x105e;

// The resource-kind tags are shared across the encoder groups (defined once in the parent
// module) so the guest-facing ABI has a single numbering.
pub use super::{
    RGFX_RESOURCE_KIND_BUFFER, RGFX_RESOURCE_KIND_ICB, RGFX_RESOURCE_KIND_NONE,
    RGFX_RESOURCE_KIND_TEXTURE,
};

/// `setRenderPipelineState:`. The pipeline may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetRenderPipelineState {
    /// The render pipeline state, or 0 for nil.
    pub pipeline_obj_id: u64,
}

/// `setVertexBuffer:offset:atIndex:`. The buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexBuffer {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setFragmentBuffer:offset:atIndex:`. The buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentBuffer {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setFragmentTexture:atIndex:`. The texture may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentTexture {
    /// The texture, or 0 for nil.
    pub texture_obj_id: u64,
    /// The texture binding index.
    pub index: u64,
}

/// `setViewport:`. The six `MTLViewport` fields travel as the bits of their `f64`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetViewport {
    /// `originX` as `f64` bits.
    pub origin_x_bits: u64,
    /// `originY` as `f64` bits.
    pub origin_y_bits: u64,
    /// `width` as `f64` bits.
    pub width_bits: u64,
    /// `height` as `f64` bits.
    pub height_bits: u64,
    /// `znear` as `f64` bits.
    pub znear_bits: u64,
    /// `zfar` as `f64` bits.
    pub zfar_bits: u64,
}

/// `setScissorRect:`. The four `MTLScissorRect` fields are `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetScissorRect {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

/// `drawPrimitives:vertexStart:vertexCount:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPrimitives {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The first vertex.
    pub vertex_start: u64,
    /// The vertices to draw.
    pub vertex_count: u64,
}

/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`.
/// The index buffer must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPrimitives {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The indices to draw.
    pub index_count: u64,
    /// `MTLIndexType`.
    pub index_type: u64,
    /// The index buffer (must not be nil).
    pub index_buffer_obj_id: u64,
    /// The offset into the index buffer.
    pub index_buffer_offset: u64,
}

// ---------------------------------------------------------------------------
// The rest of the render-encoder op bodies, in `RenderCmd`'s order. Scalar
// arguments are `u64` (`NSUInteger` and the raw Metal enums), floats travel as
// `u32`/`u64` bits, and object references are `u64` `obj_id` (0 = nil). The
// array-tailed ops (`SetVertexBuffers`, `UseResources`, `SetViewports`, …) are a
// fixed head struct then a trailing element array at the head's `size_of`, their
// count in the head, exactly like the render pass's colour attachments.
// ---------------------------------------------------------------------------

/// `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPrimitivesInstances {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The first vertex.
    pub vertex_start: u64,
    /// The vertices to draw.
    pub vertex_count: u64,
    /// The instances to draw.
    pub instance_count: u64,
}

/// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPrimitivesBase {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The first vertex.
    pub vertex_start: u64,
    /// The vertices to draw.
    pub vertex_count: u64,
    /// The instances to draw.
    pub instance_count: u64,
    /// The first instance.
    pub base_instance: u64,
}

/// `drawPrimitives:indirectBuffer:indirectBufferOffset:`. The indirect buffer must
/// name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPrimitivesIndirect {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The indirect-arguments buffer (must not be nil).
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
}

/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:`.
/// The index buffer must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPrimitivesInstances {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The indices to draw.
    pub index_count: u64,
    /// `MTLIndexType`.
    pub index_type: u64,
    /// The index buffer (must not be nil).
    pub index_buffer_obj_id: u64,
    /// The offset into the index buffer.
    pub index_buffer_offset: u64,
    /// The instances to draw.
    pub instance_count: u64,
}

/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`.
/// The index buffer must name an object; `base_vertex` is signed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPrimitivesBase {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The indices to draw.
    pub index_count: u64,
    /// `MTLIndexType`.
    pub index_type: u64,
    /// The index buffer (must not be nil).
    pub index_buffer_obj_id: u64,
    /// The offset into the index buffer.
    pub index_buffer_offset: u64,
    /// The instances to draw.
    pub instance_count: u64,
    /// The value added to each index before fetching a vertex (`NSInteger`).
    pub base_vertex: i64,
    /// The first instance.
    pub base_instance: u64,
}

/// `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:`.
/// Both buffers must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPrimitivesIndirect {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// `MTLIndexType`.
    pub index_type: u64,
    /// The index buffer (must not be nil).
    pub index_buffer_obj_id: u64,
    /// The offset into the index buffer.
    pub index_buffer_offset: u64,
    /// The indirect-arguments buffer (must not be nil).
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
}

/// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:`.
/// The patch index buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPatches {
    /// Control points per patch.
    pub number_of_patch_control_points: u64,
    /// The first patch.
    pub patch_start: u64,
    /// The patches to draw.
    pub patch_count: u64,
    /// The patch index buffer, or 0 for nil.
    pub patch_index_buffer_obj_id: u64,
    /// The offset into the patch index buffer.
    pub patch_index_buffer_offset: u64,
    /// The instances to draw.
    pub instance_count: u64,
    /// The first instance.
    pub base_instance: u64,
}

/// `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:`.
/// The patch index buffer may be nil; the indirect buffer must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPatchesIndirect {
    /// Control points per patch.
    pub number_of_patch_control_points: u64,
    /// The patch index buffer, or 0 for nil.
    pub patch_index_buffer_obj_id: u64,
    /// The offset into the patch index buffer.
    pub patch_index_buffer_offset: u64,
    /// The indirect-arguments buffer (must not be nil).
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
}

/// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:`.
/// The patch index buffer may be nil; the control-point index buffer must name an
/// object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPatches {
    /// Control points per patch.
    pub number_of_patch_control_points: u64,
    /// The first patch.
    pub patch_start: u64,
    /// The patches to draw.
    pub patch_count: u64,
    /// The patch index buffer, or 0 for nil.
    pub patch_index_buffer_obj_id: u64,
    /// The offset into the patch index buffer.
    pub patch_index_buffer_offset: u64,
    /// The control-point index buffer (must not be nil).
    pub control_point_index_buffer_obj_id: u64,
    /// The offset into the control-point index buffer.
    pub control_point_index_buffer_offset: u64,
    /// The instances to draw.
    pub instance_count: u64,
    /// The first instance.
    pub base_instance: u64,
}

/// `drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:`.
/// The patch index buffer may be nil; the control-point index and indirect buffers
/// must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPatchesIndirect {
    /// Control points per patch.
    pub number_of_patch_control_points: u64,
    /// The patch index buffer, or 0 for nil.
    pub patch_index_buffer_obj_id: u64,
    /// The offset into the patch index buffer.
    pub patch_index_buffer_offset: u64,
    /// The control-point index buffer (must not be nil).
    pub control_point_index_buffer_obj_id: u64,
    /// The offset into the control-point index buffer.
    pub control_point_index_buffer_offset: u64,
    /// The indirect-arguments buffer (must not be nil).
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
}

/// `dispatchThreadsPerTile:`. The `MTLSize` fields are `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDispatchThreadsPerTile {
    /// `threadsPerTile.width`.
    pub width: u64,
    /// `threadsPerTile.height`.
    pub height: u64,
    /// `threadsPerTile.depth`.
    pub depth: u64,
}

/// `dispatchThreadsPerTile:inRegion:` (SPI). The region is in pixels.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDispatchThreadsPerTileInRegion {
    /// `threadsPerTile.width`.
    pub width: u64,
    /// `threadsPerTile.height`.
    pub height: u64,
    /// `threadsPerTile.depth`.
    pub depth: u64,
    /// `region.origin.x`.
    pub origin_x: u64,
    /// `region.origin.y`.
    pub origin_y: u64,
    /// `region.origin.z`.
    pub origin_z: u64,
    /// `region.size.width`.
    pub region_width: u64,
    /// `region.size.height`.
    pub region_height: u64,
    /// `region.size.depth`.
    pub region_depth: u64,
}

/// `dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:` (SPI). The index is
/// a `uint32_t`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDispatchThreadsPerTileInRegionIndex {
    /// `threadsPerTile.width`.
    pub width: u64,
    /// `threadsPerTile.height`.
    pub height: u64,
    /// `threadsPerTile.depth`.
    pub depth: u64,
    /// `region.origin.x`.
    pub origin_x: u64,
    /// `region.origin.y`.
    pub origin_y: u64,
    /// `region.origin.z`.
    pub origin_z: u64,
    /// `region.size.width`.
    pub region_width: u64,
    /// `region.size.height`.
    pub region_height: u64,
    /// `region.size.depth`.
    pub region_depth: u64,
    /// `renderTargetArrayIndex`.
    pub render_target_array_index: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_4c: u32,
}

/// `executeCommandsInBuffer:withRange:`. The indirect command buffer must name an
/// object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpExecuteCommandsInBuffer {
    /// The indirect command buffer (must not be nil).
    pub indirect_command_buffer_obj_id: u64,
    /// `range.location`.
    pub range_location: u64,
    /// `range.length`.
    pub range_length: u64,
}

/// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`. Both the indirect
/// command buffer and the range buffer must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpExecuteCommandsIndirect {
    /// The indirect command buffer (must not be nil).
    pub indirect_command_buffer_obj_id: u64,
    /// The buffer holding the execution range (must not be nil).
    pub indirect_range_buffer_obj_id: u64,
    /// The offset into the range buffer.
    pub indirect_buffer_offset: u64,
}

/// `memoryBarrierWithScope:afterStages:beforeStages:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpMemoryBarrierWithScope {
    /// `MTLBarrierScope`.
    pub scope: u64,
    /// `MTLRenderStages` the barrier is after.
    pub after: u64,
    /// `MTLRenderStages` the barrier is before.
    pub before: u64,
}

/// `updateFence:afterStages:`. The fence must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpUpdateFence {
    /// The fence (must not be nil).
    pub fence_obj_id: u64,
    /// `MTLRenderStages`.
    pub stages: u64,
}

/// `waitForFence:beforeStages:`. The fence must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpWaitForFence {
    /// The fence (must not be nil).
    pub fence_obj_id: u64,
    /// `MTLRenderStages`.
    pub stages: u64,
}

/// `setCullMode:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetCullMode {
    /// `MTLCullMode`.
    pub cull_mode: u64,
}

/// `setFrontFacingWinding:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFrontFacingWinding {
    /// `MTLWinding`.
    pub front_facing_winding: u64,
}

/// `setTriangleFillMode:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTriangleFillMode {
    /// `MTLTriangleFillMode`.
    pub fill_mode: u64,
}

/// `setDepthClipModeSPI:` (SPI).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthClipModeSpi {
    /// `MTLDepthClipMode`.
    pub mode: u64,
}

/// `setProvokingVertexMode:` (SPI).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetProvokingVertexMode {
    /// The provoking-vertex mode.
    pub mode: u64,
}

/// `setDepthStoreAction:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthStoreAction {
    /// `MTLStoreAction`.
    pub store_action: u64,
}

/// `setStencilStoreAction:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetStencilStoreAction {
    /// `MTLStoreAction`.
    pub store_action: u64,
}

/// `setStencilStoreActionOptions:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetStencilStoreActionOptions {
    /// `MTLStoreActionOptions`.
    pub store_action_options: u64,
}

/// `setDepthStoreActionOptions:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthStoreActionOptions {
    /// `MTLStoreActionOptions`.
    pub store_action_options: u64,
}

/// `setColorStoreAction:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetColorStoreAction {
    /// `MTLStoreAction`.
    pub store_action: u64,
    /// The colour attachment index.
    pub color_attachment_index: u64,
}

/// `setColorStoreActionOptions:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetColorStoreActionOptions {
    /// `MTLStoreActionOptions`.
    pub store_action_options: u64,
    /// The colour attachment index.
    pub color_attachment_index: u64,
}

/// `setVisibilityResultMode:offset:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVisibilityResultMode {
    /// `MTLVisibilityResultMode`.
    pub mode: u64,
    /// The offset into the visibility result buffer.
    pub offset: u64,
}

/// `setVertexAmplificationMode:value:` (SPI).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexAmplificationMode {
    /// The amplification mode.
    pub mode: u64,
    /// The amplification value.
    pub value: u64,
}

/// `setVertexBufferOffset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexBufferOffset {
    /// The new offset.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setFragmentBufferOffset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentBufferOffset {
    /// The new offset.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setTileBufferOffset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTileBufferOffset {
    /// The new offset.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setVertexBufferOffset:attributeStride:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexBufferOffsetAttributeStride {
    /// The new offset.
    pub offset: u64,
    /// The attribute stride.
    pub stride: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setThreadgroupMemoryLength:offset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetThreadgroupMemoryLength {
    /// The threadgroup memory length.
    pub length: u64,
    /// The offset.
    pub offset: u64,
    /// The binding index.
    pub index: u64,
}

/// `setAlphaTestReferenceValue:` (SPI). The value travels as its `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetAlphaTestReferenceValue {
    /// `referenceValue` as `f32` bits.
    pub reference_value_bits: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
}

/// `setLineWidth:` (SPI). The width travels as its `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetLineWidth {
    /// `lineWidth` as `f32` bits.
    pub line_width_bits: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
}

/// `setPointSize:` (SPI). The size travels as its `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetPointSize {
    /// `pointSize` as `f32` bits.
    pub point_size_bits: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
}

/// `setTessellationFactorScale:`. The scale travels as its `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTessellationFactorScale {
    /// `scale` as `f32` bits.
    pub scale_bits: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
}

/// `setBlendColorRed:green:blue:alpha:`. Each channel travels as its `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetBlendColorRed {
    /// `red` as `f32` bits.
    pub red_bits: u32,
    /// `green` as `f32` bits.
    pub green_bits: u32,
    /// `blue` as `f32` bits.
    pub blue_bits: u32,
    /// `alpha` as `f32` bits.
    pub alpha_bits: u32,
}

/// `setDepthBias:slopeScale:clamp:`. Each argument travels as its `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthBias {
    /// `depthBias` as `f32` bits.
    pub depth_bias_bits: u32,
    /// `slopeScale` as `f32` bits.
    pub slope_scale_bits: u32,
    /// `clamp` as `f32` bits.
    pub clamp_bits: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_0c: u32,
}

/// `setViewportTransformEnabled:` (SPI). `enabled` is a `BOOL` (0 or 1).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetViewportTransformEnabled {
    /// `enabled` (0 or 1).
    pub enabled: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
}

/// `setPrimitiveRestartEnabled:index:` (SPI). `enabled` is a `BOOL` (0 or 1).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetPrimitiveRestartEnabled {
    /// The restart index.
    pub index: u64,
    /// `enabled` (0 or 1).
    pub enabled: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_0c: u32,
}

/// `setStencilFrontReferenceValue:backReferenceValue:`. Each value is a `uint32_t`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetStencilFrontReferenceValue {
    /// The front reference value.
    pub front_reference_value: u32,
    /// The back reference value.
    pub back_reference_value: u32,
}

/// `setDepthCleared` (SPI, no arguments).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthCleared {}

/// `setStencilCleared` (SPI, no arguments).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetStencilCleared {}

/// `setDepthStencilState:`. The state may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthStencilState {
    /// The depth/stencil state, or 0 for nil.
    pub depth_stencil_state_obj_id: u64,
}

/// `setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:` (SPI). The
/// texture may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetColorResolveTexture {
    /// The resolve texture, or 0 for nil.
    pub texture_obj_id: u64,
    /// The slice.
    pub slice: u64,
    /// The depth plane.
    pub depth_plane: u64,
    /// The mip level.
    pub level: u64,
    /// The colour attachment index.
    pub index: u64,
    /// `yInvert` (0 or 1).
    pub y_invert: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_2c: u32,
}

/// `setDepthResolveTexture:slice:depthPlane:level:yInvert:` (SPI). The texture may be
/// nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetDepthResolveTexture {
    /// The resolve texture, or 0 for nil.
    pub texture_obj_id: u64,
    /// The slice.
    pub slice: u64,
    /// The depth plane.
    pub depth_plane: u64,
    /// The mip level.
    pub level: u64,
    /// `yInvert` (0 or 1).
    pub y_invert: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_24: u32,
}

/// `setStencilResolveTexture:slice:depthPlane:level:yInvert:` (SPI). The texture may
/// be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetStencilResolveTexture {
    /// The resolve texture, or 0 for nil.
    pub texture_obj_id: u64,
    /// The slice.
    pub slice: u64,
    /// The depth plane.
    pub depth_plane: u64,
    /// The mip level.
    pub level: u64,
    /// `yInvert` (0 or 1).
    pub y_invert: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_24: u32,
}

/// `setTessellationFactorBuffer:offset:instanceStride:`. The buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTessellationFactorBuffer {
    /// The tessellation factor buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The instance stride.
    pub instance_stride: u64,
}

// ---- The array-tailed ops: a fixed head then a trailing element array. ----

/// One element of a buffer-binding array ([`RgfxOpSetVertexBuffers`] and the like):
/// a buffer (0 = nil) and its offset.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferBinding {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
}

/// One element of an attribute-stride buffer-binding array
/// ([`RgfxOpSetVertexBuffersAttributeStrides`]): a buffer (0 = nil), its offset and
/// its attribute stride.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferStrideBinding {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The attribute stride.
    pub stride: u64,
}

/// One element of a texture-binding array ([`RgfxOpSetVertexTextures`] and the like):
/// a texture (0 = nil).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureBinding {
    /// The texture, or 0 for nil.
    pub texture_obj_id: u64,
}

/// One element of a sampler-binding array ([`RgfxOpSetVertexSamplerStates`] and the
/// like): a sampler (0 = nil).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSamplerBinding {
    /// The sampler, or 0 for nil.
    pub sampler_obj_id: u64,
}

/// One element of a sampler-with-LOD-clamps array
/// ([`RgfxOpSetVertexSamplerStatesLodClamps`] and the like): a sampler (0 = nil) and
/// its LOD clamps, which travel as their `f32` bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSamplerLodBinding {
    /// The sampler, or 0 for nil.
    pub sampler_obj_id: u64,
    /// `lodMinClamp` as `f32` bits.
    pub lod_min_clamp_bits: u32,
    /// `lodMaxClamp` as `f32` bits.
    pub lod_max_clamp_bits: u32,
}

/// One element of a scissor-rect array ([`RgfxOpSetScissorRects`]): an
/// `MTLScissorRect`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxScissorRectElem {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

/// One element of a viewport array ([`RgfxOpSetViewports`]): an `MTLViewport`, its
/// `f64` fields as bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxViewportElem {
    /// `originX` as `f64` bits.
    pub origin_x_bits: u64,
    /// `originY` as `f64` bits.
    pub origin_y_bits: u64,
    /// `width` as `f64` bits.
    pub width_bits: u64,
    /// `height` as `f64` bits.
    pub height_bits: u64,
    /// `znear` as `f64` bits.
    pub znear_bits: u64,
    /// `zfar` as `f64` bits.
    pub zfar_bits: u64,
}

/// One element of a vertex-amplification view-mapping array
/// ([`RgfxOpSetVertexAmplificationCount`]): an `MTLVertexAmplificationViewMapping`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxViewMapping {
    /// `viewportArrayIndexOffset`.
    pub viewport_array_index_offset: u32,
    /// `renderTargetArrayIndexOffset`.
    pub render_target_array_index_offset: u32,
}

/// One element of a heap array ([`RgfxOpUseHeaps`] and the like): a heap (0 = nil).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxHeapBinding {
    /// The heap, or 0 for nil.
    pub heap_obj_id: u64,
}

/// One element of a resource array ([`RgfxOpUseResources`] and the like): a resource
/// (0 = nil) and its [kind](RGFX_RESOURCE_KIND_BUFFER), which says how the host
/// resolves it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxResourceBinding {
    /// The resource, or 0 for nil.
    pub obj_id: u64,
    /// `RGFX_RESOURCE_KIND_*`: whether a buffer, a texture or an indirect command
    /// buffer.
    pub kind: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_0c: u32,
}

/// `setVertexBuffers:offsets:withRange:`. `range.length` [`RgfxBufferBinding`]
/// records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexBuffers {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxBufferBinding`] records that follow.
    pub range_length: u64,
}

/// `setFragmentBuffers:offsets:withRange:`. `range.length` [`RgfxBufferBinding`]
/// records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentBuffers {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxBufferBinding`] records that follow.
    pub range_length: u64,
}

/// `setTileBuffers:offsets:withRange:`. `range.length` [`RgfxBufferBinding`] records
/// follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTileBuffers {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxBufferBinding`] records that follow.
    pub range_length: u64,
}

/// `setVertexBuffers:offsets:attributeStrides:withRange:`. `range.length`
/// [`RgfxBufferStrideBinding`] records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexBuffersAttributeStrides {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxBufferStrideBinding`] records that follow.
    pub range_length: u64,
}

/// `setVertexTextures:withRange:`. `range.length` [`RgfxTextureBinding`] records
/// follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexTextures {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxTextureBinding`] records that follow.
    pub range_length: u64,
}

/// `setFragmentTextures:withRange:`. `range.length` [`RgfxTextureBinding`] records
/// follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentTextures {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxTextureBinding`] records that follow.
    pub range_length: u64,
}

/// `setTileTextures:withRange:`. `range.length` [`RgfxTextureBinding`] records follow
/// at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTileTextures {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxTextureBinding`] records that follow.
    pub range_length: u64,
}

/// `setVertexSamplerStates:withRange:`. `range.length` [`RgfxSamplerBinding`] records
/// follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexSamplerStates {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerBinding`] records that follow.
    pub range_length: u64,
}

/// `setFragmentSamplerStates:withRange:`. `range.length` [`RgfxSamplerBinding`]
/// records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentSamplerStates {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerBinding`] records that follow.
    pub range_length: u64,
}

/// `setTileSamplerStates:withRange:`. `range.length` [`RgfxSamplerBinding`] records
/// follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTileSamplerStates {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerBinding`] records that follow.
    pub range_length: u64,
}

/// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. `range.length`
/// [`RgfxSamplerLodBinding`] records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexSamplerStatesLodClamps {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerLodBinding`] records that follow.
    pub range_length: u64,
}

/// `setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. `range.length`
/// [`RgfxSamplerLodBinding`] records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentSamplerStatesLodClamps {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerLodBinding`] records that follow.
    pub range_length: u64,
}

/// `setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. `range.length`
/// [`RgfxSamplerLodBinding`] records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetTileSamplerStatesLodClamps {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerLodBinding`] records that follow.
    pub range_length: u64,
}

/// `setScissorRects:count:`. `count` [`RgfxScissorRectElem`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetScissorRects {
    /// The [`RgfxScissorRectElem`] records that follow.
    pub count: u64,
}

/// `setViewports:count:`. `count` [`RgfxViewportElem`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetViewports {
    /// The [`RgfxViewportElem`] records that follow.
    pub count: u64,
}

/// `setVertexAmplificationCount:viewMappings:`. `count` [`RgfxViewMapping`] records
/// follow at `size_of::<Self>()`; the amplification count is `count`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexAmplificationCount {
    /// The [`RgfxViewMapping`] records that follow, and the amplification count.
    pub count: u64,
}

/// `useHeaps:count:`. `count` [`RgfxHeapBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpUseHeaps {
    /// The [`RgfxHeapBinding`] records that follow.
    pub count: u64,
}

/// `useHeaps:count:stages:`. `count` [`RgfxHeapBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpUseHeapsStages {
    /// The [`RgfxHeapBinding`] records that follow.
    pub count: u64,
    /// `MTLRenderStages`.
    pub stages: u64,
}

/// `useResources:count:usage:`. `count` [`RgfxResourceBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpUseResources {
    /// The [`RgfxResourceBinding`] records that follow.
    pub count: u64,
    /// `MTLResourceUsage`.
    pub usage: u64,
}

/// `useResources:count:usage:stages:`. `count` [`RgfxResourceBinding`] records follow
/// at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpUseResourcesStages {
    /// The [`RgfxResourceBinding`] records that follow.
    pub count: u64,
    /// `MTLResourceUsage`.
    pub usage: u64,
    /// `MTLRenderStages`.
    pub stages: u64,
}

// ---- The object and mesh stages and the mesh draws. ----

/// `setObjectBuffers:offsets:withRange:`. `range.length` [`RgfxBufferBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetObjectBuffers {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxBufferBinding`] records that follow.
    pub range_length: u64,
}

/// `setMeshBuffers:offsets:withRange:`. `range.length` [`RgfxBufferBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetMeshBuffers {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxBufferBinding`] records that follow.
    pub range_length: u64,
}

/// `setObjectBufferOffset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetObjectBufferOffset {
    /// The new offset.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setMeshBufferOffset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetMeshBufferOffset {
    /// The new offset.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setObjectTextures:withRange:`. `range.length` [`RgfxTextureBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetObjectTextures {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxTextureBinding`] records that follow.
    pub range_length: u64,
}

/// `setMeshTextures:withRange:`. `range.length` [`RgfxTextureBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetMeshTextures {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxTextureBinding`] records that follow.
    pub range_length: u64,
}

/// `setObjectSamplerStates:withRange:`. `range.length` [`RgfxSamplerBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetObjectSamplerStates {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerBinding`] records that follow.
    pub range_length: u64,
}

/// `setMeshSamplerStates:withRange:`. `range.length` [`RgfxSamplerBinding`] records follow at
/// `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetMeshSamplerStates {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerBinding`] records that follow.
    pub range_length: u64,
}

/// `setObjectSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. `range.length`
/// [`RgfxSamplerLodBinding`] records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetObjectSamplerStatesLodClamps {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerLodBinding`] records that follow.
    pub range_length: u64,
}

/// `setMeshSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. `range.length`
/// [`RgfxSamplerLodBinding`] records follow at `size_of::<Self>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetMeshSamplerStatesLodClamps {
    /// `range.location`: the first binding index.
    pub range_location: u64,
    /// `range.length`: the [`RgfxSamplerLodBinding`] records that follow.
    pub range_length: u64,
}

/// `setObjectThreadgroupMemoryLength:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetObjectThreadgroupMemoryLength {
    /// The threadgroup memory length.
    pub length: u64,
    /// The binding index.
    pub index: u64,
}

/// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`. The `MTLSize`
/// fields are `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawMeshThreadgroups {
    /// `threadgroupsPerGrid`: width, height and depth.
    pub threadgroups_per_grid: [u64; 3],
    /// `threadsPerObjectThreadgroup`: width, height and depth.
    pub threads_per_object_threadgroup: [u64; 3],
    /// `threadsPerMeshThreadgroup`: width, height and depth.
    pub threads_per_mesh_threadgroup: [u64; 3],
}

/// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`. The `MTLSize` fields
/// are `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawMeshThreads {
    /// `threadsPerGrid`: width, height and depth.
    pub threads_per_grid: [u64; 3],
    /// `threadsPerObjectThreadgroup`: width, height and depth.
    pub threads_per_object_threadgroup: [u64; 3],
    /// `threadsPerMeshThreadgroup`: width, height and depth.
    pub threads_per_mesh_threadgroup: [u64; 3],
}

/// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
/// The indirect buffer holds an `MTLDispatchThreadgroupsIndirectArguments`; nil (0) faults the
/// buffer at the op.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawMeshThreadgroupsIndirect {
    /// The indirect buffer.
    pub indirect_buffer_obj_id: u64,
    /// The indirect arguments' offset in it.
    pub indirect_buffer_offset: u64,
    /// `threadsPerObjectThreadgroup`: width, height and depth.
    pub threads_per_object_threadgroup: [u64; 3],
    /// `threadsPerMeshThreadgroup`: width, height and depth.
    pub threads_per_mesh_threadgroup: [u64; 3],
}

// The render ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxOpSetRenderPipelineState>() == 8);
    assert!(size_of::<RgfxOpSetVertexBuffer>() == 24);
    assert!(size_of::<RgfxOpSetFragmentBuffer>() == 24);
    assert!(size_of::<RgfxOpSetFragmentTexture>() == 16);
    assert!(size_of::<RgfxOpSetViewport>() == 48);
    assert!(size_of::<RgfxOpSetScissorRect>() == 32);
    assert!(size_of::<RgfxOpDrawPrimitives>() == 24);
    assert!(size_of::<RgfxOpDrawIndexedPrimitives>() == 40);
    // The render tags stay in the render group.
    assert!(
        super::op_group(RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE) == super::RGFX_OP_GROUP_RENDER
    );
    assert!(super::op_group(RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES) == super::RGFX_OP_GROUP_RENDER);
};

// The rest of the render ops' layout (and the array elements').
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxOpDrawPrimitivesInstances>() == 32);
    assert!(size_of::<RgfxOpDrawPrimitivesBase>() == 40);
    assert!(size_of::<RgfxOpDrawPrimitivesIndirect>() == 24);
    assert!(size_of::<RgfxOpDrawIndexedPrimitivesInstances>() == 48);
    assert!(size_of::<RgfxOpDrawIndexedPrimitivesBase>() == 64);
    assert!(size_of::<RgfxOpDrawIndexedPrimitivesIndirect>() == 48);
    assert!(size_of::<RgfxOpDrawPatches>() == 56);
    assert!(size_of::<RgfxOpDrawPatchesIndirect>() == 40);
    assert!(size_of::<RgfxOpDrawIndexedPatches>() == 72);
    assert!(size_of::<RgfxOpDrawIndexedPatchesIndirect>() == 56);
    assert!(size_of::<RgfxOpDispatchThreadsPerTile>() == 24);
    assert!(size_of::<RgfxOpDispatchThreadsPerTileInRegion>() == 72);
    assert!(size_of::<RgfxOpDispatchThreadsPerTileInRegionIndex>() == 80);
    assert!(size_of::<RgfxOpExecuteCommandsInBuffer>() == 24);
    assert!(size_of::<RgfxOpExecuteCommandsIndirect>() == 24);
    assert!(size_of::<RgfxOpMemoryBarrierWithScope>() == 24);
    assert!(size_of::<RgfxOpUpdateFence>() == 16);
    assert!(size_of::<RgfxOpWaitForFence>() == 16);
    assert!(size_of::<RgfxOpSetCullMode>() == 8);
    assert!(size_of::<RgfxOpSetFrontFacingWinding>() == 8);
    assert!(size_of::<RgfxOpSetTriangleFillMode>() == 8);
    assert!(size_of::<RgfxOpSetDepthClipModeSpi>() == 8);
    assert!(size_of::<RgfxOpSetProvokingVertexMode>() == 8);
    assert!(size_of::<RgfxOpSetDepthStoreAction>() == 8);
    assert!(size_of::<RgfxOpSetStencilStoreAction>() == 8);
    assert!(size_of::<RgfxOpSetStencilStoreActionOptions>() == 8);
    assert!(size_of::<RgfxOpSetDepthStoreActionOptions>() == 8);
    assert!(size_of::<RgfxOpSetColorStoreAction>() == 16);
    assert!(size_of::<RgfxOpSetColorStoreActionOptions>() == 16);
    assert!(size_of::<RgfxOpSetVisibilityResultMode>() == 16);
    assert!(size_of::<RgfxOpSetVertexAmplificationMode>() == 16);
    assert!(size_of::<RgfxOpSetVertexBufferOffset>() == 16);
    assert!(size_of::<RgfxOpSetFragmentBufferOffset>() == 16);
    assert!(size_of::<RgfxOpSetTileBufferOffset>() == 16);
    assert!(size_of::<RgfxOpSetVertexBufferOffsetAttributeStride>() == 24);
    assert!(size_of::<RgfxOpSetThreadgroupMemoryLength>() == 24);
    assert!(size_of::<RgfxOpSetAlphaTestReferenceValue>() == 8);
    assert!(size_of::<RgfxOpSetLineWidth>() == 8);
    assert!(size_of::<RgfxOpSetPointSize>() == 8);
    assert!(size_of::<RgfxOpSetTessellationFactorScale>() == 8);
    assert!(size_of::<RgfxOpSetBlendColorRed>() == 16);
    assert!(size_of::<RgfxOpSetDepthBias>() == 16);
    assert!(size_of::<RgfxOpSetViewportTransformEnabled>() == 8);
    assert!(size_of::<RgfxOpSetPrimitiveRestartEnabled>() == 16);
    assert!(size_of::<RgfxOpSetStencilFrontReferenceValue>() == 8);
    assert!(size_of::<RgfxOpSetDepthCleared>() == 0);
    assert!(size_of::<RgfxOpSetStencilCleared>() == 0);
    assert!(size_of::<RgfxOpSetDepthStencilState>() == 8);
    assert!(size_of::<RgfxOpSetColorResolveTexture>() == 48);
    assert!(size_of::<RgfxOpSetDepthResolveTexture>() == 40);
    assert!(size_of::<RgfxOpSetStencilResolveTexture>() == 40);
    assert!(size_of::<RgfxOpSetTessellationFactorBuffer>() == 24);
    // The array-tailed heads and their elements.
    assert!(size_of::<RgfxBufferBinding>() == 16);
    assert!(size_of::<RgfxBufferStrideBinding>() == 24);
    assert!(size_of::<RgfxTextureBinding>() == 8);
    assert!(size_of::<RgfxSamplerBinding>() == 8);
    assert!(size_of::<RgfxSamplerLodBinding>() == 16);
    assert!(size_of::<RgfxScissorRectElem>() == 32);
    assert!(size_of::<RgfxViewportElem>() == 48);
    assert!(size_of::<RgfxViewMapping>() == 8);
    assert!(size_of::<RgfxHeapBinding>() == 8);
    assert!(size_of::<RgfxResourceBinding>() == 16);
    assert!(size_of::<RgfxOpSetVertexBuffers>() == 16);
    assert!(size_of::<RgfxOpSetFragmentBuffers>() == 16);
    assert!(size_of::<RgfxOpSetTileBuffers>() == 16);
    assert!(size_of::<RgfxOpSetVertexBuffersAttributeStrides>() == 16);
    assert!(size_of::<RgfxOpSetVertexTextures>() == 16);
    assert!(size_of::<RgfxOpSetFragmentTextures>() == 16);
    assert!(size_of::<RgfxOpSetTileTextures>() == 16);
    assert!(size_of::<RgfxOpSetVertexSamplerStates>() == 16);
    assert!(size_of::<RgfxOpSetFragmentSamplerStates>() == 16);
    assert!(size_of::<RgfxOpSetTileSamplerStates>() == 16);
    assert!(size_of::<RgfxOpSetVertexSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxOpSetFragmentSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxOpSetTileSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxOpSetScissorRects>() == 8);
    assert!(size_of::<RgfxOpSetViewports>() == 8);
    assert!(size_of::<RgfxOpSetVertexAmplificationCount>() == 8);
    assert!(size_of::<RgfxOpUseHeaps>() == 8);
    assert!(size_of::<RgfxOpUseHeapsStages>() == 16);
    assert!(size_of::<RgfxOpUseResources>() == 16);
    assert!(size_of::<RgfxOpUseResourcesStages>() == 24);
    // Every new render tag stays in the render group and is unique within it.
    assert!(
        super::op_group(RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS)
            == super::RGFX_OP_GROUP_RENDER
    );
    assert!(RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS == 0x104f);
    assert!(super::op_group(RGFX_OP_RENDER_RESET_STATE) == super::RGFX_OP_GROUP_RENDER);
    assert!(size_of::<RgfxOpSetObjectBuffers>() == 16);
    assert!(size_of::<RgfxOpSetMeshBuffers>() == 16);
    assert!(size_of::<RgfxOpSetObjectBufferOffset>() == 16);
    assert!(size_of::<RgfxOpSetMeshBufferOffset>() == 16);
    assert!(size_of::<RgfxOpSetObjectTextures>() == 16);
    assert!(size_of::<RgfxOpSetMeshTextures>() == 16);
    assert!(size_of::<RgfxOpSetObjectSamplerStates>() == 16);
    assert!(size_of::<RgfxOpSetMeshSamplerStates>() == 16);
    assert!(size_of::<RgfxOpSetObjectSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxOpSetMeshSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxOpSetObjectThreadgroupMemoryLength>() == 16);
    assert!(size_of::<RgfxOpDrawMeshThreadgroups>() == 72);
    assert!(size_of::<RgfxOpDrawMeshThreads>() == 72);
    assert!(size_of::<RgfxOpDrawMeshThreadgroupsIndirect>() == 64);
    // The mesh stages' tags follow RESET_STATE in a run of their own.
    assert!(RGFX_OP_RENDER_SET_OBJECT_BUFFERS == RGFX_OP_RENDER_RESET_STATE + 1);
    assert!(RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT == 0x105e);
    assert!(
        super::op_group(RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT)
            == super::RGFX_OP_GROUP_RENDER
    );
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// Each render op body round-trips, floats travelling as bits.
    #[test]
    fn render_ops_round_trip() {
        let vb = RgfxOpSetVertexBuffer {
            buffer_obj_id: 4,
            offset: 256,
            index: 2,
        };
        assert_eq!(
            RgfxOpSetVertexBuffer::read_from_bytes(vb.as_bytes()).unwrap(),
            vb
        );

        let vp = RgfxOpSetViewport {
            origin_x_bits: 0.0f64.to_bits(),
            origin_y_bits: 0.0f64.to_bits(),
            width_bits: 1920.0f64.to_bits(),
            height_bits: 1080.0f64.to_bits(),
            znear_bits: 0.0f64.to_bits(),
            zfar_bits: 1.0f64.to_bits(),
        };
        assert_eq!(
            RgfxOpSetViewport::read_from_bytes(vp.as_bytes()).unwrap(),
            vp
        );
        assert_eq!(f64::from_bits(vp.width_bits), 1920.0);

        let di = RgfxOpDrawIndexedPrimitives {
            primitive_type: 3,
            index_count: 6,
            index_type: 1,
            index_buffer_obj_id: 9,
            index_buffer_offset: 0,
        };
        assert_eq!(
            RgfxOpDrawIndexedPrimitives::read_from_bytes(di.as_bytes()).unwrap(),
            di
        );

        let mesh = RgfxOpDrawMeshThreadgroupsIndirect {
            indirect_buffer_obj_id: 3,
            indirect_buffer_offset: 16,
            threads_per_object_threadgroup: [2, 1, 1],
            threads_per_mesh_threadgroup: [4, 1, 1],
        };
        assert_eq!(
            RgfxOpDrawMeshThreadgroupsIndirect::read_from_bytes(mesh.as_bytes()).unwrap(),
            mesh
        );
        let threads = RgfxOpDrawMeshThreads {
            threads_per_grid: [10, 1, 1],
            threads_per_object_threadgroup: [1, 1, 1],
            threads_per_mesh_threadgroup: [4, 1, 1],
        };
        // The three sizes lie in the op's order, each width first.
        assert_eq!(&threads.as_bytes()[..8], &10u64.to_le_bytes());
        assert_eq!(&threads.as_bytes()[48..56], &4u64.to_le_bytes());
    }

    /// A scalar, a signed, a float-bits and a bool-bearing op round-trip.
    #[test]
    fn scalar_render_ops_round_trip() {
        let base = RgfxOpDrawIndexedPrimitivesBase {
            primitive_type: 3,
            index_count: 6,
            index_type: 1,
            index_buffer_obj_id: 9,
            index_buffer_offset: 0,
            instance_count: 2,
            base_vertex: -4,
            base_instance: 1,
        };
        assert_eq!(
            RgfxOpDrawIndexedPrimitivesBase::read_from_bytes(base.as_bytes()).unwrap(),
            base
        );
        assert_eq!(base.base_vertex, -4);

        let bias = RgfxOpSetDepthBias {
            depth_bias_bits: 1.5f32.to_bits(),
            slope_scale_bits: 2.0f32.to_bits(),
            clamp_bits: 0.25f32.to_bits(),
            reserved_0c: 0,
        };
        assert_eq!(
            RgfxOpSetDepthBias::read_from_bytes(bias.as_bytes()).unwrap(),
            bias
        );
        assert_eq!(f32::from_bits(bias.slope_scale_bits), 2.0);

        let restart = RgfxOpSetPrimitiveRestartEnabled {
            index: 0xffff,
            enabled: 1,
            reserved_0c: 0,
        };
        assert_eq!(
            RgfxOpSetPrimitiveRestartEnabled::read_from_bytes(restart.as_bytes()).unwrap(),
            restart
        );

        // The no-arg ops have an empty body.
        assert!(RgfxOpSetDepthCleared::default().as_bytes().is_empty());
        assert!(RgfxOpSetStencilCleared::default().as_bytes().is_empty());
    }

    /// An array-tailed op's head and its trailing elements round-trip, the elements
    /// at the head's `size_of`.
    #[test]
    fn array_tailed_render_ops_round_trip() {
        let head = RgfxOpSetVertexBuffers {
            range_location: 2,
            range_length: 2,
        };
        let a = RgfxBufferBinding {
            buffer_obj_id: 7,
            offset: 16,
        };
        let b = RgfxBufferBinding {
            buffer_obj_id: 0,
            offset: 0,
        };
        let mut body = head.as_bytes().to_vec();
        body.extend_from_slice(a.as_bytes());
        body.extend_from_slice(b.as_bytes());
        assert_eq!(body.len(), 16 + 2 * 16);

        let (got_head, _) = RgfxOpSetVertexBuffers::read_from_prefix(&body).unwrap();
        assert_eq!(got_head, head);
        let off = size_of::<RgfxOpSetVertexBuffers>();
        let stride = size_of::<RgfxBufferBinding>();
        let e0 = RgfxBufferBinding::read_from_bytes(&body[off..off + stride]).unwrap();
        let e1 = RgfxBufferBinding::read_from_bytes(&body[off + stride..off + 2 * stride]).unwrap();
        assert_eq!((e0, e1), (a, b));

        // A resource binding carries its kind, and a viewport's fields travel as bits.
        let res = RgfxResourceBinding {
            obj_id: 5,
            kind: RGFX_RESOURCE_KIND_TEXTURE,
            reserved_0c: 0,
        };
        assert_eq!(
            RgfxResourceBinding::read_from_bytes(res.as_bytes()).unwrap(),
            res
        );
        let vp = RgfxViewportElem {
            width_bits: 1920.0f64.to_bits(),
            ..RgfxViewportElem::default()
        };
        assert_eq!(f64::from_bits(vp.width_bits), 1920.0);
        assert_eq!(
            RgfxViewportElem::read_from_bytes(vp.as_bytes()).unwrap(),
            vp
        );
    }
}
