/*
 * RgfxWireLayout: the wire structs' sizes, asserted at the guest's own compile.
 *
 * reims_graphics_wire.h is generated from crates/reims-graphics/src/wire.rs and
 * drift-checked against it by `cargo xtask header --check`. wire.rs asserts every
 * struct's size in a `const _` block; this is the same wall on the C side, so a C
 * header that ever disagrees with the sizes the guest and host both rely on fails
 * the bundle's build here, not silently at run time on the wire.
 */

#include "reims_graphics_wire.h"

#include "RgfxEncoder.hpp"

// Transport (stage 2).
static_assert(sizeof(RgfxRingHeader) == 256, "RgfxRingHeader");
static_assert(__builtin_offsetof(RgfxRingHeader, consumer) == 0x40, "ring consumer line");
static_assert(__builtin_offsetof(RgfxRingHeader, consumer_idle) == 0x80, "ring consumer_idle line");
static_assert(__builtin_offsetof(RgfxRingHeader, armed) == 0xc0, "ring armed line");
static_assert(sizeof(RgfxMsgHeader) == 16, "RgfxMsgHeader");

// Stage-3 object lifecycle and completion records.
static_assert(sizeof(RgfxObjectDestroy) == 24, "RgfxObjectDestroy");
static_assert(sizeof(RgfxObjectCreate) == 24, "RgfxObjectCreate");
static_assert(sizeof(RgfxSharedEventCreate) == 32, "RgfxSharedEventCreate");
static_assert(sizeof(RgfxSharedEventSetValue) == 32, "RgfxSharedEventSetValue");
static_assert(sizeof(RgfxSharedEventName) == 40, "RgfxSharedEventName");
static_assert(sizeof(RgfxSharedEventImport) == 40, "RgfxSharedEventImport");
static_assert(sizeof(RgfxSharedEventUnname) == 32, "RgfxSharedEventUnname");
static_assert(sizeof(RgfxSharedEventValue) == 32, "RgfxSharedEventValue");
static_assert(sizeof(RgfxQueueCreate) == 56, "RgfxQueueCreate");
static_assert(sizeof(RgfxHeapCreate) == 48, "RgfxHeapCreate");
static_assert(sizeof(RgfxHeapOverHostMemory) == 56, "RgfxHeapOverHostMemory");
static_assert(sizeof(RgfxBufferCreate) == 48, "RgfxBufferCreate");
static_assert(sizeof(RgfxBufferOverHostMemory) == 56, "RgfxBufferOverHostMemory");
static_assert(sizeof(RgfxBufferAtOffset) == 56, "RgfxBufferAtOffset");
static_assert(sizeof(RgfxCbComplete) == 48, "RgfxCbComplete");
static_assert(sizeof(RgfxEventSignalled) == 32, "RgfxEventSignalled");
static_assert(sizeof(RgfxTextureDescriptor) == 48, "RgfxTextureDescriptor");
static_assert(sizeof(RgfxTextureCreate) == 104, "RgfxTextureCreate");
static_assert(sizeof(RgfxTextureFromBuffer) == 104, "RgfxTextureFromBuffer");
static_assert(sizeof(RgfxTextureFromIOSurface) == 120, "RgfxTextureFromIOSurface");
static_assert(sizeof(RgfxSurfaceWriter) == 16, "RgfxSurfaceWriter");
static_assert(sizeof(RgfxSurfaceWriters) == 64, "RgfxSurfaceWriters");
static_assert(sizeof(RgfxSamplerCreate) == 72, "RgfxSamplerCreate");
static_assert(sizeof(RgfxStencilDescriptor) == 24, "RgfxStencilDescriptor");
static_assert(sizeof(RgfxDepthStencilCreate) == 80, "RgfxDepthStencilCreate");
static_assert(sizeof(RgfxLibraryCreate) == 72, "RgfxLibraryCreate");
static_assert(RGFX_LIBRARY_BYTES_OFFSET == sizeof(RgfxLibraryCreate), "library bytes offset");
static_assert(sizeof(RgfxLibraryFromRegion) == 112, "RgfxLibraryFromRegion");
static_assert(sizeof(RgfxLibraryLookup) == 64, "RgfxLibraryLookup");
static_assert(sizeof(RgfxFormatAnswers) == 64, "RgfxFormatAnswers");
static_assert(sizeof(RgfxDeviceInfo) == 1432 + 64 * RGFX_DEVICE_INFO_FORMATS, "RgfxDeviceInfo");
static_assert(__builtin_offsetof(RgfxDeviceInfo, formats) == 1432, "RgfxDeviceInfo formats");
static_assert(__builtin_offsetof(RgfxDeviceInfo, family_answers) == 976,
              "RgfxDeviceInfo family_answers");
static_assert(__builtin_offsetof(RgfxDeviceInfo, name) == 1112, "RgfxDeviceInfo name");

// Stage-3 pipeline creates and reflection.
static_assert(sizeof(RgfxFunctionConstant) == 40, "RgfxFunctionConstant");
static_assert(sizeof(RgfxColorAttachmentDesc) == 40, "RgfxColorAttachmentDesc");
static_assert(sizeof(RgfxVertexBufferLayout) == 16, "RgfxVertexBufferLayout");
static_assert(sizeof(RgfxVertexAttribute) == 16, "RgfxVertexAttribute");
static_assert(sizeof(RgfxLinkedFunction) == 48, "RgfxLinkedFunction");
static_assert(sizeof(RgfxRenderPipelineCreate) == 464, "RgfxRenderPipelineCreate");
static_assert(__builtin_offsetof(RgfxRenderPipelineCreate, colors) == 128,
              "RgfxRenderPipelineCreate colors");
static_assert(sizeof(RgfxComputePipelineCreate) == 96, "RgfxComputePipelineCreate");
static_assert(sizeof(RgfxPipelineReflection) == 80, "RgfxPipelineReflection");
static_assert(sizeof(RgfxRenderPipelineCreate) % 8 == 0, "RgfxRenderPipelineCreate aligned");
static_assert(sizeof(RgfxComputePipelineCreate) % 8 == 0, "RgfxComputePipelineCreate aligned");
static_assert(sizeof(RgfxPipelineReflection) % 8 == 0, "RgfxPipelineReflection aligned");

// Every message size is a multiple of 8, as a ring message must be.
static_assert(sizeof(RgfxObjectDestroy) % 8 == 0, "RgfxObjectDestroy aligned");
static_assert(sizeof(RgfxQueueCreate) % 8 == 0, "RgfxQueueCreate aligned");
static_assert(sizeof(RgfxTextureCreate) % 8 == 0, "RgfxTextureCreate aligned");
static_assert(sizeof(RgfxDepthStencilCreate) % 8 == 0, "RgfxDepthStencilCreate aligned");

// --- Stage-3 command-buffer op stream (RgfxEncoder) --------------------------
// The framing the encoder writes and the host walks (decode::cmdbuf).
static_assert(sizeof(RgfxOpHeader) == 8, "RgfxOpHeader");
static_assert(RGFX_OP_BODY_OFFSET == 8, "op body offset");
static_assert(sizeof(RgfxCommandBuffer) == 40, "RgfxCommandBuffer");
static_assert(RGFX_COMMAND_BUFFER_OPS_OFFSET == 40, "command-buffer ops offset");
static_assert(RGFX_COMMAND_BUFFER_OPS_OFFSET == sizeof(RgfxCommandBuffer),
              "ops start right after the command-buffer header");
static_assert(sizeof(RgfxComputePassDesc) == 8, "RgfxComputePassDesc");
static_assert(sizeof(RgfxOpEvent) == 16, "RgfxOpEvent");
static_assert(sizeof(RgfxOpSurfaceUse) == 24, "RgfxOpSurfaceUse");

// The render-pass head and its records (ENCODER_BEGIN_RENDER body).
static_assert(sizeof(RgfxRenderPassDesc) == 152, "RgfxRenderPassDesc");
static_assert(RGFX_RENDER_PASS_COLOR_OFFSET == 152, "render pass colour offset");
static_assert(RGFX_RENDER_PASS_COLOR_OFFSET == sizeof(RgfxRenderPassDesc),
              "colour attachments start right after the pass head");
static_assert(sizeof(RgfxRenderPassColorAttachment) == 88, "RgfxRenderPassColorAttachment");
static_assert(sizeof(RgfxRenderPassDepthAttachment) == 64, "RgfxRenderPassDepthAttachment");
static_assert(sizeof(RgfxRenderPassStencilAttachment) == 64, "RgfxRenderPassStencilAttachment");

// The array-element records the render array ops' tails carry.
static_assert(sizeof(RgfxBufferBinding) == 16, "RgfxBufferBinding");
static_assert(sizeof(RgfxBufferStrideBinding) == 24, "RgfxBufferStrideBinding");
static_assert(sizeof(RgfxTextureBinding) == 8, "RgfxTextureBinding");
static_assert(sizeof(RgfxSamplerBinding) == 8, "RgfxSamplerBinding");
static_assert(sizeof(RgfxSamplerLodBinding) == 16, "RgfxSamplerLodBinding");
static_assert(sizeof(RgfxResourceBinding) == 16, "RgfxResourceBinding");
static_assert(sizeof(RgfxHeapBinding) == 8, "RgfxHeapBinding");
static_assert(sizeof(RgfxScissorRectElem) == 32, "RgfxScissorRectElem");
static_assert(sizeof(RgfxViewportElem) == 48, "RgfxViewportElem");
static_assert(sizeof(RgfxViewMapping) == 8, "RgfxViewMapping");

// A representative set of render-op body sizes the encoder writes (the header's
// own sizes are drift-checked against wire.rs by `cargo xtask header --check`;
// these guard the encoder's reliance on them for its op `len`).
static_assert(sizeof(RgfxOpSetRenderPipelineState) == 8, "RgfxOpSetRenderPipelineState");
static_assert(sizeof(RgfxOpSetVertexBuffer) == 24, "RgfxOpSetVertexBuffer");
static_assert(sizeof(RgfxOpSetViewport) == 48, "RgfxOpSetViewport");
static_assert(sizeof(RgfxOpSetScissorRect) == 32, "RgfxOpSetScissorRect");
static_assert(sizeof(RgfxOpDrawPrimitives) == 24, "RgfxOpDrawPrimitives");
static_assert(sizeof(RgfxOpDrawIndexedPrimitives) == 40, "RgfxOpDrawIndexedPrimitives");
static_assert(sizeof(RgfxOpDrawIndexedPrimitivesBase) == 64, "RgfxOpDrawIndexedPrimitivesBase");
static_assert(sizeof(RgfxOpDrawIndexedPatches) == 72, "RgfxOpDrawIndexedPatches");
static_assert(sizeof(RgfxOpDispatchThreadsPerTileInRegionIndex) == 80,
              "RgfxOpDispatchThreadsPerTileInRegionIndex");
static_assert(sizeof(RgfxOpSetVertexBuffers) == 16, "RgfxOpSetVertexBuffers");
static_assert(sizeof(RgfxOpUseResources) == 16, "RgfxOpUseResources render head");
static_assert(sizeof(RgfxOpUseResourcesStages) == 24, "RgfxOpUseResourcesStages");

// The compute-encoder op bodies (`RgfxComputeOp*`) and element records, from the
// committed header (crates/reims-graphics/src/ops/compute.rs). These sizes are that
// file's `const _` asserts.
static_assert(sizeof(RgfxComputeOpSetComputePipelineState) == 8, "compute pipeline state");
static_assert(sizeof(RgfxComputeOpSetBufferOffset) == 16, "compute buffer offset");
static_assert(sizeof(RgfxComputeOpSetBufferOffsetAttributeStride) == 24,
              "compute buffer offset stride");
static_assert(sizeof(RgfxComputeOpSetBuffers) == 16, "compute set buffers head");
static_assert(sizeof(RgfxComputeBufferBinding) == 16, "compute buffer binding");
static_assert(sizeof(RgfxComputeOpSetBuffersAttributeStrides) == 16,
              "compute set buffers stride head");
static_assert(sizeof(RgfxComputeBufferStrideBinding) == 24,
              "compute buffer stride binding");
static_assert(sizeof(RgfxComputeOpSetTexture) == 16, "compute set texture");
static_assert(sizeof(RgfxComputeOpSetTextures) == 16, "compute set textures head");
static_assert(sizeof(RgfxComputeTextureBinding) == 8, "compute texture binding");
static_assert(sizeof(RgfxComputeOpSetSamplerStates) == 16, "compute set samplers head");
static_assert(sizeof(RgfxComputeSamplerBinding) == 8, "compute sampler binding");
static_assert(sizeof(RgfxComputeOpSetSamplerStatesLodClamps) == 16,
              "compute set samplers lod head");
static_assert(sizeof(RgfxComputeSamplerLodBinding) == 16, "compute sampler lod");
static_assert(sizeof(RgfxComputeOpSetThreadgroupMemoryLength) == 16,
              "compute threadgroup memory length");
static_assert(sizeof(RgfxComputeOpSetStageInRegion) == 48, "compute stage-in region");
static_assert(sizeof(RgfxComputeOpSetStageInRegionIndirect) == 16,
              "compute stage-in region indirect");
static_assert(sizeof(RgfxComputeOpSetImageblockWidth) == 16, "compute imageblock width");
static_assert(sizeof(RgfxComputeOpDispatchThreadgroups) == 48, "compute dispatch tg");
static_assert(sizeof(RgfxComputeOpDispatchThreadgroupsIndirect) == 40,
              "compute dispatch tg indirect");
static_assert(sizeof(RgfxComputeOpDispatchThreads) == 48, "compute dispatch threads");
static_assert(sizeof(RgfxComputeOpDispatchThreadsIndirect) == 16,
              "compute dispatch threads indirect");
static_assert(sizeof(RgfxComputeOpUseResources) == 16, "compute use resources head");
static_assert(sizeof(RgfxComputeResourceRef) == 16, "compute resource ref");
static_assert(sizeof(RgfxComputeOpUseHeaps) == 8, "compute use heaps head");
static_assert(sizeof(RgfxComputeHeapBinding) == 8, "compute heap binding");
static_assert(sizeof(RgfxComputeOpMemoryBarrierResources) == 8,
              "compute memory barrier resources head");
static_assert(sizeof(RgfxComputeOpMemoryBarrierScope) == 8, "compute memory barrier scope");
static_assert(sizeof(RgfxComputeOpUpdateFence) == 8, "compute update fence");
static_assert(sizeof(RgfxComputeOpWaitForFence) == 8, "compute wait for fence");
static_assert(sizeof(RgfxComputeOpExecuteCommandsIndirect) == 24,
              "compute execute commands indirect");
static_assert(sizeof(RgfxComputeOpExecuteCommandsInBuffer) == 24,
              "compute execute commands in buffer");
static_assert(sizeof(RgfxComputeOpEncodeStartIf) == 32, "compute encode start if");
static_assert(sizeof(RgfxComputeOpEncodeStartWhile) == 32, "compute encode start while");

// The blit-encoder op bodies (`RgfxBlitOp*`), from the committed header
// (crates/reims-graphics/src/ops/blit.rs). These sizes are that file's asserts.
static_assert(sizeof(RgfxBlitOpCopyBufferToBuffer) == 40, "blit copy b2b");
static_assert(sizeof(RgfxBlitOpCopyBufferToTexture) == 104, "blit copy b2t");
static_assert(sizeof(RgfxBlitOpCopyBufferToTextureOptions) == 112, "blit copy b2t options");
static_assert(sizeof(RgfxBlitOpCopyTextureToBuffer) == 104, "blit copy t2b");
static_assert(sizeof(RgfxBlitOpCopyTextureToBufferOptions) == 112, "blit copy t2b options");
static_assert(sizeof(RgfxBlitOpCopyTextureRegion) == 120, "blit copy region");
static_assert(sizeof(RgfxBlitOpCopyTextureRegionOptions) == 128, "blit copy region options");
static_assert(sizeof(RgfxBlitOpCopyTextureSlices) == 64, "blit copy slices");
static_assert(sizeof(RgfxBlitOpCopyIndirectCommandBuffer) == 40, "blit copy icb");
static_assert(sizeof(RgfxBlitOpFillBuffer) == 32, "blit fill buffer");
static_assert(sizeof(RgfxBlitOpFillBufferPattern4) == 32, "blit fill buffer pattern4");
static_assert(sizeof(RgfxBlitOpFillTexture) == 80, "blit fill texture head");
static_assert(RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET == sizeof(RgfxBlitOpFillTexture),
              "blit fill texture bytes offset");
static_assert(sizeof(RgfxBlitOpFillTextureColor) == 112, "blit fill texture color");
static_assert(sizeof(RgfxBlitOpGenerateMipmaps) == 8, "blit generate mipmaps");
static_assert(sizeof(RgfxBlitOpInvalidateCompressedTexture) == 8, "blit invalidate");
static_assert(sizeof(RgfxBlitOpInvalidateCompressedTextureSlice) == 24,
              "blit invalidate slice");
static_assert(sizeof(RgfxBlitOpOptimizeForCpu) == 8, "blit optimize cpu");
static_assert(sizeof(RgfxBlitOpOptimizeForCpuSlice) == 24, "blit optimize cpu slice");
static_assert(sizeof(RgfxBlitOpOptimizeForGpu) == 8, "blit optimize gpu");
static_assert(sizeof(RgfxBlitOpOptimizeForGpuSlice) == 24, "blit optimize gpu slice");
static_assert(sizeof(RgfxBlitOpOptimizeIndirectCommandBuffer) == 24, "blit optimize icb");
static_assert(sizeof(RgfxBlitOpResetCommandsInBuffer) == 24, "blit reset commands");
static_assert(sizeof(RgfxBlitOpSynchronizeResource) == 16, "blit synchronize resource");
static_assert(sizeof(RgfxBlitOpSynchronizeTexture) == 24, "blit synchronize texture");
static_assert(sizeof(RgfxBlitOpUpdateFence) == 8, "blit update fence");
static_assert(sizeof(RgfxBlitOpWaitForFence) == 8, "blit wait for fence");
