/*
 * RgfxCaps: the guest bundle's view of the host device's capabilities, read once
 * at device init from the RgfxDeviceInfo caps blob the host writes into the
 * device's region (docs/graphics-kext/design.md, Decided 10; the host filler is
 * crates/reims-vgpu/src/graphics/device_info.rs).
 *
 * The blob holds every value Metal or the Vulkan platform answers for a device
 * that does not change while the device lives, but where the lane's own device
 * answers otherwise (the host's device_info.rs says where). The bundle's
 * MTLDevice answers supportsFamily:, supportsFeatureSet:, its identity, the
 * limits and the feature queries from here, so the guest sees the host's
 * answers. A feature the host platform cannot answer (a runtime gap) is 0 in the
 * blob and so reads unsupported here, never a spurious yes (the owner's rule).
 *
 * The enum order is the host's field order in device_info.rs: this file names
 * each slot so the Metal shell reads caps by name, never by a magic index. This
 * is our own code; it depends only on the committed wire header.
 */

#ifndef RGFX_CAPS_HPP_
#define RGFX_CAPS_HPP_

#include <cstdint>

#include "reims_graphics_wire.h"

/// The 27 NSUInteger limits, in the host `Limits` field order.
enum RgfxLimit {
	kRgfxLimitDeviceLinearTextureAlignmentBytes = 0,
	kRgfxLimitIOSurfaceReadOnlyTextureAlignmentBytes,
	kRgfxLimitIOSurfaceTextureAlignmentBytes,
	kRgfxLimitLinearTextureAlignmentBytes,
	kRgfxLimitMaxComputeBuffers,
	kRgfxLimitMaxComputeLocalMemorySizes,
	kRgfxLimitMaxComputeSamplers,
	kRgfxLimitMaxComputeTextures,
	kRgfxLimitMaxComputeThreadgroupMemory,
	kRgfxLimitMaxComputeThreadgroupMemoryAlignmentBytes,
	kRgfxLimitMaxFragmentBuffers,
	kRgfxLimitMaxFragmentSamplers,
	kRgfxLimitMaxFragmentTextures,
	kRgfxLimitMaxPredicatedNestingDepth,
	kRgfxLimitMaxRasterizationRateLayerCount,
	kRgfxLimitMaxTextureHeight2D,
	kRgfxLimitMaxTextureLayers,
	kRgfxLimitMaxTextureWidth2D,
	kRgfxLimitMaxThreadgroupMemoryLength,
	kRgfxLimitMaxTileBuffers,
	kRgfxLimitMaxTileSamplers,
	kRgfxLimitMaxTileTextures,
	kRgfxLimitMaxTotalComputeThreadsPerThreadgroup,
	kRgfxLimitMaxVertexBuffers,
	kRgfxLimitMaxVertexSamplers,
	kRgfxLimitMaxVertexTextures,
	kRgfxLimitMaxViewportCount,
	kRgfxLimitCount
};

/// The 25 BOOL properties, in the host `Flags` field order.
enum RgfxFlag {
	kRgfxFlagAreRasterOrderGroupsSupported = 0,
	kRgfxFlagIsDepth24Stencil8PixelFormatSupported,
	kRgfxFlagIsFramebufferReadSupported,
	kRgfxFlagIsLargeMRTSupported,
	kRgfxFlagIsRGB10A2GammaSupported,
	kRgfxFlagSupportsBufferWithAddressRanges,
	kRgfxFlagSupportsBufferWithIOSurface,
	kRgfxFlagSupportsCommandBufferJump,
	kRgfxFlagSupportsComputeCompressedTextureWrite,
	kRgfxFlagSupportsConditionalLoadStore,
	kRgfxFlagSupportsDynamicAttributeStride,
	kRgfxFlagSupportsExtendedXR10Formats,
	kRgfxFlagSupportsFloat16BcubicFiltering,
	kRgfxFlagSupportsHeapWithAddressRanges,
	kRgfxFlagSupportsImageBlocks,
	kRgfxFlagSupportsMemoryOrderAtomics,
	kRgfxFlagSupportsNativeHardwareFP16,
	kRgfxFlagSupportsNonSquareTileShaders,
	kRgfxFlagSupportsProgrammableSamplePositions,
	kRgfxFlagSupportsResourceDetachBacking,
	kRgfxFlagSupportsSharedTextureHandles,
	kRgfxFlagSupportsSimdReduction,
	kRgfxFlagSupportsSimdShuffleAndFill,
	kRgfxFlagSupportsTexture2DMultisampleArray,
	kRgfxFlagSupportsTileShaders,
	kRgfxFlagCount
};

/// The 19 integer identity answers, in the host `IdentitySizes` field order.
enum RgfxSize {
	kRgfxSizeRecommendedMaxWorkingSetSize = 0,
	kRgfxSizeLocation,
	kRgfxSizeLocationNumber,
	kRgfxSizeMaxTransferRate,
	kRgfxSizePeerGroupId,
	kRgfxSizePeerIndex,
	kRgfxSizePeerCount,
	kRgfxSizeMaxBufferLength,
	kRgfxSizeMaxArgumentBufferSamplerCount,
	kRgfxSizeArgumentBuffersSupport,
	kRgfxSizeReadWriteTextureSupport,
	kRgfxSizeSparseTileSizeInBytes,
	kRgfxSizeMaximumConcurrentCompilationTaskCount,
	kRgfxSizeLargestVertexAmplificationCount,
	kRgfxSizeLargestRasterizationRateMapLayerCount,
	kRgfxSizeFeatureProfile,
	kRgfxSizeHalfFpConfig,
	kRgfxSizeSingleFpConfig,
	kRgfxSizeDoubleFpConfig,
	kRgfxSizeCount
};

/// The 19 BOOL identity answers, in the host `IdentityFlags` field order.
enum RgfxIdentityFlag {
	kRgfxIdentityIsLowPower = 0,
	kRgfxIdentityIsHeadless,
	kRgfxIdentityIsRemovable,
	kRgfxIdentityHasUnifiedMemory,
	kRgfxIdentitySupports32BitFloatFiltering,
	kRgfxIdentitySupports32BitMSAA,
	kRgfxIdentitySupportsQueryTextureLOD,
	kRgfxIdentitySupportsBCTextureCompression,
	kRgfxIdentitySupportsPullModelInterpolation,
	kRgfxIdentityAreBarycentricCoordsSupported,
	kRgfxIdentitySupportsShaderBarycentricCoordinates,
	kRgfxIdentitySupportsDynamicLibraries,
	kRgfxIdentitySupportsRenderDynamicLibraries,
	kRgfxIdentitySupportsRaytracing,
	kRgfxIdentitySupportsFunctionPointers,
	kRgfxIdentitySupportsFunctionPointersFromRender,
	kRgfxIdentitySupportsRaytracingFromRender,
	kRgfxIdentitySupportsPrimitiveMotionBlur,
	kRgfxIdentitySupportsPlacementSparse,
	kRgfxIdentityCount
};

class RgfxCaps {
public:
	RgfxCaps();

	/// Takes the caps blob the host wrote. Returns false (and leaves the caps
	/// empty) when `info`'s version is not the one this build knows: the layout
	/// may differ, so nothing is trusted.
	bool load(const RgfxDeviceInfo &info);

	/// Whether a caps blob of a known version has been loaded.
	bool valid() const { return m_valid; }

	/// `supportsFamily:` of the raw `MTLGPUFamily` `family`: the blob's answer, and no for a
	/// value it does not list (one that names no family the host knows).
	bool supportsFamily(int64_t family) const;
	/// `supportsFeatureSet:` of the raw `MTLFeatureSet` `set`, as `supportsFamily:`.
	bool supportsFeatureSet(uint64_t set) const;
	/// `supportsCounterSampling:` of the raw `MTLCounterSamplingPoint` `point` (0..4; any
	/// other is no).
	bool supportsCounterSampling(uint64_t point) const;
	/// `minimumTextureBufferAlignmentForPixelFormat:` of the raw `MTLPixelFormat` `format`; 0
	/// where the blob has no answer (a format Metal asserts on, among them).
	uint64_t textureBufferAlignment(uint64_t format) const;
	/// What the host's textures over a buffer's bytes ask of their offset and rows: the largest of
	/// `minimumLinearTextureAlignmentForPixelFormat:`(BGRA8Unorm) and every format's
	/// texture-buffer alignment, the blob's answers; 0 where it has no linear answer.
	uint64_t linearTextureAlignment() const;
	/// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:` of `format` at
	/// `samples` (1, 2 or 4) in pages of the raw `MTLSparsePageSize` `page` (101, 102 or 103),
	/// the same for every texture type, into `out` (width, height, depth); false, with `out`
	/// all 0, where the blob has no answer.
	bool sparseTile(uint64_t format, uint64_t samples, int64_t page, uint64_t out[3]) const;
	/// One of the 15 integer identity answers.
	uint64_t size(RgfxSize which) const;
	/// One of the 19 BOOL identity answers.
	bool identityFlag(RgfxIdentityFlag which) const;
	/// `supportsTextureSampleCount:` of `count` (1, 2, 4 or 8; any other is no).
	bool supportsSampleCount(uint32_t count) const;
	/// `supportsPrimitiveType:` of the raw `MTLPrimitiveType` `type` (0..9).
	bool supportsPrimitiveType(uint32_t type) const;
	/// `supportsTextureWriteRoundingMode:` of `mode` (1 or 2; any other is no).
	bool supportsWriteRoundingMode(uint32_t mode) const;
	/// One of the 27 NSUInteger limits.
	uint64_t limit(RgfxLimit which) const;
	/// One of the 25 BOOL properties.
	bool flag(RgfxFlag which) const;

	/// `registryID`.
	uint64_t registryId() const { return m_info.registry_id; }
	/// `gpuCoreCount`, or 0 when the host does not answer it.
	uint64_t gpuCoreCount() const { return m_info.gpu_core_count; }
	/// `maxThreadsPerThreadgroup`'s `dim` (0 = width, 1 = height, 2 = depth).
	uint64_t maxThreadsPerThreadgroup(uint32_t dim) const;
	/// `maxVertexAmplificationCount`.
	uint64_t maxVertexAmplificationCount() const { return m_info.max_vertex_amplification_count; }
	/// The alignment of `heapBufferSizeAndAlignWithLength:options:`(`vm_page_size`, Private),
	/// or 0 where the host had no answer.
	uint64_t heapBufferPageAlign() const { return m_info.heap_buffer_page_private[1]; }
	/// The device name, NUL-terminated (a display string, never read back).
	const char *name() const { return m_name; }
	/// The device's `architecture.name`, NUL-terminated; empty where the host has none.
	const char *architectureName() const { return m_architectureName; }

private:
	bool m_valid;
	RgfxDeviceInfo m_info;
	char m_name[RGFX_DEVICE_NAME_LEN + 1];
	char m_architectureName[RGFX_DEVICE_ARCHITECTURE_NAME_LEN + 1];
};

#endif // RGFX_CAPS_HPP_
