/*
 * RgfxCaps: the guest bundle's view of the host device's capabilities, read once
 * at device init from the RgfxDeviceInfo caps blob the host writes into the
 * device's region (docs/graphics-kext/design.md, Decided 10; the host filler is
 * crates/reims-vgpu/src/graphics/device_info.rs).
 *
 * The blob mirrors every value Metal or the Vulkan platform answers for a device
 * that does not change while the device lives. The bundle's MTLDevice answers
 * supportsFamily:, the limits and the feature queries from here, so the guest
 * sees exactly the host device's answers. A feature the host platform cannot
 * answer (a runtime gap) is 0 in the blob and so reads unsupported here, never a
 * spurious yes (the owner's rule).
 *
 * The enum order is the host's field order in device_info.rs: this file names
 * each slot so the Metal shell reads caps by name, never by a magic index. This
 * is our own code; it depends only on the committed wire header.
 */

#ifndef RGFX_CAPS_HPP_
#define RGFX_CAPS_HPP_

#include <cstdint>

#include "reims_graphics_wire.h"

/// The 7 GPU families the host asks `supportsFamily:` of, in the blob's order.
enum RgfxFamily {
	kRgfxFamilyApple5 = 0,
	kRgfxFamilyApple6,
	kRgfxFamilyApple7,
	kRgfxFamilyApple8,
	kRgfxFamilyApple9,
	kRgfxFamilyApple10,
	kRgfxFamilyMinus1009, // the -1009 family bit
	kRgfxFamilyCount
};

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

class RgfxCaps {
public:
	RgfxCaps();

	/// Takes the caps blob the host wrote. Returns false (and leaves the caps
	/// empty) when `info`'s version is not the one this build knows: the layout
	/// may differ, so nothing is trusted.
	bool load(const RgfxDeviceInfo &info);

	/// Whether a caps blob of a known version has been loaded.
	bool valid() const { return m_valid; }

	/// `supportsFamily:` of `family`.
	bool supportsFamily(RgfxFamily family) const;
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
	/// The device name, NUL-terminated (a display string, never read back).
	const char *name() const { return m_name; }

private:
	bool m_valid;
	RgfxDeviceInfo m_info;
	char m_name[RGFX_DEVICE_NAME_LEN + 1];
};

#endif // RGFX_CAPS_HPP_
