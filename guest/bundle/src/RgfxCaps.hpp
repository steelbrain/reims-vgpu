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
#include <string>

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

/// The MTLDeviceSPI `supports*` queries the device answers itself (RgfxMetalDevice.mm), where
/// the IOGPU base answers NO from its all-zero device info and the M3 Max, the AMD 5300M and the
/// Intel UHD 630 all answer YES (measured, tools/base_answers_probe.m). `supportsX` is kRgfxSpiX.
enum RgfxSpiFeature : int {
	// What the wire carries, on a host of the Mac2 family (RgfxCaps.cpp says what of the wire).
	kRgfxSpiArgumentBuffers = 0,
	kRgfxSpiBaseVertexInstanceDrawing,
	kRgfxSpiBinaryArchives,
	kRgfxSpiBlackOrWhiteSamplerBorderColors,
	kRgfxSpiCombinedMSAAStoreAndResolveAction,
	kRgfxSpiComputeMemoryBarrier,
	kRgfxSpiConcurrentComputeDispatch,
	kRgfxSpiCountingOcclusionQuery,
	kRgfxSpiDepthClipMode,
	kRgfxSpiIndirectDrawAndDispatch,
	kRgfxSpiIndirectStageInRegion,
	kRgfxSpiIndirectTessellation,
	kRgfxSpiLayeredRendering,
	kRgfxSpiLinearTextureFromSharedBuffer,
	kRgfxSpiMSAADepthResolve,
	kRgfxSpiMSAADepthResolveFilter,
	kRgfxSpiMSAAStencilResolve,
	kRgfxSpiMSAAStencilResolveFilter,
	kRgfxSpiMirrorClampToEdgeSamplerMode,
	kRgfxSpiNonUniformThreadgroupSize,
	kRgfxSpiRenderPassWithoutRenderTarget,
	kRgfxSpiResourceHeaps,
	kRgfxSpiSamplerCompareFunction,
	kRgfxSpiTessellation,
	kRgfxSpiTextureSwizzle,
	kRgfxSpiViewportAndScissorArray,
	// Features of the Metal Shading Language the host runs, on a host of the Mac2 family.
	kRgfxSpiArrayOfSamplers,
	kRgfxSpiArrayOfTextures,
	kRgfxSpiFragmentBufferWrites,
	kRgfxSpiInt64,
	kRgfxSpiInvariantVertexPosition,
	kRgfxSpiQuadGroup,
	kRgfxSpiQuadShufflesAndBroadcast,
	kRgfxSpiRenderTextureWrites,
	kRgfxSpiSIMDGroup,
	kRgfxSpiSIMDShufflesAndBroadcast,
	kRgfxSpiShaderMinLODClamp,
	kRgfxSpiStencilFeedback,
	kRgfxSpiTextureCubeArray,
	// The host's own answers, in the device info's flags.
	kRgfxSpiDynamicAttributeStride,
	kRgfxSpiRasterOrderGroups,
	kRgfxSpiSIMDReduction,
	kRgfxSpiTexture2DMultisampleArray,
	// The host's read-write texture tier.
	kRgfxSpiReadWriteTextureArguments,
	kRgfxSpiCount
};

/// The MTLDeviceSPI `familyName` of a device named `name`: the name without its vendor (a first
/// word "Apple", "AMD", "Intel", "Intel(R)" or "NVIDIA") and without its model number (the first
/// later word that starts with a digit, and every word after it), as Metal's devices answer
/// (measured: "M3 Max" for "Apple M3 Max", "Radeon Pro" for "AMD Radeon Pro 5300M", "UHD
/// Graphics" for "Intel(R) UHD Graphics 630"). Where that leaves nothing, the whole name.
std::string rgfxFamilyNameOf(const char *name);

/// The MTLDeviceSPI `vendorName` of a device named `name`: the vendor its first word names, in
/// the words Metal's devices answer, "NVIDIA", "AMD" or "Intel(R)" (measured: "AMD" for the AMD
/// Radeon Pro 5300M, "Intel(R)" for the Intel UHD Graphics 630, macOS 26.7.1), for a name whose
/// first word is "NVIDIA", "AMD", "Intel" or "Intel(R)"; "AMD" for any other. The system's GL
/// renderer (AppleMetalOpenGLRenderer) picks its settings by finding "AMD" or "ATI", "NVIDIA" or
/// "Intel" in the answer, else by the Apple GPU families 7 and 5, and ends every process that
/// makes a GL context on any other device ("Unknown GPU!!", read from its code; measured with the
/// IOGPU base's "Apple" beside Mac2 alone, macOS 26.6.1): a GPU of another vendor answers the one
/// the x86 Macs' discrete GPUs answer, which the renderer runs gl_cases' cases with as it does
/// with the others (tools/gl_cases.m, measured: macOS 26.6.1 on an RTX 5080 host).
std::string rgfxVendorNameOf(const char *name);

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
	/// `minimumLinearTextureAlignmentForPixelFormat:` of the raw `MTLPixelFormat` `format`: the
	/// blob's `linearTextureAlignmentBytes`, for every format. A Radeon Pro 5300M, a UHD 630 and an
	/// M3 Max each answer one value for every format they make linear textures of, and end the
	/// process on any other (Metal's assertion, measured: compressed, depth and stencil formats,
	/// Invalid, and formats the device lacks), never 0; so a guest reaches no other format on a
	/// real Mac, and one that does gets no 0 to align up by.
	uint64_t minimumLinearTextureAlignment(uint64_t format) const;
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
	/// What the device answers to the MTLDeviceSPI query `which` (RgfxSpiFeature); no for a
	/// value past the list, and for every query with no caps.
	bool spiFeature(RgfxSpiFeature which) const;

	/// `registryID`.
	uint64_t registryId() const { return m_info.registry_id; }
	/// FEATURES_LO's low byte (RGFX_FEATURE_*), as the host wrote it; 0 from an older host.
	uint8_t featuresLo() const { return m_valid ? m_info.features_lo : 0; }
	/// Takes the kext's answer to the features call (the accelerator's Features selector): `kr`
	/// its return, 0 for success, `word` the device's feature word and `kext` the kext's own. A
	/// kext older than the call answers kIOReturnUnsupported; on that, or any failure, the device's
	/// word is `featuresLo()`'s bits 0 to 2, the ones the host acts on as it offers them, and the
	/// kext's is 0.
	void takeFeatures(int kr, uint64_t word, uint64_t kext);
	/// The RGFX_FEATURE_* bits this process works by: bits 0 to 2 as the host offers them, and
	/// from bit 3 the ones the kext took for its bundle (takeFeatures); 0 before takeFeatures.
	uint64_t features() const { return m_features; }
	/// The kext's own features (kRgfxKextFeature*, the kext's RgfxUserClientABI.hpp): the user-client calls
	/// and flags it carries out beyond the first kext with the features call; 0 before
	/// takeFeatures.
	uint64_t kextFeatures() const { return m_kextFeatures; }
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
	uint64_t m_features;
	uint64_t m_kextFeatures;
	RgfxDeviceInfo m_info;
	char m_name[RGFX_DEVICE_NAME_LEN + 1];
	char m_architectureName[RGFX_DEVICE_ARCHITECTURE_NAME_LEN + 1];
};

#endif // RGFX_CAPS_HPP_
