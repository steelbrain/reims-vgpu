/*
 * RgfxCaps: reading the RgfxDeviceInfo caps blob and answering the device's
 * feature queries from it (RgfxCaps.hpp).
 */

#include "RgfxCaps.hpp"

#include <algorithm>

#include <cstring>
#include <vector>

// The enum counts match the blob's arrays: a drift in either is caught here, at
// the guest's own compile, beside wire.rs's assertions on the host side.
static_assert(RGFX_DEVICE_INFO_SPARSE_TILES == 9, "3 sample counts at 3 pages");
static_assert(kRgfxSizeCount == RGFX_DEVICE_INFO_SIZES,
              "size count must match RgfxDeviceInfo::sizes");
static_assert(kRgfxSizeMaxBufferLength == RGFX_DEVICE_INFO_MAX_BUFFER_LENGTH,
              "maxBufferLength where the kext reads it");
static_assert(kRgfxIdentityCount == RGFX_DEVICE_INFO_IDENTITY_FLAGS,
              "identity flag count must match RgfxDeviceInfo::identity_flags");
static_assert(sizeof(RgfxDeviceInfo::family_values) / sizeof(int64_t) ==
                  sizeof(RgfxDeviceInfo::family_answers),
              "a family answer per family value");
static_assert(sizeof(RgfxDeviceInfo::feature_set_values) / sizeof(uint64_t) ==
                  sizeof(RgfxDeviceInfo::feature_set_answers),
              "a feature set answer per feature set value");
static_assert(kRgfxLimitCount == sizeof(RgfxDeviceInfo::limits) / sizeof(uint64_t),
              "limit count must match RgfxDeviceInfo::limits");
static_assert(kRgfxFlagCount == sizeof(RgfxDeviceInfo::flags),
              "flag count must match RgfxDeviceInfo::flags");

RgfxCaps::RgfxCaps() : m_valid(false), m_features(0), m_kextFeatures(0)
{
	std::memset(&m_info, 0, sizeof(m_info));
	m_name[0] = '\0';
	m_architectureName[0] = '\0';
}

namespace {
// `len` bytes of `text` (at most `cap`) into `out`, NUL-terminated.
void copyText(char *out, const uint8_t *text, uint32_t len, uint32_t cap)
{
	if (len > cap)
		len = cap;
	std::memcpy(out, text, len);
	out[len] = '\0';
}
} // namespace

bool RgfxCaps::load(const RgfxDeviceInfo &info)
{
	if (info.version != RGFX_DEVICE_INFO_VERSION) {
		m_valid = false;
		return false;
	}
	m_info = info;
	// Keep NUL-terminated copies of the names for the C string accessors; the blob
	// stores each one's bytes in use, the rest zero.
	copyText(m_name, info.name, info.name_len, RGFX_DEVICE_NAME_LEN);
	copyText(m_architectureName, info.architecture_name, info.architecture_name_len,
	         RGFX_DEVICE_ARCHITECTURE_NAME_LEN);
	if (m_info.family_count > RGFX_DEVICE_INFO_FAMILIES)
		m_info.family_count = RGFX_DEVICE_INFO_FAMILIES;
	if (m_info.feature_set_count > RGFX_DEVICE_INFO_FEATURE_SETS)
		m_info.feature_set_count = RGFX_DEVICE_INFO_FEATURE_SETS;
	if (m_info.format_count > RGFX_DEVICE_INFO_FORMATS)
		m_info.format_count = RGFX_DEVICE_INFO_FORMATS;
	m_valid = true;
	return true;
}

void RgfxCaps::takeFeatures(int kr, uint64_t word, uint64_t kext)
{
	constexpr uint64_t kOffered =
	    RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_SURFACE_CACHE | RGFX_FEATURE_CB_SUBMITTED;
	m_features = kr == 0 ? word : featuresLo() & kOffered;
	m_kextFeatures = kr == 0 ? kext : 0;
}

bool RgfxCaps::supportsFamily(int64_t family) const
{
	for (uint32_t k = 0; k < m_info.family_count; k++) {
		if (m_info.family_values[k] == family)
			return m_info.family_answers[k] != 0;
	}
	return false;
}

bool RgfxCaps::supportsFeatureSet(uint64_t set) const
{
	for (uint32_t k = 0; k < m_info.feature_set_count; k++) {
		if (m_info.feature_set_values[k] == set)
			return m_info.feature_set_answers[k] != 0;
	}
	return false;
}

bool RgfxCaps::supportsCounterSampling(uint64_t point) const
{
	if (point >= RGFX_DEVICE_INFO_COUNTER_POINTS)
		return false;
	return m_info.counter_sampling[point] != 0;
}

namespace {
// The blob's answers of `format`, or null where it lists none.
const RgfxFormatAnswers *formatAnswers(const RgfxDeviceInfo &info, uint64_t format)
{
	for (uint32_t k = 0; k < info.format_count; k++) {
		if (info.formats[k].format == format)
			return &info.formats[k];
	}
	return nullptr;
}
} // namespace

uint64_t RgfxCaps::textureBufferAlignment(uint64_t format) const
{
	const RgfxFormatAnswers *a = formatAnswers(m_info, format);
	return a ? a->texture_buffer_alignment : 0;
}

uint64_t RgfxCaps::linearTextureAlignment() const
{
	uint64_t align = m_info.min_linear_alignment_bgra8;
	if (align == 0)
		return 0;
	for (uint32_t k = 0; k < m_info.format_count; k++)
		align = std::max<uint64_t>(align, m_info.formats[k].texture_buffer_alignment);
	return align;
}

uint64_t RgfxCaps::minimumLinearTextureAlignment(uint64_t format) const
{
	(void)format;
	return limit(kRgfxLimitLinearTextureAlignmentBytes);
}

bool RgfxCaps::sparseTile(uint64_t format, uint64_t samples, int64_t page, uint64_t out[3]) const
{
	out[0] = out[1] = out[2] = 0;
	// Sample counts 1, 2 and 4 (major), pages 101, 102 and 103, as the blob holds them.
	int s = samples == 1 ? 0 : samples == 2 ? 1 : samples == 4 ? 2 : -1;
	int p = page >= 101 && page <= 103 ? static_cast<int>(page - 101) : -1;
	const RgfxFormatAnswers *a = formatAnswers(m_info, format);
	if (a == nullptr || s < 0 || p < 0)
		return false;
	const uint16_t *tile = a->sparse_tiles[s * 3 + p];
	if (tile[0] == 0 && tile[1] == 0 && tile[2] == 0)
		return false;
	out[0] = tile[0];
	out[1] = tile[1];
	out[2] = tile[2];
	return true;
}

uint64_t RgfxCaps::size(RgfxSize which) const
{
	if (which < 0 || which >= kRgfxSizeCount)
		return 0;
	return m_info.sizes[which];
}

bool RgfxCaps::identityFlag(RgfxIdentityFlag which) const
{
	if (which < 0 || which >= kRgfxIdentityCount)
		return false;
	return m_info.identity_flags[which] != 0;
}

bool RgfxCaps::supportsSampleCount(uint32_t count) const
{
	switch (count) {
	case 1:
		return m_info.sample_counts[0] != 0;
	case 2:
		return m_info.sample_counts[1] != 0;
	case 4:
		return m_info.sample_counts[2] != 0;
	case 8:
		return m_info.sample_counts[3] != 0;
	default:
		return false;
	}
}

bool RgfxCaps::supportsPrimitiveType(uint32_t type) const
{
	if (type >= sizeof(m_info.primitive_types))
		return false;
	return m_info.primitive_types[type] != 0;
}

bool RgfxCaps::supportsWriteRoundingMode(uint32_t mode) const
{
	// The blob carries modes 1 and 2 at indices 0 and 1.
	if (mode == 1 || mode == 2)
		return m_info.write_rounding_modes[mode - 1] != 0;
	return false;
}

uint64_t RgfxCaps::limit(RgfxLimit which) const
{
	if (which < 0 || which >= kRgfxLimitCount)
		return 0;
	return m_info.limits[which];
}

bool RgfxCaps::flag(RgfxFlag which) const
{
	if (which < 0 || which >= kRgfxFlagCount)
		return false;
	return m_info.flags[which] != 0;
}

// The Mac2 family's raw MTLGPUFamily value. Every x86 Mac's GPU and Apple silicon's Macs are of
// it, and its feature table holds each feature below.
constexpr int64_t kRgfxFamilyMac2 = 2002;

bool RgfxCaps::spiFeature(RgfxSpiFeature which) const
{
	if (!m_valid)
		return false;
	switch (which) {
	// What the wire carries: argument encoders and buffers; the draws with a base vertex and
	// instance (RENDER_DRAW_*_BASE); binary archives; a sampler's border colour, address modes and
	// compare function, as given; store actions as given, with the attachments' resolve textures
	// (RENDER_SET_*_RESOLVE_TEXTURE) and the depth and stencil resolve filters;
	// COMPUTE_MEMORY_BARRIER_*; a compute pass's dispatch type, which the device runs in order,
	// as the concurrent type permits; RENDER_SET_VISIBILITY_RESULT_MODE, the counting mode among
	// them; RENDER_SET_DEPTH_CLIP_MODE_SPI; the indirect draws and dispatches, the indirect
	// stage-in region and patch draws; a render pass's renderTargetArrayLength, and its
	// renderTargetWidth, renderTargetHeight and defaultRasterSampleCount with no attachment;
	// textures over a Shared buffer's bytes (RGFX_CMD_TEXTURE_FROM_BUFFER); COMPUTE_DISPATCH_THREADS;
	// heaps and their resources; the patch draws and tessellation factors; swizzles;
	// RENDER_SET_VIEWPORTS and RENDER_SET_SCISSOR_RECTS. The host's device carries them out where
	// it is of the Mac2 family.
	case kRgfxSpiArgumentBuffers:
	case kRgfxSpiBaseVertexInstanceDrawing:
	case kRgfxSpiBinaryArchives:
	case kRgfxSpiBlackOrWhiteSamplerBorderColors:
	case kRgfxSpiCombinedMSAAStoreAndResolveAction:
	case kRgfxSpiComputeMemoryBarrier:
	case kRgfxSpiConcurrentComputeDispatch:
	case kRgfxSpiCountingOcclusionQuery:
	case kRgfxSpiDepthClipMode:
	case kRgfxSpiIndirectDrawAndDispatch:
	case kRgfxSpiIndirectStageInRegion:
	case kRgfxSpiIndirectTessellation:
	case kRgfxSpiLayeredRendering:
	case kRgfxSpiLinearTextureFromSharedBuffer:
	case kRgfxSpiMSAADepthResolve:
	case kRgfxSpiMSAADepthResolveFilter:
	case kRgfxSpiMSAAStencilResolve:
	case kRgfxSpiMSAAStencilResolveFilter:
	case kRgfxSpiMirrorClampToEdgeSamplerMode:
	case kRgfxSpiNonUniformThreadgroupSize:
	case kRgfxSpiRenderPassWithoutRenderTarget:
	case kRgfxSpiResourceHeaps:
	case kRgfxSpiSamplerCompareFunction:
	case kRgfxSpiTessellation:
	case kRgfxSpiTextureSwizzle:
	case kRgfxSpiViewportAndScissorArray:
	// Features of the Metal Shading Language, which the guest's compiler takes from the app and
	// the host's platform runs: arrays of samplers and textures, buffer and texture writes from
	// fragment functions, 64-bit integers, invariant positions, quad-group and SIMD-group
	// functions, the min_lod_clamp sample option, stencil feedback and cube arrays.
	case kRgfxSpiArrayOfSamplers:
	case kRgfxSpiArrayOfTextures:
	case kRgfxSpiFragmentBufferWrites:
	case kRgfxSpiInt64:
	case kRgfxSpiInvariantVertexPosition:
	case kRgfxSpiQuadGroup:
	case kRgfxSpiQuadShufflesAndBroadcast:
	case kRgfxSpiRenderTextureWrites:
	case kRgfxSpiSIMDGroup:
	case kRgfxSpiSIMDShufflesAndBroadcast:
	case kRgfxSpiShaderMinLODClamp:
	case kRgfxSpiStencilFeedback:
	case kRgfxSpiTextureCubeArray:
		return supportsFamily(kRgfxFamilyMac2);
	// The host's own answers, for features the wire carries: buffers' attribute strides (the
	// *_ATTRIBUTE_STRIDE ops), raster order groups (areRasterOrderGroupsSupported), the SIMD-group
	// reductions, and 2D multisample array textures.
	case kRgfxSpiDynamicAttributeStride:
		return flag(kRgfxFlagSupportsDynamicAttributeStride);
	case kRgfxSpiRasterOrderGroups:
		return flag(kRgfxFlagAreRasterOrderGroupsSupported);
	case kRgfxSpiSIMDReduction:
		return flag(kRgfxFlagSupportsSimdReduction);
	case kRgfxSpiTexture2DMultisampleArray:
		return flag(kRgfxFlagSupportsTexture2DMultisampleArray);
	// Read-write texture arguments, at the host's read-write texture tier (Tier1 is 1).
	case kRgfxSpiReadWriteTextureArguments:
		return size(kRgfxSizeReadWriteTextureSupport) >= 1;
	case kRgfxSpiCount:
		break;
	}
	return false;
}

uint64_t RgfxCaps::maxThreadsPerThreadgroup(uint32_t dim) const
{
	if (dim >= 3)
		return 0;
	return m_info.max_threads_per_threadgroup[dim];
}

std::string rgfxVendorNameOf(const char *name)
{
	std::string first;
	for (const char *c = name; *c != '\0' && *c != ' '; c++)
		first += *c;
	if (first == "NVIDIA")
		return "NVIDIA";
	if (first == "Intel" || first == "Intel(R)")
		return "Intel(R)";
	return "AMD";
}

std::string rgfxFamilyNameOf(const char *name)
{
	std::vector<std::string> words;
	std::string word;
	for (const char *c = name; ; c++) {
		if (*c == ' ' || *c == '\0') {
			if (!word.empty())
				words.push_back(word);
			word.clear();
			if (*c == '\0')
				break;
		} else {
			word += *c;
		}
	}
	size_t first = 0;
	if (!words.empty()) {
		for (const char *vendor : {"Apple", "AMD", "Intel", "Intel(R)", "NVIDIA"}) {
			if (words[0] == vendor) {
				first = 1;
				break;
			}
		}
	}
	std::string out;
	for (size_t k = first; k < words.size(); k++) {
		if (k > first && words[k][0] >= '0' && words[k][0] <= '9')
			break;
		if (!out.empty())
			out += ' ';
		out += words[k];
	}
	return out.empty() ? std::string(name) : out;
}
