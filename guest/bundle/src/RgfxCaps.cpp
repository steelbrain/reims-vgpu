/*
 * RgfxCaps: reading the RgfxDeviceInfo caps blob and answering the device's
 * feature queries from it (RgfxCaps.hpp).
 */

#include "RgfxCaps.hpp"

#include <cstring>

// The enum counts match the blob's arrays: a drift in either is caught here, at
// the guest's own compile, beside wire.rs's assertions on the host side.
static_assert(kRgfxFamilyCount == sizeof(RgfxDeviceInfo::families),
              "family count must match RgfxDeviceInfo::families");
static_assert(kRgfxLimitCount == sizeof(RgfxDeviceInfo::limits) / sizeof(uint64_t),
              "limit count must match RgfxDeviceInfo::limits");
static_assert(kRgfxFlagCount == sizeof(RgfxDeviceInfo::flags),
              "flag count must match RgfxDeviceInfo::flags");

RgfxCaps::RgfxCaps() : m_valid(false)
{
	std::memset(&m_info, 0, sizeof(m_info));
	m_name[0] = '\0';
}

bool RgfxCaps::load(const RgfxDeviceInfo &info)
{
	if (info.version != RGFX_DEVICE_INFO_VERSION) {
		m_valid = false;
		return false;
	}
	m_info = info;
	// Keep a NUL-terminated copy of the name for the C string accessor; the blob
	// stores name_len bytes, the rest zero, and a longer name was truncated.
	uint32_t len = info.name_len;
	if (len > RGFX_DEVICE_NAME_LEN)
		len = RGFX_DEVICE_NAME_LEN;
	std::memcpy(m_name, info.name, len);
	m_name[len] = '\0';
	m_valid = true;
	return true;
}

bool RgfxCaps::supportsFamily(RgfxFamily family) const
{
	if (family < 0 || family >= kRgfxFamilyCount)
		return false;
	return m_info.families[family] != 0;
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

uint64_t RgfxCaps::maxThreadsPerThreadgroup(uint32_t dim) const
{
	if (dim >= 3)
		return 0;
	return m_info.max_threads_per_threadgroup[dim];
}
