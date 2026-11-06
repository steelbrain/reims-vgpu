// RgfxGammaCapability: the display's gamma capability, the one entry RgfxDeviceControl answers to
// request 2 of attribute 0x711. IOPresentment turns each entry into a gamma dictionary for the
// planes it names, and CoreDisplay gives a plane gamma only from such an entry: a display whose
// entry names no plane gets "Attempting to get capabilities from plane with no gamma" from
// CoreDisplay, which then fades and calibrates the display in WindowServer's own frames rather
// than through the display pipe's gamma selectors (RgfxPipeColor.hpp).
//
// An entry, as macOS 26.7.1's IOPresentment reads it
// (IOPresentmentCapabilitiesDisplayPipe::IOPresentmentCreateGammaDictionary, read):
//   0x00  field_00, its flags (AGDCFlagsToIOPresentment)
//   0x04  field_04, the planes it applies to, bit i for plane i
//   0x0c  field_0c, a float: kIOPresentmentNonAtomicApplicationPeriod
//   0x10  field_10, one type bit: kIOPresentmentType (AGDCGammaTypeToIOPresentment)
//   0x14  field_14: kIOPresentmentFixedFormat
//   0x18  field_18: kIOPresentmentConfigurationID
//   0x20  field_20, a float: kIOPresentmentMinimumValue
//   0x24  field_24, a float: kIOPresentmentMaximumValue
//   0x28  field_28: kIOPresentmentMaximumSize, a table's most entries
// The answer holds at most 8 entries 0x32c bytes apart, from the answer's 0x0c, and their count
// at its 0x08. The table types are 0x8, 0x10, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000 and
// 0x2000, whose tables IOPresentment sends through the gamma selectors (5 to 7); type 1 is a
// configuration, which it sends through selector 11, which the pipe refuses. CoreDisplay takes
// the deepest table type a plane's entries give, 0x10 as 8 bits a channel, and tables of that
// entry's MaximumSize entries; a plane with no entry has no gamma, and one whose entries' sizes
// are 0 no table (CoreDisplay::IOPCapabilities::GammaChannelTableEntryCountAndType, read).
//
// The entry is the first one macOS 26.7.1's AMDRadeonX6000Framebuffer answers
// (AMDRadeonX6000_AmdAgdcServices::getFbCapabilityForGamma, read): flags 0x88005, plane 0, type
// 0x10, values 0.0 to 1.0 and tables of 256 entries (its ASIC's capability 0xf). The AMD
// adds a second entry of type 1 where its display path has it (its capability 0x17); ours has
// none, so the count is 1.
//
// Header-only and free of the kernel's headers, so its native test
// (tests/gamma_capability_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_GAMMA_CAPABILITY_HPP_
#define RGFX_GAMMA_CAPABILITY_HPP_

#include <stdint.h>

#include "RgfxPipeColor.hpp"

// Where the answer's entries are, and how many it holds.
constexpr uint32_t kRgfxCapabilityCount = 0x08;
constexpr uint32_t kRgfxCapabilityEntries = 0x0c;
constexpr uint32_t kRgfxCapabilityStride = 0x32c;
constexpr uint32_t kRgfxCapabilityMaxEntries = 8;

// The gamma entry's values.
constexpr uint32_t kRgfxGammaCapabilityFlags = 0x88005;
constexpr uint64_t kRgfxGammaCapabilityPlanes = 1;
constexpr uint32_t kRgfxGammaCapabilityType = 0x10;
constexpr uint32_t kRgfxGammaCapabilityFloatZero = 0x00000000;
constexpr uint32_t kRgfxGammaCapabilityFloatOne = 0x3f800000;
constexpr uint64_t kRgfxGammaCapabilityTableEntries = 256;

static_assert(kRgfxGammaCapabilityTableEntries <= kRgfxGammaMaxEntries,
              "the pipe takes a table of the most entries the display declares");

struct __attribute__((packed)) RgfxGammaCapability {
	uint32_t field_00;
	uint64_t field_04;
	uint32_t field_0c;
	uint32_t field_10;
	uint32_t field_14;
	uint64_t field_18;
	uint32_t field_20;
	uint32_t field_24;
	uint64_t field_28;
};
static_assert(sizeof(RgfxGammaCapability) == 0x30, "a gamma entry's fields are 0x30 bytes");

// The display's gamma entry; the count to answer with it is 1.
inline RgfxGammaCapability rgfxGammaCapability()
{
	RgfxGammaCapability g = {};
	g.field_00 = kRgfxGammaCapabilityFlags;
	g.field_04 = kRgfxGammaCapabilityPlanes;
	g.field_0c = kRgfxGammaCapabilityFloatZero;
	g.field_10 = kRgfxGammaCapabilityType;
	g.field_20 = kRgfxGammaCapabilityFloatZero;
	g.field_24 = kRgfxGammaCapabilityFloatOne;
	g.field_28 = kRgfxGammaCapabilityTableEntries;
	return g;
}

#endif // RGFX_GAMMA_CAPABILITY_HPP_
