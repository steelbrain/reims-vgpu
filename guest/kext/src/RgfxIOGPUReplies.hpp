// RgfxIOGPUReplies: the two calls Apple's IOGPUMetalDevice base makes on the IOGPU device user
// client (kRgfxIOGPUDeviceUserClientType) as it builds the Metal device, from
// IOGPUDeviceCreateWithOptions inside [super initWithAcceleratorPort:], and the kext's replies
// (RgfxAcceleratorUserClient::deviceMethod). Measured: selector 2, the device's info (536 bytes
// out), then selector 0, its configuration (64 bytes out), each with no scalars and no struct
// in. Each must succeed at its exact size, or the device is nil. Header-only and free of the
// kernel's headers, so the native test (tests/user_client_words_test.cpp) checks the bytes the
// kext answers.
//
// Our own code.
#ifndef RGFX_IOGPU_REPLIES_HPP_
#define RGFX_IOGPU_REPLIES_HPP_

#include <stdint.h>

constexpr uint32_t kRgfxIOGPUSelConfig = 0;
constexpr uint32_t kRgfxIOGPUSelDeviceInfo = 2;
constexpr uint32_t kRgfxIOGPUConfigSize = 64;
constexpr uint32_t kRgfxIOGPUDeviceInfoSize = 536;

// The configuration's dedicatedMemorySize (its u64 at 0x10, which the guest's Metal device
// answers): the host's working set where its GPU has memory of its own, else 0
// (RgfxMemorySize.hpp). Its shared-memory size (at 0x20) is the region's: BAR 2's size, from the
// device.
constexpr uint32_t kRgfxIOGPUConfigDedicatedMemory = 0x10;
constexpr uint32_t kRgfxIOGPUConfigSharedMemory = 0x20;

// The reply's size where a call on the IOGPU device client is one of the two (`selector` with no
// scalars in or out, no struct in, and struct out of exactly the reply's size), else 0: a call
// the kext does not answer.
inline uint32_t rgfxIOGPUDeviceReplySize(uint32_t selector, uint32_t scalarsIn, uint32_t scalarsOut,
                                         uint64_t structIn, uint64_t structOut)
{
	if (scalarsIn != 0 || scalarsOut != 0 || structIn != 0)
		return 0;
	if (selector == kRgfxIOGPUSelConfig && structOut == kRgfxIOGPUConfigSize)
		return kRgfxIOGPUConfigSize;
	if (selector == kRgfxIOGPUSelDeviceInfo && structOut == kRgfxIOGPUDeviceInfoSize)
		return kRgfxIOGPUDeviceInfoSize;
	return 0;
}

// The reply to selector `selector` (one rgfxIOGPUDeviceReplySize answers), into `out`, of that
// size, for `dedicatedBytes` of dedicated memory and a device region of `regionSize` bytes. The
// configuration: the dedicated-memory and the shared-memory sizes, little-endian, the rest 0. The device info: all 0. Its u32 at
// 0x14 counts the u32 entries of a table at 0x18, which the base reads, so the device has none.
inline void rgfxIOGPUDeviceReply(uint32_t selector, uint8_t *out, uint64_t dedicatedBytes,
                                 uint64_t regionSize)
{
	const uint32_t size =
	    selector == kRgfxIOGPUSelConfig ? kRgfxIOGPUConfigSize : kRgfxIOGPUDeviceInfoSize;
	for (uint32_t i = 0; i < size; i++)
		out[i] = 0;
	if (selector != kRgfxIOGPUSelConfig)
		return;
	for (uint32_t i = 0; i < 8; i++) {
		out[kRgfxIOGPUConfigDedicatedMemory + i] = static_cast<uint8_t>(dedicatedBytes >> (8 * i));
		out[kRgfxIOGPUConfigSharedMemory + i] = static_cast<uint8_t>(regionSize >> (8 * i));
	}
}

#endif // RGFX_IOGPU_REPLIES_HPP_
