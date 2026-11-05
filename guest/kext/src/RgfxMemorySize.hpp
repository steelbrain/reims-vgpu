// RgfxMemorySize: the GPU memory the kext states to the guest, from the one answer the host gives
// for it in the device info (RgfxDeviceInfo, written at RGFX_DEVICE_INFO_OFFSET when the device is
// made): its Metal device's recommendedMaxWorkingSetSize (sizes[0]) and hasUnifiedMemory
// (identity_flags[3]), the answers the bundle gives for the guest's device.
//
// Measured on an Intel Mac under macOS 26.7.1, both GPUs:
//   - AMD Radeon Pro 5300M: recommendedMaxWorkingSetSize 4278190080, hasUnifiedMemory NO,
//     dedicatedMemorySize 4278190080; its IOPCIDevice states VRAM,totalMB 4080 (the working set
//     in MiB), which is what system_profiler reports as "VRAM (Total)".
//   - Intel UHD Graphics 630: recommendedMaxWorkingSetSize 1610612736, hasUnifiedMemory YES,
//     dedicatedMemorySize 0; its accelerator states VRAM,totalMB 1536.
// So VRAM,totalMB is the working set in MiB, and dedicatedMemorySize is the working set where the
// GPU has memory of its own, else 0. IOFBMemorySize is a different thing: the AMD's framebuffer
// publishes its aperture's size (256 MiB), and the Intel's, as ours, publishes none
// (RgfxScanoutOwner.hpp).
//
// Header-only and free of the kernel's headers, so its native test (tests/memory_size_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_MEMORY_SIZE_HPP_
#define RGFX_MEMORY_SIZE_HPP_

#include "reims_graphics_wire.h"

#include <stddef.h>
#include <stdint.h>

// The positions of the two answers in the device info, in the host's IdentitySizes and
// IdentityFlags field orders.
static constexpr uint32_t kRgfxDeviceInfoWorkingSetSize = 0;   // recommended_max_working_set_size
static constexpr uint32_t kRgfxDeviceInfoHasUnifiedMemory = 3; // has_unified_memory

// The two answers' bytes in RgfxDeviceInfo: the span the kext reads of it.
static constexpr uint64_t kRgfxMemorySizeFirst =
    offsetof(RgfxDeviceInfo, sizes) + kRgfxDeviceInfoWorkingSetSize * sizeof(uint64_t);
static constexpr uint64_t kRgfxMemorySizeEnd =
    offsetof(RgfxDeviceInfo, identity_flags) + kRgfxDeviceInfoHasUnifiedMemory + 1;

struct RgfxMemorySize {
	// VRAM,totalMB: the working set in MiB; 0 where the host gave no working set, and then the
	// kext states none.
	uint32_t vramMB;
	// The IOGPU configuration's dedicatedMemorySize (its u64 at 0x10): the working set where the
	// GPU has memory of its own, else 0.
	uint64_t dedicatedBytes;
};

inline RgfxMemorySize rgfxMemorySize(uint64_t workingSet, bool unified)
{
	const uint64_t mb = workingSet >> 20;
	RgfxMemorySize m;
	m.vramMB = mb > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(mb);
	m.dedicatedBytes = unified ? 0 : workingSet;
	return m;
}

// The answers out of the device info's bytes from kRgfxMemorySizeFirst to kRgfxMemorySizeEnd.
inline RgfxMemorySize rgfxMemorySizeFrom(const uint8_t *span)
{
	uint64_t workingSet = 0;
	for (unsigned i = 0; i < sizeof(workingSet); i++)
		workingSet |= static_cast<uint64_t>(span[i]) << (8 * i);
	const bool unified = span[kRgfxMemorySizeEnd - 1 - kRgfxMemorySizeFirst] != 0;
	return rgfxMemorySize(workingSet, unified);
}

#endif // RGFX_MEMORY_SIZE_HPP_
