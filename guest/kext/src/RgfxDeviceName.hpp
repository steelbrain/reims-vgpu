// RgfxDeviceName: the GPU name the kext states to the guest, from the host's device name in the
// device info (RgfxDeviceInfo's name_len and name, written at RGFX_DEVICE_INFO_OFFSET when the
// device is made): the name the bundle answers for the guest's Metal device.
//
// A Mac's GPU states its name as its IOPCIDevice's `model`, an OSData of the name's bytes, which
// System Information shows as the GPU's "Chipset Model"; without one it shows "Graphics" and the
// memory instead. Measured on an Intel Mac under macOS 26.7.1, both GPUs: the Intel UHD Graphics
// 630's `model` is its 22 bytes and a NUL, the AMD Radeon Pro 5300M's its 20 bytes with none,
// and System Information names both. So the kext sets `model` from the host's name as the
// Intel's is: its first name_len bytes (at most RGFX_DEVICE_NAME_LEN), cut at a NUL among them,
// then a NUL. A host that gave no name (name_len 0, or a NUL first) has none stated.
//
// Header-only and free of the kernel's headers, so its native test (tests/device_name_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DEVICE_NAME_HPP_
#define RGFX_DEVICE_NAME_HPP_

#include "reims_graphics_wire.h"

#include <stddef.h>
#include <stdint.h>

// The bytes of RgfxDeviceInfo the kext reads for the name: from name_len to the name's end.
static constexpr uint64_t kRgfxDeviceNameFirst = offsetof(RgfxDeviceInfo, name_len);
static constexpr uint64_t kRgfxDeviceNameEnd = offsetof(RgfxDeviceInfo, name) + RGFX_DEVICE_NAME_LEN;

// The most bytes of `model`: a whole name and its NUL.
static constexpr size_t kRgfxDeviceModelMax = RGFX_DEVICE_NAME_LEN + 1;

// `model`'s bytes into `out`, from a name_len and the name field's RGFX_DEVICE_NAME_LEN bytes:
// how many, its NUL counted, or 0 when the host gave no name.
inline size_t rgfxDeviceModel(uint32_t nameLen, const uint8_t *name, uint8_t (&out)[kRgfxDeviceModelMax])
{
	const size_t limit = nameLen < RGFX_DEVICE_NAME_LEN ? nameLen : RGFX_DEVICE_NAME_LEN;
	size_t n = 0;
	while (n < limit && name[n]) {
		out[n] = name[n];
		n++;
	}
	if (!n)
		return 0;
	out[n] = 0;
	return n + 1;
}

// `model`'s bytes out of the device info's bytes from kRgfxDeviceNameFirst to kRgfxDeviceNameEnd.
inline size_t rgfxDeviceModelFrom(const uint8_t *span, uint8_t (&out)[kRgfxDeviceModelMax])
{
	uint32_t nameLen = 0;
	for (unsigned i = 0; i < sizeof(nameLen); i++)
		nameLen |= static_cast<uint32_t>(span[i]) << (8 * i);
	return rgfxDeviceModel(nameLen, span + (offsetof(RgfxDeviceInfo, name) - kRgfxDeviceNameFirst), out);
}

#endif // RGFX_DEVICE_NAME_HPP_
