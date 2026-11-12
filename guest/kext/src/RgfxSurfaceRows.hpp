// RgfxSurfaceRows: the alignment the kext has IOSurface give the bytes per row it works out for a
// surface, from the host's answer in the device info (RgfxDeviceInfo's linear_row_alignment,
// written at RGFX_DEVICE_INFO_OFFSET when the device is made): the bytes the host's GPU aligns a
// linear BGRA8 image's rows to, to make one over a surface's bytes in place there. The host
// answers it where it offers the surfaces' pages, on a GPU of any memory, else 0.
//
// IOSurface aligns the bytes per row it works out for a surface, and answers
// IOSurfaceGetPropertyAlignment(kIOSurfaceBytesPerRow), to the cache line where no accelerator
// asks for more: 64 bytes on an Intel Mac (measured under macOS 26.7.1, its 630 and 5300M both),
// 128 on an Apple M3 Max. Metal's minimumLinearTextureAlignmentForPixelFormat: of BGRA8Unorm is
// 256 bytes on the 630 and the 5300M (measured), so 256-byte rows lie inside what Macs answer.
// The kext asks for the host's alignment where it is more than 64 bytes, a power of two and at
// most a page; any other answer leaves IOSurface's 64.
//
// Header-only and free of the kernel's headers, so its native test (tests/surface_rows_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_SURFACE_ROWS_HPP_
#define RGFX_SURFACE_ROWS_HPP_

#include "reims_graphics_wire.h"

#include <stddef.h>
#include <stdint.h>

// The bytes of RgfxDeviceInfo the kext reads for the alignment.
static constexpr uint64_t kRgfxSurfaceRowsFirst = offsetof(RgfxDeviceInfo, linear_row_alignment);
static constexpr uint64_t kRgfxSurfaceRowsEnd = kRgfxSurfaceRowsFirst + sizeof(uint32_t);

// IOSurface's own alignment of the bytes per row it works out, on an x86-64 Mac: its cache line.
static constexpr uint32_t kRgfxSurfaceRowsLeast = 64;

// The most the kext asks for: a page.
static constexpr uint32_t kRgfxSurfaceRowsMost = 4096;

// The alignment of the bytes per row IOSurface works out, from the host's answer.
inline uint32_t rgfxSurfaceRowAlignment(uint32_t host)
{
	const bool power = host != 0 && (host & (host - 1)) == 0;
	if (!power || host <= kRgfxSurfaceRowsLeast || host > kRgfxSurfaceRowsMost)
		return kRgfxSurfaceRowsLeast;
	return host;
}

// The alignment out of the device info's bytes from kRgfxSurfaceRowsFirst to kRgfxSurfaceRowsEnd.
inline uint32_t rgfxSurfaceRowAlignmentFrom(const uint8_t *span)
{
	uint32_t host = 0;
	for (unsigned i = 0; i < sizeof(host); i++)
		host |= static_cast<uint32_t>(span[i]) << (8 * i);
	return rgfxSurfaceRowAlignment(host);
}

#endif // RGFX_SURFACE_ROWS_HPP_
