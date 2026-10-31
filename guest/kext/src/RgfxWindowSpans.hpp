// RgfxWindowSpans: which spans of the region a client's window (RgfxAcceleratorUserClient.hpp)
// maps in. A span is whole pages past the region's reserved head (the rings and the device
// page, which no process maps), inside the window. Header-only and free of the kernel's
// headers, so its native test (tests/window_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_WINDOW_SPANS_HPP_
#define RGFX_WINDOW_SPANS_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The window's page: a span starts and ends on one, so no page mapped in holds a byte of
// anything else's.
constexpr uint64_t kRgfxWindowPage = 0x1000;

// Whether the `length` bytes at region offset `offset` are a span a window of `windowSize`
// bytes maps in.
inline bool rgfxWindowSpan(uint64_t windowSize, uint64_t offset, uint64_t length)
{
	return length && offset % kRgfxWindowPage == 0 && length % kRgfxWindowPage == 0 &&
	       offset >= RGFX_ALLOCATIONS_OFFSET && offset <= windowSize &&
	       length <= windowSize - offset;
}

#endif // RGFX_WINDOW_SPANS_HPP_
