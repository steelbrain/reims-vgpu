// RgfxWindowSpans: which spans of the region a client's window (RgfxAcceleratorUserClient.hpp)
// maps in, and what the window's own memory is. A span is whole pages past the region's reserved
// head (the rings and the device page, which no process maps), inside the window. Header-only
// and free of the kernel's headers, so its native test (tests/window_test.cpp) builds it as the
// kext does.
//
// Our own code.
#ifndef RGFX_WINDOW_SPANS_HPP_
#define RGFX_WINDOW_SPANS_HPP_

#include <stdint.h>

#include "RgfxRegionMapping.hpp"
#include "reims_graphics_wire.h"

// The window's page, a process's page (RgfxRegionMapping.hpp): a span starts and ends on one, so
// no page mapped in, or taken back, holds a byte of anything else's.
constexpr uint64_t kRgfxWindowPage = kRgfxProcessPage;

// The window's own memory, in IOBufferMemoryDescriptor's option bits (IOKit's
// IOMemoryDescriptor.h and IOBufferMemoryDescriptor.h; RgfxAcceleratorUserClientWindow.cpp
// asserts each is IOKit's): the kernel's to read and write, pageable, so it holds no page until
// one is read, and purgeable, never made volatile. The purgeable bit decides what the window
// costs the process (read from the public xnu source). vm_map_enter charges an IOKit mapping of
// anonymous memory to the task's footprint at its whole size as it is made
// (vm_map_iokit_mapped_region, the iokit_mapped ledger), but not a mapping of a purgeable
// object: that object's resident pages are charged to its owner, which mach_make_memory_entry
// makes the task that creates the memory, the process whose context open makes the window. So
// the window costs the process only the pages it reads there, and not the region's size.
constexpr uint32_t kRgfxWindowDirectionInOut = 0x3; // kIODirectionInOut
constexpr uint32_t kRgfxWindowPageable = 0x20;      // kIOMemoryPageable
constexpr uint32_t kRgfxWindowPurgeable = 0x40;     // kIOMemoryPurgeable
constexpr uint32_t kRgfxWindowMemoryOptions =
    kRgfxWindowDirectionInOut | kRgfxWindowPageable | kRgfxWindowPurgeable;

// Whether the `length` bytes at region offset `offset` are a span a window of `windowSize`
// bytes maps in.
inline bool rgfxWindowSpan(uint64_t windowSize, uint64_t offset, uint64_t length)
{
	return length && offset % kRgfxWindowPage == 0 && length % kRgfxWindowPage == 0 &&
	       offset >= RGFX_ALLOCATIONS_OFFSET && offset <= windowSize &&
	       length <= windowSize - offset;
}

#endif // RGFX_WINDOW_SPANS_HPP_
