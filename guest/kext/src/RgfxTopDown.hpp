// RgfxTopDown: the order an x86-64 kext faults a range's pages in before it wires them, from its
// last page down to its first (rgfxTopDown).
//
// An x86-64 macOS kernel hands out its free pages a 16 KiB clump at a time, each clump's pages in
// ascending order on its free queue, but refills each CPU's list of free pages last in, first out
// (vm_page_list_push), so the pages a range faults in from its first page up come out each clump
// backwards: a page a run of guest RAM, which the host maps a run at a time. Faulted from the last
// page down, the clump's pages land in ascending order, four pages a run. **Measured** 2026-11-14
// on an x86-64 macOS 26.7.2 guest on homelab-ms01-1's Iris Xe (jobs m3-max-31194 and 31227): the
// boot's 16,201-page view went from 1.000 runs a page to 0.251, Mission Control's 7,261-page
// capture surface from 1.000 to 0.252 and 0.312, and the views' mappings the host held at the
// desktop from 36,897 to 12,441, the walk's steps and Geekbench 7.1's times within their spread.
//
// Header-only and free of the kernel's headers, so its native test (tests/top_down_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_TOP_DOWN_HPP_
#define RGFX_TOP_DOWN_HPP_

#include <stdint.h>

// Calls `visit(at)` with the first byte of each page of `page` bytes (a power of two) that the
// `length` bytes at `address` touch, from the page of their last byte down to the page of their
// first, until `visit` answers false: the pages a wire of the range takes, as IOKit rounds it out.
// Answers how many pages `visit` took. None where the range is empty, or wraps or ends in the
// address space's last page.
template <typename Visit>
inline uint64_t rgfxTopDown(uint64_t address, uint64_t length, uint64_t page, Visit &&visit)
{
	if (length == 0 || page == 0 || (page & (page - 1)) != 0)
		return 0;
	const uint64_t mask = page - 1;
	if (address > UINT64_MAX - length || address + length > UINT64_MAX - mask)
		return 0;
	const uint64_t first = address & ~mask;
	uint64_t at = (address + length + mask) & ~mask;
	uint64_t taken = 0;
	while (at > first) {
		at -= page;
		if (!visit(at))
			break;
		taken++;
	}
	return taken;
}

#endif // RGFX_TOP_DOWN_HPP_
