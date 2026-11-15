// RgfxDoorbellPage: where the process rings its queues' doorbells (RGFX_FEATURE_DOORBELL_PAGES;
// design.md, "Doorbell pages").
//
// The kext binds a doorbell page of the device's BAR 4 to the process's context and maps it into
// the process, uncached, where the bundle asks for it (kRgfxAccelMethodOpenContext's seventh
// output, kRgfxKextFeatureDoorbellPage). A 32-bit store of a queue's ring id at the page's first
// byte then wakes the host's consumer of the ring, as the kext's doorbell call's write of
// RGFX_REG_DOORBELL does, with no call into the kext: the store exits to the hypervisor, which
// hands it to the host. A context with no page (a kext or host without the pages, or every page
// held) rings through the kext's call, as before them.
//
// Header-only and free of IOKit, so the shell tests drive it as the device does.
//
// Our own code; it depends only on the user client's ABI.
#ifndef RGFX_DOORBELL_PAGE_HPP_
#define RGFX_DOORBELL_PAGE_HPP_

#include <cstdint>

#include "../../kext/src/RgfxUserClientABI.hpp"

// How many outputs the bundle asks OpenContext for, given the kext's own word: the span where the
// kext answers it, and with it the doorbell page where the kext answers that.
inline uint32_t rgfxOpenContextOutputs(uint64_t kextFeatures)
{
	if ((kextFeatures & kRgfxKextFeatureContextSpan) == 0)
		return 5;
	return (kextFeatures & kRgfxKextFeatureDoorbellPage) != 0 ? 7 : 6;
}

// The doorbell page OpenContext's `count` outputs name: null where they are fewer than seven or the
// seventh is 0 (the kext mapped none).
inline volatile uint32_t *rgfxDoorbellPageOf(const uint64_t *outputs, uint32_t count)
{
	if (count < 7 || outputs[6] == 0)
		return nullptr;
	return reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(outputs[6]));
}

// Rings ring `ringId` at the doorbell page `page`: one 32-bit store at its first byte, after
// every store before it, the ring's published producer among them. On arm64 the region's release
// stores order those for the guest's CPUs alone, and the device is none of them, so a DMB OSHST
// comes first, as the kext's register writes take (RgfxRegionMapping.hpp); and the store is one STR
// of a W register at a base register's address, the one form a hypervisor decodes from its
// abort's syndrome. x86-64 keeps every store in order, and takes the volatile store, which a
// compiler barrier keeps after the stores before it.
inline void rgfxDoorbellPageRing(volatile uint32_t *page, uint32_t ringId)
{
#if defined(__aarch64__)
	__builtin_arm_dmb(2); // DMB OSHST
	__asm__ volatile("str %w0, [%1]" : : "rZ"(ringId), "r"(page) : "memory");
#else
	__asm__ volatile("" : : : "memory");
	*page = ringId;
#endif
}

#endif // RGFX_DOORBELL_PAGE_HPP_
