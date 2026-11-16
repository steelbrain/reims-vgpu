// RgfxAllocationCount: the bytes a device's resources take in this process, which its
// currentAllocatedSize answers (RgfxMetalDevice.mm).
//
// What a Mac counts there, measured on both x86-64 Macs' GPUs (oracle_allocsize, AMD 5300M and
// Intel 630, macOS 26.7.1), each object made, then freed:
// - a resource the device makes counts its allocatedSize while it lives, and leaves the count
//   when it is freed, at once, with no GPU work in flight: buffers of every storage mode, by
//   length, by bytes and no-copy over the app's pages (each its whole pages); textures, an
//   IOSurface's among them; indirect command buffers; visible function tables;
// - a heap counts its size, and what is made in it adds nothing;
// - a texture view, a texture over a buffer, samplers, depth/stencil states, events, fences,
//   pipelines and queues add nothing of their own.
// Beyond that each GPU counts memory of its driver's own, which the two do not agree on: the
// 5300M 544768 bytes for the first queue and 32768 for the first sampler, the 630 8699904 from
// the device's making, both some for a command buffer's run. The device counts none of it.
//
// Header-only, for the native tests. Our own code.

#ifndef RGFX_ALLOCATION_COUNT_HPP_
#define RGFX_ALLOCATION_COUNT_HPP_

#include <atomic>
#include <cstdint>

class RgfxAllocationCount {
public:
	void add(uint64_t bytes) { bytes_.fetch_add(bytes, std::memory_order_relaxed); }
	void remove(uint64_t bytes) { bytes_.fetch_sub(bytes, std::memory_order_relaxed); }
	uint64_t bytes() const { return bytes_.load(std::memory_order_relaxed); }

private:
	std::atomic<uint64_t> bytes_{0};
};

#endif
