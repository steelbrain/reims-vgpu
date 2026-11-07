/*
 * RgfxLibraryStream: a library streamed through a staging window made for it
 * (RgfxLibraryStream.hpp).
 */

#include "RgfxLibraryStream.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <os/log.h>
#include <unistd.h>

#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

RgfxStagingShape rgfxStagingShape(uint64_t length, uint64_t chunkMax)
{
	const uint64_t chunks = (length + chunkMax - 1) / chunkMax;
	const uint64_t even = (length + chunks - 1) / chunks;
	RgfxStagingShape s = {};
	s.chunk = (even + kRgfxStagingPage - 1) / kRgfxStagingPage * kRgfxStagingPage;
	s.blocks = chunks > 1 ? 2 : 1;
	s.bytes = kRgfxStagingPage + s.blocks * s.chunk;
	return s;
}

bool rgfxStagingSmaller(uint64_t *chunkMax)
{
	if (*chunkMax <= kRgfxStagingPage)
		return false;
	const uint64_t next = *chunkMax / 4;
	*chunkMax = next < kRgfxStagingPage ? kRgfxStagingPage : next;
	return true;
}

namespace {

// One stream's window: its block, its chunks' size and blocks, and each block's place, its ack
// slot and the ack its last chunk asked for (0: none).
struct Window {
	uint64_t offset = 0;
	RgfxStagingShape shape = {};
	uint64_t blockOffset[2] = {};
	volatile uint8_t *block[2] = {};
	uint64_t ackOffset[2] = {};
	volatile uint64_t *ack[2] = {};
	uint64_t expected[2] = {};
};

// Streams in the process go one at a time, so it holds one window at most.
std::mutex &streamLock()
{
	static std::mutex *lock = new std::mutex(); // never destroyed: libraries come at any time
	return *lock;
}

// Each chunk's ack value, unique in the process.
uint64_t g_nextToken = 1;

// The window for `length` bytes: the shape for the largest chunks the region has room for.
bool makeWindow(RgfxRegion &region, uint64_t length, Window *w)
{
	uint64_t chunkMax = kRgfxStagingChunkMax;
	uint64_t lastBytes = 0;
	do {
		const RgfxStagingShape s = rgfxStagingShape(length, chunkMax);
		if (s.bytes == lastBytes)
			continue; // a library smaller than the chunk: the same window again
		lastBytes = s.bytes;
		// The chunks, which the process only writes, past the acks' page, which it reads:
		// write-combined where the kext maps parts so.
		RgfxRegion::Block b =
		    region.allocate(s.bytes, RgfxRegion::kDefaultBlockAlign, kRgfxStagingPage);
		if (!b.ok)
			continue;
		w->offset = b.offset;
		w->shape = s;
		for (uint32_t i = 0; i < 2; i++) {
			w->ackOffset[i] = b.offset + 64 * static_cast<uint64_t>(i);
			w->ack[i] = reinterpret_cast<volatile uint64_t *>(b.ptr + 64 * i);
			__atomic_store_n(w->ack[i], 0, __ATOMIC_RELEASE);
			w->blockOffset[i] = b.offset + kRgfxStagingPage + s.chunk * i;
			w->block[i] = b.ptr + kRgfxStagingPage + s.chunk * i;
		}
		if (s.chunk < rgfxStagingShape(length, kRgfxStagingChunkMax).chunk) {
			static std::atomic<bool> logged{false};
			if (!logged.exchange(true))
				os_log(OS_LOG_DEFAULT,
				       "RGFX: a library of %llu bytes streams in chunks of %llu bytes: the region "
				       "had no room for a larger window (logged once)",
				       (unsigned long long)length, (unsigned long long)s.chunk);
		}
		return true;
	} while (rgfxStagingSmaller(&chunkMax));
	static std::atomic<bool> logged{false};
	if (!logged.exchange(true))
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: no room in the region for a library's staging window, even of %llu "
		             "bytes: the library of %llu bytes is not made (logged once)",
		             (unsigned long long)rgfxStagingShape(length, kRgfxStagingPage).bytes,
		             (unsigned long long)length);
	return false;
}

// Waits, `awaitUs` at most, for the host to have copied block `i`'s last chunk out.
bool awaitBlock(const Window &w, uint32_t i, uint64_t awaitUs)
{
	const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(awaitUs);
	for (;;) {
		if (__atomic_load_n(w.ack[i], __ATOMIC_ACQUIRE) == w.expected[i])
			return true;
		if (std::chrono::steady_clock::now() >= end)
			return false;
		usleep(200);
	}
}

} // namespace

bool rgfxStreamLibrary(RgfxRegion &region, RgfxSerializer &ser, const uint8_t *bytes,
                       uint64_t length, const uint8_t *key, RgfxObjId *outObjId,
                       uint32_t *outStatus, uint64_t awaitUs)
{
	if (bytes == nullptr || length == 0)
		return false;
	std::lock_guard<std::mutex> guard(streamLock());
	Window w;
	if (!makeWindow(region, length, &w))
		return false;
	const uint64_t chunk = w.shape.chunk;
	const RgfxObjId objId = ser.allocLibraryId();
	bool streamed = true, copiedOut = true;
	uint32_t i = 0;
	for (uint64_t at = 0; at < length; at += chunk, i = (i + 1) % w.shape.blocks) {
		const uint64_t n = length - at < chunk ? length - at : chunk;
		if (!awaitBlock(w, i, awaitUs)) {
			streamed = copiedOut = false;
			break;
		}
		std::memcpy(const_cast<uint8_t *>(w.block[i]), bytes + at, n);
		const uint64_t token = g_nextToken++;
		bool posted = false;
		for (int r = 0; r < 2000 && !posted; r++) {
			posted = ser.postLibraryChunk(objId, w.blockOffset[i], n, length, at, w.ackOffset[i],
			                              token, key, outStatus);
			if (!posted)
				usleep(1000);
		}
		if (!posted) {
			streamed = false;
			break;
		}
		w.expected[i] = token;
	}
	// The window goes back once the host has copied out every chunk it took (the last chunk's
	// answer comes once the library is made from them all, so this rarely waits).
	for (uint32_t b = 0; copiedOut && b < w.shape.blocks; b++)
		copiedOut = awaitBlock(w, b, awaitUs);
	if (copiedOut) {
		region.release(w.offset);
	} else {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true))
			os_log_error(OS_LOG_DEFAULT,
			             "RGFX: the host did not copy a library's chunk out of its staging window "
			             "in time: the window's %llu bytes are kept (logged once)",
			             (unsigned long long)w.shape.bytes);
	}
	if (!streamed)
		return false;
	*outObjId = objId;
	return true;
}
