/*
 * RgfxLibraryStream: a library too large for one ring message, streamed to the host in chunks
 * (RGFX_CMD_LIBRARY_FROM_REGION) through a staging window of the device region.
 *
 * THE WINDOW. One block of the region: a page of ack slots (one a cache line), then one chunk
 * block, or two used in turn where the library takes more than one chunk, so the copy of one
 * chunk overlaps the host's copy-out of the one before. The host writes a block's ack slot once
 * it has copied its chunk out, and the block is written again only then. The window is made for
 * one stream, sized to its library: chunks of the library split evenly into as few chunks of at
 * most kRgfxStagingChunkMax as hold it, each a whole number of pages, so a window is never more
 * than the library and three pages. It goes back to the region once the host has copied out every
 * chunk, so a process holds no window between streams.
 *
 * Where the region has no room for that window, smaller chunks are asked for, down to a page
 * each, so a library is streamed while any room for a window of two pages and an ack page is
 * left; only then is it refused.
 *
 * Streams are serialised in the process, so it holds at most one window at a time.
 *
 * No window is kept for the next stream: what a window costs is small beside the stream
 * (measured on the RTX 5080 host, 42 apps and 7 WebGL and WebGPU pages open, 75 processes with a
 * context: 52 streams in 10 processes, at most 12 in one; a window's take from the kext 69 us at
 * the median and 374 us at most, its give 26 us and 345 us, 6.1 ms in all against the streams'
 * 1.6 s, the host's compiles in them). That boot refused no region block, where the window each
 * process kept for its life had held 576 of the region's 731 MiB outside WindowServer's reserve,
 * and refused blocks from 41 processes on.
 *
 * Our own code; depends only on the committed wire header.
 */

#ifndef RGFX_LIBRARY_STREAM_HPP_
#define RGFX_LIBRARY_STREAM_HPP_

#include <cstdint>

#include "RgfxObjectModel.hpp"

class RgfxRegion;
class RgfxSerializer;

/// A page: the ack slots' block and the unit every chunk is a multiple of.
constexpr uint64_t kRgfxStagingPage = 0x1000;
/// The most bytes one chunk carries: 32 MiB, so QuartzCore's 160 MB library goes in five chunks
/// per block.
constexpr uint64_t kRgfxStagingChunkMax = 32ull << 20;

/// A staging window's shape: its chunks' size, how many chunk blocks it holds (1 or 2), and its
/// bytes, the ack page with them.
struct RgfxStagingShape {
	uint64_t chunk;
	uint32_t blocks;
	uint64_t bytes;
};

/// The shape of the window that streams `length` bytes (more than 0) in chunks of at most
/// `chunkMax` bytes (a multiple of a page, at least one): as few chunks as hold the library,
/// split evenly and rounded up to whole pages, and a second block only where there are two or
/// more.
RgfxStagingShape rgfxStagingShape(uint64_t length, uint64_t chunkMax);

/// The chunk sizes a stream asks for in turn where the region refuses the window before: from
/// kRgfxStagingChunkMax down to a page, each a quarter of the one before. `*chunkMax` becomes the
/// next; false after the last.
bool rgfxStagingSmaller(uint64_t *chunkMax);

/// How long a stream waits for the host to copy a block's chunk out, at most: ten seconds.
constexpr uint64_t kRgfxStagingAwaitUs = 10ull * 1000 * 1000;

/// Streams the `length` bytes at `bytes` to the host as one library, through a window of
/// `region` made for it: each chunk copied into the next block once the host has copied out what
/// it last held, then posted, carrying the content key where `key` (32 bytes) is not null. The
/// last chunk's answer is `*outStatus`, and the library's id `*outObjId`. The window goes back
/// once every chunk posted is copied out, which the host's acks say; one whose acks do not come
/// within `awaitUs` is kept, so nothing the host may still read is given out again. False where
/// the region has room for no window, where the host did not take a chunk, or where it did not
/// copy one out within `awaitUs`.
bool rgfxStreamLibrary(RgfxRegion &region, RgfxSerializer &ser, const uint8_t *bytes,
                       uint64_t length, const uint8_t *key, RgfxObjId *outObjId,
                       uint32_t *outStatus, uint64_t awaitUs = kRgfxStagingAwaitUs);

#endif // RGFX_LIBRARY_STREAM_HPP_
