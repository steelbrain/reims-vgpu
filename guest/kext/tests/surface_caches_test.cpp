// The native test of RgfxSurfaceCaches.hpp: a lock of a cached surface the GPU wrote asks one
// download and goes on once it is answered, other locks waiting for it meanwhile; a write named
// during a download is copied by the next; a read-write lock moves the CPU generation; a lock
// waits for the block's release; and a surface that is not cached does nothing past the writes.
#include "RgfxSurfaceCaches.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_caches_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kSurface = 0x1000, kOther = 0x2000, kNotCached = 0x3000;
constexpr uint64_t kBlock = 0x40000, kOtherBlock = 0x80000;
constexpr uint32_t kId = 42, kOtherId = 43;
} // namespace

int main()
{
	RgfxCachedSurface entries[2] = {};
	RgfxCacheTable t = {entries, 0, 2};
	uint64_t offset = 0, mark = 0;

	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	check(c != nullptr && c->cpuGen == 0 && c->writes == 0, "a cached surface starts at 0");
	check(rgfxCacheAdd(t, kSurface, kOtherBlock, kOtherId) == nullptr, "a key is added once");
	check(rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId) != nullptr, "a second surface");
	check(rgfxCacheAdd(t, 0x9000, 0x90000, 9) == nullptr, "no room past the capacity");

	// Nothing written: a lock goes on, asking nothing.
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn, "nothing to copy");
	check(rgfxCacheNext(t, kNotCached, &offset, &mark) == kRgfxCacheGoOn,
	      "a surface not cached goes on");
	check(!rgfxCacheNoteWrite(t, kNotCached), "a write of a surface not cached is not kept");

	// A read-write lock moves the generation, of its own surface only.
	check(rgfxCacheReadWriteLock(t, kSurface) == 1, "the first read-write lock: generation 1");
	check(rgfxCacheReadWriteLock(t, kSurface) == 2, "the second: generation 2");
	check(rgfxCacheFind(t, kOther)->cpuGen == 0, "the other surface's stays");
	check(rgfxCacheReadWriteLock(t, kNotCached) == 0, "a surface not cached has none");

	// The GPU writes it: the next lock asks a download, and another waits for its answer.
	check(rgfxCacheNoteWrite(t, kSurface), "a write is named");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload && offset == kBlock &&
	          mark == 1,
	      "a lock after a write asks the download of the surface's block");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "a second lock waits for the first's download");
	check(rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheGoOn,
	      "the other surface's lock does not");

	// A write named while the download is in flight is still to copy after its answer.
	check(rgfxCacheNoteWrite(t, kSurface), "a write named during the download");
	rgfxCacheDownloaded(t, kSurface, mark, true);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload && mark == 2,
	      "the later write asks a download of its own");
	rgfxCacheDownloaded(t, kSurface, mark, true);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn,
	      "every write copied: the lock goes on");

	// A download not answered OK copied nothing: the next lock asks again.
	check(rgfxCacheNoteWrite(t, kSurface), "another write");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload, "its download");
	rgfxCacheDownloaded(t, kSurface, mark, false);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload,
	      "a refused download is asked again");
	rgfxCacheDownloaded(t, kSurface, mark, true);

	// The block goes: locks wait for its release's answer, and the surface leaves the table.
	check(!rgfxCacheReleasing(t, 0x123000), "a block of no cached surface");
	check(rgfxCacheOfSurface(t, kId) == rgfxCacheFind(t, kSurface), "found by IOSurfaceID");
	check(rgfxCacheReleasing(t, kBlock), "the cached block's release is posted");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "a lock waits for the release");
	check(rgfxCacheOfSurface(t, kId) == nullptr, "a surface whose block goes has no generation");
	void *map = nullptr;
	rgfxCacheRemove(t, rgfxCacheAtBlock(t, kBlock), &map);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn,
	      "answered: the lock goes on, the surface no longer cached");
	check(t.count == 1 && rgfxCacheFind(t, kOther) != nullptr, "the other surface stays");

	std::printf("surface_caches_test: OK (%d checks)\n", checks);
	return 0;
}
