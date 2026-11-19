// block_zero_probe: whether a region block one process filled and let go of reads zero in the
// next process the kext gives its offset (the kext's RgfxBlockZero.hpp), for each kind of block
// in the region, on the kext's own selectors (guest/kext/src/RgfxUserClientABI.hpp), with no Metal.
// Each step is its own process, so the bytes cross processes as an app's would:
//   block_zero_probe fill KIND free|close   takes a block of KIND, stores 0xa5 over every byte its
//                                            window maps, then frees it (Free, or SurfaceBlockRelease
//                                            for a surface's copy) or leaves that to its close, and
//                                            exits. Prints "FILL KIND at 0xOFFSET, N bytes".
//   block_zero_probe take KIND OFFSET       takes blocks of KIND, holding each one in front of
//                                            OFFSET, until one is at OFFSET (3 s at most, as the
//                                            device answers the release), and counts the bytes of
//                                            it its window maps that do not read zero. Prints
//                                            "ZERO OK KIND ..." or "ZERO FAIL KIND: why", exit 1.
//   block_zero_probe cost [N]               N bodies (64 KiB, 256 KiB and 1 MiB, write-combined
//                                            whole) taken and freed, each take's call timed: the
//                                            mean, p50, p99 and most in microseconds by size.
// KIND: body (64 KiB, write-combined whole, as a command buffer's body), staging (a write-back
// page and 256 KiB write-combined, as a library's staging chunk), slab (1 MiB write-back), copy (a
// 1920x1080 BGRA IOSurface's copy block).
// Build: xcrun clang -arch x86_64 -O2 -mmacosx-version-min=11.0 -framework IOKit
//        -framework IOSurface -framework CoreFoundation block_zero_probe.c -o block_zero_probe
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../kext/src/RgfxUserClientABI.hpp"

enum { kPage = 0x1000, kMaxHeld = 64 };

struct Client {
	io_connect_t conn;
	uint64_t window;
	int writeCombined;
};

static uint64_t nowNs(void)
{
	static mach_timebase_info_data_t tb;
	if (!tb.denom)
		mach_timebase_info(&tb);
	return mach_absolute_time() * tb.numer / tb.denom;
}

static int open_(struct Client *c)
{
	memset(c, 0, sizeof(*c));
	io_service_t svc =
	    IOServiceGetMatchingService(MACH_PORT_NULL, IOServiceMatching("RgfxAccelerator"));
	if (!svc)
		return 0;
	kern_return_t kr = IOServiceOpen(svc, mach_task_self(), kRgfxAccelUserClientType, &c->conn);
	IOObjectRelease(svc);
	if (kr != KERN_SUCCESS)
		return 0;
	uint64_t features[3] = {0};
	uint32_t n = 3;
	if (IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodFeatures, NULL, 0, features, &n) ==
	    KERN_SUCCESS)
		c->writeCombined = (features[1] & kRgfxKextFeatureWriteCombined) != 0;
	uint64_t ctx[5] = {0};
	n = 5;
	if (IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodOpenContext, NULL, 0, ctx, &n) !=
	    KERN_SUCCESS)
		return 0;
	c->window = ctx[3];
	return 1;
}

struct Kind {
	const char *name;
	uint64_t size;
	uint64_t writeCombinedFrom; // UINT64_MAX: write-back whole
	int copy;
};

static const struct Kind kinds[] = {
    {"body", 64 << 10, 0, 0},
    {"staging", kPage + (256 << 10), kPage, 0},
    {"slab", 1 << 20, UINT64_MAX, 0},
    {"copy", 1920ull * 1080 * 4, 0, 1},
};

static const struct Kind *kindNamed(const char *name)
{
	for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); i++)
		if (!strcmp(kinds[i].name, name))
			return &kinds[i];
	return NULL;
}

// A block of `k`: its region offset and the bytes its window maps; for a copy, the surface it
// copies, kept for the process's life.
static kern_return_t take(struct Client *c, const struct Kind *k, uint64_t *offset,
                          uint64_t *bytes)
{
	uint32_t n = 1;
	if (!k->copy) {
		const uint64_t in[3] = {k->size, kPage, k->writeCombinedFrom};
		*bytes = (k->size + kPage - 1) & ~(uint64_t)(kPage - 1);
		return IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodAlloc, in,
		                                 c->writeCombined ? 3 : 2, offset, &n);
	}
	const int width = 1920, height = 1080, bpe = 4;
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(
	    NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	const int fmt = 0x42475241 /* 'BGRA' */, rowBytes = width * bpe;
	CFNumberRef w = CFNumberCreate(NULL, kCFNumberIntType, &width);
	CFNumberRef h = CFNumberCreate(NULL, kCFNumberIntType, &height);
	CFNumberRef e = CFNumberCreate(NULL, kCFNumberIntType, &bpe);
	CFNumberRef r = CFNumberCreate(NULL, kCFNumberIntType, &rowBytes);
	CFNumberRef f = CFNumberCreate(NULL, kCFNumberIntType, &fmt);
	CFDictionarySetValue(props, kIOSurfaceWidth, w);
	CFDictionarySetValue(props, kIOSurfaceHeight, h);
	CFDictionarySetValue(props, kIOSurfaceBytesPerElement, e);
	CFDictionarySetValue(props, kIOSurfaceBytesPerRow, r);
	CFDictionarySetValue(props, kIOSurfacePixelFormat, f);
	IOSurfaceRef s = IOSurfaceCreate(props);
	CFRelease(w);
	CFRelease(h);
	CFRelease(e);
	CFRelease(r);
	CFRelease(f);
	CFRelease(props);
	if (!s)
		return kIOReturnNoMemory;
	IOSurfaceLock(s, 0, NULL);
	const uint64_t plane = (uint64_t)(uintptr_t)IOSurfaceGetBaseAddress(s);
	const uint64_t length = (uint64_t)height * rowBytes;
	memset((void *)(uintptr_t)plane, 0x3c, length);
	IOSurfaceUnlock(s, 0, NULL);
	const uint64_t size = (length + kPage - 1) & ~(uint64_t)(kPage - 1);
	const uint64_t in[3] = {plane, length, size};
	uint64_t out[4] = {0};
	n = 4;
	kern_return_t kr =
	    IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodSurfaceBlock, in, 3, out, &n);
	*offset = out[0];
	*bytes = size;
	return kr;
}

static kern_return_t letGo(struct Client *c, const struct Kind *k, uint64_t offset)
{
	return IOConnectCallScalarMethod(
	    c->conn, k->copy ? kRgfxAccelMethodSurfaceBlockRelease : kRgfxAccelMethodFree, &offset, 1,
	    NULL, NULL);
}

static int fill(const struct Kind *k, int close_)
{
	struct Client c;
	if (!open_(&c)) {
		printf("FILL FAIL %s: no context\n", k->name);
		return 1;
	}
	uint64_t offset = 0, bytes = 0;
	kern_return_t kr = take(&c, k, &offset, &bytes);
	if (kr != KERN_SUCCESS) {
		printf("FILL FAIL %s: take 0x%x\n", k->name, kr);
		return 1;
	}
	memset((void *)(uintptr_t)(c.window + offset), 0xa5, bytes);
	__sync_synchronize();
	printf("FILL %s at 0x%llx, %llu bytes, %s\n", k->name, (unsigned long long)offset,
	       (unsigned long long)bytes, close_ ? "left to the close" : "freed");
	if (!close_) {
		kr = letGo(&c, k, offset);
		if (kr != KERN_SUCCESS) {
			printf("FILL FAIL %s: free 0x%x\n", k->name, kr);
			return 1;
		}
	}
	fflush(stdout);
	// The process's close frees what it holds; exit closes the connection.
	return 0;
}

static int takeAt(const struct Kind *k, uint64_t want)
{
	struct Client c;
	if (!open_(&c)) {
		printf("ZERO FAIL %s: no context\n", k->name);
		return 1;
	}
	uint64_t held[kMaxHeld];
	uint32_t holding = 0;
	const uint64_t until = nowNs() + 3000000000ull;
	uint64_t offset = 0, bytes = 0;
	int found = 0, tries = 0;
	while (!found && nowNs() < until && holding < kMaxHeld) {
		tries++;
		kern_return_t kr = take(&c, k, &offset, &bytes);
		if (kr != KERN_SUCCESS) {
			printf("ZERO FAIL %s: take 0x%x\n", k->name, kr);
			return 1;
		}
		if (offset == want) {
			found = 1;
		} else if (offset < want) {
			// A gap in front of it: held, so the next take looks past it.
			held[holding++] = offset;
		} else {
			// Past it: its release not yet answered, or another process's block there.
			letGo(&c, k, offset);
			usleep(50000);
		}
	}
	if (!found) {
		printf("ZERO SKIP %s: no block at 0x%llx in %d takes (%u held in front)\n", k->name,
		       (unsigned long long)want, tries, holding);
		return 2;
	}
	const volatile uint8_t *p = (const volatile uint8_t *)(uintptr_t)(c.window + offset);
	uint64_t nonzero = 0, filled = 0;
	for (uint64_t i = 0; i < bytes; i++) {
		const uint8_t v = p[i];
		nonzero += v != 0;
		filled += v == 0xa5;
	}
	for (uint32_t i = 0; i < holding; i++)
		letGo(&c, k, held[i]);
	if (nonzero) {
		printf("ZERO FAIL %s: at 0x%llx, %llu of %llu bytes not zero (%llu of the filler's)\n",
		       k->name, (unsigned long long)offset, (unsigned long long)nonzero,
		       (unsigned long long)bytes, (unsigned long long)filled);
		return 1;
	}
	printf("ZERO OK %s: at 0x%llx, all %llu bytes zero, in %d takes\n", k->name,
	       (unsigned long long)offset, (unsigned long long)bytes, tries);
	return 0;
}

static int cmpU64(const void *a, const void *b)
{
	const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return x < y ? -1 : x > y;
}

static int cost(int count)
{
	struct Client c;
	if (!open_(&c)) {
		printf("COST FAIL: no context\n");
		return 1;
	}
	static const uint64_t sizes[] = {64 << 10, 256 << 10, 1 << 20};
	uint64_t *ns = calloc((size_t)count, sizeof(uint64_t));
	if (!ns)
		return 1;
	for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
		const struct Kind body = {"body", sizes[s], 0, 0};
		uint64_t sum = 0;
		for (int i = 0; i < count; i++) {
			uint64_t offset = 0, bytes = 0;
			const uint64_t t0 = nowNs();
			kern_return_t kr = take(&c, &body, &offset, &bytes);
			ns[i] = nowNs() - t0;
			sum += ns[i];
			if (kr != KERN_SUCCESS) {
				printf("COST FAIL: take of %llu KiB 0x%x\n", (unsigned long long)(sizes[s] >> 10),
				       kr);
				return 1;
			}
			// Written once, as a body is.
			((volatile uint8_t *)(uintptr_t)(c.window + offset))[0] = 1;
			letGo(&c, &body, offset);
		}
		qsort(ns, (size_t)count, sizeof(uint64_t), cmpU64);
		printf("COST body %llu KiB: %d takes, mean %.1f us, p50 %.1f us, p99 %.1f us, max %.1f us\n",
		       (unsigned long long)(sizes[s] >> 10), count, (double)sum / count / 1e3,
		       (double)ns[count / 2] / 1e3, (double)ns[(count * 99) / 100] / 1e3,
		       (double)ns[count - 1] / 1e3);
		usleep(200000);
	}
	free(ns);
	printf("COST OK\n");
	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 4 && !strcmp(argv[1], "fill") && kindNamed(argv[2]) &&
	    (!strcmp(argv[3], "free") || !strcmp(argv[3], "close")))
		return fill(kindNamed(argv[2]), !strcmp(argv[3], "close"));
	if (argc == 4 && !strcmp(argv[1], "take") && kindNamed(argv[2]))
		return takeAt(kindNamed(argv[2]), strtoull(argv[3], NULL, 0));
	if (argc >= 2 && !strcmp(argv[1], "cost")) {
		const long n = argc >= 3 ? strtol(argv[2], NULL, 0) : 200;
		if (n < 1 || n > 100000) {
			fprintf(stderr, "block_zero_probe cost: N from 1 to 100000\n");
			return 2;
		}
		return cost((int)n);
	}
	fprintf(stderr, "usage: block_zero_probe fill body|staging|slab|copy free|close\n"
	                "       block_zero_probe take body|staging|slab|copy OFFSET\n"
	                "       block_zero_probe cost [N]\n");
	return 2;
}
