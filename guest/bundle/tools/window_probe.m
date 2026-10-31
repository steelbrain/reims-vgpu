// window_probe: what of the reims-graphics device region each process maps, on the kext's own
// selectors (guest/kext/src/RgfxAccelerator.hpp), with no Metal.
//   window_probe check     the isolation of the windows, across processes:
//       1. the whole region does not map, the device info comes from its selector, and the
//          window holds nothing of the region's head (the rings, the device page);
//       2. a block this process writes reads zeros in another process's window at its offset,
//          where a write faults, and the other may not free it;
//       3. once this process frees it, the block reads zeros here too, and a write faults;
//       4. an IOSurface plane's shared block holds what one holder wrote for another holder,
//          and reads zeros for a process that does not hold the surface, which gets a block of
//          its own for pages of its own; once this process lets go, it reads zeros here.
//     Prints "WINDOW OK" when every step holds, else "WINDOW FAIL: why".
//   window_probe bench [N]  N (2000 by default) allocs and frees of a 64 KiB block, one after
//     the other: the microseconds a pair takes from this process, the kext's window work
//     included (which the kext logs at the close).
// The other processes are this binary again, started with "peer-*" arguments.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework IOKit -framework IOSurface
//        -framework Foundation window_probe.m -o window_probe
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <setjmp.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
// The guests' SDK still has global surfaces, which the macOS 27 SDK deprecates.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern char **environ;

enum {
	kUserClientType = 0x52474658,
	kMemoryRegion = 0,
	kOpenContext = 0x100,
	kAlloc = 0x101,
	kFree = 0x102,
	kSurfaceBlock = 0x109,
	kSurfaceBlockRelease = 0x10a,
	kDeviceInfo = 0x10b,
};
static const uint64_t kHead = 0x400000;       // RGFX_ALLOCATIONS_OFFSET
static const uint64_t kDeviceInfoAt = 0x18100; // RGFX_DEVICE_INFO_OFFSET
static const size_t kDeviceInfoSize = 11640;   // sizeof(RgfxDeviceInfo)
static const uint64_t kBlock = 0x10000;

struct Conn {
	io_connect_t conn;
	volatile uint8_t *base;
	uint64_t size;
};

static int openConn(struct Conn *c)
{
	io_service_t svc =
	    IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("RgfxAccelerator"));
	if (!svc)
		return -1;
	kern_return_t kr = IOServiceOpen(svc, mach_task_self(), kUserClientType, &c->conn);
	IOObjectRelease(svc);
	if (kr != KERN_SUCCESS)
		return -1;
	uint64_t out[5] = {};
	uint32_t n = 5;
	kr = IOConnectCallScalarMethod(c->conn, kOpenContext, NULL, 0, out, &n);
	if (kr != KERN_SUCCESS) {
		// A kext with no window answers three.
		n = 3;
		kr = IOConnectCallScalarMethod(c->conn, kOpenContext, NULL, 0, out, &n);
		if (kr != KERN_SUCCESS)
			return -1;
		c->base = NULL;
		c->size = 0;
		return 0;
	}
	c->base = (volatile uint8_t *)(uintptr_t)out[3];
	c->size = out[4];
	return 0;
}

static int alloc(struct Conn *c, uint64_t size, uint64_t *offset)
{
	uint64_t in[2] = {size, 0x1000};
	uint32_t n = 1;
	return IOConnectCallScalarMethod(c->conn, kAlloc, in, 2, offset, &n) == KERN_SUCCESS ? 0 : -1;
}

static kern_return_t freeBlock(struct Conn *c, uint64_t offset)
{
	return IOConnectCallScalarMethod(c->conn, kFree, &offset, 1, NULL, NULL);
}

// The bytes at `p` that are not `value`, of `n`.
static size_t notAll(volatile const uint8_t *p, size_t n, uint8_t value)
{
	size_t bad = 0;
	for (size_t i = 0; i < n; i++)
		bad += p[i] != value;
	return bad;
}

static sigjmp_buf gFault;
static void onFault(int sig)
{
	(void)sig;
	siglongjmp(gFault, 1);
}

// Whether a write of one byte at `p` faults.
static int writeFaults(volatile uint8_t *p)
{
	struct sigaction sa = {}, oldSegv, oldBus;
	sa.sa_handler = onFault;
	sigaction(SIGSEGV, &sa, &oldSegv);
	sigaction(SIGBUS, &sa, &oldBus);
	int faulted = 0;
	if (sigsetjmp(gFault, 1) == 0)
		*p = 0x5a;
	else
		faulted = 1;
	sigaction(SIGSEGV, &oldSegv, NULL);
	sigaction(SIGBUS, &oldBus, NULL);
	return faulted;
}

static int spawnWait(char *const argv[])
{
	pid_t pid;
	if (posix_spawn(&pid, argv[0], NULL, NULL, argv, environ) != 0)
		return -1;
	int status = 0;
	if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status))
		return -1;
	return WEXITSTATUS(status);
}

// peer-block OFFSET: another process's block at OFFSET reads zeros here, a write faults, and
// this process may not free it.
static int peerBlock(uint64_t offset)
{
	struct Conn c;
	if (openConn(&c) != 0 || !c.base) {
		printf("  peer: no context\n");
		return 1;
	}
	size_t bad = notAll(c.base + offset, kBlock, 0);
	int faulted = writeFaults(c.base + offset);
	kern_return_t freed = freeBlock(&c, offset);
	printf("  peer: the other's block reads %zu nonzero bytes here, a write %s, its free 0x%x\n",
	       bad, faulted ? "faults" : "lands", freed);
	return bad == 0 && faulted && freed != KERN_SUCCESS ? 0 : 1;
}

// The shared block of `s`'s plane 0, through `c`: its offset, and whether this ask made it.
static int surfaceBlock(struct Conn *c, const void *plane, uint64_t length, uint64_t *offset,
                        int *fresh)
{
	uint64_t size = (length + 16 + 0xfff) & ~0xfffull;
	uint64_t in[3] = {(uintptr_t)plane, length, size};
	uint64_t out[4] = {};
	uint32_t n = 4;
	kern_return_t kr = IOConnectCallScalarMethod(c->conn, kSurfaceBlock, in, 3, out, &n);
	if (kr != KERN_SUCCESS)
		return -1;
	*offset = out[0];
	*fresh = (int)out[3];
	return 0;
}

// peer-holder SID OFFSET PATTERN: a holder of surface SID gets the block at OFFSET, which holds
// PATTERN.
static int peerHolder(uint32_t sid, uint64_t offset, uint8_t pattern)
{
	struct Conn c;
	IOSurfaceRef s = IOSurfaceLookup(sid);
	if (openConn(&c) != 0 || !c.base || !s) {
		printf("  holder: no context or no surface\n");
		return 1;
	}
	uint64_t length = IOSurfaceGetBytesPerRow(s) * IOSurfaceGetHeight(s);
	uint64_t at = 0;
	int fresh = 0;
	if (surfaceBlock(&c, IOSurfaceGetBaseAddress(s), length, &at, &fresh) != 0) {
		printf("  holder: no block\n");
		return 1;
	}
	size_t bad = notAll(c.base + at, length, pattern);
	printf("  holder: block 0x%llx (%s), %zu bytes not the writer's\n", (unsigned long long)at,
	       fresh ? "made" : "the writer's", bad);
	return at == offset && !fresh && bad == 0 ? 0 : 1;
}

// peer-other OFFSET LENGTH: a process that does not hold the surface reads zeros at its block,
// and pages of its own get a block of their own.
static int peerOther(uint64_t offset, uint64_t length)
{
	struct Conn c;
	if (openConn(&c) != 0 || !c.base) {
		printf("  other: no context\n");
		return 1;
	}
	size_t bad = notAll(c.base + offset, length, 0);
	void *mine = NULL;
	if (posix_memalign(&mine, 0x1000, length) != 0)
		return 1;
	memset(mine, 0x11, length);
	uint64_t at = 0;
	int fresh = 0;
	int asked = surfaceBlock(&c, mine, length, &at, &fresh);
	printf("  other: the surface's block reads %zu nonzero bytes here; its own pages get block "
	       "0x%llx (%s)\n",
	       bad, (unsigned long long)at, asked ? "none" : fresh ? "made" : "another's");
	return bad == 0 && asked == 0 && at != offset && fresh ? 0 : 1;
}

static int check(const char *self)
{
	struct Conn c;
	if (openConn(&c) != 0) {
		printf("WINDOW FAIL: no context\n");
		return 1;
	}
	if (!c.base) {
		printf("WINDOW FAIL: the kext gives no window\n");
		return 1;
	}
	// 1. The region does not map whole; the device info comes from its selector; the window
	// holds nothing of the head.
	mach_vm_address_t addr = 0;
	mach_vm_size_t size = 0;
	kern_return_t mapped =
	    IOConnectMapMemory64(c.conn, kMemoryRegion, mach_task_self(), &addr, &size, kIOMapAnywhere);
	uint8_t *info = calloc(1, kDeviceInfoSize);
	size_t infoSize = kDeviceInfoSize;
	kern_return_t infoKr =
	    IOConnectCallStructMethod(c.conn, kDeviceInfo, NULL, 0, info, &infoSize);
	size_t infoBytes = notAll(info, kDeviceInfoSize, 0);
	size_t headBytes = notAll(c.base, kHead, 0);
	size_t infoHere = notAll(c.base + (kDeviceInfoAt & ~0xfffull), 0x2000, 0);
	printf("1 window 0x%llx, 0x%llx bytes; the region's map 0x%x; device info 0x%x, %zu bytes, "
	       "%zu nonzero; the head reads %zu nonzero bytes here (the device page's %zu)\n",
	       (unsigned long long)(uintptr_t)c.base, (unsigned long long)c.size, mapped, infoKr,
	       infoSize, infoBytes, headBytes, infoHere);
	if (mapped == KERN_SUCCESS || infoKr != KERN_SUCCESS || infoSize != kDeviceInfoSize ||
	    infoBytes == 0 || headBytes != 0) {
		printf("WINDOW FAIL: 1\n");
		return 1;
	}
	// 2. A block of ours is no one else's to see.
	uint64_t off = 0;
	if (alloc(&c, kBlock, &off) != 0) {
		printf("WINDOW FAIL: 2 no block\n");
		return 1;
	}
	memset((void *)(c.base + off), 0xa5, kBlock);
	size_t mine = notAll(c.base + off, kBlock, 0xa5);
	printf("2 block 0x%llx written here (%zu bytes not)\n", (unsigned long long)off, mine);
	char offArg[32];
	snprintf(offArg, sizeof(offArg), "%llu", (unsigned long long)off);
	char *peer[] = {(char *)self, "peer-block", offArg, NULL};
	if (mine != 0 || spawnWait(peer) != 0) {
		printf("WINDOW FAIL: 2\n");
		return 1;
	}
	// 3. Freed, it is not ours to see either.
	kern_return_t freed = freeBlock(&c, off);
	size_t after = notAll(c.base + off, kBlock, 0);
	int faulted = writeFaults(c.base + off);
	printf("3 freed (0x%x): it reads %zu nonzero bytes here, a write %s\n", freed, after,
	       faulted ? "faults" : "lands");
	if (freed != KERN_SUCCESS || after != 0 || !faulted) {
		printf("WINDOW FAIL: 3\n");
		return 1;
	}
	// 4. A surface plane's block is its holders' alone.
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @256,
		(id)kIOSurfaceHeight : @64,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
		(id)kIOSurfaceIsGlobal : @YES,
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	if (!s) {
		printf("WINDOW FAIL: 4 no surface\n");
		return 1;
	}
	uint64_t length = IOSurfaceGetBytesPerRow(s) * IOSurfaceGetHeight(s);
	uint64_t sblock = 0;
	int fresh = 0;
	if (surfaceBlock(&c, IOSurfaceGetBaseAddress(s), length, &sblock, &fresh) != 0 || !fresh) {
		printf("WINDOW FAIL: 4 no block for the plane\n");
		return 1;
	}
	memset((void *)(c.base + sblock), 0x3c, length);
	printf("4 surface %u's plane: block 0x%llx, %llu bytes written here\n", IOSurfaceGetID(s),
	       (unsigned long long)sblock, (unsigned long long)length);
	char sidArg[32], sbArg[32], lenArg[32];
	snprintf(sidArg, sizeof(sidArg), "%u", IOSurfaceGetID(s));
	snprintf(sbArg, sizeof(sbArg), "%llu", (unsigned long long)sblock);
	snprintf(lenArg, sizeof(lenArg), "%llu", (unsigned long long)length);
	char *holder[] = {(char *)self, "peer-holder", sidArg, sbArg, "60", NULL};
	char *other[] = {(char *)self, "peer-other", sbArg, lenArg, NULL};
	if (spawnWait(holder) != 0 || spawnWait(other) != 0) {
		printf("WINDOW FAIL: 4\n");
		return 1;
	}
	kern_return_t let = IOConnectCallScalarMethod(c.conn, kSurfaceBlockRelease, &sblock, 1, NULL,
	                                              NULL);
	size_t gone = notAll(c.base + sblock, length, 0);
	printf("4 let go (0x%x): the plane's block reads %zu nonzero bytes here\n", let, gone);
	if (let != KERN_SUCCESS || gone != 0) {
		printf("WINDOW FAIL: 4 after the let-go\n");
		return 1;
	}
	CFRelease(s);
	IOServiceClose(c.conn);
	printf("WINDOW OK\n");
	return 0;
}

static int bench(long n)
{
	struct Conn c;
	if (openConn(&c) != 0) {
		printf("bench: no context\n");
		return 1;
	}
	mach_timebase_info_data_t tb;
	mach_timebase_info(&tb);
	long failed = 0;
	uint64_t inAlloc = 0, inFree = 0;
	uint64_t start = mach_absolute_time();
	for (long i = 0; i < n; i++) {
		uint64_t off = 0;
		uint64_t t0 = mach_absolute_time();
		int a = alloc(&c, kBlock, &off);
		uint64_t t1 = mach_absolute_time();
		kern_return_t f = a == 0 ? freeBlock(&c, off) : KERN_FAILURE;
		uint64_t t2 = mach_absolute_time();
		inAlloc += t1 - t0;
		inFree += t2 - t1;
		if (a != 0 || f != KERN_SUCCESS)
			failed++;
	}
	double us = (double)(mach_absolute_time() - start) * tb.numer / tb.denom / 1000.0;
	double allocUs = (double)inAlloc * tb.numer / tb.denom / 1000.0 / (double)n;
	double freeUs = (double)inFree * tb.numer / tb.denom / 1000.0 / (double)n;
	printf("bench: %ld allocs and frees of 64 KiB in %.0f us, %.2f us a pair (an alloc %.2f, a "
	       "free %.2f; %s), failed=%ld\n",
	       n, us, us / (double)n, allocUs, freeUs, c.base ? "with a window" : "no window", failed);
	IOServiceClose(c.conn);
	return failed != 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		setvbuf(stdout, NULL, _IONBF, 0);
		if (argc >= 3 && strcmp(argv[1], "peer-block") == 0)
			return peerBlock(strtoull(argv[2], NULL, 0));
		if (argc >= 5 && strcmp(argv[1], "peer-holder") == 0)
			return peerHolder((uint32_t)strtoul(argv[2], NULL, 0), strtoull(argv[3], NULL, 0),
			                  (uint8_t)strtoul(argv[4], NULL, 0));
		if (argc >= 4 && strcmp(argv[1], "peer-other") == 0)
			return peerOther(strtoull(argv[2], NULL, 0), strtoull(argv[3], NULL, 0));
		if (argc >= 2 && strcmp(argv[1], "bench") == 0)
			return bench(argc >= 3 ? strtol(argv[2], NULL, 0) : 2000);
		return check(argv[0]);
	}
}
