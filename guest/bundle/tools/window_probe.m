// window_probe: what of the reims-graphics device region each process maps, on the kext's own
// selectors (guest/kext/src/RgfxUserClientABI.hpp), with no Metal.
//   window_probe check     the isolation of the windows, across processes:
//       1. the whole region does not map, the device info comes from its selector, and the
//          window holds nothing of the region's head (the rings, the device page);
//       2. a block this process writes reads zeros in another process's window at its offset,
//          where a write faults, and the other may not free it;
//       3. once this process frees it, the block reads zeros here too, and a write faults;
//       4. an IOSurface plane's shared block holds what one holder wrote for another holder,
//          and reads zeros for a process that does not hold the surface, which gets a block of
//          its own for pages of its own; once this process lets go, it reads zeros here.
//       5. the display's pixels are the scanout's owner's alone (the WindowServer that holds the
//          framebuffer's server connection): another process's lock of a surface over the
//          scanout is refused and maps nothing, the scanout's grant to its context is refused
//          and its window reads zeros there, where a write faults, and the framebuffer's VRAM
//          range does not map on the framebuffer's shared connection.
//     Prints "WINDOW OK" when every step holds, else "WINDOW FAIL: why".
//   window_probe scanout   step 5 alone, in a process of its own.
//   window_probe former-owner   as root, with no WindowServer holding the framebuffer's server
//     connection: this process owns the scanout through that connection, locks a surface over
//     it and is granted it, closes the connection, and once another process's connection makes
//     that process the owner, its lock mapping faults and its window no longer holds the
//     scanout's span. Prints "FORMER OK", "FORMER FAIL: why" or "FORMER SETUP: why".
//   window_probe bench [N]  N (2000 by default) allocs and frees of a 64 KiB block, one after
//     the other: the microseconds a pair takes from this process, the kext's window work
//     included (which the kext logs at the close).
// The other processes are this binary again, started with "peer-*" arguments.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework IOKit -framework IOSurface
//        -framework Foundation window_probe.m -o window_probe
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/graphics/IOAccelSurfaceConnect.h>
#import <IOKit/graphics/IOGraphicsTypes.h>
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

#include "../../kext/src/RgfxUserClientABI.hpp"
// The guests' SDK still has global surfaces, which the macOS 27 SDK deprecates.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern char **environ;

// The whole region's memory type (the kext's kRgfxAccelMemoryRegion), which it refuses.
enum {
	kMemoryRegion = 0,
};
static const uint64_t kHead = 0x400000;       // RGFX_ALLOCATIONS_OFFSET
static const uint64_t kDeviceInfoAt = 0x18100; // RGFX_DEVICE_INFO_OFFSET
#define RGFX_PROBE_SCANOUT_SPAN 0x8000000ull     // RGFX_SCANOUT_SPAN
static const size_t kDeviceInfoSize = 11672;   // sizeof(RgfxDeviceInfo)
static const uint64_t kBlock = 0x10000;
// The accelerator's IOAccel surface client (kIOAccelSurfaceClientType), and IOGraphics' memory
// type for a framebuffer's VRAM range on its user clients (kIOFBVRAMMemory, in IOGraphics' open
// source, IOGraphicsTypesPrivate.h).
static const uint32_t kSurfaceClientType = 0;
static const uint32_t kFramebufferVram = 110;

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
	kern_return_t kr = IOServiceOpen(svc, mach_task_self(), kRgfxAccelUserClientType, &c->conn);
	IOObjectRelease(svc);
	if (kr != KERN_SUCCESS)
		return -1;
	uint64_t out[5] = {};
	uint32_t n = 5;
	kr = IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodOpenContext, NULL, 0, out, &n);
	if (kr != KERN_SUCCESS) {
		// A kext with no window answers three.
		n = 3;
		kr = IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodOpenContext, NULL, 0, out, &n);
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
	return IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodAlloc, in, 2, offset, &n) == KERN_SUCCESS ? 0 : -1;
}

static kern_return_t freeBlock(struct Conn *c, uint64_t offset)
{
	return IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodFree, &offset, 1, NULL, NULL);
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

// Whether a read of one byte at `p` faults.
static int readFaults(volatile const uint8_t *p)
{
	struct sigaction sa = {}, oldSegv, oldBus;
	sa.sa_handler = onFault;
	sigaction(SIGSEGV, &sa, &oldSegv);
	sigaction(SIGBUS, &sa, &oldBus);
	int faulted = 0;
	if (sigsetjmp(gFault, 1) == 0)
		(void)*p;
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
	kern_return_t kr = IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodSurfaceBlock, in, 3, out, &n);
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

// A surface of the accelerator's, with no backing, on framebuffer 0: `sid`, BGRA.
static io_connect_t openSurface(uint32_t sid)
{
	io_service_t svc =
	    IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("RgfxAccelerator"));
	if (!svc)
		return IO_OBJECT_NULL;
	io_connect_t s = IO_OBJECT_NULL;
	kern_return_t kr = IOServiceOpen(svc, mach_task_self(), kSurfaceClientType, &s);
	IOObjectRelease(svc);
	if (kr != KERN_SUCCESS)
		return IO_OBJECT_NULL;
	uint64_t idMode[2] = {sid, kIOAccelSurfaceModeColorDepthBGRA32};
	if (IOConnectCallScalarMethod(s, kIOAccelSurfaceSetIDMode, idMode, 2, NULL, NULL) !=
	    KERN_SUCCESS) {
		IOServiceClose(s);
		return IO_OBJECT_NULL;
	}
	return s;
}

// Shapes surface `s` to `w` x `h` at the display's origin and write-locks it: the lock's answer,
// its information in `*info`. The lock is let go at once.
static kern_return_t lockShape(io_connect_t s, int w, int h, IOAccelSurfaceInformation *info)
{
	IOAccelDeviceRegion region = {};
	region.num_rects = 0;
	region.bounds = (IOAccelBounds){0, 0, (SInt16)w, (SInt16)h};
	uint64_t shape[2] = {0, 0}; // options, framebuffer 0
	kern_return_t kr = IOConnectCallMethod(s, kIOAccelSurfaceSetShape, shape, 2, &region,
	                                       sizeof(region), NULL, NULL, NULL, NULL);
	if (kr != KERN_SUCCESS)
		return kr;
	memset(info, 0, sizeof(*info));
	size_t infoSize = sizeof(*info);
	uint64_t options = 0;
	kr = IOConnectCallMethod(s, kIOAccelSurfaceWriteLockOptions, &options, 1, NULL, 0, NULL, NULL,
	                         info, &infoSize);
	if (kr == KERN_SUCCESS)
		IOConnectCallScalarMethod(s, kIOAccelSurfaceWriteUnlockOptions, &options, 1, NULL, NULL);
	return kr;
}

// The largest `n` in [1, limit] for which `fits(n)` holds, `fits` holding up to it and not past.
static int largest(io_connect_t s, int limit, int width)
{
	int lo = 1, hi = limit;
	while (lo < hi) {
		int mid = lo + (hi - lo + 1) / 2;
		IOAccelSurfaceInformation info;
		kern_return_t kr = width ? lockShape(s, mid, 1, &info) : lockShape(s, 1, mid, &info);
		if (kr == KERN_SUCCESS)
			lo = mid;
		else
			hi = mid - 1;
	}
	return lo;
}

// peer-scanout: this process does not hold the framebuffer's server connection, so it reaches
// none of the display's pixels: its lock of a surface over the scanout is refused, its context
// is not granted the scanout and its window reads zeros there, and the framebuffer's VRAM range
// does not map for it on the shared connection.
static int peerScanout(void)
{
	struct Conn c;
	if (openConn(&c) != 0 || !c.base) {
		printf("  scanout: no context\n");
		return 1;
	}
	uint32_t sid = 0x52000000u | ((uint32_t)getpid() & 0xffffffu);
	io_connect_t s = openSurface(sid);
	if (s == IO_OBJECT_NULL) {
		printf("  scanout: no surface\n");
		return 1;
	}
	int failed = 0;
	// a. The lock of a corner of the display.
	IOAccelSurfaceInformation info;
	kern_return_t locked = lockShape(s, 16, 16, &info);
	printf("  scanout: a lock of 16x16 at 0,0: 0x%x, address 0x%llx\n", locked,
	       (unsigned long long)info.address[0]);
	int w = 16, h = 16;
	if (locked == KERN_SUCCESS) {
		// The display's pixels reach this process: how much of them, and the display's shape
		// for the grant below.
		volatile const uint8_t *px = (volatile const uint8_t *)(uintptr_t)info.address[0];
		printf("  scanout: the lock maps the display here (%zu nonzero bytes in its first row)\n",
		       px ? notAll(px, 64, 0) : 0);
		w = largest(s, 16384, 1);
		h = largest(s, 16384, 0);
		printf("  scanout: the display is %dx%d\n", w, h);
		failed = 1;
	}
	// b. The scanout's grant to this process's context, for the surface shaped as the whole
	// display (as far as this process could tell its shape).
	lockShape(s, w, h, &info);
	uint64_t in[2] = {sid, 0};
	uint64_t out[5] = {};
	uint32_t n = 5;
	kern_return_t granted = IOConnectCallScalarMethod(c.conn, kRgfxAccelMethodSurfaceScanout, in, 2, out, &n);
	uint64_t at = granted == KERN_SUCCESS ? out[0] : kHead;
	uint64_t length = granted == KERN_SUCCESS && out[1] ? out[1] : 0x10000;
	if (at >= c.size || length > c.size - at)
		at = kHead, length = 0x10000;
	size_t bad = notAll(c.base + at, length, 0);
	int faulted = writeFaults(c.base + at);
	printf("  scanout: the grant 0x%x (span 0x%llx, 0x%llx bytes); the window reads %zu nonzero "
	       "bytes there, a write %s\n",
	       granted, (unsigned long long)at, (unsigned long long)length, bad,
	       faulted ? "faults" : "lands");
	if (granted == KERN_SUCCESS || bad != 0 || !faulted)
		failed = 1;
	IOServiceClose(s);
	// c. The framebuffer's VRAM range, on its shared connection, which IOGraphics opens for any
	// task while WindowServer holds the server connection.
	io_service_t fb =
	    IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("ReimsGraphics"));
	io_connect_t shared = IO_OBJECT_NULL;
	kern_return_t opened =
	    fb ? IOServiceOpen(fb, mach_task_self(), kIOFBSharedConnectType, &shared) : KERN_FAILURE;
	if (fb)
		IOObjectRelease(fb);
	mach_vm_address_t addr = 0;
	mach_vm_size_t size = 0;
	kern_return_t vram = opened == KERN_SUCCESS
	                         ? IOConnectMapMemory64(shared, kFramebufferVram, mach_task_self(),
	                                                &addr, &size, kIOMapAnywhere)
	                         : KERN_FAILURE;
	if (vram == KERN_SUCCESS) {
		// What the range reaches: the scanout at its start, and past it the accelerator's
		// blocks, every process's.
		size_t onScreen = notAll((volatile const uint8_t *)(uintptr_t)addr, 0x100000, 0);
		size_t past = size > RGFX_PROBE_SCANOUT_SPAN
		                  ? notAll((volatile const uint8_t *)(uintptr_t)addr +
		                               RGFX_PROBE_SCANOUT_SPAN,
		                           size - RGFX_PROBE_SCANOUT_SPAN, 0)
		                  : 0;
		printf("  scanout: the VRAM range maps here: 0x%llx bytes, %zu nonzero bytes in its "
		       "first MiB, %zu past the scanout's span\n",
		       (unsigned long long)size, onScreen, past);
		failed = 1;
	} else {
		printf("  scanout: the shared connection 0x%x, its VRAM range 0x%x (not mapped)\n",
		       opened, vram);
	}
	if (shared != IO_OBJECT_NULL)
		IOServiceClose(shared);
	IOServiceClose(c.conn);
	return failed;
}

// The framebuffer's server connection (kIOFBServerConnectType), which makes this task the
// scanout's owner; IO_OBJECT_NULL where it is not made (WindowServer holds it), its answer in
// `*kr`.
static io_connect_t openServer(kern_return_t *kr)
{
	io_service_t fb =
	    IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("ReimsGraphics"));
	io_connect_t server = IO_OBJECT_NULL;
	*kr = fb ? IOServiceOpen(fb, mach_task_self(), kIOFBServerConnectType, &server)
	         : KERN_FAILURE;
	if (fb)
		IOObjectRelease(fb);
	return *kr == KERN_SUCCESS ? server : IO_OBJECT_NULL;
}

// peer-owner: opens the framebuffer's server connection, so the scanout's owner moves to this
// task, and closes it.
static int peerOwner(void)
{
	kern_return_t kr;
	io_connect_t server = openServer(&kr);
	printf("  former: the next owner's server connection: 0x%x\n", kr);
	if (server == IO_OBJECT_NULL)
		return 1;
	usleep(200000);
	IOServiceClose(server);
	return 0;
}

// former-owner: with no WindowServer holding the framebuffer's server connection (stopped, and
// this run as root), this task takes the connection, locks a surface over the scanout and is
// granted the scanout, then closes the connection: its holds stay while no task owns the
// scanout. Once another task's connection makes that task the owner, this task's lock mapping
// faults and its window reads zeros at the scanout's span, where a write faults.
static int formerOwner(const char *self)
{
	kern_return_t kr;
	io_connect_t server = openServer(&kr);
	if (server == IO_OBJECT_NULL) {
		printf("FORMER SETUP: the server connection 0x%x (does WindowServer hold it?)\n", kr);
		return 2;
	}
	struct Conn c;
	uint32_t sid = 0x46000000u | ((uint32_t)getpid() & 0xffffffu);
	io_connect_t s = IO_OBJECT_NULL;
	if (openConn(&c) != 0 || !c.base || (s = openSurface(sid)) == IO_OBJECT_NULL) {
		printf("FORMER SETUP: no context or surface\n");
		IOServiceClose(server);
		return 2;
	}
	IOAccelSurfaceInformation info;
	kern_return_t locked = lockShape(s, 16, 16, &info);
	int w = locked == KERN_SUCCESS ? largest(s, 16384, 1) : 0;
	int h = locked == KERN_SUCCESS ? largest(s, 16384, 0) : 0;
	locked = lockShape(s, w ? w : 16, h ? h : 16, &info);
	volatile const uint8_t *px = (volatile const uint8_t *)(uintptr_t)info.address[0];
	uint64_t in[2] = {sid, 0};
	uint64_t out[5] = {};
	uint32_t n = 5;
	kern_return_t granted = IOConnectCallScalarMethod(c.conn, kRgfxAccelMethodSurfaceScanout, in, 2, out, &n);
	uint64_t at = out[0], length = out[1];
	printf("  former: as the owner: the lock of %dx%d 0x%x at 0x%llx, the grant 0x%x (span "
	       "0x%llx, 0x%llx bytes)\n",
	       w, h, locked, (unsigned long long)info.address[0], granted, (unsigned long long)at,
	       (unsigned long long)length);
	if (locked != KERN_SUCCESS || !px || granted != KERN_SUCCESS || at >= c.size ||
	    length > c.size - at) {
		printf("FORMER SETUP: the owner was not given the scanout\n");
		IOServiceClose(server);
		return 2;
	}
	IOServiceClose(server);
	usleep(200000);
	int before = readFaults(px);
	size_t windowBefore = notAll(c.base + at, 4096, 0);
	printf("  former: the connection closed, no owner: the lock mapping %s, the window holds %zu "
	       "nonzero bytes of the span's first page\n",
	       before ? "faults" : "reads", windowBefore);
	char *next[] = {(char *)self, "peer-owner", NULL};
	if (spawnWait(next) != 0) {
		printf("FORMER SETUP: the next owner's connection was not made\n");
		return 2;
	}
	int after = readFaults(px);
	size_t windowAfter = notAll(c.base + at, length, 0);
	int writeFault = writeFaults(c.base + at);
	printf("  former: after the owner moved: the lock mapping %s, the window holds %zu nonzero "
	       "bytes of the span, a write there %s\n",
	       after ? "faults" : "reads", windowAfter, writeFault ? "faults" : "lands");
	IOServiceClose(s);
	IOServiceClose(c.conn);
	if (!after || windowAfter != 0 || !writeFault) {
		printf("FORMER FAIL: the former owner keeps the scanout\n");
		return 1;
	}
	printf("FORMER OK\n");
	return 0;
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
	    IOConnectCallStructMethod(c.conn, kRgfxAccelMethodDeviceInfo, NULL, 0, info, &infoSize);
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
	kern_return_t let = IOConnectCallScalarMethod(c.conn, kRgfxAccelMethodSurfaceBlockRelease, &sblock, 1, NULL,
	                                              NULL);
	size_t gone = notAll(c.base + sblock, length, 0);
	printf("4 let go (0x%x): the plane's block reads %zu nonzero bytes here\n", let, gone);
	if (let != KERN_SUCCESS || gone != 0) {
		printf("WINDOW FAIL: 4 after the let-go\n");
		return 1;
	}
	CFRelease(s);
	IOServiceClose(c.conn);
	// 5. The display's pixels are not this process's to reach, nor another's.
	char *scanout[] = {(char *)self, "peer-scanout", NULL};
	if (spawnWait(scanout) != 0) {
		printf("WINDOW FAIL: 5\n");
		return 1;
	}
	printf("5 the scanout reaches no process but its owner\n");
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
		if (argc >= 2 && strcmp(argv[1], "peer-scanout") == 0)
			return peerScanout();
		if (argc >= 2 && strcmp(argv[1], "scanout") == 0) {
			char *scanout[] = {argv[0], "peer-scanout", NULL};
			int r = spawnWait(scanout);
			printf(r == 0 ? "SCANOUT OK\n" : "SCANOUT FAIL\n");
			return r != 0;
		}
		if (argc >= 2 && strcmp(argv[1], "peer-owner") == 0)
			return peerOwner();
		if (argc >= 2 && strcmp(argv[1], "former-owner") == 0)
			return formerOwner(argv[0]);
		if (argc >= 2 && strcmp(argv[1], "bench") == 0)
			return bench(argc >= 3 ? strtol(argv[2], NULL, 0) : 2000);
		return check(argv[0]);
	}
}
