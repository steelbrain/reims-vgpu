// shmem: how much CPU-visible Metal memory one process holds before Metal answers nil, what each
// make costs while it holds the rest, and what the memory costs the system (the process's
// footprint, the system's wired pages) at each step: made, used by the GPU, freed. To set the
// graphics lane's Shared and Managed memory against real Macs.
//   shmem [-d NAME] [-cap GIB] [-count N] [-time S] [-guard GIB] [-use] [-check] [-touch] CASE...
//     -d NAME    the first of MTLCopyAllDevices whose name holds NAME (default: the system's).
//     -cap GIB   the most bytes a hold case makes (default 8).
//     -count N   the most buffers a hold case makes (default 1,000,000).
//     -time S    the most seconds a hold case makes buffers for (default 20).
//     -guard GIB a hold case stops once the system's free, inactive, speculative and purgeable
//                pages sum to less than this (default 6), so a run never pushes a Mac to swap.
//     -use       after a hold, one command buffer fills 4 bytes of each held buffer (at most
//                65,536 of them, spread evenly), committed and waited: what a GPU use costs and
//                wires.
//     -check     after a hold (and its use), the CPU writes the first word of each held buffer, one
//                command buffer copies it to the second word and fills the last 4 bytes of each
//                (and synchronizes a Managed one), and the CPU reads both back: bad = the buffers
//                whose words are not what was written, and the command buffer's error.
//     -touch     the CPU writes one byte a page of each buffer as it is made.
//     -nofree    hold and align cases keep their buffers for the process's life, which its exit
//                frees through the context's close: no free per buffer.
//     -rounds N  the rounds of a zero case (default 200).
// The cases:
//   info                 the device's answers and the system's memory and wire limits.
//   hold.MODE.SIZE       buffers of SIZE (4k, 64k, 1m, 2m, 16m, 64m, 256m) and MODE (shared,
//                        managed, private) made and held until nil or a limit: n, bytes, stop
//                        (nil, cap, count, time, guard), make_us p50/p99/max and those of the
//                        first and the last tenth, then the footprint, wired and
//                        currentAllocatedSize deltas (MiB) after the makes, after the use, and
//                        after all are freed, and the free time.
//   max.MODE             one buffer of maxBufferLength, and one of a page more: ok or nil, us.
//   zero.MODE.SIZE       -rounds rounds of a make, a check that every 64th byte reads zero, a fill of
//                        0xa5, a free: dirty = the makes that read a byte not zero.
//   align.MODE.SIZE      64 buffers of SIZE held: allocatedSize, how many contents pointers and
//                        gpuAddresses are page aligned, the smallest gap between two contents
//                        and two gpuAddresses, and how many share a page with another.
//   superpage            (x86_64) 2 MiB superpages (VM_FLAGS_SUPERPAGE_SIZE_2MB) allocated
//                        until the kernel refuses or the cap: n, us p50/p99, wired delta, then
//                        freed. No Metal.
//   spmark.N             (x86_64) N superpages and N 2 MiB ranges of plain pages, each 4 KiB page
//                        marked (spmarkCase), held -time seconds for a host scan of guest RAM.
// One line a case: "shm: NAME key=value ...". Build (on a Mac):
//   xcrun clang -fobjc-arc -O2 -arch x86_64 -mmacosx-version-min=13.0 shmem.m -framework Metal \
//     -framework Foundation -o shmem-x86_64        (and -arch arm64 for Apple silicon)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

static double gCapGiB = 8, gTimeS = 20, gGuardGiB = 6;
static unsigned long gCount = 1000000;
static BOOL gUse = NO, gCheck = NO, gTouch = NO, gNoFree = NO;
static int gRounds = 200;

static double nowUs(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1000.0;
}

static uint64_t footprint(void)
{
	task_vm_info_data_t info;
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
		return 0;
	return info.phys_footprint;
}

static vm_statistics64_data_t vmStats(void)
{
	vm_statistics64_data_t s = {0};
	mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
	host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s, &count);
	return s;
}

static uint64_t pageBytes(void)
{
	vm_size_t p = 0;
	host_page_size(mach_host_self(), &p);
	return p;
}

static uint64_t wiredBytes(void)
{
	return (uint64_t)vmStats().wire_count * pageBytes();
}

static uint64_t availBytes(void)
{
	vm_statistics64_data_t s = vmStats();
	return ((uint64_t)s.free_count + s.inactive_count + s.speculative_count + s.purgeable_count) *
	       pageBytes();
}

static double mib(int64_t b) { return (double)b / (1 << 20); }

static int cmpD(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

// The q-th quantile of n samples (sorted in place).
static double quant(double *v, size_t n, double q)
{
	if (n == 0)
		return 0;
	qsort(v, n, sizeof *v, cmpD);
	size_t i = (size_t)(q * (n - 1) + 0.5);
	return v[i];
}

static uint64_t sizeOf(NSString *s)
{
	uint64_t v = strtoull(s.UTF8String, NULL, 10);
	if ([s hasSuffix:@"k"])
		return v << 10;
	if ([s hasSuffix:@"m"])
		return v << 20;
	if ([s hasSuffix:@"g"])
		return v << 30;
	return v;
}

static MTLResourceOptions modeOf(NSString *s, BOOL *ok)
{
	*ok = YES;
	if ([s isEqualToString:@"shared"])
		return MTLResourceStorageModeShared;
	if ([s isEqualToString:@"private"])
		return MTLResourceStorageModePrivate;
#if TARGET_OS_OSX
	if ([s isEqualToString:@"managed"])
		return MTLResourceStorageModeManaged;
#endif
	*ok = NO;
	return 0;
}

static uint64_t sysctlU64(const char *name)
{
	uint64_t v = 0;
	size_t len = sizeof v;
	if (sysctlbyname(name, &v, &len, NULL, 0) != 0)
		return 0;
	if (len == 4)
		return (uint32_t)v;
	return v;
}

static void info(id<MTLDevice> d)
{
	printf("shm: info name=\"%s\" unified=%d lowpower=%d removable=%d location=%lu "
	       "maxBufferLength=%llu (%.1f MiB) recommendedMaxWorkingSetSize=%llu (%.1f MiB) "
	       "currentAllocatedSize=%.1fMiB maxTransferRate=%llu hasUnifiedMemory=%d\n",
	       d.name.UTF8String, d.hasUnifiedMemory, d.lowPower, d.removable,
	       (unsigned long)d.location, (unsigned long long)d.maxBufferLength,
	       mib((int64_t)d.maxBufferLength), (unsigned long long)d.recommendedMaxWorkingSetSize,
	       mib((int64_t)d.recommendedMaxWorkingSetSize), mib((int64_t)d.currentAllocatedSize),
	       (unsigned long long)d.maxTransferRate, d.hasUnifiedMemory);
	printf("shm: info memsize=%.1fGiB page=%llu global_user_wire_limit=%.1fGiB "
	       "user_wire_limit=%.1fGiB wired=%.1fMiB avail=%.1fMiB footprint=%.1fMiB\n",
	       sysctlU64("hw.memsize") / 1073741824.0, (unsigned long long)pageBytes(),
	       sysctlU64("vm.global_user_wire_limit") / 1073741824.0,
	       sysctlU64("vm.user_wire_limit") / 1073741824.0, mib((int64_t)wiredBytes()),
	       mib((int64_t)availBytes()), mib((int64_t)footprint()));
	static const uint64_t sizes[] = {256, 4096, 65536, 1 << 20, 2 << 20, 64 << 20};
	for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
		MTLSizeAndAlign sa = [d heapBufferSizeAndAlignWithLength:sizes[i]
		                                                 options:MTLResourceStorageModePrivate];
		printf("shm: info heapBufferSizeAndAlign.private.%llu size=%llu align=%llu\n",
		       (unsigned long long)sizes[i], (unsigned long long)sa.size,
		       (unsigned long long)sa.align);
	}
}

// Fills 4 bytes of each of (at most 65,536 of) `held` in one command buffer, waited.
static double useAll(id<MTLDevice> d, NSArray<id<MTLBuffer>> *held, unsigned *used, NSString **err)
{
	*used = 0;
	if (held.count == 0)
		return 0;
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	NSUInteger step = held.count > 65536 ? (held.count + 65535) / 65536 : 1;
	for (NSUInteger i = 0; i < held.count; i += step) {
		[b fillBuffer:held[i] range:NSMakeRange(0, 4) value:0x5a];
		(*used)++;
	}
	[b endEncoding];
	double t0 = nowUs();
	[cb commit];
	[cb waitUntilCompleted];
	double t = nowUs() - t0;
	if (cb.status != MTLCommandBufferStatusCompleted)
		*err = [NSString stringWithFormat:@"status %ld %@", (long)cb.status,
		                                  cb.error.localizedDescription];
	return t;
}

// The CPU writes each held buffer's first word; one command buffer copies it to the second word
// and fills the last 4 bytes; the CPU reads both. Answers the buffers whose words differ.
static double checkAll(id<MTLDevice> d, NSArray<id<MTLBuffer>> *held, uint64_t size,
                       BOOL managed, unsigned *bad, NSString **err)
{
	*bad = 0;
	const uint64_t last = size / 4 - 1;
	for (NSUInteger i = 0; i < held.count; i++) {
		uint32_t *c = (uint32_t *)held[i].contents;
		c[0] = (uint32_t)i ^ 0xc0ffee00u;
		c[1] = 0;
		c[last] = 0;
		if (managed) {
			[held[i] didModifyRange:NSMakeRange(0, 8)];
			[held[i] didModifyRange:NSMakeRange(size - 4, 4)];
		}
	}
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	for (NSUInteger i = 0; i < held.count; i++) {
		[b copyFromBuffer:held[i] sourceOffset:0 toBuffer:held[i] destinationOffset:4 size:4];
		[b fillBuffer:held[i] range:NSMakeRange(size - 4, 4) value:0x5a];
#if TARGET_OS_OSX
		if (managed)
			[b synchronizeResource:held[i]];
#endif
	}
	[b endEncoding];
	double t0 = nowUs();
	[cb commit];
	[cb waitUntilCompleted];
	double t = nowUs() - t0;
	if (cb.status != MTLCommandBufferStatusCompleted)
		*err = [NSString stringWithFormat:@"status %ld %@", (long)cb.status,
		                                  cb.error.localizedDescription];
	for (NSUInteger i = 0; i < held.count; i++) {
		const uint32_t *c = (const uint32_t *)held[i].contents;
		*bad += c[1] != ((uint32_t)i ^ 0xc0ffee00u) || c[last] != 0x5a5a5a5au;
	}
	return t;
}

static void hold(id<MTLDevice> d, NSString *name, MTLResourceOptions opts, uint64_t size)
{
	const uint64_t cap = (uint64_t)(gCapGiB * 1073741824.0);
	const uint64_t guard = (uint64_t)(gGuardGiB * 1073741824.0);
	unsigned long maxN = gCount;
	if (cap / size < maxN)
		maxN = cap / size;
	double *lat = calloc(maxN + 1, sizeof *lat);
	NSMutableArray<id<MTLBuffer>> *held = [NSMutableArray arrayWithCapacity:maxN];
	const uint64_t f0 = footprint(), w0 = wiredBytes(), a0 = d.currentAllocatedSize;
	const uint64_t page = pageBytes();
	const char *stop = "cap";
	unsigned long n = 0, nilRetryOk = 0;
	const double start = nowUs();
	for (;;) {
		if (n >= maxN) {
			stop = n >= gCount ? "count" : "cap";
			break;
		}
		if (nowUs() - start > gTimeS * 1e6) {
			stop = "time";
			break;
		}
		if ((n & 63) == 0 && availBytes() < guard) {
			stop = "guard";
			break;
		}
		id<MTLBuffer> b;
		double t0, t1;
		@autoreleasepool {
			t0 = nowUs();
			b = [d newBufferWithLength:size options:opts];
			t1 = nowUs();
		}
		if (b == nil) {
			stop = "nil";
			for (int r = 0; r < 3; r++) {
				@autoreleasepool {
					id<MTLBuffer> again = [d newBufferWithLength:size options:opts];
					if (again != nil)
						nilRetryOk++;
				}
			}
			break;
		}
		lat[n++] = t1 - t0;
		if (gTouch && (opts & MTLResourceStorageModeMask) != MTLResourceStorageModePrivate) {
			volatile uint8_t *c = (volatile uint8_t *)b.contents;
			for (uint64_t o = 0; o < size; o += page)
				c[o] = 1;
		}
		[held addObject:b];
	}
	const double makeS = (nowUs() - start) / 1e6;
	const uint64_t f1 = footprint(), w1 = wiredBytes(), a1 = d.currentAllocatedSize;
	size_t tenth = n / 10 ? n / 10 : (n ? 1 : 0);
	double *head = calloc(tenth + 1, sizeof *head), *tail = calloc(tenth + 1, sizeof *tail);
	memcpy(head, lat, tenth * sizeof *lat);
	memcpy(tail, lat + (n - tenth), tenth * sizeof *lat);
	double p50 = quant(lat, n, 0.5), p99 = quant(lat, n, 0.99), mx = n ? lat[n - 1] : 0;
	double h50 = quant(head, tenth, 0.5), t50 = quant(tail, tenth, 0.5);
	printf("shm: %s n=%lu bytes=%.1fMiB stop=%s nil_retry_ok=%lu make_us=%.1f make_p99=%.1f "
	       "make_max=%.1f first10_us=%.1f last10_us=%.1f make_s=%.2f footprint=%+.1fMiB "
	       "wired=%+.1fMiB allocated=%+.1fMiB\n",
	       name.UTF8String, n, mib((int64_t)(n * size)), stop, nilRetryOk, p50, p99, mx, h50, t50,
	       makeS, mib((int64_t)(f1 - f0)), mib((int64_t)(w1 - w0)), mib((int64_t)(a1 - a0)));
	fflush(stdout);
	if (gUse && n) {
		unsigned used = 0;
		NSString *err = nil;
		double t = useAll(d, held, &used, &err);
		const uint64_t f2 = footprint(), w2 = wiredBytes(), a2 = d.currentAllocatedSize;
		printf("shm: %s.use buffers=%u cb_ms=%.1f footprint=%+.1fMiB wired=%+.1fMiB "
		       "allocated=%+.1fMiB%s%s\n",
		       name.UTF8String, used, t / 1000, mib((int64_t)(f2 - f0)), mib((int64_t)(w2 - w0)),
		       mib((int64_t)(a2 - a0)), err ? " err=" : "", err ? err.UTF8String : "");
		fflush(stdout);
		// A second after the use, idle: does the driver let the wiring go?
		usleep(1000000);
		printf("shm: %s.idle wired=%+.1fMiB footprint=%+.1fMiB\n", name.UTF8String,
		       mib((int64_t)(wiredBytes() - w0)), mib((int64_t)(footprint() - f0)));
	}
	if (gCheck && n && (opts & MTLResourceStorageModeMask) != MTLResourceStorageModePrivate) {
		unsigned bad = 0;
		NSString *err = nil;
		const BOOL managed = (opts & MTLResourceStorageModeMask) != MTLResourceStorageModeShared;
		double t = checkAll(d, held, size, managed, &bad, &err);
		printf("shm: %s.check buffers=%lu bad=%u cb_ms=%.1f wired=%+.1fMiB allocated=%+.1fMiB%s%s\n",
		       name.UTF8String, (unsigned long)held.count, bad, t / 1000,
		       mib((int64_t)(wiredBytes() - w0)), mib((int64_t)(d.currentAllocatedSize - a0)),
		       err ? " err=" : "", err ? err.UTF8String : "");
		fflush(stdout);
	}
	if (gNoFree) {
		// Kept for the process's life: its exit frees them through the context's close.
		CFBridgingRetain(held);
		printf("shm: %s.free skipped (-nofree)\n", name.UTF8String);
		free(lat);
		free(head);
		free(tail);
		return;
	}
	double tf = nowUs();
	[held removeAllObjects];
	tf = nowUs() - tf;
	usleep(200000);
	printf("shm: %s.free all_ms=%.1f per_us=%.2f footprint=%+.1fMiB wired=%+.1fMiB "
	       "allocated=%+.1fMiB\n",
	       name.UTF8String, tf / 1000, n ? tf / n : 0, mib((int64_t)(footprint() - f0)),
	       mib((int64_t)(wiredBytes() - w0)), mib((int64_t)(d.currentAllocatedSize - a0)));
	fflush(stdout);
	free(lat);
	free(head);
	free(tail);
}

static void maxCase(id<MTLDevice> d, NSString *name, MTLResourceOptions opts)
{
	const uint64_t page = pageBytes();
	uint64_t sizes[2] = {d.maxBufferLength, d.maxBufferLength + page};
	for (int i = 0; i < 2; i++) {
		@autoreleasepool {
			double t0 = nowUs();
			id<MTLBuffer> b = [d newBufferWithLength:sizes[i] options:opts];
			double t = nowUs() - t0;
			printf("shm: %s.%s length=%llu %s us=%.1f\n", name.UTF8String, i ? "over" : "at",
			       (unsigned long long)sizes[i], b ? "ok" : "nil", t);
			fflush(stdout);
		}
	}
}

static void zeroCase(id<MTLDevice> d, NSString *name, MTLResourceOptions opts, uint64_t size)
{
	unsigned dirty = 0, made = 0;
	for (int r = 0; r < gRounds; r++) {
		@autoreleasepool {
			id<MTLBuffer> b = [d newBufferWithLength:size options:opts];
			if (b == nil)
				continue;
			made++;
			volatile uint8_t *c = (volatile uint8_t *)b.contents;
			for (uint64_t o = 0; o < size; o += 64)
				if (c[o] != 0) {
					dirty++;
					break;
				}
			memset((void *)c, 0xa5, size);
		}
	}
	printf("shm: %s made=%u dirty=%u\n", name.UTF8String, made, dirty);
}

static int cmpU(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return x < y ? -1 : x > y;
}

static void alignCase(id<MTLDevice> d, NSString *name, MTLResourceOptions opts, uint64_t size)
{
	enum { N = 64 };
	NSMutableArray *held = [NSMutableArray array];
	uint64_t cp[N], gp[N];
	unsigned n = 0, cAligned = 0, gAligned = 0;
	uint64_t allocated = 0;
	const uint64_t page = 4096;
	for (unsigned i = 0; i < N; i++) {
		id<MTLBuffer> b = [d newBufferWithLength:size options:opts];
		if (b == nil)
			break;
		[held addObject:b];
		cp[n] = (uint64_t)b.contents;
		gp[n] = b.gpuAddress;
		allocated = b.allocatedSize;
		cAligned += (cp[n] & (page - 1)) == 0;
		gAligned += (gp[n] & (page - 1)) == 0;
		n++;
	}
	qsort(cp, n, sizeof *cp, cmpU);
	qsort(gp, n, sizeof *gp, cmpU);
	uint64_t cGap = UINT64_MAX, gGap = UINT64_MAX;
	unsigned sharePage = 0;
	for (unsigned i = 1; i < n; i++) {
		if (cp[i] - cp[i - 1] < cGap)
			cGap = cp[i] - cp[i - 1];
		if (gp[i] - gp[i - 1] < gGap)
			gGap = gp[i] - gp[i - 1];
		if (cp[i] / page == (cp[i - 1] + size - 1) / page)
			sharePage++;
	}
	if (gNoFree)
		CFBridgingRetain(held);
	printf("shm: %s n=%u allocatedSize=%llu contents_page_aligned=%u gpu_page_aligned=%u "
	       "min_contents_gap=%llu min_gpu_gap=%llu share_page=%u contents_low=0x%llx "
	       "gpu_low=0x%llx\n",
	       name.UTF8String, n, (unsigned long long)allocated, cAligned, gAligned,
	       (unsigned long long)cGap, (unsigned long long)gGap, sharePage,
	       (unsigned long long)(cp[0] & 0xffff), (unsigned long long)(gp[0] & 0xffff));
}

static void superpageCase(void)
{
#if defined(__x86_64__)
	const uint64_t sp = 2 << 20;
	const uint64_t cap = (uint64_t)(gCapGiB * 1073741824.0);
	const uint64_t guard = (uint64_t)(gGuardGiB * 1073741824.0);
	unsigned long maxN = cap / sp, n = 0;
	double *lat = calloc(maxN + 1, sizeof *lat);
	mach_vm_address_t *at = calloc(maxN + 1, sizeof *at);
	const uint64_t w0 = wiredBytes(), f0 = footprint();
	kern_return_t last = KERN_SUCCESS;
	const char *stop = "cap";
	const double start = nowUs();
	while (n < maxN) {
		if (availBytes() < guard) {
			stop = "guard";
			break;
		}
		if (nowUs() - start > gTimeS * 1e6) {
			stop = "time";
			break;
		}
		mach_vm_address_t a = 0;
		double t0 = nowUs();
		last = mach_vm_allocate(mach_task_self(), &a, sp,
		                        VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_2MB |
		                            VM_MAKE_TAG(VM_MEMORY_IOACCELERATOR));
		double t = nowUs() - t0;
		if (last != KERN_SUCCESS) {
			stop = "refused";
			break;
		}
		lat[n] = t;
		at[n++] = a;
	}
	const uint64_t w1 = wiredBytes(), f1 = footprint();
	unsigned zeros = 0;
	for (unsigned long i = 0; i < n; i++)
		zeros += ((volatile uint8_t *)at[i])[sp - 1] == 0;
	double t0 = nowUs();
	for (unsigned long i = 0; i < n; i++)
		mach_vm_deallocate(mach_task_self(), at[i], sp);
	double tf = nowUs() - t0;
	double p50 = quant(lat, n, 0.5), p99 = quant(lat, n, 0.99);
	printf("shm: superpage n=%lu bytes=%.1fMiB stop=%s kr=%d us=%.1f p99=%.1f wired=%+.1fMiB "
	       "footprint=%+.1fMiB zero=%u free_us=%.1f\n",
	       n, mib((int64_t)(n * sp)), stop, last, p50, p99, mib((int64_t)(w1 - w0)),
	       mib((int64_t)(f1 - f0)), zeros, n ? tf / n : 0);
	free(lat);
	free(at);
#else
	printf("shm: superpage SKIP (x86_64 only)\n");
#endif
}

// spmark.N: N 2 MiB superpages and N 2 MiB ranges of plain pages (vm_allocate, each page
// written), every 4 KiB page of both marked with "RGSPMARK", the process id, the kind (1 a
// superpage, 0 plain), the range's index and the page's index, then held for -time seconds, so a
// scan of the guest's RAM from the host finds where each page lies in guest-physical memory.
static void spmarkCase(unsigned long n)
{
#if defined(__x86_64__)
	const uint64_t sp = 2 << 20;
	mach_vm_address_t *at = calloc(2 * n, sizeof *at);
	unsigned long made[2] = {0, 0};
	for (int kind = 1; kind >= 0; kind--) {
		for (unsigned long i = 0; i < n; i++) {
			mach_vm_address_t a = 0;
			int flags = VM_FLAGS_ANYWHERE | (kind ? VM_FLAGS_SUPERPAGE_SIZE_2MB : 0);
			if (mach_vm_allocate(mach_task_self(), &a, sp, flags) != KERN_SUCCESS)
				break;
			at[kind * n + i] = a;
			for (uint64_t p = 0; p < sp / 4096; p++) {
				uint64_t *w = (uint64_t *)(a + p * 4096);
				memcpy(w, "RGSPMARK", 8);
				w[1] = (uint64_t)getpid();
				w[2] = (uint64_t)kind;
				w[3] = i;
				w[4] = p;
			}
			made[kind]++;
		}
	}
	printf("shm: spmark superpages=%lu plain=%lu pid=%d ready, holding %.0f s\n", made[1],
	       made[0], getpid(), gTimeS);
	fflush(stdout);
	usleep((useconds_t)(gTimeS * 1e6));
	for (unsigned long i = 0; i < 2 * n; i++)
		if (at[i])
			mach_vm_deallocate(mach_task_self(), at[i], sp);
	free(at);
	printf("shm: spmark freed\n");
#else
	(void)n;
	printf("shm: spmark SKIP (x86_64 only)\n");
#endif
}

int main(int argc, const char **argv)
{
	@autoreleasepool {
		const char *want = NULL;
		NSMutableArray<NSString *> *cases = [NSMutableArray array];
		for (int i = 1; i < argc; i++) {
			if (!strcmp(argv[i], "-d") && i + 1 < argc)
				want = argv[++i];
			else if (!strcmp(argv[i], "-cap") && i + 1 < argc)
				gCapGiB = atof(argv[++i]);
			else if (!strcmp(argv[i], "-count") && i + 1 < argc)
				gCount = strtoul(argv[++i], NULL, 10);
			else if (!strcmp(argv[i], "-time") && i + 1 < argc)
				gTimeS = atof(argv[++i]);
			else if (!strcmp(argv[i], "-guard") && i + 1 < argc)
				gGuardGiB = atof(argv[++i]);
			else if (!strcmp(argv[i], "-use"))
				gUse = YES;
			else if (!strcmp(argv[i], "-check"))
				gCheck = YES;
			else if (!strcmp(argv[i], "-touch"))
				gTouch = YES;
			else if (!strcmp(argv[i], "-nofree"))
				gNoFree = YES;
			else if (!strcmp(argv[i], "-rounds") && i + 1 < argc)
				gRounds = atoi(argv[++i]);
			else
				[cases addObject:@(argv[i])];
		}
		id<MTLDevice> d = nil;
		if (want) {
			for (id<MTLDevice> c in MTLCopyAllDevices())
				if (strstr(c.name.UTF8String, want)) {
					d = c;
					break;
				}
		} else {
			d = MTLCreateSystemDefaultDevice();
		}
		BOOL needDevice = NO;
		for (NSString *c in cases)
			needDevice |= ![c isEqualToString:@"superpage"] && ![c hasPrefix:@"spmark"];
		if (d == nil && needDevice) {
			fprintf(stderr, "shmem: no device\n");
			return 1;
		}
		printf("shm: start device=\"%s\" cap=%.1fGiB count=%lu time=%.0fs guard=%.1fGiB use=%d "
		       "check=%d touch=%d\n",
		       d ? d.name.UTF8String : "-", gCapGiB, gCount, gTimeS, gGuardGiB, gUse, gCheck,
		       gTouch);
		for (NSString *c in cases) {
			NSArray<NSString *> *p = [c componentsSeparatedByString:@"."];
			BOOL ok = YES;
			@autoreleasepool {
				if ([c isEqualToString:@"info"])
					info(d);
				else if ([c isEqualToString:@"superpage"])
					superpageCase();
				else if (p.count == 2 && [p[0] isEqualToString:@"spmark"])
					spmarkCase(strtoul(p[1].UTF8String, NULL, 10));
				else if (p.count == 3 && [p[0] isEqualToString:@"hold"]) {
					MTLResourceOptions o = modeOf(p[1], &ok);
					if (ok)
						hold(d, c, o, sizeOf(p[2]));
				} else if (p.count == 2 && [p[0] isEqualToString:@"max"]) {
					MTLResourceOptions o = modeOf(p[1], &ok);
					if (ok)
						maxCase(d, c, o);
				} else if (p.count == 3 && [p[0] isEqualToString:@"zero"]) {
					MTLResourceOptions o = modeOf(p[1], &ok);
					if (ok)
						zeroCase(d, c, o, sizeOf(p[2]));
				} else if (p.count == 3 && [p[0] isEqualToString:@"align"]) {
					MTLResourceOptions o = modeOf(p[1], &ok);
					if (ok)
						alignCase(d, c, o, sizeOf(p[2]));
				} else
					ok = NO;
			}
			if (!ok)
				printf("shm: %s SKIP unknown case\n", c.UTF8String);
			fflush(stdout);
		}
	}
	return 0;
}
