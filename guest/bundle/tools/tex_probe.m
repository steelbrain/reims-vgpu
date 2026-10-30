// tex_probe: textures on the reims-graphics Metal device, with their resource ids, in one process.
//   tex_probe copy      a 64 x 64 BGRA8 Private texture: one blit copies a Shared buffer's pattern
//                       into it and back out into a second buffer, which is checked byte for byte
//   tex_probe hold N    N 4 x 4 textures made and held at once, then the copy above: prints how
//                       many were made, and their lowest and highest gpuResourceID, which must be
//                       distinct, non-zero and below 16381 (the host device's limit)
//   tex_probe handler N the copy above, waited for only by its completed handler (no
//                       waitUntilCompleted), then N command buffers each filling a Shared buffer,
//                       committed at once and waited for only by their handlers, which must run
//                       off the main thread, once each, in commit order, the buffer filled by the
//                       last; then 2 s idle, in which the process must use under 20 ms of CPU (the
//                       completion thread sleeps in the kext)
//   tex_probe big N     one command buffer of N fills, each 4 bytes of a Shared buffer with its
//                       own value, far larger than its queue's ring for large N, waited for by its
//                       handler alone: every word must hold its fill
// Prints "TEX OK" when every check passes, else "TEX FAIL: why" and exits 1. A completion 20 s
// late prints HANG and exits 3. The copy waits with waitUntilCompleted, or with its completed
// handler alone (`handler`).
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        tex_probe.m -o tex_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <sys/resource.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

static const unsigned long kLimit = 16381;

// One phase's watchdog: it ends the process with HANG if `done` is not set 20 s after it starts.
struct watch {
	atomic_int done;
	const char *what;
};

static void *watchdog(void *arg)
{
	struct watch *w = arg;
	sleep(20);
	if (!atomic_load(&w->done)) {
		printf("HANG: %s never completed\n", w->what);
		fflush(stdout);
		_exit(3);
	}
	return NULL;
}

// Starts a watchdog over `what`; set the returned watch's `done` when it completes.
static struct watch *watch_start(const char *what)
{
	struct watch *w = calloc(1, sizeof(*w)); // the watchdog may outlive the phase: never freed
	atomic_init(&w->done, 0);
	w->what = what;
	pthread_t dog;
	pthread_create(&dog, NULL, watchdog, w);
	pthread_detach(dog);
	return w;
}

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int fail(const char *why)
{
	printf("TEX FAIL: %s\n", why);
	return 1;
}

// Waits for the semaphore a command buffer's completed handler signals, and nothing else.
static void wait_by_handler(dispatch_semaphore_t done)
{
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
}

// The copy through a Private texture, checked byte for byte; its texture's resource id printed.
// `byHandler` waits for it with a completed handler alone.
static int copy_through_texture(id<MTLDevice> device, id<MTLCommandQueue> queue, int byHandler)
{
	const NSUInteger side = 64, row = side * 4, size = row * side;
	id<MTLBuffer> src = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (src == nil || dst == nil)
		return fail("no Shared buffer");
	uint8_t *s = (uint8_t *)[src contents];
	for (NSUInteger i = 0; i < size; i++)
		s[i] = (uint8_t)(i * 7 + 3);
	memset([dst contents], 0, size);
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> tex = [device newTextureWithDescriptor:d];
	if (tex == nil)
		return fail("no texture");
	printf("texture gpuResourceID=%llu\n", (unsigned long long)tex.gpuResourceID._impl);
	if (tex.gpuResourceID._impl == 0 || tex.gpuResourceID._impl >= kLimit)
		return fail("texture resource id 0 or past the limit");
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit copyFromBuffer:src
	           sourceOffset:0
	      sourceBytesPerRow:row
	    sourceBytesPerImage:size
	             sourceSize:MTLSizeMake(side, side, 1)
	              toTexture:tex
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	[blit copyFromTexture:tex
	                 sourceSlice:0
	                 sourceLevel:0
	                sourceOrigin:MTLOriginMake(0, 0, 0)
	                  sourceSize:MTLSizeMake(side, side, 1)
	                    toBuffer:dst
	           destinationOffset:0
	      destinationBytesPerRow:row
	    destinationBytesPerImage:size];
	[blit endEncoding];
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	__block int onMain = 0;
	if (byHandler) {
		[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
			(void)b;
			onMain = pthread_main_np();
			dispatch_semaphore_signal(done);
		}];
	}
	struct watch *w = watch_start("the copy's command buffer");
	[cb commit];
	if (byHandler)
		wait_by_handler(done);
	else
		[cb waitUntilCompleted];
	atomic_store(&w->done, 1);
	if (byHandler && onMain)
		return fail("the completed handler ran on the main thread");
	printf("status=%ld (4=completed,5=error)\n", (long)cb.status);
	if (cb.status != MTLCommandBufferStatusCompleted)
		return fail(cb.error ? [[cb.error description] UTF8String] : "not completed");
	if (memcmp([src contents], [dst contents], size) != 0) {
		const uint8_t *o = (const uint8_t *)[dst contents];
		printf("dst[0]=0x%02x expected 0x%02x\n", o[0], s[0]);
		return fail("the bytes back from the texture differ");
	}
	return 0;
}

// `n` command buffers, each filling a Shared buffer with its own value, committed back to back
// and waited for only by their completed handlers: each runs once, off the main thread, in
// commit order, and the buffer holds the last one's value.
static int handlers_in_order(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	id<MTLBuffer> buf = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
	if (buf == nil)
		return fail("no Shared buffer");
	memset([buf contents], 0, 256);
	long *order = calloc((size_t)n, sizeof(long));
	__block atomic_long ran;
	atomic_init(&ran, 0);
	__block atomic_int offMain;
	atomic_init(&offMain, 1);
	dispatch_semaphore_t all = dispatch_semaphore_create(0);
	struct watch *w = watch_start("the fills' command buffers");
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit fillBuffer:buf range:NSMakeRange(0, 256) value:(uint8_t)(i + 1)];
		[blit endEncoding];
		[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
			long at = atomic_fetch_add(&ran, 1);
			if (at < n)
				order[at] = i;
			if (pthread_main_np() || b.status != MTLCommandBufferStatusCompleted)
				atomic_store(&offMain, 0);
			if (at + 1 == n)
				dispatch_semaphore_signal(all);
		}];
		[cb commit];
	}
	dispatch_semaphore_wait(all, DISPATCH_TIME_FOREVER);
	atomic_store(&w->done, 1);
	double took = now_ms() - start;
	usleep(100000); // a handler run twice would land here
	long total = atomic_load(&ran);
	long misplaced = 0;
	for (long i = 0; i < n; i++)
		misplaced += order[i] != i;
	free(order);
	uint8_t last = ((const uint8_t *)[buf contents])[255];
	printf("handlers ran=%ld of %ld in %.1f ms, out of order=%ld, off main and completed=%d, "
	       "buf=0x%02x\n",
	       total, n, took, misplaced, atomic_load(&offMain), last);
	if (total != n)
		return fail("a completed handler ran more than once");
	if (misplaced)
		return fail("the completed handlers ran out of commit order");
	if (!atomic_load(&offMain))
		return fail("a handler ran on the main thread or saw a buffer not completed");
	if (last != (uint8_t)n)
		return fail("the buffer does not hold the last fill");
	return 0;
}

// One command buffer of `n` 4-byte fills, each of its own value, waited for by its handler.
static int one_big_buffer(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	id<MTLBuffer> buf = [device newBufferWithLength:(NSUInteger)(4 * n)
	                                        options:MTLResourceStorageModeShared];
	if (buf == nil)
		return fail("no Shared buffer");
	memset([buf contents], 0, (size_t)(4 * n));
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	for (long i = 0; i < n; i++)
		[blit fillBuffer:buf range:NSMakeRange((NSUInteger)(4 * i), 4) value:(uint8_t)(i * 7 + 1)];
	[blit endEncoding];
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
		(void)b;
		dispatch_semaphore_signal(done);
	}];
	struct watch *w = watch_start("the big command buffer");
	double start = now_ms();
	[cb commit];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	atomic_store(&w->done, 1);
	double took = now_ms() - start;
	const uint8_t *b = (const uint8_t *)[buf contents];
	long wrong = 0;
	for (long i = 0; i < 4 * n; i++)
		wrong += b[i] != (uint8_t)((i / 4) * 7 + 1);
	printf("big: %ld fills in %.1f ms, status=%ld, wrong bytes=%ld\n", n, took, (long)cb.status,
	       wrong);
	if (cb.status != MTLCommandBufferStatusCompleted)
		return fail(cb.error ? [[cb.error description] UTF8String] : "not completed");
	if (wrong)
		return fail("a fill of the big command buffer is missing");
	return 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		int handler = argc >= 2 && strcmp(argv[1], "handler") == 0;
		int big = argc >= 2 && strcmp(argv[1], "big") == 0;
		if (argc < 2 ||
		    (strcmp(argv[1], "copy") != 0 && strcmp(argv[1], "hold") != 0 && !handler && !big) ||
		    (strcmp(argv[1], "copy") != 0 && argc < 3)) {
			fprintf(stderr, "usage: tex_probe copy | hold N | handler N | big N\n");
			return 2;
		}
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil)
			return fail("no Metal device");
		printf("device=%s\n", [[device name] UTF8String]);
		id<MTLCommandQueue> queue = [device newCommandQueue];
		if (queue == nil)
			return fail("no queue");
		NSMutableArray<id<MTLTexture>> *held = [NSMutableArray array];
		if (strcmp(argv[1], "hold") == 0) {
			long n = strtol(argv[2], NULL, 10);
			MTLTextureDescriptor *d =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                                                       width:4
			                                                      height:4
			                                                   mipmapped:NO];
			d.storageMode = MTLStorageModePrivate;
			d.usage = MTLTextureUsageShaderRead;
			unsigned long lo = ~0ul, hi = 0;
			uint8_t *seen = calloc(kLimit, 1);
			int distinct = 1;
			for (long i = 0; i < n; i++) {
				id<MTLTexture> t = [device newTextureWithDescriptor:d];
				if (t == nil)
					break;
				unsigned long id = (unsigned long)t.gpuResourceID._impl;
				if (id == 0 || id >= kLimit) {
					free(seen);
					printf("texture %ld: gpuResourceID=%lu\n", i, id);
					return fail("a resource id is 0 or past the limit");
				}
				if (seen[id])
					distinct = 0;
				seen[id] = 1;
				lo = id < lo ? id : lo;
				hi = id > hi ? id : hi;
				[held addObject:t];
			}
			free(seen);
			printf("held=%lu of %ld, gpuResourceID %lu to %lu, distinct=%d\n",
			       (unsigned long)held.count, n, held.count ? lo : 0ul, hi, distinct);
			if (!distinct)
				return fail("two live textures answer one resource id");
			// Every id held: give one back so the copy's texture has one, as an app would.
			if ((long)held.count < n && held.count > 0)
				[held removeLastObject];
		}
		if (big) {
			int r = one_big_buffer(device, queue, strtol(argv[2], NULL, 10));
			if (r != 0)
				return r;
			printf("TEX OK\n");
			return 0;
		}
		int r = copy_through_texture(device, queue, handler);
		if (r != 0)
			return r;
		if (handler) {
			r = handlers_in_order(device, queue, strtol(argv[2], NULL, 10));
			if (r != 0)
				return r;
			struct rusage before, after;
			getrusage(RUSAGE_SELF, &before);
			sleep(2);
			getrusage(RUSAGE_SELF, &after);
			long us = (after.ru_utime.tv_sec - before.ru_utime.tv_sec) * 1000000L +
			          (after.ru_utime.tv_usec - before.ru_utime.tv_usec) +
			          (after.ru_stime.tv_sec - before.ru_stime.tv_sec) * 1000000L +
			          (after.ru_stime.tv_usec - before.ru_stime.tv_usec);
			printf("idle cpu=%ld us over 2 s\n", us);
			if (us > 20000)
				return fail("the process uses CPU while idle");
		}
		printf("TEX OK\n");
	}
	return 0;
}
