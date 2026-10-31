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
//   tex_probe churn N   N times: a new 64 x 64 Private texture, one command buffer copying a
//                       pattern of its own through it, committed, and the texture let go before
//                       the buffer completes; each buffer must complete with its own bytes back
//                       (its texture outlives it on the host), and N past the device's 16320
//                       resource ids needs each texture's id back once it is gone
//   tex_probe big N     one command buffer of N fills, each 4 bytes of a Shared buffer with its
//                       own value, far larger than its queue's ring for large N, waited for by its
//                       handler alone: every word must hold its fill
//   tex_probe kinds N   N rounds, each of one object of every kind made and let go while the
//                       round's command buffer is in flight: a new queue, a 4 MiB Shared buffer
//                       (a region block), a 1 MiB Shared heap with a buffer and a texture placed
//                       in it, a 256 MiB Private heap with a texture placed in it, a Private
//                       buffer of a page and a byte, a 64 x 64 IOSurface texture (its copy is a
//                       region block), a Private texture, a sampler, a depth-stencil state, an
//                       event, a fence and a shared event. The buffer fills the 4 MiB buffer and copies a
//                       pattern through the Private texture into it, the copy out waiting on
//                       the fence the copy in updates; each must complete with those bytes in
//                       the Shared buffer, so N past 256 needs each queue to end, and N past
//                       256 needs the blocks back (the region is 1 GiB), and N past 250 needs
//                       the Private heaps' GPU addresses back (64 GiB a process)
//   tex_probe sharedheap N  N rounds of a 1 MiB Shared Placement heap with buffers placed in
//                       it: A of 64 KiB at 0, B of 64 KiB at 64 KiB and C of 4 KiB at 0, over
//                       A's first bytes. Each placed buffer must be the heap's bytes at its
//                       offset (C's contents A's, B's 64 KiB past them) at the heap's GPU address
//                       plus the offset; the CPU writes a pattern through A, which C reads, and
//                       one command buffer copies A to B and fills C's second half: B must hold
//                       the CPU's pattern and A the GPU's fill, each seen through the other's
//                       contents, as Metal's placed resources alias their heap's memory
// Prints "TEX OK" when every check passes, else "TEX FAIL: why" and exits 1. A completion 20 s
// late prints HANG and exits 3. The copy waits with waitUntilCompleted, or with its completed
// handler alone (`handler`).
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        -framework IOSurface tex_probe.m -o tex_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
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

// `n` textures, each let go right after the command buffer that copies through it is committed.
static int churn(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	const NSUInteger side = 64, row = side * 4, size = row * side;
	id<MTLBuffer> src = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (src == nil || dst == nil)
		return fail("no Shared buffer");
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead;
	double start = now_ms();
	long bad = 0, failed = 0, noTexture = 0;
	for (long i = 0; i < n; i++) {
		struct watch *w = watch_start("a churn command buffer");
		memset([src contents], (int)(i * 13 + 5), size);
		memset([dst contents], 0, size);
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		__block MTLCommandBufferStatus status = MTLCommandBufferStatusNotEnqueued;
		@autoreleasepool {
			id<MTLTexture> tex = [device newTextureWithDescriptor:d];
			if (tex == nil) {
				noTexture++;
				atomic_store(&w->done, 1);
				continue;
			}
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
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				status = b.status;
				dispatch_semaphore_signal(done);
			}];
			[cb commit];
		} // the texture goes now, while its buffer is in flight
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
		atomic_store(&w->done, 1);
		if (status != MTLCommandBufferStatusCompleted)
			failed++;
		else if (memcmp([src contents], [dst contents], size) != 0)
			bad++;
	}
	printf("churn: %ld textures in %.0f ms, failed=%ld, wrong bytes=%ld, no texture=%ld\n", n,
	       now_ms() - start, failed, bad, noTexture);
	if (noTexture)
		return fail("a texture could not be made: an id never came back");
	if (failed)
		return fail("a command buffer failed: its texture went before it ran");
	if (bad)
		return fail("a copy came back wrong");
	return 0;
}

// `n` rounds of one object of every kind, each let go while the round's command buffer runs.
static int kinds(id<MTLDevice> device, long n)
{
	const NSUInteger side = 64, row = side * 4, size = row * side, big = 4u << 20;
	id<MTLBuffer> src = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (src == nil)
		return fail("no Shared buffer");
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead;
	MTLTextureDescriptor *sd =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	sd.usage = MTLTextureUsageShaderRead;
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(side),
		(id)kIOSurfaceHeight : @(side),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	double start = now_ms();
	long failed = 0, bad = 0, missing = 0;
	for (long i = 0; i < n; i++) {
		struct watch *w = watch_start("a kinds command buffer");
		uint8_t fill = (uint8_t)(i * 7 + 3), pattern = (uint8_t)(i * 13 + 5);
		memset([src contents], pattern, size);
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		__block MTLCommandBufferStatus status = MTLCommandBufferStatusNotEnqueued;
		id<MTLBuffer> out = nil; // checked once the round's objects are gone
		@autoreleasepool {
			id<MTLCommandQueue> queue = [device newCommandQueue];
			id<MTLBuffer> shared = [device newBufferWithLength:big
			                                          options:MTLResourceStorageModeShared];
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 1u << 20;
			hd.storageMode = MTLStorageModeShared;
			hd.type = MTLHeapTypePlacement; // resources at offsets the probe names
			id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
			id<MTLBuffer> placed = [heap newBufferWithLength:4096
			                                         options:MTLResourceStorageModeShared
			                                          offset:0];
			// A texture placed in a heap takes the heap's storage mode, as Metal requires.
			MTLTextureDescriptor *placedDesc = [sd copy];
			placedDesc.storageMode = MTLStorageModeShared;
			id<MTLTexture> placedTex = [heap newTextureWithDescriptor:placedDesc offset:0x10000];
			// A 256 MiB Private heap a round: 1000 rounds go round the process's 64 GiB of GPU
			// addresses four times, so its picks must come back as the heaps go.
			MTLHeapDescriptor *pd = [MTLHeapDescriptor new];
			pd.size = 256u << 20;
			pd.storageMode = MTLStorageModePrivate;
			pd.type = MTLHeapTypePlacement;
			id<MTLHeap> privateHeap = [device newHeapWithDescriptor:pd];
			id<MTLTexture> privateTex = [privateHeap newTextureWithDescriptor:d offset:0];
			id<MTLBuffer> deviceBuffer = [device newBufferWithLength:0x4001
			                                                  options:MTLResourceStorageModePrivate];
			IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
			id<MTLTexture> surfaceTex =
			    surface ? [device newTextureWithDescriptor:sd iosurface:surface plane:0] : nil;
			if (surface)
				CFRelease(surface);
			id<MTLTexture> tex = [device newTextureWithDescriptor:d];
			id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
			id<MTLDepthStencilState> depth =
			    [device newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]];
			id<MTLEvent> event = [device newEvent];
			id<MTLFence> fence = [device newFence];
			id<MTLSharedEvent> sharedEvent = [device newSharedEvent];
			if (queue == nil || shared == nil || heap == nil || placed == nil || placedTex == nil ||
			    privateHeap == nil || privateTex == nil || deviceBuffer == nil ||
			    surfaceTex == nil || tex == nil || sampler == nil || depth == nil || event == nil ||
			    fence == nil || sharedEvent == nil) {
				printf("round %ld: queue=%d shared=%d heap=%d placed=%d placedTex=%d "
				       "privateHeap=%d privateTex=%d deviceBuffer=%d surfaceTex=%d "
				       "tex=%d sampler=%d depth=%d event=%d fence=%d sharedEvent=%d\n",
				       i, queue != nil, shared != nil, heap != nil, placed != nil, placedTex != nil,
				       privateHeap != nil, privateTex != nil, deviceBuffer != nil,
				       surfaceTex != nil, tex != nil, sampler != nil, depth != nil, event != nil,
				       fence != nil, sharedEvent != nil);
				missing++;
				atomic_store(&w->done, 1);
				break;
			}
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit fillBuffer:shared range:NSMakeRange(0, big) value:fill];
			[blit fillBuffer:deviceBuffer range:NSMakeRange(0, 0x4001) value:fill];
			[blit copyFromBuffer:src
			           sourceOffset:0
			      sourceBytesPerRow:row
			    sourceBytesPerImage:size
			             sourceSize:MTLSizeMake(side, side, 1)
			              toTexture:privateTex
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit copyFromBuffer:src
			           sourceOffset:0
			      sourceBytesPerRow:row
			    sourceBytesPerImage:size
			             sourceSize:MTLSizeMake(side, side, 1)
			              toTexture:tex
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit updateFence:fence];
			[blit endEncoding];
			blit = [cb blitCommandEncoder];
			[blit waitForFence:fence];
			[blit copyFromTexture:tex
			                 sourceSlice:0
			                 sourceLevel:0
			                sourceOrigin:MTLOriginMake(0, 0, 0)
			                  sourceSize:MTLSizeMake(side, side, 1)
			                    toBuffer:shared
			           destinationOffset:0
			      destinationBytesPerRow:row
			    destinationBytesPerImage:size];
			[blit endEncoding];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				status = b.status;
				dispatch_semaphore_signal(done);
			}];
			[cb commit];
			out = shared; // kept to check its bytes; every other object goes now
		}
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
		atomic_store(&w->done, 1);
		if (missing)
			break;
		@autoreleasepool {
			const uint8_t *b = (const uint8_t *)[out contents];
			long wrong = 0;
			for (NSUInteger k = 0; k < big; k++)
				wrong += b[k] != (k < size ? pattern : fill);
			if (status != MTLCommandBufferStatusCompleted)
				failed++;
			else if (wrong)
				bad++;
			out = nil; // its block goes back for the next round
		}
	}
	printf("kinds: %ld rounds in %.0f ms (%.1f GiB of 4 MiB Shared buffers), failed=%ld, wrong=%ld, "
	       "missing=%ld\n",
	       n, now_ms() - start, (double)n * big / (1u << 30), failed, bad, missing);
	if (missing)
		return fail("an object could not be made: a queue or a block never came back");
	if (failed)
		return fail("a command buffer failed: an object went before it ran");
	if (bad)
		return fail("a Shared buffer came back wrong");
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

// `n` rounds of a Shared heap whose placed buffers alias its bytes, the CPU's and the GPU's
// writes each seen through the other buffers over them.
static int shared_heap(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	const NSUInteger half = 0x10000, small = 0x1000;
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		@autoreleasepool {
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 1u << 20;
			hd.storageMode = MTLStorageModeShared;
			hd.type = MTLHeapTypePlacement;
			id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
			id<MTLBuffer> a = [heap newBufferWithLength:half options:shared offset:0];
			id<MTLBuffer> b = [heap newBufferWithLength:half options:shared offset:half];
			id<MTLBuffer> c = [heap newBufferWithLength:small options:shared offset:0];
			if (heap == nil || a == nil || b == nil || c == nil) {
				printf("round %ld: heap=%d a=%d b=%d c=%d\n", i, heap != nil, a != nil, b != nil,
				       c != nil);
				return fail("no Shared heap or placed buffer");
			}
			uint8_t *pa = (uint8_t *)[a contents], *pb = (uint8_t *)[b contents];
			uint8_t *pc = (uint8_t *)[c contents];
			if (pa == NULL || pc != pa || pb != pa + half) {
				printf("round %ld: contents a=%p b=%p c=%p\n", i, pa, pb, pc);
				return fail("placed buffers are not the heap's bytes at their offsets");
			}
			if (b.gpuAddress != a.gpuAddress + half || c.gpuAddress != a.gpuAddress ||
			    a.gpuAddress == 0) {
				printf("round %ld: gpuAddress a=0x%llx b=0x%llx c=0x%llx\n", i, a.gpuAddress,
				       b.gpuAddress, c.gpuAddress);
				return fail("placed buffers are not at the heap's address plus their offsets");
			}
			const uint8_t pattern = (uint8_t)(i * 13 + 5), fill = (uint8_t)(i * 7 + 3);
			memset(pa, pattern, half);
			if (pc[small - 1] != pattern)
				return fail("the CPU's write through one placed buffer is not another's");
			struct watch *w = watch_start("a shared heap command buffer");
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:half];
			[blit fillBuffer:c range:NSMakeRange(small / 2, small / 2) value:fill];
			[blit endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			atomic_store(&w->done, 1);
			if (cb.status != MTLCommandBufferStatusCompleted)
				return fail("a shared heap command buffer failed");
			long wrong = 0;
			for (NSUInteger k = 0; k < half; k++)
				wrong += pb[k] != pattern;
			for (NSUInteger k = 0; k < small; k++)
				wrong += pa[k] != (k < small / 2 ? pattern : fill);
			if (wrong) {
				printf("round %ld: %ld wrong bytes (b[0]=0x%02x a[0x800]=0x%02x)\n", i, wrong,
				       pb[0], pa[small / 2]);
				return fail("the GPU's reads and writes of the heap's bytes are not the CPU's");
			}
		}
	}
	printf("sharedheap: %ld rounds in %.0f ms\n", n, now_ms() - start);
	return 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		int handler = argc >= 2 && strcmp(argv[1], "handler") == 0;
		int big = argc >= 2 && strcmp(argv[1], "big") == 0;
		int churning = argc >= 2 && strcmp(argv[1], "churn") == 0;
		int everyKind = argc >= 2 && strcmp(argv[1], "kinds") == 0;
		int sharedHeap = argc >= 2 && strcmp(argv[1], "sharedheap") == 0;
		if (argc < 2 ||
		    (strcmp(argv[1], "copy") != 0 && strcmp(argv[1], "hold") != 0 && !handler && !big &&
		     !churning && !everyKind && !sharedHeap) ||
		    (strcmp(argv[1], "copy") != 0 && argc < 3)) {
			fprintf(stderr,
			        "usage: tex_probe copy | hold N | handler N | big N | churn N | kinds N | "
			        "sharedheap N\n");
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
		if (big || churning || everyKind || sharedHeap) {
			long n = strtol(argv[2], NULL, 10);
			int r = big          ? one_big_buffer(device, queue, n)
			        : churning   ? churn(device, queue, n)
			        : sharedHeap ? shared_heap(device, queue, n)
			                     : kinds(device, n);
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
