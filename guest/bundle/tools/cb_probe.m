// cb_probe: command buffers committed and waited for on the reims-graphics Metal device, in one
// process, with a watchdog that names the command buffer whose completion never came.
//   cb_probe one N       N tiny command buffers on one queue, each committed and waited for
//   cb_probe two N       N tiny command buffers alternating over two queues, each waited for
//   cb_probe queues N    N queues, each made just before its first command buffer, which is
//                        committed at once and waited for (a compositor's first frame)
// A command buffer is one blit filling 256 bytes of a Shared buffer with its index. A
// completion 10 s late prints HANG and exits 3; the buffer's bytes are checked after each.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        cb_probe.m -o cb_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static _Atomic long g_done;    // command buffers completed
static _Atomic long g_waiting; // the one being waited for (-1: none)
static _Atomic int g_queue;

static void *watchdog(void *arg)
{
	long last = -1;
	int still = 0;
	for (;;) {
		sleep(1);
		long d = atomic_load(&g_done);
		if (d == last && atomic_load(&g_waiting) >= 0) {
			if (++still >= 10) {
				printf("HANG: command buffer %ld (queue %d) never completed; %ld completed before it\n",
				       atomic_load(&g_waiting), atomic_load(&g_queue), d);
				fflush(stdout);
				_exit(3);
			}
		} else {
			still = 0;
		}
		last = d;
	}
	return NULL;
}

static int run(id<MTLCommandQueue> q, id<MTLBuffer> buf, long i, int queueIndex)
{
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b fillBuffer:buf range:NSMakeRange(0, 256) value:(uint8_t)(i & 0xff)];
	[b endEncoding];
	atomic_store(&g_queue, queueIndex);
	atomic_store(&g_waiting, i);
	[cb commit];
	[cb waitUntilCompleted];
	atomic_store(&g_waiting, -1);
	atomic_fetch_add(&g_done, 1);
	if (cb.status != MTLCommandBufferStatusCompleted) {
		printf("command buffer %ld: status %ld\n", i, (long)cb.status);
		return 1;
	}
	const uint8_t *p = buf.contents;
	for (int k = 0; k < 256; k++) {
		if (p[k] != (uint8_t)(i & 0xff)) {
			printf("command buffer %ld: byte %d is 0x%02x\n", i, k, p[k]);
			return 1;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	@autoreleasepool {
		if (argc < 3) {
			printf("usage: cb_probe one|two|queues N\n");
			return 2;
		}
		const char *mode = argv[1];
		long n = atol(argv[2]);
		id<MTLDevice> dev = nil;
		for (id<MTLDevice> d in MTLCopyAllDevices())
			if ([NSStringFromClass([d class]) isEqualToString:@"RgfxMetalDevice"]) dev = d;
		if (dev == nil) {
			printf("no RgfxMetalDevice\n");
			return 2;
		}
		atomic_store(&g_waiting, -1);
		pthread_t t;
		pthread_create(&t, NULL, watchdog, NULL);
		id<MTLBuffer> buf = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
		NSTimeInterval t0 = [NSDate timeIntervalSinceReferenceDate];
		int fails = 0;
		if (strcmp(mode, "queues") == 0) {
			NSMutableArray *keep = [NSMutableArray array];
			for (long i = 0; i < n && !fails; i++) {
				id<MTLCommandQueue> q = [dev newCommandQueue];
				[keep addObject:q];
				fails += run(q, buf, i, (int)i);
			}
		} else {
			id<MTLCommandQueue> q0 = [dev newCommandQueue];
			id<MTLCommandQueue> q1 = strcmp(mode, "two") == 0 ? [dev newCommandQueue] : q0;
			for (long i = 0; i < n && !fails; i++) {
				int which = (int)(i & 1) && q1 != q0;
				fails += run(which ? q1 : q0, buf, i, which);
			}
		}
		double s = [NSDate timeIntervalSinceReferenceDate] - t0;
		printf("RESULT %s %s: %ld command buffers completed in %.2f s (%.1f us each)\n",
		       fails ? "FAIL" : "PASS", mode, atomic_load(&g_done), s,
		       atomic_load(&g_done) ? s * 1e6 / atomic_load(&g_done) : 0.0);
		return fails ? 1 : 0;
	}
}
