// oracle_locks: whether Metal takes a lock an app can hold on an object it hands the app, for
// the bundle's app-lock tests (tests/queue_tests.mm, AN APP'S LOCKS). In each step the main thread
// holds @synchronized on one object and waits for Metal's work on it, which another thread does
// where the work needs one:
//   buffer          the buffer's commit and waitUntilCompleted under the buffer's lock
//   buffer-handler  the same, the buffer with a completed handler
//   buffer-handler-lock  the same, the completed handler taking @synchronized on the buffer too:
//                   the app's own deadlock, as the wait returns only after the handler; Metal's
//                   wait never returns (measured: the M3 Max, macOS 27.0.1), nor does the bundle's
//   queue           a buffer of the queue made, a blit encoded, committed and waited for on another
//                   thread, under the queue's lock
//   function        a compute pipeline made from the function on another thread, and the
//                   function's bitCodeHash where it answers it, under the function's lock
//   device          a queue and a buffer made, and a Shared texture's replaceRegion: and getBytes:,
//                   on another thread, under the device's lock
//   oracle_locks DEVICE [STEP]   every step (or the one named), each in a child of its own
// and prints `STEP: finished in N ms` or `STEP: TIMEOUT, no end within 10 s (killed)`, or how its
// child ended. Build: build.sh.
#import "oracle_common.h"
#import <objc/message.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;

// Runs `work` on a thread of its own and waits for it, the caller holding whatever lock it holds.
static void on_other_thread(void (^work)(void))
{
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	NSThread *t = [[NSThread alloc] initWithBlock:^{
		@autoreleasepool {
			work();
		}
		dispatch_semaphore_signal(done);
	}];
	[t start];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
}

static int step(const char *name)
{
	id<MTLCommandQueue> queue = [D newCommandQueue];
	if (!strcmp(name, "buffer-handler-lock")) {
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		const void *mine = (__bridge const void *)cb;
		__block int ran = 0, same = 0;
		[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
			same = (__bridge const void *)x == mine;
			@synchronized(x) {
				ran = 1;
			}
		}];
		@synchronized(cb) {
			[cb commit];
			[cb waitUntilCompleted];
		}
		printf("  status=%ld handler=%d handler's buffer is the buffer=%d\n", (long)cb.status, ran, same);
		return cb.status == MTLCommandBufferStatusCompleted ? 0 : 1;
	}
	if (!strcmp(name, "buffer") || !strcmp(name, "buffer-handler")) {
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		__block int ran = 0;
		if (!strcmp(name, "buffer-handler")) {
			[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				ran = 1;
			}];
		}
		@synchronized(cb) {
			[cb commit];
			[cb waitUntilCompleted];
		}
		printf("  status=%ld handler=%d\n", (long)cb.status, ran);
		return cb.status == MTLCommandBufferStatusCompleted ? 0 : 1;
	}
	if (!strcmp(name, "queue")) {
		__block long status = -1;
		id<MTLBuffer> a = [D newBufferWithLength:256 options:MTLResourceStorageModeShared];
		id<MTLBuffer> b = [D newBufferWithLength:256 options:MTLResourceStorageModeShared];
		@synchronized(queue) {
			on_other_thread(^{
				id<MTLCommandBuffer> cb = [queue commandBuffer];
				id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
				[blit copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:256];
				[blit endEncoding];
				[cb commit];
				[cb waitUntilCompleted];
				status = (long)cb.status;
			});
		}
		printf("  status=%ld\n", status);
		return status == MTLCommandBufferStatusCompleted ? 0 : 1;
	}
	if (!strcmp(name, "function")) {
		NSError *err = nil;
		id<MTLLibrary> lib = [D newLibraryWithSource:@"kernel void k(device uint *o [[buffer(0)]], "
		                                             @"uint i [[thread_position_in_grid]]) { o[i] = i; }"
		                                    options:nil
		                                      error:&err];
		id<MTLFunction> fn = [lib newFunctionWithName:@"k"];
		if (fn == nil) {
			printf("  no function: %s\n", err.description.UTF8String);
			return 1;
		}
		__block int made = 0, hashed = -1;
		@synchronized(fn) {
			on_other_thread(^{
				NSError *e = nil;
				made = [D newComputePipelineStateWithFunction:fn error:&e] != nil;
				SEL s = NSSelectorFromString(@"bitCodeHash");
				if ([(id)fn respondsToSelector:s])
					hashed = ((const void *(*)(id, SEL))objc_msgSend)(fn, s) != NULL;
			});
		}
		printf("  pipeline=%d bitCodeHash=%d (-1: not answered)\n", made, hashed);
		return made ? 0 : 1;
	}
	if (!strcmp(name, "device")) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:4
		                                                      height:4
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeShared;
		id<MTLTexture> tex = [D newTextureWithDescriptor:td];
		__block int copied = 0;
		@synchronized(D) {
			on_other_thread(^{
				id<MTLCommandQueue> q = [D newCommandQueue];
				id<MTLBuffer> stage = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
				uint8_t in[64], out[64] = {0};
				for (int i = 0; i < 64; i++)
					in[i] = (uint8_t)i;
				[tex replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:in bytesPerRow:16];
				[tex getBytes:out bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
				copied = q != nil && stage != nil && memcmp(in, out, 64) == 0;
			});
		}
		printf("  copied=%d\n", copied);
		return copied ? 0 : 1;
	}
	fprintf(stderr, "no step %s\n", name);
	return 2;
}

// Runs STEP in a child, which must end within 10 s.
static int run_child(const char *self, const char *device, const char *name)
{
	char *args[] = {(char *)self, (char *)device, "--step", (char *)name, NULL};
	pid_t pid = 0;
	if (posix_spawn(&pid, self, NULL, NULL, args, environ) != 0) {
		printf("%s: no child\n", name);
		return 1;
	}
	double t0 = oracle_now_ms();
	int status = 0;
	pid_t ended = 0;
	while (ended == 0 && oracle_now_ms() - t0 < 10000.0) {
		ended = waitpid(pid, &status, WNOHANG);
		if (ended == 0)
			usleep(10000);
	}
	if (ended == 0) {
		kill(pid, SIGKILL);
		waitpid(pid, &status, 0);
		printf("%s: TIMEOUT, no end within 10 s (killed)\n", name);
		return 1;
	}
	const int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
	printf("%s: %s in %.0f ms (status %d)\n", name, ok ? "finished" : "FAILED", oracle_now_ms() - t0,
	       status);
	return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [STEP]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 3 && !strcmp(argv[2], "--step")) {
		@autoreleasepool {
			return step(argv[3]);
		}
	}
	oracle_header("oracle_locks", D);
	D = nil;
	if (argc > 2)
		return run_child(argv[0], argv[1], argv[2]);
	const char *steps[] = {"buffer", "buffer-handler", "buffer-handler-lock", "queue", "function",
	                       "device"};
	int failed = 0;
	for (size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++)
		failed += run_child(argv[0], argv[1], steps[i]);
	printf("# %d of %zu steps did not finish\n", failed, sizeof(steps) / sizeof(steps[0]));
	return failed != 0;
}
