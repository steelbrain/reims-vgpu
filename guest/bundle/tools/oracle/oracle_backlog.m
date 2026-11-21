// oracle_backlog: whether a GPU read of an IOSurface by one process waits out another process's
// backlog when that process's writer of the surface sits at the tail of its queue, whether it
// also waits for what that queue commits after the writer, and whether a writer committed by a
// plain commit is ordered at all. dgpu-oracle.md Q48 holds what it found.
//   oracle_backlog DEVICE ARM [ROUNDS [TARGET_MS]]
// Process A (this one) commits 8 busy command buffers of about TARGET_MS (default 20) of GPU time
// each on one queue (a spin over an unrelated texture), then a ninth that writes 1000 + round into
// surface S (on odd rounds the unrelated surface C instead: the control, nothing for B to wait for).
// S and C are 1024 x 1024 R32Uint. The spin is calibrated first: 2000 steps run three times, the
// last two's mean GPU time scaled to TARGET_MS; the header line prints the spin and the GPU time
// of one buffer of it.
//   ARM 1: the ninth by commitAndWaitUntilSubmitted.
//   ARM 2: as 1, then 4 more busy buffers committed after the ninth, before B's read.
//   ARM 3: the ninth by plain commit.
// Then process B (a second process, the same binary, told over a pipe) commits a blit that copies
// S into a buffer, by commitAndWaitUntilSubmitted. Each round prints, from B's commit: B's call
// return, scheduled and completed; B's completion against A's ninth's completion (positive:
// after); the value B read (1000 + round: after A's write; else before it); A's ninth's call
// return; the 8 busy buffers' mean GPU time; for arm 2 B's completion against the first and the
// last of A's 4 later buffers; B's final status. Times are mach absolute time, one clock for both
// processes.
// Q48 ran each arm on each GPU, off screen, as:
//   caffeinate -s ./oracle_backlog DEVICE ARM 6 20      (DEVICE 5300M or 630)
// Build: see README.md.
#import "oracle_common.h"
#import <IOSurface/IOSurface.h>
#include <objc/message.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
enum { kSide = 1024 };

static NSString *const kSource = @"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"static uint spinFor(uint2 g, uint spin) { uint x = g.x * 2654435761u + g.y;\n"
	"  for (uint i = 0; i < spin; i++) x = x * 1664525u + 1013904223u; return x; }\n"
	"kernel void put(texture2d<uint, access::write> t [[texture(0)]],\n"
	"  constant uint &v [[buffer(0)]], constant uint &spin [[buffer(1)]],\n"
	"  uint2 g [[thread_position_in_grid]]) {\n"
	"  uint x = spinFor(g, spin); t.write(uint4(v + (x == 7u ? 1u : 0u)), g); }\n";

static id<MTLTexture> over(id<MTLDevice> dev, IOSurfaceRef s)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
	                                                                             width:kSide height:kSide mipmapped:NO];
	d.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
	d.storageMode = MTLStorageModeManaged;
	id<MTLTexture> t = [dev newTextureWithDescriptor:d iosurface:s plane:0];
	if (!t) {
		d.storageMode = MTLStorageModeShared;
		t = [dev newTextureWithDescriptor:d iosurface:s plane:0];
	}
	return t;
}

static IOSurfaceRef newSurface(void)
{
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(kSide), (id)kIOSurfaceHeight : @(kSide), (id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'L032'), (id)kIOSurfaceIsGlobal : @YES,
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	IOSurfaceLock(s, 0, NULL);
	memset(IOSurfaceGetBaseAddress(s), 0, IOSurfaceGetAllocSize(s));
	IOSurfaceUnlock(s, 0, NULL);
	return s;
}

static BOOL cwus(id<MTLCommandBuffer> cb)
{
	SEL sel = NSSelectorFromString(@"commitAndWaitUntilSubmitted");
	if (![(id)cb respondsToSelector:sel]) {
		[cb commit];
		return NO;
	}
	((BOOL (*)(id, SEL))objc_msgSend)(cb, sel);
	return YES;
}

// B: per round, on "g", blit S into a buffer by commitAndWaitUntilSubmitted; write back
// "commit_abs ret_ms sched_ms done_abs value\n".
static int sideB(const char *device, uint32_t sid, int in, int out, int rounds)
{
	id<MTLDevice> dev = oracle_device(device);
	IOSurfaceRef s = IOSurfaceLookup(sid);
	id<MTLTexture> t = over(dev, s);
	id<MTLCommandQueue> q = [dev newCommandQueue];
	id<MTLBuffer> buf = [dev newBufferWithLength:kSide * kSide * 4 options:MTLResourceStorageModeShared];
	for (int r = 0; r < rounds; r++) {
		char c;
		if (read(in, &c, 1) != 1)
			break;
		@autoreleasepool {
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			        sourceSize:MTLSizeMake(kSide, kSide, 1) toBuffer:buf destinationOffset:0
			destinationBytesPerRow:kSide * 4 destinationBytesPerImage:kSide * kSide * 4];
			[b endEncoding];
			__block double sched = 0, done = 0;
			const double t0 = oracle_now_ms();
			[cb addScheduledHandler:^(id<MTLCommandBuffer> x) { sched = oracle_now_ms(); }];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> x) { done = oracle_now_ms(); }];
			cwus(cb);
			const double ret = oracle_now_ms() - t0;
			[cb waitUntilCompleted];
			usleep(2000);
			char line[256];
			int n = snprintf(line, sizeof line, "%.3f %.3f %.3f %.3f %u %ld\n", t0, ret, sched - t0, done,
			                 ((uint32_t *)buf.contents)[0], (long)cb.status);
			write(out, line, n);
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc == 7 && strcmp(argv[2], "B") == 0)
			return sideB(argv[1], (uint32_t)atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
		if (argc < 3) {
			fprintf(stderr, "usage: %s DEVICE 1|2|3 [ROUNDS [TARGET_MS]]\n", argv[0]);
			return 2;
		}
		const int arm = atoi(argv[2]);
		const int rounds = argc > 3 ? atoi(argv[3]) : 6;
		const double target = argc > 4 ? atof(argv[4]) : 20.0;
		id<MTLDevice> dev = oracle_device(argv[1]);
		oracle_header("oracle_backlog", dev);
		IOSurfaceRef s = newSurface(), cs = newSurface();
		NSError *err = nil;
		id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
		id<MTLComputePipelineState> put =
			[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"put"] error:&err];
		if (!put) {
			printf("  ! %s\n", err.description.UTF8String);
			return 2;
		}
		id<MTLTexture> ts = over(dev, s), tc = over(dev, cs);
		MTLTextureDescriptor *bd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
		                                                                             width:kSide height:kSide mipmapped:NO];
		bd.usage = MTLTextureUsageShaderWrite;
		bd.storageMode = MTLStorageModePrivate;
		id<MTLTexture> tb = [dev newTextureWithDescriptor:bd];
		id<MTLCommandQueue> q = [dev newCommandQueueWithMaxCommandBufferCount:64];
		id<MTLCommandBuffer> (^busy)(uint32_t) = ^id<MTLCommandBuffer>(uint32_t spin) {
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			uint32_t v = 7;
			[e setComputePipelineState:put];
			[e setTexture:tb atIndex:0];
			[e setBytes:&v length:4 atIndex:0];
			[e setBytes:&spin length:4 atIndex:1];
			[e dispatchThreads:MTLSizeMake(kSide, kSide, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[e endEncoding];
			return cb;
		};
		// Calibrate: spin 2000, three times, the last two's mean, scaled to the target.
		double per = 0;
		for (int i = 0; i < 3; i++) {
			id<MTLCommandBuffer> cb = busy(2000);
			[cb commit];
			[cb waitUntilCompleted];
			if (i > 0)
				per += (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 2;
		}
		uint32_t spin = (uint32_t)(2000.0 * target / per);
		{
			id<MTLCommandBuffer> cb = busy(spin);
			[cb commit];
			[cb waitUntilCompleted];
			printf("# arm %d, %d rounds; busy spin %u: %.2f ms GPU (target %.1f)\n", arm, rounds, spin,
			       (cb.GPUEndTime - cb.GPUStartTime) * 1e3, target);
		}
		int toB[2], fromB[2];
		pipe(toB);
		pipe(fromB);
		char sid[16], rfd[16], wfd[16], nr[16];
		snprintf(sid, sizeof sid, "%u", IOSurfaceGetID(s));
		snprintf(rfd, sizeof rfd, "%d", toB[0]);
		snprintf(wfd, sizeof wfd, "%d", fromB[1]);
		snprintf(nr, sizeof nr, "%d", rounds);
		char *args[] = {argv[0], argv[1], "B", sid, rfd, wfd, nr, NULL};
		pid_t pid = 0;
		posix_spawn(&pid, argv[0], NULL, NULL, args, environ);
		close(toB[0]);
		close(fromB[1]);
		FILE *fb = fdopen(fromB[0], "r");
		sleep(2);
		for (int r = 0; r < rounds; r++) {
			@autoreleasepool {
				const bool control = r % 2;
				NSMutableArray *ahead = [NSMutableArray array], *later = [NSMutableArray array];
				__block double ninthDone = 0;
				double *laterDone = calloc(4, sizeof(double));
				for (int i = 0; i < 8; i++) {
					id<MTLCommandBuffer> cb = busy(spin);
					[cb commit];
					[ahead addObject:cb];
				}
				id<MTLCommandBuffer> nin = [q commandBuffer];
				id<MTLComputeCommandEncoder> e = [nin computeCommandEncoder];
				uint32_t v = 1000 + (uint32_t)r, one = 1;
				[e setComputePipelineState:put];
				[e setTexture:control ? tc : ts atIndex:0];
				[e setBytes:&v length:4 atIndex:0];
				[e setBytes:&one length:4 atIndex:1];
				[e dispatchThreads:MTLSizeMake(kSide, kSide, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
				[e endEncoding];
				[nin addCompletedHandler:^(id<MTLCommandBuffer> x) { ninthDone = oracle_now_ms(); }];
				const double n0 = oracle_now_ms();
				if (arm == 3)
					[nin commit];
				else
					cwus(nin);
				const double ninthRet = oracle_now_ms() - n0;
				if (arm == 2) {
					for (int i = 0; i < 4; i++) {
						id<MTLCommandBuffer> cb = busy(spin);
						[cb addCompletedHandler:^(id<MTLCommandBuffer> x) { laterDone[i] = oracle_now_ms(); }];
						[cb commit];
						[later addObject:cb];
					}
				}
				write(toB[1], "g", 1);
				double bCommit = 0, bRet = 0, bSched = 0, bDone = 0;
				unsigned val = 0;
				long bst = 0;
				if (fscanf(fb, "%lf %lf %lf %lf %u %ld", &bCommit, &bRet, &bSched, &bDone, &val, &bst) != 6) {
					printf("  ! no answer from B\n");
					break;
				}
				[nin waitUntilCompleted];
				for (id<MTLCommandBuffer> cb in later)
					[cb waitUntilCompleted];
				usleep(2000);
				double bgpu = 0;
				for (id<MTLCommandBuffer> cb in ahead)
					bgpu += (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 8;
				printf("round %d arm %d (ninth writes %s): ninth %s returned %.3f ms; B returned %.3f ms, "
				       "scheduled %.3f ms, completed %.2f ms after its commit, %+.2f ms from A's ninth's "
				       "completion (%s), read %u (%s); A's ninth completed %.2f ms after B's commit; busy GPU "
				       "mean %.2f ms",
				       r, arm, control ? "C, control" : "S", arm == 3 ? "commit" : "cwus", ninthRet, bRet, bSched,
				       bDone - bCommit, bDone - ninthDone, bDone > ninthDone ? "after" : "BEFORE", val,
				       val == 1000 + (unsigned)r ? "after A's write" : "not A's write",
				       ninthDone - bCommit, bgpu);
				if (arm == 2)
					printf("; B vs A's 4 later: %+.2f ms from the first's completion, %+.2f from the last's (%s)",
					       bDone - laterDone[0], bDone - laterDone[3],
					       bDone < laterDone[0] ? "B BEFORE all 4" : bDone > laterDone[3] ? "B after all 4"
					                                                                 : "B among them");
				printf("; B status %ld\n", bst);
				free(laterDone);
				usleep(200000);
			}
		}
		close(toB[1]);
		int st;
		waitpid(pid, &st, 0);
	}
	return 0;
}
