// oracle_order: how the device orders a GPU access to an IOSurface after another queue's or
// another process's pending access to it, and when it then answers the later command buffer
// submitted and scheduled.
//   oracle_order DEVICE WHERE MODE [ROUNDS [SPIN]]
// The long side commits a compute pass over surface S that spins SPIN steps a thread (default
// 60000: about 210 ms on the 5300M) before it writes or reads S; on odd rounds it runs over an
// unrelated surface C instead, so the short side's access has nothing to wait for. Right after the
// long side's commit, the short side commits a blit over S with commitAndWaitUntilSubmitted.
// WHERE:
//   xproc     the short side is a second process, with a texture of its own over S (looked up by
//             its id), told over a pipe once the long side has committed
//   xqueue    the short side is a second queue of this process, with a texture of its own over S
//   xqueue1   the same, through the long side's texture object
// MODE:
//   raw   the long side writes 1000 + round into S; the short side copies S into a buffer
//   war   the long side copies S into a buffer; the short side writes 2000 + round into S
//   waw   both write S
// Each round prints the short side's times from its commit: when commitAndWaitUntilSubmitted
// returns and the buffer's status then, when its scheduled handler runs, when its completed handler
// runs, and for raw the value it read (1000 + round: after the long write; anything else: before
// it); then the long side's commit-to-completion and GPU time. The short side's line comes first.
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
                                  "  uint x = spinFor(g, spin); t.write(uint4(v + (x == 7u ? 1u : 0u)), g); }\n"
                                  "kernel void get(texture2d<uint, access::read> t [[texture(0)]],\n"
                                  "  device uint *out [[buffer(0)]], constant uint &spin [[buffer(1)]],\n"
                                  "  uint2 g [[thread_position_in_grid]]) {\n"
                                  "  uint x = spinFor(g, spin); out[g.y * 1024 + g.x] = t.read(g).x + (x == 7u ? 1u : 0u); }\n";

// A texture over plane 0 of `s`, Managed where the device takes it, else Shared.
static id<MTLTexture> over(id<MTLDevice> dev, IOSurfaceRef s)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
	                                                                             width:kSide
	                                                                            height:kSide
	                                                                         mipmapped:NO];
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
		(id)kIOSurfaceWidth : @(kSide),
		(id)kIOSurfaceHeight : @(kSide),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'L032'),
		(id)kIOSurfaceIsGlobal : @YES,
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	IOSurfaceLock(s, 0, NULL);
	memset(IOSurfaceGetBaseAddress(s), 0, IOSurfaceGetAllocSize(s));
	IOSurfaceUnlock(s, 0, NULL);
	return s;
}

static const char *statusName(MTLCommandBufferStatus s)
{
	switch (s) {
	case MTLCommandBufferStatusNotEnqueued: return "NotEnqueued";
	case MTLCommandBufferStatusEnqueued: return "Enqueued";
	case MTLCommandBufferStatusCommitted: return "Committed";
	case MTLCommandBufferStatusScheduled: return "Scheduled";
	case MTLCommandBufferStatusCompleted: return "Completed";
	case MTLCommandBufferStatusError: return "Error";
	}
	return "?";
}

// The short side's blit over `t` in round `r` (reads: `t` into `buf`; else `buf`, filled with
// 2000 + r, into `t`), committed by commitAndWaitUntilSubmitted; prints its times.
static void shortBlit(id<MTLCommandQueue> q, id<MTLTexture> t, id<MTLBuffer> buf, bool reads, const char *who, int r)
{
	if (!reads) {
		uint32_t *p = (uint32_t *)buf.contents;
		for (int i = 0; i < kSide * kSide; i++)
			p[i] = 2000 + (uint32_t)r;
	}
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	if (reads)
		[b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
		          sourceSize:MTLSizeMake(kSide, kSide, 1) toBuffer:buf destinationOffset:0
		destinationBytesPerRow:kSide * 4 destinationBytesPerImage:kSide * kSide * 4];
	else
		[b copyFromBuffer:buf sourceOffset:0 sourceBytesPerRow:kSide * 4 sourceBytesPerImage:kSide * kSide * 4
		       sourceSize:MTLSizeMake(kSide, kSide, 1) toTexture:t destinationSlice:0 destinationLevel:0
		destinationOrigin:MTLOriginMake(0, 0, 0)];
	[b endEncoding];
	__block double sched = 0, done = 0;
	const double t0 = oracle_now_ms();
	[cb addScheduledHandler:^(id<MTLCommandBuffer> c) {
		sched = oracle_now_ms() - t0;
	}];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
		done = oracle_now_ms() - t0;
	}];
	SEL sel = NSSelectorFromString(@"commitAndWaitUntilSubmitted");
	const bool submits = [(id)cb respondsToSelector:sel];
	if (submits)
		((BOOL (*)(id, SEL))objc_msgSend)(cb, sel);
	else
		[cb commit];
	const double ret = oracle_now_ms() - t0;
	const MTLCommandBufferStatus st = cb.status;
	[cb waitUntilCompleted];
	usleep(2000);
	printf("  short (%s) %s: %s %.2f ms (status %s), scheduled %.2f ms, completed %.2f ms", who,
	       reads ? "read S" : "wrote S", submits ? "submitted-return" : "commit-return", ret, statusName(st),
	       sched, done);
	if (reads)
		printf(", read %u", ((uint32_t *)buf.contents)[0]);
	printf("\n");
}

static int shortSide(const char *device, const char *mode, uint32_t sid, int in, int out, int rounds)
{
	id<MTLDevice> dev = oracle_device(device);
	IOSurfaceRef s = IOSurfaceLookup(sid);
	id<MTLTexture> t = over(dev, s);
	id<MTLCommandQueue> q = [dev newCommandQueue];
	id<MTLBuffer> buf = [dev newBufferWithLength:kSide * kSide * 4 options:MTLResourceStorageModeShared];
	const bool reads = strcmp(mode, "raw") == 0;
	for (int r = 0; r < rounds; r++) {
		char c;
		if (read(in, &c, 1) != 1)
			break;
		shortBlit(q, t, buf, reads, "other process", r);
		write(out, "d", 1);
	}
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc == 8 && strcmp(argv[2], "short") == 0)
			return shortSide(argv[1], argv[3], (uint32_t)atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
			                 atoi(argv[7]));
		if (argc < 4) {
			fprintf(stderr, "usage: %s DEVICE xproc|xqueue|xqueue1 raw|war|waw [ROUNDS [SPIN]]\n", argv[0]);
			return 2;
		}
		const char *where = argv[2], *mode = argv[3];
		const int rounds = argc > 4 ? atoi(argv[4]) : 6;
		const uint32_t spin = argc > 5 ? (uint32_t)atoi(argv[5]) : 60000;
		const bool xproc = strcmp(where, "xproc") == 0, sameTex = strcmp(where, "xqueue1") == 0;
		const bool war = strcmp(mode, "war") == 0, raw = strcmp(mode, "raw") == 0;
		id<MTLDevice> dev = oracle_device(argv[1]);
		oracle_header("oracle_order", dev);
		printf("# %s %s, %d rounds, spin %u\n", where, mode, rounds, spin);
		IOSurfaceRef s = newSurface(), cs = newSurface();
		NSError *err = nil;
		id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
		id<MTLComputePipelineState> put =
			[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"put"] error:&err];
		id<MTLComputePipelineState> get =
			[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"get"] error:&err];
		if (!put || !get) {
			printf("  ! %s\n", err.description.UTF8String);
			return 2;
		}
		id<MTLTexture> ts = over(dev, s), tc = over(dev, cs);
		id<MTLTexture> ts2 = sameTex ? ts : over(dev, s);
		id<MTLCommandQueue> q = [dev newCommandQueue], q2 = [dev newCommandQueue];
		id<MTLBuffer> out = [dev newBufferWithLength:kSide * kSide * 4 options:MTLResourceStorageModeShared];
		id<MTLBuffer> buf2 = [dev newBufferWithLength:kSide * kSide * 4 options:MTLResourceStorageModeShared];
		int toShort[2], fromShort[2];
		pid_t pid = 0;
		if (xproc) {
			pipe(toShort);
			pipe(fromShort);
			char sid[16], rfd[16], wfd[16], nr[16];
			snprintf(sid, sizeof sid, "%u", IOSurfaceGetID(s));
			snprintf(rfd, sizeof rfd, "%d", toShort[0]);
			snprintf(wfd, sizeof wfd, "%d", fromShort[1]);
			snprintf(nr, sizeof nr, "%d", rounds);
			char *args[] = {argv[0], argv[1], "short", (char *)mode, sid, rfd, wfd, nr, NULL};
			posix_spawn(&pid, argv[0], NULL, NULL, args, environ);
			close(toShort[0]);
			close(fromShort[1]);
			sleep(2);
		}
		for (int r = 0; r < rounds; r++) {
			const bool control = r % 2;
			id<MTLTexture> t = control ? tc : ts;
			const uint32_t v = 1000 + (uint32_t)r;
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:war ? get : put];
			[e setTexture:t atIndex:0];
			if (war)
				[e setBuffer:out offset:0 atIndex:0];
			else
				[e setBytes:&v length:4 atIndex:0];
			[e setBytes:&spin length:4 atIndex:1];
			[e dispatchThreads:MTLSizeMake(kSide, kSide, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[e endEncoding];
			const double t0 = oracle_now_ms();
			[cb commit];
			if (xproc)
				write(toShort[1], "g", 1);
			else
				shortBlit(q2, ts2, buf2, raw, "other queue", r);
			[cb waitUntilCompleted];
			const double ms = oracle_now_ms() - t0;
			if (xproc) {
				char c;
				read(fromShort[0], &c, 1);
			}
			printf("round %d (long side on %s): long %s in %.1f ms (GPU %.1f ms)\n", r,
			       control ? "the control C" : "S", war ? "read" : "wrote", ms,
			       (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
			usleep(200000);
		}
		if (xproc) {
			close(toShort[1]);
			int st;
			waitpid(pid, &st, 0);
		}
	}
	return 0;
}
