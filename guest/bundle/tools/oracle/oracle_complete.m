// oracle_complete: what a resource answers to the resource SPI isComplete, isWriteComplete and
// waitUntilComplete while a command buffer that uses it is in flight, and whether a command
// buffer of a second queue waits for one of the first queue that writes what it reads.
//   oracle_complete DEVICE answers [MS [sched]]
//   oracle_complete DEVICE xqueue [MS [ROUNDS]]
// A command buffer of each case spins first, about MS ms (default 300 for answers, 200 for
// xqueue; a compute kernel's loop, calibrated first, over a scratch buffer of its own).
//
// answers: each case makes the resource, commits a buffer that spins and then uses it as the case
// says, reads isComplete and isWriteComplete at once (with "sched", once waitUntilScheduled
// returns), times waitUntilComplete and reads the buffer's status after it, then reads the two
// answers again once the buffer completed. Cases:
//   tex-render-<storage>   a 64 x 64 BGRA8 texture a render pass clears after the spin
//   tex-write-<storage>    the same texture written by the kernel itself (access::write)
//   tex-read-<storage>     the same texture only read by the kernel
//   tex-blit-<storage>     the texture the destination of a blit after the spin
//   buf-write-<storage>    a 4 KiB buffer the kernel writes
//   buf-read-<storage>     the same buffer the kernel only reads
//   tex-uncommitted        a texture a render pass clears in a buffer not yet committed (the wait
//                          on another thread, 2 s at most)
//   tex-other-queue        a texture a completed buffer cleared, asked while a buffer of a second
//                          queue that names nothing of it is in flight
// <storage> is managed and private.
//
// xqueue: each round commits on queue A a buffer that spins and then clears a Managed texture,
// waits until it is scheduled, then commits on queue B the case's buffer and waits until it
// completes, and counts the rounds that found A's buffer completed when B's wait returned
// (ROUNDS of them, default 5). Cases for B's buffer:
//   empty          no encoder
//   blit-empty     a blit encoder with no command
//   sync-managed   a blit synchronizeTexture:slice:level: of the texture A's buffer clears (Apple's
//                  GL renderer's read-back)
//   sync-other     the same of a Managed texture A's buffer never names
//   compute-tiny   a compute dispatch of one thread over a buffer of its own
//   wuc-sync       sync-managed, with the texture's waitUntilComplete first
// Build: see README.md.
#import "oracle_common.h"

@interface NSObject (OracleCompleteSpi)
- (BOOL)isComplete;
- (BOOL)isWriteComplete;
- (void)waitUntilComplete;
@end

static NSString *const kSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "kernel void spin(device float *s [[buffer(0)]], constant uint &n [[buffer(1)]],\n"
     "                 uint g [[thread_position_in_grid]]) {\n"
     "  float a = s[g];\n  for (uint i = 0; i < n; i++) a = fma(a, 1.0000001f, 0.5f);\n"
     "  s[g] = a; }\n"
     "kernel void twrite(texture2d<float, access::write> t [[texture(0)]],\n"
     "                   device float *s [[buffer(0)]], constant uint &n [[buffer(1)]],\n"
     "                   uint2 g [[thread_position_in_grid]]) {\n"
     "  float a = s[g.x];\n  for (uint i = 0; i < n; i++) a = fma(a, 1.0000001f, 0.5f);\n"
     "  t.write(float4(fract(a), 0.25, 0.5, 1.0), g); }\n"
     "kernel void tread(texture2d<float, access::read> t [[texture(0)]],\n"
     "                  device float *s [[buffer(0)]], constant uint &n [[buffer(1)]],\n"
     "                  uint2 g [[thread_position_in_grid]]) {\n"
     "  float a = t.read(g).x;\n  for (uint i = 0; i < n; i++) a = fma(a, 1.0000001f, 0.5f);\n"
     "  s[g.x] += a; }\n"
     "kernel void bwrite(device float *b [[buffer(2)]], device float *s [[buffer(0)]],\n"
     "                   constant uint &n [[buffer(1)]], uint g [[thread_position_in_grid]]) {\n"
     "  float a = s[g];\n  for (uint i = 0; i < n; i++) a = fma(a, 1.0000001f, 0.5f);\n"
     "  b[g] = a; }\n"
     "kernel void bread(const device float *b [[buffer(2)]], device float *s [[buffer(0)]],\n"
     "                  constant uint &n [[buffer(1)]], uint g [[thread_position_in_grid]]) {\n"
     "  float a = b[g];\n  for (uint i = 0; i < n; i++) a = fma(a, 1.0000001f, 0.5f);\n"
     "  s[g] = a; }\n";

static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQ, gQ2;
static id<MTLComputePipelineState> gSpin, gTW, gTR, gBW, gBR;
static id<MTLBuffer> gScratch, gTiny;
static uint32_t gN = 1000;
static int gSched;

static void encodeSpin(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> p, id<MTLTexture> t,
                       id<MTLBuffer> b, uint32_t n, NSUInteger threads, int twoD)
{
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:p];
	[e setBuffer:(threads == 1 ? gTiny : gScratch) offset:0 atIndex:0];
	[e setBytes:&n length:4 atIndex:1];
	if (t)
		[e setTexture:t atIndex:0];
	if (b)
		[e setBuffer:b offset:0 atIndex:2];
	if (twoD)
		[e dispatchThreads:MTLSizeMake(64, 64, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
	else
		[e dispatchThreads:MTLSizeMake(threads, 1, 1)
		    threadsPerThreadgroup:MTLSizeMake(threads < 64 ? threads : 64, 1, 1)];
	[e endEncoding];
}

static void calibrate(double ms)
{
	for (int i = 0; i < 12; i++) {
		id<MTLCommandBuffer> cb = [gQ commandBuffer];
		encodeSpin(cb, gSpin, nil, nil, gN, 1024, 0);
		const double t0 = oracle_now_ms();
		[cb commit];
		[cb waitUntilCompleted];
		const double t = oracle_now_ms() - t0;
		if (t > ms * 0.8 && t < ms * 1.5)
			break;
		double f = t > 0.5 ? ms / t : 8;
		if (f > 8)
			f = 8;
		gN = (uint32_t)(gN * f) + 1;
	}
}

static void answer(const char *name, id r, id<MTLCommandBuffer> cb)
{
	if (gSched)
		[cb waitUntilScheduled];
	const BOOL c0 = [r isComplete], w0 = [r isWriteComplete];
	const MTLCommandBufferStatus s0 = cb.status;
	const double t0 = oracle_now_ms();
	[r waitUntilComplete];
	const double waited = oracle_now_ms() - t0;
	const MTLCommandBufferStatus s1 = cb.status;
	[cb waitUntilCompleted];
	const BOOL c1 = [r isComplete], w1 = [r isWriteComplete];
	printf("%-20s asked: isComplete %d isWriteComplete %d (buffer status %lu); waitUntilComplete "
	       "%.1f ms, buffer status after %lu; once complete: %d %d\n",
	       name, c0, w0, (unsigned long)s0, waited, (unsigned long)s1, c1, w1);
}

static id<MTLTexture> newTexture(MTLStorageMode mode, MTLTextureUsage extra)
{
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO];
	d.storageMode = mode;
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | extra;
	return [gDev newTextureWithDescriptor:d];
}

static void clearInto(id<MTLCommandBuffer> cb, id<MTLTexture> t)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = t;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 1);
	[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
}

static void answers(void)
{
	const MTLStorageMode modes[2] = {MTLStorageModeManaged, MTLStorageModePrivate};
	const char *modeNames[2] = {"managed", "private"};
	for (int m = 0; m < 2; m++) {
		char name[64];
		id<MTLTexture> t = newTexture(modes[m], 0);
		id<MTLCommandBuffer> cb = [gQ commandBuffer];
		encodeSpin(cb, gSpin, nil, nil, gN, 1024, 0);
		clearInto(cb, t);
		[cb commit];
		snprintf(name, sizeof name, "tex-render-%s", modeNames[m]);
		answer(name, t, cb);
		t = newTexture(modes[m], MTLTextureUsageShaderWrite);
		cb = [gQ commandBuffer];
		encodeSpin(cb, gTW, t, nil, gN, 0, 1);
		[cb commit];
		snprintf(name, sizeof name, "tex-write-%s", modeNames[m]);
		answer(name, t, cb);
		t = newTexture(modes[m], 0);
		cb = [gQ commandBuffer];
		encodeSpin(cb, gTR, t, nil, gN, 0, 1);
		[cb commit];
		snprintf(name, sizeof name, "tex-read-%s", modeNames[m]);
		answer(name, t, cb);
		t = newTexture(modes[m], 0);
		id<MTLBuffer> src = [gDev newBufferWithLength:64 * 64 * 4
		                                      options:MTLResourceStorageModeShared];
		cb = [gQ commandBuffer];
		encodeSpin(cb, gSpin, nil, nil, gN, 1024, 0);
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:src
		           sourceOffset:0
		      sourceBytesPerRow:256
		    sourceBytesPerImage:256 * 64
		             sourceSize:MTLSizeMake(64, 64, 1)
		              toTexture:t
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[blit endEncoding];
		[cb commit];
		snprintf(name, sizeof name, "tex-blit-%s", modeNames[m]);
		answer(name, t, cb);
		const MTLResourceOptions options =
		    m == 0 ? MTLResourceStorageModeManaged : MTLResourceStorageModePrivate;
		id<MTLBuffer> buffer = [gDev newBufferWithLength:4096 options:options];
		cb = [gQ commandBuffer];
		encodeSpin(cb, gBW, nil, buffer, gN, 1024, 0);
		[cb commit];
		snprintf(name, sizeof name, "buf-write-%s", modeNames[m]);
		answer(name, buffer, cb);
		buffer = [gDev newBufferWithLength:4096 options:options];
		cb = [gQ commandBuffer];
		encodeSpin(cb, gBR, nil, buffer, gN, 1024, 0);
		[cb commit];
		snprintf(name, sizeof name, "buf-read-%s", modeNames[m]);
		answer(name, buffer, cb);
	}
	{
		id<MTLTexture> t = newTexture(MTLStorageModeManaged, 0);
		id<MTLCommandBuffer> cb = [gQ commandBuffer];
		clearInto(cb, t);
		const BOOL c0 = [(id)t isComplete], w0 = [(id)t isWriteComplete];
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		const double t0 = oracle_now_ms();
		dispatch_async(dispatch_get_global_queue(0, 0), ^{
			[(id)t waitUntilComplete];
			dispatch_semaphore_signal(done);
		});
		const long timedOut =
		    dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
		printf("tex-uncommitted      before commit: isComplete %d isWriteComplete %d; "
		       "waitUntilComplete %s %.1f ms\n",
		       c0, w0, timedOut ? "still waiting after" : "returned in", oracle_now_ms() - t0);
		[cb commit];
		[cb waitUntilCompleted];
		if (timedOut)
			printf("tex-uncommitted      after the commit completed: the wait %s\n",
			       dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC))
			           ? "still waits"
			           : "returned");
	}
	{
		id<MTLTexture> t = newTexture(MTLStorageModeManaged, 0);
		id<MTLCommandBuffer> a = [gQ commandBuffer];
		clearInto(a, t);
		[a commit];
		[a waitUntilCompleted];
		id<MTLCommandBuffer> b = [gQ2 commandBuffer];
		encodeSpin(b, gSpin, nil, nil, gN, 1024, 0);
		[b commit];
		answer("tex-other-queue", t, b);
	}
}

static void xqueue(int rounds)
{
	const char *cases[] = {"empty",       "blit-empty",   "sync-managed",
	                       "sync-other",  "compute-tiny", "wuc-sync"};
	for (int c = 0; c < 6; c++) {
		int aDone = 0;
		double bSum = 0, aSum = 0;
		for (int r = 0; r < rounds; r++) {
			id<MTLTexture> drawn = newTexture(MTLStorageModeManaged, 0);
			id<MTLTexture> other = newTexture(MTLStorageModeManaged, 0);
			id<MTLCommandBuffer> a = [gQ commandBuffer];
			encodeSpin(a, gSpin, nil, nil, gN, 1024, 0);
			clearInto(a, drawn);
			const double t0 = oracle_now_ms();
			[a commit];
			[a waitUntilScheduled];
			id<MTLCommandBuffer> b = [gQ2 commandBuffer];
			if (c == 5)
				[(id)drawn waitUntilComplete];
			if (c == 1 || c == 2 || c == 3 || c == 5) {
				id<MTLBlitCommandEncoder> e = [b blitCommandEncoder];
				if (c == 2 || c == 5)
					[e synchronizeTexture:drawn slice:0 level:0];
				if (c == 3)
					[e synchronizeTexture:other slice:0 level:0];
				[e endEncoding];
			} else if (c == 4) {
				encodeSpin(b, gSpin, nil, nil, 1, 1, 0);
			}
			[b commit];
			[b waitUntilCompleted];
			const double tb = oracle_now_ms() - t0;
			aDone += a.status == MTLCommandBufferStatusCompleted;
			[a waitUntilCompleted];
			bSum += tb;
			aSum += oracle_now_ms() - t0;
		}
		printf("%-13s A completed when B's wait returned: %d of %d; B's wait returned %.1f ms "
		       "after A's commit, A's %.1f ms (means)\n",
		       cases[c], aDone, rounds, bSum / rounds, aSum / rounds);
	}
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		setvbuf(stdout, NULL, _IOLBF, 0);
		if (argc < 3 || (strcmp(argv[2], "answers") != 0 && strcmp(argv[2], "xqueue") != 0)) {
			fprintf(stderr, "usage: %s DEVICE answers [MS [sched]] | DEVICE xqueue [MS [ROUNDS]]\n",
			        argv[0]);
			return 2;
		}
		gDev = oracle_device(argv[1]);
		const int xq = strcmp(argv[2], "xqueue") == 0;
		const double ms = argc > 3 ? atof(argv[3]) : (xq ? 200 : 300);
		gSched = !xq && argc > 4 && strcmp(argv[4], "sched") == 0;
		const int rounds = xq && argc > 4 ? atoi(argv[4]) : 5;
		oracle_header("oracle_complete", gDev);
		NSError *err = nil;
		id<MTLLibrary> lib = [gDev newLibraryWithSource:kSource options:nil error:&err];
		if (!lib) {
			printf("library: %s\n", err.description.UTF8String);
			return 1;
		}
		NSString *names[] = {@"spin", @"twrite", @"tread", @"bwrite", @"bread"};
		id<MTLComputePipelineState> made[5];
		for (int i = 0; i < 5; i++) {
			made[i] = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]]
			                                              error:&err];
			if (!made[i]) {
				printf("pipeline %s: %s\n", names[i].UTF8String, err.description.UTF8String);
				return 1;
			}
		}
		gSpin = made[0], gTW = made[1], gTR = made[2], gBW = made[3], gBR = made[4];
		gQ = [gDev newCommandQueue];
		gQ2 = [gDev newCommandQueue];
		gScratch = [gDev newBufferWithLength:64 * 64 * 4 options:MTLResourceStorageModePrivate];
		gTiny = [gDev newBufferWithLength:256 options:MTLResourceStorageModePrivate];
		calibrate(ms);
		printf("spin of %u steps; %s\n", gN,
		       xq ? "queue B's buffer after queue A's is scheduled"
		          : (gSched ? "each answer once the buffer is scheduled (waitUntilScheduled)"
		                    : "each answer right after the commit"));
		if (xq)
			xqueue(rounds);
		else
			answers();
		printf("done\n");
	}
	return 0;
}
