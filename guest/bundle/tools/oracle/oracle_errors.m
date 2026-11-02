// oracle_errors: the errors and refusals Metal gives for common failures, with the whole NSError
// (domain, code, userInfo, the underlying error and the encoders' execution status).
//   oracle_errors DEVICE [CASE [ARG]]   every case but timeout (or the one named), each in a child
// Cases:
//   purgeable     a buffer and a texture set purgeable Empty, then used by a command buffer
//   creates       oversized creates: a buffer past maxBufferLength, a 32768 x 32768 texture, a
//                 heap past the working set
//   residency GIB GIB 1 GiB Private buffers made and all filled in one command buffer
//   timeout SEC   one kernel that runs about SEC seconds (the GPU's watchdog may end it); run it
//                 only on a GPU that does not drive a display
// Build: see README.md.
#import "oracle_common.h"
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
static const char *gArgv0;

static void print_error(const char *indent, NSError *e)
{
	if (!e) {
		printf("%serror nil\n", indent);
		return;
	}
	printf("%sdomain %s code %ld\n", indent, e.domain.UTF8String, (long)e.code);
	for (NSString *k in e.userInfo) {
		id v = e.userInfo[k];
		if ([v isKindOfClass:[NSError class]]) {
			printf("%s  %s:\n", indent, k.UTF8String);
			char deeper[32];
			snprintf(deeper, sizeof deeper, "%s    ", indent);
			print_error(deeper, v);
		} else if ([v isKindOfClass:[NSArray class]]) {
			printf("%s  %s: [%lu]\n", indent, k.UTF8String, (unsigned long)[v count]);
			for (id x in v) {
				if ([x conformsToProtocol:@protocol(MTLCommandBufferEncoderInfo)]) {
					id<MTLCommandBufferEncoderInfo> info = x;
					printf("%s    encoder '%s' status %ld signposts %lu\n", indent,
					       info.label.UTF8String, (long)info.errorState,
					       (unsigned long)info.debugSignposts.count);
				} else {
					printf("%s    %s\n", indent, [x description].UTF8String);
				}
			}
		} else {
			printf("%s  %s: %s\n", indent, k.UTF8String, [v description].UTF8String);
		}
	}
}

static id<MTLCommandBuffer> new_cb(void)
{
	MTLCommandBufferDescriptor *cd = [MTLCommandBufferDescriptor new];
	cd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
	return [Q commandBufferWithDescriptor:cd];
}

static void finish(const char *what, id<MTLCommandBuffer> cb)
{
	double t0 = oracle_now_ms();
	[cb commit];
	[cb waitUntilCompleted];
	printf("  %s: status %ld after %.1f ms (GPU %.1f ms)\n", what, (long)cb.status,
	       oracle_now_ms() - t0, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
	print_error("    ", cb.error);
}

static void case_purgeable(void)
{
	printf("case purgeable\n");
	id<MTLBuffer> b = [D newBufferWithLength:1 << 20 options:MTLResourceStorageModePrivate];
	MTLPurgeableState old = [b setPurgeableState:MTLPurgeableStateVolatile];
	MTLPurgeableState was = [b setPurgeableState:MTLPurgeableStateEmpty];
	MTLPurgeableState now = [b setPurgeableState:MTLPurgeableStateKeepCurrent];
	printf("  buffer purgeable states: %lu -> %lu -> now %lu\n", (unsigned long)old,
	       (unsigned long)was, (unsigned long)now);
	id<MTLBuffer> out = [D newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = new_cb();
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	e.label = @"copy from emptied buffer";
	[e copyFromBuffer:b sourceOffset:0 toBuffer:out destinationOffset:0 size:1 << 20];
	[e endEncoding];
	finish("copy from an emptied buffer", cb);
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:256
	                                                      height:256
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	id<MTLTexture> t = [D newTextureWithDescriptor:td];
	[t setPurgeableState:MTLPurgeableStateVolatile];
	[t setPurgeableState:MTLPurgeableStateEmpty];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = t;
	rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	cb = new_cb();
	id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:rp];
	r.label = @"pass on emptied texture";
	[r endEncoding];
	finish("render pass on an emptied texture", cb);
}

static void case_creates(void)
{
	printf("case creates (maxBufferLength %lu)\n", (unsigned long)D.maxBufferLength);
	printf("  buffer of maxBufferLength + 4096: ");
	fflush(stdout);
	id<MTLBuffer> b = [D newBufferWithLength:D.maxBufferLength + 4096
	                                 options:MTLResourceStorageModePrivate];
	printf("%s\n", b ? "made" : "nil");
	b = nil;
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = D.recommendedMaxWorkingSetSize * 4;
	hd.storageMode = MTLStorageModePrivate;
	printf("  Private heap of 4x recommendedMaxWorkingSetSize (%llu): ", (unsigned long long)hd.size);
	fflush(stdout);
	id<MTLHeap> h = [D newHeapWithDescriptor:hd];
	printf("%s\n", h ? "made" : "nil");
	h = nil;
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:32768
	                                                      height:32768
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	printf("  32768x32768 RGBA8 texture: ");
	fflush(stdout);
	id<MTLTexture> t = [D newTextureWithDescriptor:td];
	printf("%s\n", t ? "made" : "nil");
}

static void case_residency(int gib)
{
	printf("case residency: %d x 1 GiB Private buffers, all filled in one command buffer\n", gib);
	NSMutableArray *bufs = [NSMutableArray array];
	for (int i = 0; i < gib; i++) {
		id<MTLBuffer> b = [D newBufferWithLength:1ull << 30 options:MTLResourceStorageModePrivate];
		if (!b) {
			printf("  buffer %d: nil\n", i);
			break;
		}
		[bufs addObject:b];
	}
	printf("  made %lu, currentAllocatedSize %lu\n", (unsigned long)bufs.count,
	       (unsigned long)D.currentAllocatedSize);
	id<MTLCommandBuffer> cb = new_cb();
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	e.label = @"fill all";
	for (id<MTLBuffer> b in bufs)
		[e fillBuffer:b range:NSMakeRange(0, b.length) value:1];
	[e endEncoding];
	finish("fill every buffer", cb);
}

static void case_timeout(double seconds)
{
	printf("case timeout: one kernel of about %.1f s\n", seconds);
	id<MTLLibrary> lib = oracle_library(D, gArgv0, "oracle");
	NSError *err = nil;
	id<MTLComputePipelineState> p =
	    [D newComputePipelineStateWithFunction:[lib newFunctionWithName:@"progress"] error:&err];
	id<MTLBuffer> ctr = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
	id<MTLBuffer> sink = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
	// Calibrate: 100 steps of `spin`.
	uint32_t steps = 100, spin = 20000;
	double gpu = 0;
	for (int k = 0; k < 5; k++) {
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:p];
		[e setBuffer:ctr offset:0 atIndex:0];
		[e setBytes:&steps length:4 atIndex:1];
		[e setBytes:&spin length:4 atIndex:2];
		[e setBuffer:sink offset:0 atIndex:3];
		[e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[e endEncoding];
		oracle_commit_wait(cb, NULL, &gpu);
		if (gpu > 50)
			break;
		spin *= 4;
	}
	steps = (uint32_t)(100 * seconds * 1000 / gpu);
	printf("  calibration: 100 steps of %u = %.1f ms; %u steps\n", spin, gpu, steps);
	memset(ctr.contents, 0, 64);
	id<MTLCommandBuffer> cb = new_cb();
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	e.label = @"long kernel";
	[e setComputePipelineState:p];
	[e setBuffer:ctr offset:0 atIndex:0];
	[e setBytes:&steps length:4 atIndex:1];
	[e setBytes:&spin length:4 atIndex:2];
	[e setBuffer:sink offset:0 atIndex:3];
	[e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[e endEncoding];
	finish("long kernel", cb);
	printf("  the kernel reached step %u of %u\n", ((uint32_t *)ctr.contents)[0], steps);
	cb = new_cb();
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b fillBuffer:sink range:NSMakeRange(0, 64) value:3];
	[b endEncoding];
	finish("a tiny buffer on the same queue after it", cb);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	gArgv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE [ARG]]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 2 && !strcmp(argv[2], "--case")) {
		Q = [D newCommandQueue];
		const char *c = argv[3];
		if (!strcmp(c, "purgeable"))
			case_purgeable();
		else if (!strcmp(c, "creates"))
			case_creates();
		else if (!strcmp(c, "residency"))
			case_residency(argc > 4 && argv[4] ? atoi(argv[4]) : 6);
		else if (!strcmp(c, "timeout"))
			case_timeout(argc > 4 && argv[4] ? atof(argv[4]) : 10);
		else
			return 2;
		return 0;
	}
	oracle_header("oracle_errors", D);
	D = nil;
	static const char *const all[][2] = {
	    {"purgeable", NULL}, {"creates", NULL}, {"residency", "6"}, {"timeout", "10"},
	};
	for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
		if (argc > 2 ? strcmp(argv[2], all[i][0]) : !strcmp(all[i][0], "timeout"))
			continue;
		const char *arg = argc > 3 ? argv[3] : all[i][1];
		char *args[] = {argv[0], argv[1], "--case", (char *)all[i][0], (char *)arg, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("  => case %s died on signal %d\n", all[i][0], WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("  => case %s exited %d\n", all[i][0], WEXITSTATUS(st));
	}
	return 0;
}
