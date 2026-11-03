// pipelines: what shader libraries and pipeline states cost on a Metal device, cold and warm, and
// the stall the first draw or dispatch of a new pipeline takes, to compare a guest's GPU lane
// with real Macs. Every case prints the median of its numbers in ms (and the spread of the
// first); a cold case makes something no cache has seen, a warm one the same thing again.
//   pipelines [-d NAME] [-r REPS] [-s SEED] [-L DIR] [-A FILE] [-l] [CASE...]
//     -d NAME   the first of MTLCopyAllDevices whose name holds NAME (default
//               MTLCreateSystemDefaultDevice; pass -d on a dual-GPU MacBook).
//     -r REPS   repetitions per case (default 5).
//     -s SEED   the nonce base, printed in the first line (default: from the clock). Each cold
//               repetition's library is the source with nonce SEED * 4096 + n, so the same SEED
//               in a later process, a reboot or another VM makes the same libraries again: run
//               once with a fresh SEED (cold everywhere), then again with it (warm where a cache
//               outlived the process).
//     -L DIR    the metallibs `make pipelines` writes (pipelines-NN.metallib, one nonce each, of a
//               seed fixed at build time), for the lib.data cases (default: the tool's directory).
//     -A FILE   the binary archive file of archive.file (default /tmp/pipelines-SEED.archive).
//     -l        lists the devices and exits.
//     CASE      runs only the cases whose name holds one of these.
// The cases (ms each; "cold" a fresh nonce, "warm" the same object again, in this process):
//   lib.source.small, lib.source.large  newLibraryWithSource:options:error: of a source holding
//             only the quad's functions, and of the whole source: cold, then warm.
//   lib.source.async  the completion-handler form, cold: the call's return, and its handler.
//   lib.data         newLibraryWithData: of each metallib of -L: the first load in this process
//             (cold here; across processes it says whether a cache outlived the last one), then
//             the same bytes again.
//   lib.system       newLibraryWithURL: of system metallibs Apple's frameworks load (those of
//             kSystemLibraries present on this OS): the first load, then again.
//   fn.const         newFunctionWithName:constantValues:error: of a fresh tint value, sync, and
//             the completion-handler form.
//   rp.lib           a render pipeline (v_quad + f_tex, vertex descriptor, BGRA8, blended) of a
//             fresh library each repetition (library made outside the timing): a new shader's
//             first pipeline. rp.lib.pbr: the same with f_pbr.
//   rp.const         the same library, a fresh tint each repetition: a new specialization.
//   rp.state         the same functions, the next of 8 colour-attachment states (format, blend,
//             write mask) each repetition.
//   rp.warm          one descriptor made again.
//   rp.async.8       8 pipelines of fresh libraries asked at once through the completion-handler
//             form: the wall time until the last handler, and the median handler latency.
//   cp.lib, cp.const, cp.warm, cp.async.8   the same for k_blur's compute pipeline (its weights
//             a function constant).
//   draw.first       a pipeline of a fresh library, then a command buffer of one draw into a
//             256 x 256 target, commit to completed (first), then the same buffer again (next):
//             the first draw's stall is first - next.
//   draw.topology    a pipeline drawn with triangles, then its first line draw, and its next.
//   dispatch.first   the same as draw.first for k_blur's dispatch of a 256 x 256 grid.
//   archive.add      an empty binary archive: addRenderPipelineFunctionsWithDescriptor: of a fresh
//             library's pipeline (add), then that pipeline made with the archive and
//             MTLPipelineOptionFailOnBinaryArchiveMiss (hit), and a fresh library's made with no
//             archive (none).
//   archive.file     the archive of -A: where the file exists, loads it and makes its pipeline
//             (SEED's library) with FailOnBinaryArchiveMiss (hit from disk: what an app that ships
//             an archive gets); else makes and serializes it (written).
// One line a case: "pipelines: NAME OK key=value ...", or FAIL / SKIP; "# start NAME" before it.
// Build: make pipelines (build/pipelines-x86_64 and build/pipelines-metallibs/, a guest's or an
// Intel Mac's); make pipelines ARCH=arm64 for an Apple-silicon Mac.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "pipelines_source.h"

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static NSArray<NSString *> *gPatterns;
static int gReps = 5;
static uint32_t gSeed;
static uint32_t gNext; // the next nonce's low part
static NSString *gLibDir, *gArchivePath;
static NSString *gSource;

// Metallibs Apple's frameworks load at their first use, on macOS 26 and 27 alike where present.
static NSString *const kSystemLibraries[] = {
	@"/System/Library/Frameworks/CoreImage.framework/Versions/A/Resources/default.metallib",
	@"/System/Library/Frameworks/CoreImage.framework/Versions/A/Resources/ci_filters.metallib",
	@"/System/Library/Frameworks/CoreImage.framework/Versions/A/Resources/ci_stdlib.metallib",
	@"/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/"
	@"MPSImage.framework/Versions/A/Resources/default.metallib",
	@"/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/"
	@"MPSNeuralNetwork.framework/Versions/A/Resources/default.metallib",
	@"/System/Library/Frameworks/QuartzCore.framework/Versions/A/Resources/default.metallib",
	@"/System/Library/PrivateFrameworks/RenderBox.framework/Versions/A/Resources/default.metallib",
};

// --- helpers ------------------------------------------------------------------------------------

static double now(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static double ms(double since)
{
	return (now() - since) * 1e3;
}

static double median(NSArray<NSNumber *> *xs)
{
	if (xs.count == 0)
		return NAN;
	NSArray *s = [xs sortedArrayUsingSelector:@selector(compare:)];
	const NSUInteger n = s.count;
	return n % 2 ? [s[n / 2] doubleValue] : ([s[n / 2 - 1] doubleValue] + [s[n / 2] doubleValue]) / 2;
}

// "NAME=MEDIAN" with the values' spread, "NAME=MEDIAN(MIN..MAX)", in ms.
static NSString *stat(NSString *name, NSArray<NSNumber *> *xs)
{
	if (xs.count == 0)
		return [name stringByAppendingString:@"=none"];
	return [NSString stringWithFormat:@"%@=%.3f(%.3f..%.3f)", name, median(xs),
	                                  [[xs valueForKeyPath:@"@min.self"] doubleValue],
	                                  [[xs valueForKeyPath:@"@max.self"] doubleValue]];
}

static BOOL wanted(NSString *name)
{
	if (gPatterns.count == 0)
		return YES;
	for (NSString *p in gPatterns)
		if ([name containsString:p])
			return YES;
	return NO;
}

static void runCase(NSString *name, NSString * (^body)(void))
{
	if (!wanted(name))
		return;
	printf("# start %s\n", name.UTF8String);
	fflush(stdout);
	const double t0 = now();
	NSString *line = nil;
	@try {
		@autoreleasepool {
			line = body();
		}
	} @catch (NSException *e) {
		line = [@"FAIL " stringByAppendingString:e.reason ?: e.name];
	}
	if (![line hasPrefix:@"FAIL "] && ![line hasPrefix:@"SKIP "])
		line = [@"OK " stringByAppendingString:line];
	printf("pipelines: %s %s case_s=%.1f\n", name.UTF8String, line.UTF8String, now() - t0);
	fflush(stdout);
}

static uint32_t freshNonce(void)
{
	return gSeed * 4096u + (gNext++ & 4095u);
}

// The source of nonce `nonce`; `small` keeps only the quad's functions.
static NSString *sourceOf(uint32_t nonce, BOOL small)
{
	NSString *body = gSource;
	if (small) {
		// The quad's vertex and fragment functions are the source up to f_pbr's helpers.
		NSRange r = [body rangeOfString:@"struct Light {"];
		if (r.location != NSNotFound)
			body = [body substringToIndex:r.location];
	}
	return [NSString stringWithFormat:@"#define RGFX_NONCE %uu\n%@", nonce, body];
}

static id<MTLLibrary> libraryOf(uint32_t nonce, BOOL small)
{
	NSError *err = nil;
	id<MTLLibrary> lib = [gDevice newLibraryWithSource:sourceOf(nonce, small) options:nil error:&err];
	if (lib == nil)
		[NSException raise:@"pipelines" format:@"library %u: %@", nonce, err.localizedDescription];
	return lib;
}

static MTLFunctionConstantValues *tint(float t, BOOL fade)
{
	MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
	[v setConstantValue:&t type:MTLDataTypeFloat atIndex:0];
	[v setConstantValue:&fade type:MTLDataTypeBool atIndex:1];
	return v;
}

static MTLFunctionConstantValues *sharpness(float s)
{
	MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
	[v setConstantValue:&s type:MTLDataTypeFloat atIndex:2];
	return v;
}

static id<MTLFunction> function(id<MTLLibrary> lib, NSString *name, MTLFunctionConstantValues *values)
{
	NSError *err = nil;
	id<MTLFunction> f = values ? [lib newFunctionWithName:name constantValues:values error:&err]
	                           : [lib newFunctionWithName:name];
	if (f == nil)
		[NSException raise:@"pipelines" format:@"function %@: %@", name, err.localizedDescription ?: @"none"];
	return f;
}

// The 8 colour-attachment states rp.state steps through.
static void colourState(MTLRenderPipelineColorAttachmentDescriptor *c, int which)
{
	static const MTLPixelFormat formats[] = {MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA8Unorm,
	                                         MTLPixelFormatRGBA16Float, MTLPixelFormatBGRA8Unorm_sRGB};
	c.pixelFormat = formats[which % 4];
	c.blendingEnabled = YES;
	c.sourceRGBBlendFactor = (which / 4) ? MTLBlendFactorOne : MTLBlendFactorSourceAlpha;
	c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	c.sourceAlphaBlendFactor = MTLBlendFactorOne;
	c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	c.writeMask = (which / 4) ? MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue
	                          : MTLColorWriteMaskAll;
}

static MTLRenderPipelineDescriptor *quadDescriptor(id<MTLLibrary> lib, NSString *fragment,
                                                   MTLFunctionConstantValues *values, int state)
{
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = function(lib, @"v_quad", nil);
	d.fragmentFunction = function(lib, fragment, values ?: tint(1.0f, NO));
	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	vd.attributes[0].format = MTLVertexFormatFloat2;
	vd.attributes[0].offset = 0;
	vd.attributes[0].bufferIndex = 0;
	vd.attributes[1].format = MTLVertexFormatFloat2;
	vd.attributes[1].offset = 8;
	vd.attributes[1].bufferIndex = 0;
	vd.layouts[0].stride = 16;
	d.vertexDescriptor = vd;
	colourState(d.colorAttachments[0], state);
	return d;
}

static id<MTLRenderPipelineState> renderPipeline(MTLRenderPipelineDescriptor *d, MTLPipelineOption options)
{
	NSError *err = nil;
	id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:d
	                                                                     options:options
	                                                                  reflection:nil
	                                                                       error:&err];
	if (p == nil)
		[NSException raise:@"pipelines" format:@"render pipeline: %@", err.localizedDescription ?: @"nil"];
	return p;
}

static id<MTLComputePipelineState> computePipeline(id<MTLFunction> f)
{
	NSError *err = nil;
	id<MTLComputePipelineState> p = [gDevice newComputePipelineStateWithFunction:f error:&err];
	if (p == nil)
		[NSException raise:@"pipelines" format:@"compute pipeline: %@", err.localizedDescription ?: @"nil"];
	return p;
}

static void finish(id<MTLCommandBuffer> cb)
{
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		[NSException raise:@"pipelines" format:@"command buffer status %ld: %@", (long)cb.status,
		                                       cb.error.localizedDescription ?: @"no error"];
}

static id<MTLTexture> texture(MTLPixelFormat format, NSUInteger size, MTLTextureUsage usage)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                             width:size
	                                                                            height:size
	                                                                         mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = usage;
	return [gDevice newTextureWithDescriptor:d];
}

// What a draw of the quad pipeline binds.
typedef struct {
	float mvp[16];
	float color[4];
	float time;
	uint32_t lights;
	uint32_t pad[2];
} Frame;

static id<MTLBuffer> gQuad, gFrame;
static id<MTLTexture> gSampled, gTarget, gWritten;
static id<MTLSamplerState> gSampler;

static void setupDrawing(void)
{
	if (gQuad != nil)
		return;
	const float quad[] = {-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1};
	gQuad = [gDevice newBufferWithBytes:quad length:sizeof quad options:MTLResourceStorageModeShared];
	Frame f = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {1, 1, 1, 1}, 0, 0, {0, 0}};
	gFrame = [gDevice newBufferWithBytes:&f length:sizeof f options:MTLResourceStorageModeShared];
	gSampled = texture(MTLPixelFormatRGBA8Unorm, 256, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
	gTarget = texture(MTLPixelFormatBGRA8Unorm, 256, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
	gWritten = texture(MTLPixelFormatRGBA8Unorm, 256, MTLTextureUsageShaderWrite);
	MTLSamplerDescriptor *s = [MTLSamplerDescriptor new];
	s.minFilter = s.magFilter = MTLSamplerMinMagFilterLinear;
	gSampler = [gDevice newSamplerStateWithDescriptor:s];
}

// One command buffer of one draw of `p` into the target: commit to completed, in ms.
static double drawOnce(id<MTLRenderPipelineState> p, MTLPrimitiveType type)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = gTarget;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e setRenderPipelineState:p];
	[e setVertexBuffer:gQuad offset:0 atIndex:0];
	[e setVertexBuffer:gFrame offset:0 atIndex:1];
	[e setFragmentBuffer:gFrame offset:0 atIndex:0];
	[e setFragmentTexture:gSampled atIndex:0];
	[e setFragmentSamplerState:gSampler atIndex:0];
	[e drawPrimitives:type vertexStart:0 vertexCount:4];
	[e endEncoding];
	const double t0 = now();
	[cb commit];
	finish(cb);
	return ms(t0);
}

static double dispatchOnce(id<MTLComputePipelineState> p)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:p];
	[e setTexture:gSampled atIndex:0];
	[e setTexture:gWritten atIndex:1];
	[e dispatchThreadgroups:MTLSizeMake(16, 16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[e endEncoding];
	const double t0 = now();
	[cb commit];
	finish(cb);
	return ms(t0);
}

// --- the cases ----------------------------------------------------------------------------------

static NSString *libSource(BOOL small)
{
	NSMutableArray *cold = [NSMutableArray array], *warm = [NSMutableArray array];
	const uint32_t nonce = freshNonce();
	for (int r = 0; r < gReps; r++) {
		const uint32_t n = r == 0 ? nonce : freshNonce();
		NSString *src = sourceOf(n, small);
		const double t0 = now();
		NSError *err = nil;
		id<MTLLibrary> lib = [gDevice newLibraryWithSource:src options:nil error:&err];
		if (lib == nil)
			return [@"FAIL " stringByAppendingString:err.localizedDescription ?: @"nil"];
		[cold addObject:@(ms(t0))];
	}
	NSString *src = sourceOf(nonce, small);
	for (int r = 0; r < gReps; r++) {
		const double t0 = now();
		id<MTLLibrary> lib = [gDevice newLibraryWithSource:src options:nil error:nil];
		if (lib == nil)
			return @"FAIL warm library nil";
		[warm addObject:@(ms(t0))];
	}
	return [NSString stringWithFormat:@"%@ %@ source_bytes=%lu", stat(@"cold_ms", cold), stat(@"warm_ms", warm),
	                                  (unsigned long)src.length];
}

static NSString *libSourceAsync(void)
{
	NSMutableArray *ret = [NSMutableArray array], *done = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		dispatch_semaphore_t sem = dispatch_semaphore_create(0);
		__block double handled = 0;
		__block BOOL ok = NO;
		NSString *src = sourceOf(freshNonce(), NO);
		const double t0 = now();
		[gDevice newLibraryWithSource:src
		                      options:nil
		            completionHandler:^(id<MTLLibrary> lib, NSError *err) {
			            handled = now();
			            ok = lib != nil;
			            dispatch_semaphore_signal(sem);
		            }];
		[ret addObject:@(ms(t0))];
		dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
		if (!ok)
			return @"FAIL a library nil";
		[done addObject:@((handled - t0) * 1e3)];
	}
	return [NSString stringWithFormat:@"%@ %@", stat(@"return_ms", ret), stat(@"handler_ms", done)];
}

static NSString *libData(void)
{
	NSArray<NSString *> *names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:gLibDir error:nil]
	    filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"SELF ENDSWITH '.metallib' AND SELF BEGINSWITH 'pipelines-'"]];
	if (names.count == 0)
		return [NSString stringWithFormat:@"SKIP no pipelines-*.metallib in %@", gLibDir];
	names = [names sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray *cold = [NSMutableArray array], *warm = [NSMutableArray array];
	NSMutableArray *coldRp = [NSMutableArray array];
	NSUInteger bytes = 0;
	for (NSString *name in names) {
		NSData *data = [NSData dataWithContentsOfFile:[gLibDir stringByAppendingPathComponent:name]];
		bytes = data.length;
		dispatch_data_t dd = dispatch_data_create(data.bytes, data.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		NSError *err = nil;
		double t0 = now();
		id<MTLLibrary> lib = [gDevice newLibraryWithData:dd error:&err];
		if (lib == nil)
			return [NSString stringWithFormat:@"FAIL %@: %@", name, err.localizedDescription];
		[cold addObject:@(ms(t0))];
		t0 = now();
		id<MTLLibrary> again = [gDevice newLibraryWithData:dd error:&err];
		if (again == nil)
			return [NSString stringWithFormat:@"FAIL %@ again: %@", name, err.localizedDescription];
		[warm addObject:@(ms(t0))];
		// The library's first pipeline: across processes, whether a cache kept its compile.
		MTLRenderPipelineDescriptor *d = quadDescriptor(lib, @"f_tex", nil, 0);
		t0 = now();
		renderPipeline(d, MTLPipelineOptionNone);
		[coldRp addObject:@(ms(t0))];
	}
	return [NSString stringWithFormat:@"%@ %@ %@ libraries=%lu bytes=%lu", stat(@"first_ms", cold),
	                                  stat(@"again_ms", warm), stat(@"first_rp_ms", coldRp),
	                                  (unsigned long)names.count, (unsigned long)bytes];
}

static NSString *libSystem(void)
{
	NSMutableArray *parts = [NSMutableArray array];
	for (size_t i = 0; i < sizeof kSystemLibraries / sizeof kSystemLibraries[0]; i++) {
		NSString *path = kSystemLibraries[i];
		if (![[NSFileManager defaultManager] fileExistsAtPath:path])
			continue;
		NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
		NSURL *url = [NSURL fileURLWithPath:path];
		NSError *err = nil;
		double t0 = now();
		id<MTLLibrary> lib = [gDevice newLibraryWithURL:url error:&err];
		const double first = ms(t0);
		t0 = now();
		id<MTLLibrary> again = [gDevice newLibraryWithURL:url error:nil];
		const double second = ms(t0);
		NSString *short_ = [[path stringByReplacingOccurrencesOfString:@"/System/Library/" withString:@""]
		    stringByReplacingOccurrencesOfString:@"/Versions/A/Resources/" withString:@":"];
		short_ = [short_ stringByReplacingOccurrencesOfString:@"Frameworks/" withString:@""];
		short_ = [short_ stringByReplacingOccurrencesOfString:@"PrivateFrameworks/" withString:@""];
		short_ = [short_ stringByReplacingOccurrencesOfString:@".framework" withString:@""];
		short_ = [short_ stringByReplacingOccurrencesOfString:@"/Versions/A/" withString:@"/"];
		if (lib == nil || again == nil)
			[parts addObject:[NSString stringWithFormat:@"%@=nil(%@)", short_,
			                                            [err.localizedDescription stringByReplacingOccurrencesOfString:@" " withString:@"_"]]];
		else
			[parts addObject:[NSString stringWithFormat:@"%@=%.2f/%.2f(%lluKiB,%lufn)", short_, first, second,
			                                            [attrs[NSFileSize] unsignedLongLongValue] >> 10,
			                                            (unsigned long)lib.functionNames.count]];
	}
	if (parts.count == 0)
		return @"SKIP none of the system metallibs is present";
	return [parts componentsJoinedByString:@" "];
}

static NSString *fnConst(void)
{
	id<MTLLibrary> lib = libraryOf(freshNonce(), YES);
	NSMutableArray *sync = [NSMutableArray array], *async = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		MTLFunctionConstantValues *v = tint(1.0f + 0.001f * (float)(gNext++), r & 1);
		const double t0 = now();
		function(lib, @"f_tex", v);
		[sync addObject:@(ms(t0))];
	}
	for (int r = 0; r < gReps; r++) {
		MTLFunctionConstantValues *v = tint(2.0f + 0.001f * (float)(gNext++), r & 1);
		dispatch_semaphore_t sem = dispatch_semaphore_create(0);
		__block double handled = 0;
		__block BOOL ok = NO;
		const double t0 = now();
		[lib newFunctionWithName:@"f_tex"
		          constantValues:v
		       completionHandler:^(id<MTLFunction> f, NSError *err) {
			       handled = now();
			       ok = f != nil;
			       dispatch_semaphore_signal(sem);
		       }];
		dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
		if (!ok)
			return @"FAIL an async function nil";
		[async addObject:@((handled - t0) * 1e3)];
	}
	return [NSString stringWithFormat:@"%@ %@", stat(@"sync_ms", sync), stat(@"async_ms", async)];
}

// Render pipelines: `kind` 0 a fresh library each, 1 a fresh tint, 2 the next state, 3 the same.
static NSString *renderCase(int kind, NSString *fragment)
{
	NSMutableArray *times = [NSMutableArray array];
	id<MTLLibrary> shared = libraryOf(freshNonce(), NO);
	MTLRenderPipelineDescriptor *same = quadDescriptor(shared, fragment, nil, 0);
	if (kind == 3)
		renderPipeline(same, MTLPipelineOptionNone); // the first, untimed
	for (int r = 0; r < gReps; r++) {
		MTLRenderPipelineDescriptor *d = same;
		if (kind == 0)
			d = quadDescriptor(libraryOf(freshNonce(), NO), fragment, nil, 0);
		else if (kind == 1)
			d = quadDescriptor(shared, fragment, tint(1.0f + 0.001f * (float)(gNext++), NO), 0);
		else if (kind == 2)
			d = quadDescriptor(shared, fragment, nil, (r + 1) % 8);
		const double t0 = now();
		renderPipeline(d, MTLPipelineOptionNone);
		[times addObject:@(ms(t0))];
	}
	return stat(@"ms", times);
}

static NSString *computeCase(int kind)
{
	NSMutableArray *times = [NSMutableArray array];
	id<MTLLibrary> shared = libraryOf(freshNonce(), NO);
	id<MTLFunction> same = function(shared, @"k_blur", sharpness(1.0f));
	if (kind == 3)
		computePipeline(same);
	for (int r = 0; r < gReps; r++) {
		id<MTLFunction> f = same;
		if (kind == 0)
			f = function(libraryOf(freshNonce(), NO), @"k_blur", sharpness(1.0f));
		else if (kind == 1)
			f = function(shared, @"k_blur", sharpness(1.0f + 0.001f * (float)(gNext++)));
		const double t0 = now();
		computePipeline(f);
		[times addObject:@(ms(t0))];
	}
	return stat(@"ms", times);
}

// `count` pipelines of fresh libraries asked at once: the wall time to the last handler, and each
// handler's latency.
static NSString *asyncCase(BOOL compute, int count)
{
	NSMutableArray *walls = [NSMutableArray array], *latencies = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		NSMutableArray *descs = [NSMutableArray array];
		for (int i = 0; i < count; i++) {
			id<MTLLibrary> lib = libraryOf(freshNonce(), NO);
			[descs addObject:compute ? (id)function(lib, @"k_blur", sharpness(1.0f))
			                         : (id)quadDescriptor(lib, @"f_tex", nil, 0)];
		}
		dispatch_group_t group = dispatch_group_create();
		NSMutableArray *ends = [NSMutableArray array];
		__block int failed = 0;
		NSLock *lock = [NSLock new];
		const double t0 = now();
		for (id d in descs) {
			dispatch_group_enter(group);
			void (^done)(id, NSError *) = ^(id state, NSError *err) {
				[lock lock];
				[ends addObject:@((now() - t0) * 1e3)];
				if (state == nil)
					failed++;
				[lock unlock];
				dispatch_group_leave(group);
			};
			if (compute)
				[gDevice newComputePipelineStateWithFunction:d completionHandler:done];
			else
				[gDevice newRenderPipelineStateWithDescriptor:d completionHandler:done];
		}
		dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
		const double wall = ms(t0);
		if (failed)
			return [NSString stringWithFormat:@"FAIL %d of %d pipelines nil", failed, count];
		[walls addObject:@(wall)];
		[latencies addObjectsFromArray:ends];
	}
	return [NSString stringWithFormat:@"%@ %@", stat(@"wall_ms", walls), stat(@"handler_ms", latencies)];
}

static NSString *drawFirst(void)
{
	setupDrawing();
	NSMutableArray *create = [NSMutableArray array], *first = [NSMutableArray array], *next = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		MTLRenderPipelineDescriptor *d = quadDescriptor(libraryOf(freshNonce(), NO), @"f_tex", nil, 0);
		const double t0 = now();
		id<MTLRenderPipelineState> p = renderPipeline(d, MTLPipelineOptionNone);
		[create addObject:@(ms(t0))];
		[first addObject:@(drawOnce(p, MTLPrimitiveTypeTriangleStrip))];
		[next addObject:@(drawOnce(p, MTLPrimitiveTypeTriangleStrip))];
	}
	return [NSString stringWithFormat:@"%@ %@ %@", stat(@"create_ms", create), stat(@"first_ms", first),
	                                  stat(@"next_ms", next)];
}

static NSString *drawTopology(void)
{
	setupDrawing();
	NSMutableArray *tri = [NSMutableArray array], *first = [NSMutableArray array], *next = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		id<MTLRenderPipelineState> p =
		    renderPipeline(quadDescriptor(libraryOf(freshNonce(), NO), @"f_tex", nil, 0), MTLPipelineOptionNone);
		drawOnce(p, MTLPrimitiveTypeTriangleStrip);
		[tri addObject:@(drawOnce(p, MTLPrimitiveTypeTriangleStrip))];
		[first addObject:@(drawOnce(p, MTLPrimitiveTypeLineStrip))];
		[next addObject:@(drawOnce(p, MTLPrimitiveTypeLineStrip))];
	}
	return [NSString stringWithFormat:@"%@ %@ %@", stat(@"triangles_ms", tri), stat(@"first_lines_ms", first),
	                                  stat(@"next_lines_ms", next)];
}

static NSString *dispatchFirst(void)
{
	setupDrawing();
	NSMutableArray *create = [NSMutableArray array], *first = [NSMutableArray array], *next = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		id<MTLFunction> f = function(libraryOf(freshNonce(), NO), @"k_blur", sharpness(1.0f));
		const double t0 = now();
		id<MTLComputePipelineState> p = computePipeline(f);
		[create addObject:@(ms(t0))];
		[first addObject:@(dispatchOnce(p))];
		[next addObject:@(dispatchOnce(p))];
	}
	return [NSString stringWithFormat:@"%@ %@ %@", stat(@"create_ms", create), stat(@"first_ms", first),
	                                  stat(@"next_ms", next)];
}

static id<MTLBinaryArchive> emptyArchive(NSURL *url)
{
	MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
	ad.url = url;
	NSError *err = nil;
	id<MTLBinaryArchive> a = [gDevice newBinaryArchiveWithDescriptor:ad error:&err];
	if (a == nil)
		[NSException raise:@"pipelines" format:@"archive: %@", err.localizedDescription ?: @"nil"];
	return a;
}

static NSString *archiveAdd(void)
{
	NSMutableArray *add = [NSMutableArray array], *hit = [NSMutableArray array], *none = [NSMutableArray array];
	for (int r = 0; r < gReps; r++) {
		id<MTLBinaryArchive> a = emptyArchive(nil);
		MTLRenderPipelineDescriptor *d = quadDescriptor(libraryOf(freshNonce(), NO), @"f_tex", nil, 0);
		NSError *err = nil;
		double t0 = now();
		if (![a addRenderPipelineFunctionsWithDescriptor:d error:&err])
			return [@"FAIL add: " stringByAppendingString:err.localizedDescription ?: @"no error"];
		[add addObject:@(ms(t0))];
		d.binaryArchives = @[ a ];
		t0 = now();
		renderPipeline(d, MTLPipelineOptionFailOnBinaryArchiveMiss);
		[hit addObject:@(ms(t0))];
		MTLRenderPipelineDescriptor *other = quadDescriptor(libraryOf(freshNonce(), NO), @"f_tex", nil, 0);
		t0 = now();
		renderPipeline(other, MTLPipelineOptionNone);
		[none addObject:@(ms(t0))];
	}
	return [NSString stringWithFormat:@"%@ %@ %@", stat(@"add_ms", add), stat(@"hit_ms", hit), stat(@"none_ms", none)];
}

static NSString *archiveFile(void)
{
	NSURL *url = [NSURL fileURLWithPath:gArchivePath];
	// SEED's own library: the same in every process given the same seed.
	MTLRenderPipelineDescriptor *d = quadDescriptor(libraryOf(gSeed * 4096u + 4095u, NO), @"f_tex", nil, 0);
	if ([[NSFileManager defaultManager] fileExistsAtPath:gArchivePath]) {
		double t0 = now();
		id<MTLBinaryArchive> a = emptyArchive(url);
		const double load = ms(t0);
		d.binaryArchives = @[ a ];
		t0 = now();
		NSError *err = nil;
		id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:d
		                                                                     options:MTLPipelineOptionFailOnBinaryArchiveMiss
		                                                                  reflection:nil
		                                                                       error:&err];
		const double hit = ms(t0);
		if (p == nil)
			return [NSString stringWithFormat:@"FAIL miss from %@: %@", gArchivePath, err.localizedDescription];
		return [NSString stringWithFormat:@"load_ms=%.3f hit_ms=%.3f", load, hit];
	}
	id<MTLBinaryArchive> a = emptyArchive(nil);
	NSError *err = nil;
	const double t0 = now();
	if (![a addRenderPipelineFunctionsWithDescriptor:d error:&err])
		return [@"FAIL add: " stringByAppendingString:err.localizedDescription ?: @"no error"];
	const double add = ms(t0);
	if (![a serializeToURL:url error:&err])
		return [@"FAIL serialize: " stringByAppendingString:err.localizedDescription ?: @"no error"];
	return [NSString stringWithFormat:@"written=%@ add_ms=%.3f", gArchivePath, add];
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		NSMutableArray *patterns = [NSMutableArray array];
		const char *want = NULL;
		BOOL list = NO, seeded = NO;
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
				want = argv[++i];
			else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
				gReps = atoi(argv[++i]);
			else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
				gSeed = (uint32_t)strtoul(argv[++i], NULL, 0);
				seeded = YES;
			} else if (strcmp(argv[i], "-L") == 0 && i + 1 < argc)
				gLibDir = @(argv[++i]);
			else if (strcmp(argv[i], "-A") == 0 && i + 1 < argc)
				gArchivePath = @(argv[++i]);
			else if (strcmp(argv[i], "-l") == 0)
				list = YES;
			else
				[patterns addObject:@(argv[i])];
		}
		gPatterns = patterns;
		if (gReps < 1)
			gReps = 1;
		if (!seeded)
			gSeed = (uint32_t)(time(NULL) ^ (getpid() << 16)) & 0xfffffu;
		if (gLibDir == nil) {
			char path[4096];
			uint32_t size = sizeof path;
			gLibDir = _NSGetExecutablePath(path, &size) == 0
			              ? [@(path) stringByDeletingLastPathComponent]
			              : @".";
		}
		if (gArchivePath == nil)
			gArchivePath = [NSString stringWithFormat:@"/tmp/pipelines-%u.archive", gSeed];
		gSource = [[NSString alloc] initWithBytes:pipelines_metal length:pipelines_metal_len
		                                 encoding:NSUTF8StringEncoding];
		if (list) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				printf("pipelines: device \"%s\" lowPower=%d removable=%d unified=%d\n", d.name.UTF8String,
				       d.lowPower, d.removable, d.hasUnifiedMemory);
			return 0;
		}
		if (want != NULL) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				if (gDevice == nil && strcasestr(d.name.UTF8String, want))
					gDevice = d;
		} else {
			gDevice = MTLCreateSystemDefaultDevice();
		}
		if (gDevice == nil) {
			printf("pipelines: no Metal device%s%s\n", want ? " matching " : "", want ? want : "");
			return 2;
		}
		char model[64] = "?", build[32] = "?";
		size_t len = sizeof(model);
		sysctlbyname("hw.model", model, &len, NULL, 0);
		len = sizeof(build);
		sysctlbyname("kern.osversion", build, &len, NULL, 0);
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("pipelines: device \"%s\", %s, macOS %ld.%ld.%ld (%s), %s, reps %d, seed %u\n",
		       gDevice.name.UTF8String, model, (long)v.majorVersion, (long)v.minorVersion,
		       (long)v.patchVersion, build,
#if defined(__x86_64__)
		       "x86_64",
#else
		       "arm64",
#endif
		       gReps, gSeed);
		fflush(stdout);
		gQueue = [gDevice newCommandQueue];

		runCase(@"lib.source.small", ^{ return libSource(YES); });
		runCase(@"lib.source.large", ^{ return libSource(NO); });
		runCase(@"lib.source.async", ^{ return libSourceAsync(); });
		runCase(@"lib.data", ^{ return libData(); });
		runCase(@"lib.system", ^{ return libSystem(); });
		runCase(@"fn.const", ^{ return fnConst(); });
		runCase(@"rp.lib.tex", ^{ return renderCase(0, @"f_tex"); });
		runCase(@"rp.lib.pbr", ^{ return renderCase(0, @"f_pbr"); });
		runCase(@"rp.const", ^{ return renderCase(1, @"f_tex"); });
		runCase(@"rp.state", ^{ return renderCase(2, @"f_tex"); });
		runCase(@"rp.warm", ^{ return renderCase(3, @"f_tex"); });
		runCase(@"rp.async.8", ^{ return asyncCase(NO, 8); });
		runCase(@"cp.lib", ^{ return computeCase(0); });
		runCase(@"cp.const", ^{ return computeCase(1); });
		runCase(@"cp.warm", ^{ return computeCase(3); });
		runCase(@"cp.async.8", ^{ return asyncCase(YES, 8); });
		runCase(@"draw.first", ^{ return drawFirst(); });
		runCase(@"draw.topology", ^{ return drawTopology(); });
		runCase(@"dispatch.first", ^{ return dispatchFirst(); });
		runCase(@"archive.add", ^{ return archiveAdd(); });
		runCase(@"archive.file", ^{ return archiveFile(); });
	}
	return 0;
}
