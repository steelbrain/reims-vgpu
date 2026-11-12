// oracle_ratemap: what a device answers and draws for rasterization rate maps
// (MTLRasterizationRateMap): the layer counts it supports, the classes and members of its maps
// and their descriptors, a map's answers (screen size, granularity, layers, parameter size and
// alignment, physical sizes, both coordinate maps) and its parameter data, a pass drawn through
// the map, and the calls that may end the process. No Metal validation layer is asked for: what a
// guest's process meets. It runs the same on the guest's device.
//   oracle_ratemap DEVICE            every step; the deaths each in a child process of its own
//   oracle_ratemap DEVICE --die NAME the death NAME alone (what the children run)
// Steps and their lines:
//   supports N => YES|NO                       supportsRasterizationRateMapWithLayerCount:, 0 to 3
//   members CLASS: SEL=TYPES ...                the instance methods of a map's and a
//                                               descriptor's classes below NSObject, sorted, by
//                                               class, each with its type encoding
//   desc mutability => M | none                 a fresh descriptor's private `mutability`
//   desc minFactor => F | none                  and its private `minFactor`
//   map NAME => made CLASS | nil                maps of the shapes below
//   map NAME screen WxHxD gran WxHxD layers L params SIZE/ALIGN mutability M|none label L
//   map NAME layer I physical WxHxD
//   map NAME minFactor => F | none              the map's private `minFactor`
//   map NAME formatted => TEXT                  its private `formattedDescription:` at indent 0,
//                                               newlines as `|`
//   map NAME s2p I (X, Y) => (X', Y')           mapScreenToPhysicalCoordinates:forLayer:, at each
//                                               layer of the map
//   map NAME p2s I (X, Y) => (X', Y')           mapPhysicalToScreenCoordinates:forLayer:
//   map NAME copy at OFF => fnv HASH nonzero N  copyParameterDataToBuffer:offset: into a Shared
//                                               buffer filled with 0xa5, then the bytes it holds
//                                               over the parameter size: FNV-1a 64 and the count
//                                               of bytes that changed
//   pass NAME VIEWPORT => physical WxH fragments N
//   pass NAME VIEWPORT texel (PX, PY) => pos (X, Y) screen (SX, SY)
//                                               a pass over map NAME into a target of its layer 0
//                                               physical size, of one triangle over the whole
//                                               target, whose fragment writes its position and
//                                               the map's decode of it to the screen (the
//                                               shader's rasterization_rate_map_decoder) and
//                                               counts itself; VIEWPORT `default` sets none,
//                                               `screen` the map's screen size
//   die NAME => died on signal X | exited X      the call each names, in a child process:
//     nolayers-s2p   mapScreenToPhysicalCoordinates: of a map of no layers
//     nolayers-p2s   mapPhysicalToScreenCoordinates: of a map of no layers
//     private-copy   copyParameterDataToBuffer:offset: into a Private buffer
//     mutable-p2s    mapPhysicalToScreenCoordinates: of a Mutable map not reset (where the
//                    descriptor answers setMutability:)
//     past-s2p       mapScreenToPhysicalCoordinates: of map `one` at layer 1, past its layers
//     past-p2s       mapPhysicalToScreenCoordinates: of map `one` at layer 1
// Build: see README.md; the pass loads oracle_ratemap.metallib from beside the executable.
#import "oracle_common.h"
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static uint64_t fnv(const uint8_t *p, size_t n)
{
	uint64_t h = 1469598103934665603ull;
	for (size_t i = 0; i < n; i++) {
		h ^= p[i];
		h *= 1099511628211ull;
	}
	return h;
}

static void members(id obj)
{
	for (Class c = object_getClass(obj); c != Nil && c != [NSObject class];
	     c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *list = class_copyMethodList(c, &n);
		NSMutableArray<NSString *> *names = [NSMutableArray array];
		for (unsigned i = 0; i < n; i++)
			[names addObject:[NSString stringWithFormat:@"%s=%s", sel_getName(method_getName(list[i])),
			                                            method_getTypeEncoding(list[i])]];
		free(list);
		[names sortUsingSelector:@selector(compare:)];
		printf("members %s:", class_getName(c));
		for (NSString *name in names)
			printf(" %s", name.UTF8String);
		printf("\n");
	}
}

static NSString *mutabilityOf(id obj)
{
	if (![obj respondsToSelector:@selector(mutability)])
		return @"none";
	NSUInteger m = ((NSUInteger(*)(id, SEL))objc_msgSend)(obj, @selector(mutability));
	return [NSString stringWithFormat:@"%lu", (unsigned long)m];
}

// `obj`'s private `minFactor`, by the return type its class gives it.
static NSString *minFactorOf(id obj)
{
	Method m = class_getInstanceMethod(object_getClass(obj), @selector(minFactor));
	if (m == NULL)
		return @"none";
	char type[16] = "";
	method_getReturnType(m, type, sizeof(type));
	if (strcmp(type, "f") == 0)
		return [NSString stringWithFormat:@"%.9g (f)",
		                 ((float (*)(id, SEL))objc_msgSend)(obj, @selector(minFactor))];
	if (strcmp(type, "d") == 0)
		return [NSString stringWithFormat:@"%.17g (d)",
		                 ((double (*)(id, SEL))objc_msgSend)(obj, @selector(minFactor))];
	return [NSString stringWithFormat:@"%llu (%s)",
	                 (unsigned long long)((uint64_t (*)(id, SEL))objc_msgSend)(obj, @selector(minFactor)),
	                 type];
}

static MTLRasterizationRateLayerDescriptor *layerOf(NSUInteger h, const float *hs, NSUInteger v,
                                                    const float *vs)
{
	return [[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(h, v, 0)
	                                                             horizontal:hs
	                                                               vertical:vs];
}

// The shapes the steps make: `one`, a 256 x 128 screen of one layer, `two`, the same with a
// second layer, and `none`, a 64 x 64 screen of no layers, which only the deaths make (the AMD
// Radeon Pro 5300M's creator ends the process there, macOS 26.7.1).
static MTLRasterizationRateMapDescriptor *shape(const char *name)
{
	static const float h0[] = {1.0f, 0.25f, 1.0f};
	static const float v0[] = {1.0f, 0.5f};
	static const float h1[] = {0.5f, 1.0f};
	static const float v1[] = {0.25f, 1.0f, 0.75f};
	MTLRasterizationRateMapDescriptor *d = [[MTLRasterizationRateMapDescriptor alloc] init];
	if (strcmp(name, "none") == 0) {
		d.screenSize = MTLSizeMake(64, 64, 0);
		return d;
	}
	d.screenSize = MTLSizeMake(256, 128, 0);
	[d setLayer:layerOf(3, h0, 2, v0) atIndex:0];
	if (strcmp(name, "two") == 0)
		[d setLayer:layerOf(2, h1, 3, v1) atIndex:1];
	d.label = [NSString stringWithFormat:@"oracle-%s", name];
	return d;
}

static void answers(const char *name, id<MTLRasterizationRateMap> map, id<MTLDevice> dev)
{
	MTLSize s = map.screenSize, g = map.physicalGranularity;
	MTLSizeAndAlign pa = map.parameterBufferSizeAndAlign;
	printf("map %s screen %lux%lux%lu gran %lux%lux%lu layers %lu params %lu/%lu mutability %s "
	       "label %s\n",
	       name, (unsigned long)s.width, (unsigned long)s.height, (unsigned long)s.depth,
	       (unsigned long)g.width, (unsigned long)g.height, (unsigned long)g.depth,
	       (unsigned long)map.layerCount, (unsigned long)pa.size, (unsigned long)pa.align,
	       mutabilityOf(map).UTF8String, map.label ? map.label.UTF8String : "(nil)");
	printf("map %s minFactor => %s\n", name, minFactorOf(map).UTF8String);
	if ([(id)map respondsToSelector:@selector(formattedDescription:)]) {
		NSString *text = ((NSString * (*)(id, SEL, NSUInteger))objc_msgSend)(
		    map, @selector(formattedDescription:), 0);
		printf("map %s formatted => %s\n", name,
		       [text stringByReplacingOccurrencesOfString:@"\n" withString:@"|"].UTF8String);
	}
	NSUInteger layers = map.layerCount;
	for (NSUInteger i = 0; i < 3; i++) {
		MTLSize p = [map physicalSizeForLayer:i];
		printf("map %s layer %lu physical %lux%lux%lu\n", name, (unsigned long)i,
		       (unsigned long)p.width, (unsigned long)p.height, (unsigned long)p.depth);
	}
	if (layers == 0)
		return;
	static const float pts[][2] = {{0, 0}, {10.5f, 3.25f}, {100, 50}, {200, 100}, {255, 127}, {256, 128}};
	for (NSUInteger i = 0; i < layers; i++) {
		for (size_t k = 0; k < sizeof(pts) / sizeof(pts[0]); k++) {
			MTLCoordinate2D c = MTLCoordinate2DMake(pts[k][0], pts[k][1]);
			MTLCoordinate2D o = [map mapScreenToPhysicalCoordinates:c forLayer:i];
			printf("map %s s2p %lu (%g, %g) => (%.9g, %.9g)\n", name, (unsigned long)i, c.x, c.y,
			       o.x, o.y);
			MTLCoordinate2D back = [map mapPhysicalToScreenCoordinates:c forLayer:i];
			printf("map %s p2s %lu (%g, %g) => (%.9g, %.9g)\n", name, (unsigned long)i, c.x, c.y,
			       back.x, back.y);
		}
	}
	const NSUInteger size = pa.size;
	id<MTLBuffer> buf = [dev newBufferWithLength:size + 64 options:MTLResourceStorageModeShared];
	for (NSUInteger off = 0; off <= 4; off += 4) {
		memset(buf.contents, 0xa5, buf.length);
		[map copyParameterDataToBuffer:buf offset:off];
		const uint8_t *b = (const uint8_t *)buf.contents;
		size_t changed = 0;
		for (size_t i = 0; i < buf.length; i++)
			changed += b[i] != 0xa5;
		printf("map %s copy at %lu => fnv %016llx nonzero %zu\n", name, (unsigned long)off,
		       (unsigned long long)fnv(b + off, size), changed);
	}
}

static void pass(const char *name, id<MTLRasterizationRateMap> map, id<MTLDevice> dev,
                 id<MTLLibrary> lib, BOOL screenViewport)
{
	const char *vp = screenViewport ? "screen" : "default";
	MTLSize phys = [map physicalSizeForLayer:0];
	if (phys.width == 0 || phys.height == 0) {
		printf("pass %s %s => physical %lux%lu: no target\n", name, vp, (unsigned long)phys.width,
		       (unsigned long)phys.height);
		return;
	}
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
	                                                       width:phys.width
	                                                      height:phys.height
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> target = [dev newTextureWithDescriptor:td];
	MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
	pd.vertexFunction = [lib newFunctionWithName:@"ratemap_vs"];
	pd.fragmentFunction = [lib newFunctionWithName:@"ratemap_fs"];
	pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
	NSError *err = nil;
	id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
	if (!ps) {
		printf("pass %s %s => no pipeline: %s\n", name, vp, err.description.UTF8String);
		return;
	}
	id<MTLBuffer> params = [dev newBufferWithLength:map.parameterBufferSizeAndAlign.size
	                                        options:MTLResourceStorageModeShared];
	[map copyParameterDataToBuffer:params offset:0];
	id<MTLBuffer> count = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
	memset(count.contents, 0, 4);
	const NSUInteger rowBytes = phys.width * 16;
	id<MTLBuffer> out = [dev newBufferWithLength:rowBytes * phys.height
	                                     options:MTLResourceStorageModeShared];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = target;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.rasterizationRateMap = map;
	id<MTLCommandQueue> q = [dev newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
	if (screenViewport) {
		MTLSize s = map.screenSize;
		[re setViewport:(MTLViewport){0, 0, (double)s.width, (double)s.height, 0, 1}];
	}
	[re setRenderPipelineState:ps];
	[re setFragmentBuffer:params offset:0 atIndex:0];
	[re setFragmentBuffer:count offset:0 atIndex:1];
	[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[re endEncoding];
	id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
	[be copyFromTexture:target
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(phys.width, phys.height, 1)
	                toBuffer:out
	       destinationOffset:0
	  destinationBytesPerRow:rowBytes
	destinationBytesPerImage:rowBytes * phys.height];
	[be endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	printf("pass %s %s => physical %lux%lu fragments %u\n", name, vp, (unsigned long)phys.width,
	       (unsigned long)phys.height, *(const uint32_t *)count.contents);
	const float *px = (const float *)out.contents;
	const NSUInteger xs[] = {0, phys.width / 4, phys.width / 2, phys.width - 1};
	const NSUInteger ys[] = {0, phys.height / 2, phys.height - 1};
	for (size_t yi = 0; yi < sizeof(ys) / sizeof(ys[0]); yi++) {
		for (size_t xi = 0; xi < sizeof(xs) / sizeof(xs[0]); xi++) {
			const NSUInteger x = xs[xi], y = ys[yi];
			const float *t = px + (y * phys.width + x) * 4;
			printf("pass %s %s texel (%lu, %lu) => pos (%.9g, %.9g) screen (%.9g, %.9g)\n", name, vp,
			       (unsigned long)x, (unsigned long)y, t[0], t[1], t[2], t[3]);
		}
	}
}

static int die(id<MTLDevice> dev, const char *name)
{
	if (strncmp(name, "nolayers-", 9) == 0) {
		id<MTLRasterizationRateMap> map = [dev newRasterizationRateMapWithDescriptor:shape("none")];
		printf("die %s: map %s\n", name, map ? "made" : "nil");
		fflush(stdout);
		MTLCoordinate2D c = MTLCoordinate2DMake(1, 1);
		MTLCoordinate2D o = strcmp(name, "nolayers-s2p") == 0
		                        ? [map mapScreenToPhysicalCoordinates:c forLayer:0]
		                        : [map mapPhysicalToScreenCoordinates:c forLayer:0];
		printf("die %s => answered (%.9g, %.9g)\n", name, o.x, o.y);
		return 0;
	}
	if (strncmp(name, "past-", 5) == 0) {
		id<MTLRasterizationRateMap> map = [dev newRasterizationRateMapWithDescriptor:shape("one")];
		printf("die %s: map %s\n", name, map ? "made" : "nil");
		MTLCoordinate2D c = MTLCoordinate2DMake(10, 10);
		MTLCoordinate2D o = strcmp(name, "past-s2p") == 0
		                        ? [map mapScreenToPhysicalCoordinates:c forLayer:1]
		                        : [map mapPhysicalToScreenCoordinates:c forLayer:1];
		printf("die %s => answered (%.9g, %.9g)\n", name, o.x, o.y);
		return 0;
	}
	if (strcmp(name, "private-copy") == 0) {
		id<MTLRasterizationRateMap> map = [dev newRasterizationRateMapWithDescriptor:shape("one")];
		id<MTLBuffer> buf = [dev newBufferWithLength:map.parameterBufferSizeAndAlign.size
		                                     options:MTLResourceStorageModePrivate];
		printf("die %s: map %s buffer %s\n", name, map ? "made" : "nil", buf ? "made" : "nil");
		fflush(stdout);
		[map copyParameterDataToBuffer:buf offset:0];
		printf("die %s => returned\n", name);
		return 0;
	}
	if (strcmp(name, "mutable-p2s") == 0) {
		MTLRasterizationRateMapDescriptor *d = shape("one");
		if (![d respondsToSelector:@selector(setMutability:)]) {
			printf("die %s => the descriptor answers no setMutability:\n", name);
			return 0;
		}
		((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setMutability:),
		                                             (NSUInteger)MTLMutabilityMutable);
		id<MTLRasterizationRateMap> map = [dev newRasterizationRateMapWithDescriptor:d];
		printf("die %s: map %s mutability %s physical %lux%lu\n", name, map ? "made" : "nil",
		       mutabilityOf(map).UTF8String, (unsigned long)[map physicalSizeForLayer:0].width,
		       (unsigned long)[map physicalSizeForLayer:0].height);
		fflush(stdout);
		MTLCoordinate2D o = [map mapPhysicalToScreenCoordinates:MTLCoordinate2DMake(1, 1)
		                                               forLayer:0];
		printf("die %s => answered (%.9g, %.9g)\n", name, o.x, o.y);
		return 0;
	}
	printf("die %s => no such death\n", name);
	return 2;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE [--die NAME]\n", argv[0]);
			return 2;
		}
		id<MTLDevice> dev = oracle_device(argv[1]);
		if (argc >= 4 && strcmp(argv[2], "--die") == 0)
			return die(dev, argv[3]);
		oracle_header("oracle_ratemap", dev);
		for (NSUInteger n = 0; n <= 3; n++)
			printf("supports %lu => %s\n", (unsigned long)n,
			       [dev supportsRasterizationRateMapWithLayerCount:n] ? "YES" : "NO");
		MTLRasterizationRateMapDescriptor *fresh = [[MTLRasterizationRateMapDescriptor alloc] init];
		printf("desc mutability => %s\n", mutabilityOf(fresh).UTF8String);
		printf("desc minFactor => %s\n", minFactorOf(fresh).UTF8String);
		members(fresh);
		members([[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(1, 1, 0)]);
		id<MTLLibrary> lib = oracle_library(dev, argv[0], "oracle_ratemap");
		const char *names[] = {"one", "two"};
		BOOL listed = NO;
		for (size_t k = 0; k < sizeof(names) / sizeof(names[0]); k++) {
			const char *name = names[k];
			id<MTLRasterizationRateMap> map = [dev newRasterizationRateMapWithDescriptor:shape(name)];
			printf("map %s => %s %s\n", name, map ? "made" : "nil",
			       map ? class_getName(object_getClass(map)) : "");
			if (!map)
				continue;
			if (!listed) {
				members(map);
				listed = YES;
			}
			answers(name, map, dev);
			if (lib && map.layerCount > 0) {
				pass(name, map, dev, lib, NO);
				pass(name, map, dev, lib, YES);
			}
		}
		fflush(stdout);
		const char *deaths[] = {"nolayers-s2p", "nolayers-p2s", "private-copy", "mutable-p2s",
		                        "past-s2p", "past-p2s"};
		for (size_t k = 0; k < sizeof(deaths) / sizeof(deaths[0]); k++) {
			const char *name = deaths[k];
			int fds[2];
			pipe(fds);
			posix_spawn_file_actions_t fa;
			posix_spawn_file_actions_init(&fa);
			posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
			posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
			posix_spawn_file_actions_addclose(&fa, fds[0]);
			char *args[] = {argv[0], argv[1], (char *)"--die", (char *)name, NULL};
			pid_t pid = 0;
			posix_spawn(&pid, argv[0], &fa, NULL, args, environ);
			posix_spawn_file_actions_destroy(&fa);
			close(fds[1]);
			char buf[4096];
			ssize_t n;
			while ((n = read(fds[0], buf, sizeof(buf))) > 0)
				fwrite(buf, 1, (size_t)n, stdout);
			close(fds[0]);
			int st = 0;
			waitpid(pid, &st, 0);
			if (WIFSIGNALED(st))
				printf("die %s => died on signal %d\n", name, WTERMSIG(st));
			else
				printf("die %s => exited %d\n", name, WEXITSTATUS(st));
		}
	}
	return 0;
}
