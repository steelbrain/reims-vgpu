// oracle_spi_members: which members of each Metal protocol a GPU's own objects answer, for every
// kind of object the reims-graphics bundle makes, on each of this Mac's devices: the public
// protocol and its SPI counterpart (MTLFooSPI), with the protocols each adopts. One line per
// member, "DEVICE PROTOCOL SELECTOR required|optional yes|no [OWNER]" (yes: respondsToSelector:;
// OWNER, for a member answered, the first class of the object's chain whose own methods have it:
// the driver's class or one of Metal's base classes), under a line "# DEVICE KIND class CLASS"
// for each object. It calls no member it checks. The register
// tests/protocol_members.txt gives its verdicts as %spi-both, %spi-neither and %spi-differ.
//   oracle_spi_members answers  also calls each member the object answers that takes no argument
//     and returns a value (not void, not a struct), except those whose names act (new...,
//     commit..., end..., ...Encoder..., the list in answer()) and those that end the process
//     (kAsserts), and prints "A DEVICE KIND SELECTOR TYPE VALUE": what a fresh object of the kind
//     answers (the device's own members are not called). Each call is named first
//     ("C DEVICE KIND SELECTOR"), so one that ends the process is the last line: add it to
//     kAsserts and stop there. Run it once, never in a loop through its aborts: forty runs, each
//     ending on one of Metal's assertions, most likely wedged the oracle's GPU stack in its
//     kernel (2026-11-08: ioreg's accelerator queries hung after them).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <ctype.h>
#include <objc/runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The first class of `obj`'s chain whose own instance methods include `sel`, or "-".
static const char *owner(id obj, SEL sel) {
	for (Class c = object_getClass(obj); c; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *m = class_copyMethodList(c, &n);
		BOOL has = NO;
		for (unsigned i = 0; i < n && !has; i++) has = sel_isEqual(method_getName(m[i]), sel);
		free(m);
		if (has) return class_getName(c);
	}
	return "-";
}

static void walk(const char *dev, id obj, Protocol *p, NSMutableSet<NSString *> *seen) {
	if (!p) return;
	const char *pn = protocol_getName(p);
	NSString *name = @(pn);
	if ([seen containsObject:name]) return;
	[seen addObject:name];
	if (!strncmp(pn, "MTL", 3)) {
		for (int req = 1; req >= 0; req--) {
			unsigned n = 0;
			struct objc_method_description *d = protocol_copyMethodDescriptionList(p, req, YES, &n);
			for (unsigned i = 0; i < n; i++)
				if ([obj respondsToSelector:d[i].name])
					printf("%s %s %s %s yes %s\n", dev, pn, sel_getName(d[i].name), req ? "required" : "optional",
					       owner(obj, d[i].name));
				else
					printf("%s %s %s %s no\n", dev, pn, sel_getName(d[i].name), req ? "required" : "optional");
			free(d);
		}
	}
	unsigned n = 0;
	Protocol *__unsafe_unretained *a = protocol_copyProtocolList(p, &n);
	for (unsigned i = 0; i < n; i++) walk(dev, obj, a[i], seen);
	free(a);
}

static BOOL g_answers;

// The getters Metal's own objects assert in ("Driver needs to override this", ...), ending the
// process: "KIND SELECTOR" (measured: the AMD 5300M and the Intel 630, macOS 26.7.1;
// hasBarrier on the Intel 630's only).
static const char *const kAsserts[] = {
	"MTLFunction functionHandle", "MTLTexture sparseTextureTier", "MTLBuffer sparseBufferTier",
	"MTLDepthStencilState uniqueIdentifier", "MTLDepthStencilState resourceIndex",
	"MTLIndirectRenderCommand getScissorRects", "MTLIndirectRenderCommand drawMeshThreadgroupsArguments",
	"MTLIndirectRenderCommand drawMeshThreadsArguments", "MTLIndirectRenderCommand getBlendColor",
	"MTLIndirectRenderCommand getCullMode", "MTLIndirectRenderCommand getDepthBiasInfo",
	"MTLIndirectRenderCommand getDepthClipMode", "MTLIndirectRenderCommand getDepthStencilState",
	"MTLIndirectRenderCommand getDepthStencilStateUniqueIdentifier",
	"MTLIndirectRenderCommand getDepthTestBounds", "MTLIndirectRenderCommand getFrontFacingWinding",
	"MTLIndirectRenderCommand getStencilReferenceValues", "MTLIndirectRenderCommand getTriangleFillMode",
	"MTLIndirectRenderCommand getViewports", "MTLIndirectRenderCommand hasBarrier",
};

// What `obj` answers to the getter `sel`, by its method's return type.
static void answer(const char *dev, const char *kind, id obj, SEL sel) {
	const char *name = sel_getName(sel);
	if (strchr(name, ':') || strstr(name, "ncoder") || ![obj respondsToSelector:sel])
		return;
	// A member that acts (commits, ends, waits, makes...) is not a getter, whatever it returns.
	static const char *const verbs[] = {"new", "commit", "enqueue", "wait", "present", "encode", "end",
		"pop", "push", "insert", "dispatch", "finish", "add", "set", "use", "drop", "do", "reset",
		"copy", "fill", "generate", "optimize", "synchronize", "update", "make", "remove", "detach",
		"attach", "flush", "discard", "unload", "release", "retain", "serialize", "compile", "load",
		"purge", "evict", "__"};
	for (size_t i = 0; i < sizeof verbs / sizeof verbs[0]; i++) {
		const size_t n = strlen(verbs[i]);
		if (!strncmp(name, verbs[i], n) && (name[n] == '\0' || isupper((unsigned char)name[n]) || verbs[i][0] == '_'))
			return;
	}
	Method m = class_getInstanceMethod([obj class], sel);
	const char *types = m ? method_getTypeEncoding(m) : NULL;
	if (!types) return;
	char rt = types[0];
	while (rt == 'r' || rt == 'n' || rt == 'N' || rt == 'o' || rt == 'O' || rt == 'R' || rt == 'V')
		rt = *(++types);
	IMP imp = method_getImplementation(m);
	char line[256];
	snprintf(line, sizeof line, "%s %s", kind, name);
	for (size_t i = 0; i < sizeof kAsserts / sizeof kAsserts[0]; i++)
		if (!strcmp(line, kAsserts[i])) {
			printf("A %s %s %s ends the process (asserts)\n", dev, kind, name);
			return;
		}
	// Named first, so a member that ends the process is the last line.
	printf("C %s %s %s\n", dev, kind, name);
	fflush(stdout);
	switch (rt) {
	case '@': {
		id v = ((id (*)(id, SEL))imp)(obj, sel);
		NSString *d = v ? [[v description] stringByReplacingOccurrencesOfString:@"\n" withString:@" "] : @"nil";
		if (d.length > 160) d = [[d substringToIndex:160] stringByAppendingString:@"..."];
		printf("A %s %s %s @ %s %s\n", dev, kind, name, v ? class_getName([v class]) : "-", d.UTF8String);
		break;
	}
	case 'c': case 'B': case 'C':
		printf("A %s %s %s %c %d\n", dev, kind, name, rt, (int)((BOOL (*)(id, SEL))imp)(obj, sel));
		break;
	case 'i': case 's': case 'l':
		printf("A %s %s %s %c %ld\n", dev, kind, name, rt, (long)((int (*)(id, SEL))imp)(obj, sel));
		break;
	case 'I': case 'S': case 'L':
		printf("A %s %s %s %c %lu\n", dev, kind, name, rt, (unsigned long)((unsigned (*)(id, SEL))imp)(obj, sel));
		break;
	case 'q':
		printf("A %s %s %s q %lld\n", dev, kind, name, ((long long (*)(id, SEL))imp)(obj, sel));
		break;
	case 'Q':
		printf("A %s %s %s Q %llu\n", dev, kind, name, ((unsigned long long (*)(id, SEL))imp)(obj, sel));
		break;
	case 'f':
		printf("A %s %s %s f %g\n", dev, kind, name, ((float (*)(id, SEL))imp)(obj, sel));
		break;
	case 'd':
		printf("A %s %s %s d %g\n", dev, kind, name, ((double (*)(id, SEL))imp)(obj, sel));
		break;
	case '^': case '*':
		printf("A %s %s %s ^ %s\n", dev, kind, name, ((void *(*)(id, SEL))imp)(obj, sel) ? "non-null" : "null");
		break;
	default:
		printf("A %s %s %s %s (not called)\n", dev, kind, name, types);
	}
}

static void answersOf(const char *dev, const char *kind, id obj, Protocol *p, NSMutableSet<NSString *> *seen) {
	if (!p) return;
	NSString *name = @(protocol_getName(p));
	if ([seen containsObject:name]) return;
	[seen addObject:name];
	if ([name hasPrefix:@"MTL"]) {
		for (int req = 1; req >= 0; req--) {
			unsigned n = 0;
			struct objc_method_description *d = protocol_copyMethodDescriptionList(p, req, YES, &n);
			for (unsigned i = 0; i < n; i++) answer(dev, kind, obj, d[i].name);
			free(d);
		}
	}
	unsigned n = 0;
	Protocol *__unsafe_unretained *a = protocol_copyProtocolList(p, &n);
	for (unsigned i = 0; i < n; i++) answersOf(dev, kind, obj, a[i], seen);
	free(a);
}

static void check(const char *dev, const char *proto, id obj) {
	if (!obj) { printf("%s %s (no object)\n", dev, proto); return; }
	printf("# %s %s class %s\n", dev, proto, class_getName([obj class]));
	NSMutableSet<NSString *> *seen = [NSMutableSet set];
	walk(dev, obj, objc_getProtocol(proto), seen);
	walk(dev, obj, objc_getProtocol([@(proto) stringByAppendingString:@"SPI"].UTF8String), seen);
	// The device's SPI is its IOGPU base's on the guest (the register's %base-spi); not asked.
	if (g_answers && strcmp(proto, "MTLDevice") != 0) {
		NSMutableSet<NSString *> *asked = [NSMutableSet set];
		answersOf(dev, proto, obj, objc_getProtocol([@(proto) stringByAppendingString:@"SPI"].UTF8String), asked);
	}
}

static NSString *const kSource =
	@"#include <metal_stdlib>\nusing namespace metal;\n"
	 "struct V { float4 p [[position]]; };\n"
	 "vertex V vs(uint i [[vertex_id]]) { V v; v.p = float4(float(i), 0, 0, 1); return v; }\n"
	 "fragment float4 fs(V in [[stage_in]]) { return float4(1); }\n"
	 "struct A { device float *b [[id(0)]]; };\n"
	 "kernel void k(device float *b [[buffer(0)]], constant A &a [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = *a.b; }\n";

int main(int argc, char **argv) {
	@autoreleasepool {
		g_answers = argc > 1 && !strcmp(argv[1], "answers");
		for (id<MTLDevice> d in MTLCopyAllDevices()) {
			NSString *name = [d.name stringByReplacingOccurrencesOfString:@" " withString:@"_"];
			const char *dev = name.UTF8String;
			printf("# device %s class %s\n", dev, class_getName([(id)d class]));
			check(dev, "MTLDevice", d);
			NSError *e = nil;
			id<MTLLibrary> lib = [d newLibraryWithSource:kSource options:nil error:&e];
			if (!lib) printf("# %s library: %s\n", dev, e.description.UTF8String);
			check(dev, "MTLLibrary", lib);
			id<MTLFunction> kfn = [lib newFunctionWithName:@"k"];
			check(dev, "MTLFunction", kfn);
			check(dev, "MTLComputePipelineState", [d newComputePipelineStateWithFunction:kfn error:&e]);
			MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
			rpd.vertexFunction = [lib newFunctionWithName:@"vs"];
			rpd.fragmentFunction = [lib newFunctionWithName:@"fs"];
			rpd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
			rpd.supportIndirectCommandBuffers = YES;
			id<MTLRenderPipelineState> rps = [d newRenderPipelineStateWithDescriptor:rpd error:&e];
			check(dev, "MTLRenderPipelineState", rps);
			MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:16 height:16 mipmapped:NO];
			td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
			td.storageMode = MTLStorageModePrivate;
			id<MTLTexture> tex = [d newTextureWithDescriptor:td];
			check(dev, "MTLTexture", tex);
			id<MTLBuffer> buf = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
			check(dev, "MTLBuffer", buf);
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 1 << 20;
			hd.storageMode = MTLStorageModePrivate;
			check(dev, "MTLHeap", [d newHeapWithDescriptor:hd]);
			check(dev, "MTLSamplerState", [d newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]]);
			check(dev, "MTLDepthStencilState", [d newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]]);
			check(dev, "MTLFence", [d newFence]);
			check(dev, "MTLEvent", [d newEvent]);
			check(dev, "MTLSharedEvent", [d newSharedEvent]);
			MTLArgumentDescriptor *ad = [MTLArgumentDescriptor argumentDescriptor];
			ad.dataType = MTLDataTypePointer;
			ad.index = 0;
			check(dev, "MTLArgumentEncoder", [d newArgumentEncoderWithArguments:@[ad]]);
			MTLBinaryArchiveDescriptor *bad = [MTLBinaryArchiveDescriptor new];
			check(dev, "MTLBinaryArchive", [d newBinaryArchiveWithDescriptor:bad error:&e]);
			MTLIndirectCommandBufferDescriptor *icbd = [MTLIndirectCommandBufferDescriptor new];
			icbd.commandTypes = MTLIndirectCommandTypeDraw;
			icbd.inheritPipelineState = NO;
			icbd.inheritBuffers = NO;
			icbd.maxVertexBufferBindCount = 1;
			icbd.maxFragmentBufferBindCount = 1;
			id<MTLIndirectCommandBuffer> icb = [d newIndirectCommandBufferWithDescriptor:icbd maxCommandCount:1 options:0];
			check(dev, "MTLIndirectCommandBuffer", icb);
			check(dev, "MTLIndirectRenderCommand", [icb indirectRenderCommandAtIndex:0]);
			MTLIndirectCommandBufferDescriptor *cicbd = [MTLIndirectCommandBufferDescriptor new];
			cicbd.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
			cicbd.inheritPipelineState = NO;
			cicbd.inheritBuffers = NO;
			cicbd.maxKernelBufferBindCount = 2;
			id<MTLIndirectCommandBuffer> cicb = [d newIndirectCommandBufferWithDescriptor:cicbd maxCommandCount:1 options:0];
			check(dev, "MTLIndirectComputeCommand", [cicb indirectComputeCommandAtIndex:0]);
			id<MTLCommandQueue> q = [d newCommandQueue];
			check(dev, "MTLCommandQueue", q);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			check(dev, "MTLCommandBuffer", cb);
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = tex;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
			check(dev, "MTLRenderCommandEncoder", re);
			[re endEncoding];
			id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
			check(dev, "MTLComputeCommandEncoder", ce);
			[ce endEncoding];
			id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
			check(dev, "MTLBlitCommandEncoder", be);
			[be endEncoding];
			id<MTLParallelRenderCommandEncoder> pe = [cb parallelRenderCommandEncoderWithDescriptor:rp];
			check(dev, "MTLParallelRenderCommandEncoder", pe);
			[pe endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			check(dev, "MTLLogContainer", cb.logs);
		}
	}
	return 0;
}
