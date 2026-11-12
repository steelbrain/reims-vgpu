// oracle_fnptr: what a device answers and runs for function pointers (MTLFunctionHandle,
// MTLVisibleFunctionTable) and dynamic libraries (MTLDynamicLibrary): the support it answers, a
// pipeline's handles of its linked functions and the tables it makes, what a table answers as a
// resource, kernels and a fragment function calling through a table bound at a buffer index or
// held in an argument buffer, an entry set after the dispatch was encoded, and a kernel linked
// against a dynamic library, preloaded, loaded by its install name, or linked at a compile from
// source. No Metal validation layer is asked for: what a guest's process meets. It runs the same
// on the guest's device.
//   oracle_fnptr DEVICE
// Steps and their lines:
//   supports NAME => YES|NO                 supportsFunctionPointers, ...FromRender,
//                                           supportsDynamicLibraries, supportsRenderDynamicLibraries
//   compute pipeline => made CLASS | nil ERROR   fp_kernel, linking fp_add1 and fp_mul2
//   handle NAME => CLASS name N type T [gpuResourceID R | gpuResourceID raises X] | nil
//                                           functionHandleWithFunction:
//   handle add1 again => same|other         a second ask of the same function's handle
//   handle by name NAME => ... | raises X   functionHandleWithName: (macOS 26)
//   table N => CLASS id R allocated A options O storage S cache C hazard H heap X aliasable B
//   table N purgeable ASK => OLD then OLD2  setPurgeableState: of ASK, then of KeepCurrent
//   dispatch NAME => v0 v1 ...              out[i] = table[which[i]](i + 10) for each i; the
//                                           argument buffer's without useResource: only where
//                                           ORACLE_FNPTR_UNUSED_ARGBUF is set (it faults the 5300M)
//   size N => S                             what the shader reads of a table's size
//   argenc => length L align A              the argument encoder of fp_arg_kernel's buffer 0
//   argenc bytes => HEX                     the bytes setVisibleFunctionTable:atIndex: writes
//   render pipeline => made CLASS | nil ERROR    fp_vs and fp_fs, fp_fs linking fp_red, fp_green
//   render handle NAME STAGE => ...         functionHandleWithFunction:stage:
//   render table STAGE => CLASS id R | nil  newVisibleFunctionTableWithDescriptor:stage:
//   draw WHICH => RGBA                      a pass whose fragments call table[WHICH]; the render
//                                           steps not where ORACLE_FNPTR_NO_RENDER is set
//   dylib NAME => ...                       the dynamic library steps (case dylib below)
//   dylib refuse NAME => CLASS | nil ERROR  what newDynamicLibrary: and newDynamicLibraryWithURL:
//                                           refuse: an executable library, its file
//   dylib label => A then B                 a made library's label, then after setLabel:
//   dylib serialize WHERE => YES|NO ERROR wrote N   serializeToURL: to a new file, or to a file
//                                           in a missing directory, then the file read back
//   dylib unresolved NAME => ...            a kernel linked against a library whose install name
//                                           names no file: its pipeline, bare and with the
//                                           library preloaded
//   dylib airData => N bytes, the file's YES|NO, the same object again YES|NO, references past
//                   its pool R | not answered
//                                           the private member Metal's compile of source reads,
//                                           asked twice: the caller owns what it answers
//   dylib linked data NAME => ...           the linked kernel's library made of its file's bytes
//                                           (newLibraryWithData:): its pipeline, bare and with the
//                                           library preloaded
//   dylib source compile sent => SEL,...    the messages a compile of source sends a dynamic
//                                           library of its options' libraries (a proxy over it)
//   dylib source NAME => ...                that compile's kernel: its pipeline, bare and with
//                                           the library preloaded
// Build: see README.md; it loads oracle_fnptr.metallib, liboracle_fnptr_dyn.metallib,
// oracle_fnptr_link.metallib and oracle_fnptr_link_missing.metallib from beside the executable.
#import "oracle_common.h"
#include <objc/message.h>
#include <objc/runtime.h>

static const char *cls(id o)
{
	return o ? class_getName(object_getClass(o)) : "nil";
}

static const char *err_text(NSError *e)
{
	if (!e)
		return "(none)";
	static char buf[512];
	snprintf(buf, sizeof(buf), "%s %ld: %s", e.domain.UTF8String, (long)e.code,
	         e.localizedDescription.UTF8String);
	for (char *p = buf; *p; p++)
		if (*p == '\n')
			*p = '|';
	return buf;
}

static NSURL *beside(const char *argv0, NSString *name)
{
	NSString *exe = [NSString stringWithUTF8String:argv0];
	if (![exe containsString:@"/"])
		exe = [@"./" stringByAppendingString:exe];
	NSString *dir = [[exe stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent];
	return [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:name]];
}

static void handle_line(const char *what, id<MTLFunctionHandle> h)
{
	if (!h) {
		printf("%s => nil\n", what);
		return;
	}
	printf("%s => %s name %s type %ld", what, cls(h), h.name.UTF8String, (long)h.functionType);
	if ([(id)h respondsToSelector:@selector(gpuResourceID)]) {
		@try {
			printf(" gpuResourceID %llu", (unsigned long long)h.gpuResourceID._impl);
		} @catch (NSException *x) {
			printf(" gpuResourceID raises %s", x.name.UTF8String);
		}
	}
	printf("\n");
}

static void table_line(const char *what, id<MTLVisibleFunctionTable> t)
{
	if (!t) {
		printf("%s => nil\n", what);
		return;
	}
	printf("%s => %s id %llu allocated %lu options %lu storage %lu cache %lu hazard %lu heap %s "
	       "aliasable %d label %s\n",
	       what, cls(t), (unsigned long long)t.gpuResourceID._impl, (unsigned long)t.allocatedSize,
	       (unsigned long)t.resourceOptions, (unsigned long)t.storageMode,
	       (unsigned long)t.cpuCacheMode, (unsigned long)t.hazardTrackingMode, cls(t.heap),
	       (int)t.isAliasable, t.label ? t.label.UTF8String : "(nil)");
}

static id<MTLBuffer> words(id<MTLDevice> d, const uint32_t *w, size_t n)
{
	return [d newBufferWithBytes:w length:n * 4 options:MTLResourceStorageModeShared];
}

// Runs `pipe` over `n` threads with `bind` setting buffer 0 (and any more), `which` at 1 and
// an output at 2; `before_commit` runs after the dispatch is encoded, before the commit.
static void dispatch(const char *what, id<MTLDevice> d, id<MTLCommandQueue> q,
                     id<MTLComputePipelineState> pipe, const uint32_t *which, size_t n,
                     void (^bind)(id<MTLComputeCommandEncoder>), void (^before_commit)(void))
{
	id<MTLBuffer> w = words(d, which, n);
	id<MTLBuffer> out = [d newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
	memset(out.contents, 0xa5, n * 4);
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:pipe];
	bind(e);
	[e setBuffer:w offset:0 atIndex:1];
	[e setBuffer:out offset:0 atIndex:2];
	[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(n, 1, 1)];
	[e endEncoding];
	if (before_commit)
		before_commit();
	oracle_commit_wait(cb, NULL, NULL);
	printf("%s =>", what);
	const int32_t *o = out.contents;
	for (size_t i = 0; i < n; i++)
		printf(" %d", o[i]);
	printf("\n");
}

static void compute(id<MTLDevice> d, id<MTLLibrary> lib, id<MTLCommandQueue> q)
{
	id<MTLFunction> add1 = [lib newFunctionWithName:@"fp_add1"];
	id<MTLFunction> mul2 = [lib newFunctionWithName:@"fp_mul2"];
	id<MTLFunction> neg = [lib newFunctionWithName:@"fp_neg"];
	printf("function add1 => %s type %ld\n", cls(add1), (long)add1.functionType);
	MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
	lf.functions = @[ add1, mul2 ];
	id<MTLComputePipelineState> pipes[3] = {nil, nil, nil};
	NSString *kernels[3] = {@"fp_kernel", @"fp_arg_kernel", @"fp_size_kernel"};
	for (int k = 0; k < 3; k++) {
		MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
		cd.computeFunction = [lib newFunctionWithName:kernels[k]];
		cd.linkedFunctions = lf;
		NSError *e = nil;
		pipes[k] = [d newComputePipelineStateWithDescriptor:cd
		                                            options:MTLPipelineOptionNone
		                                         reflection:nil
		                                              error:&e];
		printf("compute pipeline %s => %s %s\n", kernels[k].UTF8String,
		       pipes[k] ? "made" : "nil", pipes[k] ? cls(pipes[k]) : err_text(e));
	}
	id<MTLComputePipelineState> p = pipes[0];
	if (!p)
		return;
	id<MTLFunctionHandle> h_add1 = [p functionHandleWithFunction:add1];
	id<MTLFunctionHandle> h_mul2 = [p functionHandleWithFunction:mul2];
	handle_line("handle add1", h_add1);
	handle_line("handle mul2", h_mul2);
	handle_line("handle neg", [p functionHandleWithFunction:neg]);
	handle_line("handle kernel", [p functionHandleWithFunction:[lib newFunctionWithName:@"fp_kernel"]]);
	printf("handle add1 again => %s\n", [p functionHandleWithFunction:add1] == h_add1 ? "same" : "other");
	if ([(id)p respondsToSelector:@selector(functionHandleWithName:)]) {
		@try {
			handle_line("handle by name fp_mul2", [p functionHandleWithName:@"fp_mul2"]);
		} @catch (NSException *x) {
			printf("handle by name fp_mul2 => raises %s\n", x.name.UTF8String);
		}
	}
	NSUInteger counts[] = {0, 1, 3, 2016, 2017, 4096};
	for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
		MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
		td.functionCount = counts[i];
		id<MTLVisibleFunctionTable> t = [p newVisibleFunctionTableWithDescriptor:td];
		char what[64];
		snprintf(what, sizeof(what), "table %lu", (unsigned long)counts[i]);
		table_line(what, t);
	}
	MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
	td.functionCount = 3;
	id<MTLVisibleFunctionTable> t = [p newVisibleFunctionTableWithDescriptor:td];
	t.label = @"oracle-table";
	table_line("table labelled", t);
	MTLPurgeableState first = [t setPurgeableState:MTLPurgeableStateVolatile];
	MTLPurgeableState then = [t setPurgeableState:MTLPurgeableStateKeepCurrent];
	printf("table purgeable volatile => %lu then %lu\n", (unsigned long)first, (unsigned long)then);
	[t setFunction:h_add1 atIndex:0];
	id<MTLFunctionHandle> range[2] = {h_mul2, h_add1};
	[t setFunctions:range withRange:NSMakeRange(1, 2)];
	uint32_t which[6] = {0, 1, 2, 1, 0, 2};
	dispatch("dispatch bound", d, q, p, which, 6, ^(id<MTLComputeCommandEncoder> e) {
		[e setVisibleFunctionTable:t atBufferIndex:0];
	}, nil);
	dispatch("dispatch range", d, q, p, which, 6, ^(id<MTLComputeCommandEncoder> e) {
		id<MTLVisibleFunctionTable> ts[1] = {t};
		[e setVisibleFunctionTables:ts withBufferRange:NSMakeRange(0, 1)];
	}, nil);
	// An entry set after the dispatch was encoded, before its commit.
	id<MTLVisibleFunctionTable> late = [p newVisibleFunctionTableWithDescriptor:td];
	[late setFunction:h_add1 atIndex:0];
	[late setFunction:h_add1 atIndex:1];
	[late setFunction:h_add1 atIndex:2];
	uint32_t zero[4] = {0, 1, 2, 0};
	dispatch("dispatch set before commit", d, q, p, zero, 4, ^(id<MTLComputeCommandEncoder> e) {
		[e setVisibleFunctionTable:late atBufferIndex:0];
	}, ^{
		[late setFunction:h_mul2 atIndex:0];
	});
	if (pipes[2]) {
		uint32_t none[1] = {0};
		dispatch("size 3", d, q, pipes[2], none, 1, ^(id<MTLComputeCommandEncoder> e) {
			[e setVisibleFunctionTable:t atBufferIndex:0];
		}, nil);
	}
	id<MTLComputePipelineState> ap = pipes[1];
	if (!ap)
		return;
	// A table of fp_arg_kernel's own, in an argument buffer, with no useResource.
	id<MTLVisibleFunctionTable> at = [ap newVisibleFunctionTableWithDescriptor:td];
	[at setFunction:[ap functionHandleWithFunction:mul2] atIndex:0];
	[at setFunction:[ap functionHandleWithFunction:add1] atIndex:1];
	[at setFunction:[ap functionHandleWithFunction:mul2] atIndex:2];
	table_line("table of arg pipeline", at);
	id<MTLArgumentEncoder> ae =
	    [[lib newFunctionWithName:@"fp_arg_kernel"] newArgumentEncoderWithBufferIndex:0];
	printf("argenc => length %lu align %lu\n", (unsigned long)ae.encodedLength,
	       (unsigned long)ae.alignment);
	id<MTLBuffer> args = [d newBufferWithLength:ae.encodedLength + 16
	                                    options:MTLResourceStorageModeShared];
	memset(args.contents, 0xa5, args.length);
	[ae setArgumentBuffer:args offset:0];
	[ae setVisibleFunctionTable:at atIndex:0];
	printf("argenc bytes =>");
	for (NSUInteger i = 0; i < ae.encodedLength; i++)
		printf("%s%02x", i % 8 ? "" : " ", ((const uint8_t *)args.contents)[i]);
	printf("\n");
	// Without useResource: the AMD Radeon Pro 5300M's command buffer times out (macOS 26.7.1), a
	// GPU fault every process on the Mac meets: run only where asked.
	if (getenv("ORACLE_FNPTR_UNUSED_ARGBUF"))
		dispatch("dispatch argbuf", d, q, ap, which, 6, ^(id<MTLComputeCommandEncoder> e) {
			[e setBuffer:args offset:0 atIndex:0];
		}, nil);
	else
		printf("dispatch argbuf => skipped (ORACLE_FNPTR_UNUSED_ARGBUF=1 runs it)\n");
	dispatch("dispatch argbuf useResource", d, q, ap, which, 6, ^(id<MTLComputeCommandEncoder> e) {
		[e setBuffer:args offset:0 atIndex:0];
		[e useResource:at usage:MTLResourceUsageRead];
	}, nil);
}

static void render(id<MTLDevice> d, id<MTLLibrary> lib, id<MTLCommandQueue> q)
{
	id<MTLFunction> red = [lib newFunctionWithName:@"fp_red"];
	id<MTLFunction> green = [lib newFunctionWithName:@"fp_green"];
	MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
	rd.vertexFunction = [lib newFunctionWithName:@"fp_vs"];
	rd.fragmentFunction = [lib newFunctionWithName:@"fp_fs"];
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
	lf.functions = @[ red, green ];
	rd.fragmentLinkedFunctions = lf;
	NSError *e = nil;
	id<MTLRenderPipelineState> p = [d newRenderPipelineStateWithDescriptor:rd error:&e];
	printf("render pipeline => %s %s\n", p ? "made" : "nil", p ? cls(p) : err_text(e));
	if (!p)
		return;
	id<MTLFunctionHandle> h_red = [p functionHandleWithFunction:red stage:MTLRenderStageFragment];
	id<MTLFunctionHandle> h_green = [p functionHandleWithFunction:green stage:MTLRenderStageFragment];
	handle_line("render handle red fragment", h_red);
	handle_line("render handle green fragment", h_green);
	handle_line("render handle red vertex", [p functionHandleWithFunction:red stage:MTLRenderStageVertex]);
	MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
	td.functionCount = 2;
	id<MTLVisibleFunctionTable> t = [p newVisibleFunctionTableWithDescriptor:td
	                                                                   stage:MTLRenderStageFragment];
	table_line("render table fragment", t);
	table_line("render table vertex",
	           [p newVisibleFunctionTableWithDescriptor:td stage:MTLRenderStageVertex]);
	if (!t)
		return;
	[t setFunction:h_red atIndex:0];
	[t setFunction:h_green atIndex:1];
	MTLTextureDescriptor *xd =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                      width:8
	                                                     height:8
	                                                  mipmapped:NO];
	xd.usage = MTLTextureUsageRenderTarget;
	xd.storageMode = MTLStorageModeManaged;
	for (uint32_t which = 0; which < 2; which++) {
		id<MTLTexture> target = [d newTextureWithDescriptor:xd];
		MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 1, 1);
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:pass];
		[re setRenderPipelineState:p];
		[re setFragmentVisibleFunctionTable:t atBufferIndex:0];
		[re setFragmentBytes:&which length:4 atIndex:1];
		[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[re endEncoding];
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b synchronizeResource:target];
		[b endEncoding];
		oracle_commit_wait(cb, NULL, NULL);
		uint8_t px[4] = {0};
		[target getBytes:px bytesPerRow:32 fromRegion:MTLRegionMake2D(3, 3, 1, 1) mipmapLevel:0];
		printf("draw %u => %u %u %u %u\n", which, px[0], px[1], px[2], px[3]);
	}
}

static void dylib_run(const char *what, id<MTLDevice> d, id<MTLCommandQueue> q,
                      MTLComputePipelineDescriptor *cd)
{
	NSError *e = nil;
	id<MTLComputePipelineState> p = [d newComputePipelineStateWithDescriptor:cd
	                                                                 options:MTLPipelineOptionNone
	                                                              reflection:nil
	                                                                   error:&e];
	printf("dylib %s pipeline => %s %s\n", what, p ? "made" : "nil", p ? cls(p) : err_text(e));
	if (!p)
		return;
	char name[96];
	snprintf(name, sizeof(name), "dylib %s dispatch", what);
	id<MTLBuffer> out = [d newBufferWithLength:16 options:MTLResourceStorageModeShared];
	memset(out.contents, 0xa5, 16);
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:p];
	[enc setBuffer:out offset:0 atIndex:0];
	[enc dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
	[enc endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	const int32_t *o = out.contents;
	printf("%s => %d %d %d %d\n", name, o[0], o[1], o[2], o[3]);
}

// A proxy over a dynamic library that keeps the name of each message sent to it, then forwards it.
@interface OracleDylibSpy : NSProxy
@property(strong) id target;
@property(strong) NSMutableArray<NSString *> *sent;
@end
@implementation OracleDylibSpy
- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel
{
	return [_target methodSignatureForSelector:sel];
}
- (void)forwardInvocation:(NSInvocation *)inv
{
	[_sent addObject:NSStringFromSelector(inv.selector)];
	[inv invokeWithTarget:_target];
}
@end

static long file_size(NSURL *url)
{
	NSDictionary *a = [[NSFileManager defaultManager] attributesOfItemAtPath:url.path error:nil];
	return a ? (long)[a fileSize] : -1L;
}

// What the dynamic library calls refuse, a library's label, serializeToURL:, and a kernel whose
// dynamic library's install name names no file.
static void dylib_more(id<MTLDevice> d, const char *argv0, id<MTLCommandQueue> q,
                       id<MTLDynamicLibrary> dl)
{
	NSError *e = nil;
	NSURL *exe_url = beside(argv0, @"oracle_fnptr.metallib");
	id<MTLLibrary> exe = [d newLibraryWithURL:exe_url error:&e];
	if (exe) {
		e = nil;
		id<MTLDynamicLibrary> r = [d newDynamicLibrary:exe error:&e];
		printf("dylib refuse executable => %s %s\n", cls(r), err_text(e));
	}
	e = nil;
	id<MTLDynamicLibrary> ru = [d newDynamicLibraryWithURL:exe_url error:&e];
	printf("dylib refuse executable file => %s %s\n", cls(ru), err_text(e));
	e = nil;
	NSURL *gone = beside(argv0, @"liboracle_fnptr_none.metallib");
	id<MTLDynamicLibrary> rg = [d newDynamicLibraryWithURL:gone error:&e];
	printf("dylib refuse missing file => %s %s\n", cls(rg), err_text(e));

	printf("dylib label => %s", dl.label ? dl.label.UTF8String : "(nil)");
	dl.label = @"oracle-dylib";
	printf(" then %s\n", dl.label ? dl.label.UTF8String : "(nil)");

	NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"oracle_fnptr_dylib"];
	[[NSFileManager defaultManager] createDirectoryAtPath:dir
	                          withIntermediateDirectories:YES
	                                           attributes:nil
	                                                error:nil];
	NSURL *to = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"written.metallib"]];
	[[NSFileManager defaultManager] removeItemAtURL:to error:nil];
	e = nil;
	BOOL ok = [dl serializeToURL:to error:&e];
	printf("dylib serialize new => %s %s wrote %ld\n", ok ? "YES" : "NO", err_text(e),
	       file_size(to));
	if (ok) {
		e = nil;
		id<MTLDynamicLibrary> back = [d newDynamicLibraryWithURL:to error:&e];
		printf("dylib serialize read back => %s installName %s error %s\n", cls(back),
		       back.installName ? back.installName.UTF8String : "(nil)", err_text(e));
	}
	NSURL *nowhere = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"none/x.metallib"]];
	e = nil;
	ok = [dl serializeToURL:nowhere error:&e];
	printf("dylib serialize missing dir => %s %s wrote %ld\n", ok ? "YES" : "NO", err_text(e),
	       file_size(nowhere));
	[[NSFileManager defaultManager] removeItemAtPath:dir error:nil];

	e = nil;
	id<MTLLibrary> miss = [d newLibraryWithURL:beside(argv0, @"oracle_fnptr_link_missing.metallib")
	                                     error:&e];
	printf("dylib unresolved library => %s error %s\n", cls(miss), err_text(e));
	if (miss) {
		MTLComputePipelineDescriptor *bare = [MTLComputePipelineDescriptor new];
		bare.computeFunction = [miss newFunctionWithName:@"fpd_kernel"];
		dylib_run("unresolved bare", d, q, bare);
		MTLComputePipelineDescriptor *pre = [MTLComputePipelineDescriptor new];
		pre.computeFunction = [miss newFunctionWithName:@"fpd_kernel"];
		pre.preloadedLibraries = @[ dl ];
		dylib_run("unresolved preloaded", d, q, pre);
	}
}

static void dylib(id<MTLDevice> d, const char *argv0, id<MTLCommandQueue> q)
{
	NSError *e = nil;
	NSURL *url = beside(argv0, @"liboracle_fnptr_dyn.metallib");
	id<MTLLibrary> lib = [d newLibraryWithURL:url error:&e];
	printf("dylib library => %s type %ld installName %s error %s\n", cls(lib),
	       lib ? (long)lib.type : -1L, lib.installName ? lib.installName.UTF8String : "(nil)",
	       err_text(e));
	if (!lib)
		return;
	e = nil;
	id<MTLDynamicLibrary> dl = [d newDynamicLibrary:lib error:&e];
	printf("dylib newDynamicLibrary => %s installName %s label %s error %s\n", cls(dl),
	       dl.installName ? dl.installName.UTF8String : "(nil)",
	       dl.label ? dl.label.UTF8String : "(nil)", err_text(e));
	e = nil;
	id<MTLDynamicLibrary> du = [d newDynamicLibraryWithURL:url error:&e];
	printf("dylib newDynamicLibraryWithURL => %s installName %s error %s\n", cls(du),
	       du.installName ? du.installName.UTF8String : "(nil)", err_text(e));
	e = nil;
	id<MTLLibrary> link = [d newLibraryWithURL:beside(argv0, @"oracle_fnptr_link.metallib") error:&e];
	printf("dylib linked library => %s type %ld error %s\n", cls(link),
	       link ? (long)link.type : -1L, err_text(e));
	if (link && dl) {
		MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
		cd.computeFunction = [link newFunctionWithName:@"fpd_kernel"];
		cd.preloadedLibraries = @[ dl ];
		dylib_run("preloaded", d, q, cd);
		MTLComputePipelineDescriptor *by_name = [MTLComputePipelineDescriptor new];
		by_name.computeFunction = [link newFunctionWithName:@"fpd_kernel"];
		dylib_run("by install name", d, q, by_name);
	}
	if (!dl)
		return;
	SEL air = sel_registerName("airData");
	NSData *file = [NSData dataWithContentsOfURL:url];
	if ([(id)dl respondsToSelector:air]) {
		// Ask twice, each in a pool of its own: an answer the caller owns outlives its pool with
		// one reference more than the object holds.
		void *(*send)(id, SEL) = (void *(*)(id, SEL))objc_msgSend;
		void *first = NULL, *second = NULL;
		@autoreleasepool {
			first = send(dl, air);
		}
		@autoreleasepool {
			second = send(dl, air);
		}
		long held = first ? CFGetRetainCount(first) : 0;
		if (second)
			CFRelease(second);
		dispatch_data_t data = (__bridge_transfer dispatch_data_t)first;
		const void *bytes = NULL;
		size_t size = 0;
		dispatch_data_t map = data ? dispatch_data_create_map(data, &bytes, &size) : nil;
		printf("dylib airData => %zu bytes, the file's %s, the same object again %s, references "
		       "past its pool %ld\n",
		       size,
		       map && size == file.length && memcmp(bytes, file.bytes, size) == 0 ? "YES" : "NO",
		       first == second ? "YES" : "NO", held);
	} else {
		printf("dylib airData => not answered\n");
	}
	NSData *link_bytes =
	    [NSData dataWithContentsOfURL:beside(argv0, @"oracle_fnptr_link.metallib")];
	dispatch_data_t link_data = dispatch_data_create(link_bytes.bytes, link_bytes.length, nil,
	                                                 DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	e = nil;
	id<MTLLibrary> link_lib = [d newLibraryWithData:link_data error:&e];
	printf("dylib linked data library => %s error %s\n", cls(link_lib), err_text(e));
	if (link_lib) {
		MTLComputePipelineDescriptor *bare = [MTLComputePipelineDescriptor new];
		bare.computeFunction = [link_lib newFunctionWithName:@"fpd_kernel"];
		dylib_run("linked data bare", d, q, bare);
		MTLComputePipelineDescriptor *pre = [MTLComputePipelineDescriptor new];
		pre.computeFunction = [link_lib newFunctionWithName:@"fpd_kernel"];
		pre.preloadedLibraries = @[ dl ];
		dylib_run("linked data preloaded", d, q, pre);
	}
	dylib_more(d, argv0, q, dl);
	OracleDylibSpy *spy = [OracleDylibSpy alloc];
	spy.target = dl;
	spy.sent = [NSMutableArray array];
	MTLCompileOptions *co = [MTLCompileOptions new];
	co.libraries = @[ (id<MTLDynamicLibrary>)spy ];
	NSString *src = @"#include <metal_stdlib>\n"
	                @"extern int fpd_triple(int x);\n"
	                @"kernel void fpd_src_kernel(device int *out [[buffer(0)]], uint i "
	                @"[[thread_position_in_grid]]) { out[i] = fpd_triple(int(i) + 20); }\n";
	e = nil;
	id<MTLLibrary> src_lib = [d newLibraryWithSource:src options:co error:&e];
	printf("dylib source compile sent => %s\n",
	       spy.sent.count ? [spy.sent componentsJoinedByString:@","].UTF8String : "(nothing)");
	printf("dylib source library => %s error %s\n", cls(src_lib), err_text(e));
	if (src_lib) {
		MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
		cd.computeFunction = [src_lib newFunctionWithName:@"fpd_src_kernel"];
		dylib_run("source", d, q, cd);
		MTLComputePipelineDescriptor *pre = [MTLComputePipelineDescriptor new];
		pre.computeFunction = [src_lib newFunctionWithName:@"fpd_src_kernel"];
		pre.preloadedLibraries = @[ dl ];
		dylib_run("source preloaded", d, q, pre);
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE\n", argv[0]);
			return 2;
		}
		id<MTLDevice> d = oracle_device(argv[1]);
		oracle_header("oracle_fnptr", d);
		printf("supports functionPointers => %s\n", d.supportsFunctionPointers ? "YES" : "NO");
		printf("supports functionPointersFromRender => %s\n",
		       d.supportsFunctionPointersFromRender ? "YES" : "NO");
		printf("supports dynamicLibraries => %s\n", d.supportsDynamicLibraries ? "YES" : "NO");
		printf("supports renderDynamicLibraries => %s\n",
		       d.supportsRenderDynamicLibraries ? "YES" : "NO");
		id<MTLCommandQueue> q = [d newCommandQueue];
		id<MTLLibrary> lib = oracle_library(d, argv[0], "oracle_fnptr");
		if (lib) {
			compute(d, lib, q);
			// The Intel UHD Graphics 630 ends the process at a render pipeline's handle
			// (macOS 26.7.1): ORACLE_FNPTR_NO_RENDER skips the render steps.
			if (getenv("ORACLE_FNPTR_NO_RENDER"))
				printf("render => skipped (ORACLE_FNPTR_NO_RENDER)\n");
			else
				render(d, lib, q);
		}
		dylib(d, argv[0], q);
	}
	return 0;
}
