// oracle_icb: what indirect command buffers do on a Mac GPU: the command types and limits the
// device makes, what the descriptor and the command setters refuse, what CPU- and GPU-encoded
// commands draw and dispatch, execution ranges, resets, copies and optimizes, when CPU writes
// reach an execution, residency, and timing.
//   oracle_icb DEVICE [CASE]            every case (or the one named), each in a child process
//   oracle_icb DEVICE --case CASE [N]   one case (or its subcase N): what the children run
// Set ORACLE_VALIDATE=1 to run the descriptor and setters subcases again under Metal's validation
// (MTL_DEBUG_LAYER=1), whose messages land in the output.
// Cases:
//   info        device answers, a fresh descriptor's defaults, an ICB's properties as a resource
//   types       which command types, alone and mixed, the device makes an ICB of
//   limits      the largest maxCommandCount made per type; size and allocatedSize
//   descriptor  each descriptor field past its range: made or nil (subcases)
//   setters     command setters past what the descriptor allows, then executed (subcases)
//   render      CPU-encoded draws: inherit variants, indexed draws, base vertex and instance
//   patches     patch and indexed patch draws
//   ranges      execute ranges, the indirect range form, commands never encoded
//   reset       resetWithRange:, a command's reset, a blit's reset, copy and optimize
//   order       CPU writes after the execute is encoded and before the commit; while in flight
//   compute     CPU-encoded dispatches: threadgroups, threads, threadgroup memory, barriers,
//               inherit variants, ranges
//   gpu         kernel-encoded render and compute commands (an ICB in an argument buffer), GPU
//               reset, a GPU-encoded ICB copied, CPU and GPU writes mixed
//   residency   an executed ICB's buffers without useResource
//   timing      CPU encode cost per command at 1024 and 16384, the commit after it, what each
//               encode draws, execution GPU time
//   freed       buffers the commands name released before an execute, a heap's among them
//   resets      what a command's reset and resetWithRange: keep of a command's pipeline state
// Render results print as the 8 cells of a 64x8 target, each the hex digit of the color index
// found there ('.' where nothing drew, '?' where something unexpected did).
// Build: see README.md.
#import "oracle_common.h"
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
static id<MTLLibrary> L;
static const char *gArgv0;

#define kCells ((NSUInteger)8)
static id<MTLBuffer> gRects, gColors, gTf, gIdx16, gIdx32, gCpIdx;
static id<MTLTexture> gTarget;
static id<MTLBuffer> gReadback;
static id<MTLRenderPipelineState> gDraw, gPatch, gPatchIdx;
static id<MTLComputePipelineState> gAdd, gMul, gTg, gSpin;

static const char *yesno(BOOL b)
{
	return b ? "yes" : "no";
}

static id<MTLBuffer> buf(NSUInteger len)
{
	id<MTLBuffer> b = [D newBufferWithLength:len options:MTLResourceStorageModeShared];
	memset(b.contents, 0, len);
	return b;
}

static id<MTLFunction> fn(const char *name)
{
	id<MTLFunction> f = [L newFunctionWithName:[NSString stringWithUTF8String:name]];
	if (!f)
		printf("  ! no function %s\n", name);
	return f;
}

static id<MTLComputePipelineState> cpso(const char *name, BOOL icb)
{
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = fn(name);
	cd.supportIndirectCommandBuffers = icb;
	NSError *e = nil;
	id<MTLComputePipelineState> p = [D newComputePipelineStateWithDescriptor:cd
	                                                                 options:0
	                                                              reflection:nil
	                                                                   error:&e];
	if (!p)
		printf("  ! compute pipeline %s: %s\n", name, e.description.UTF8String);
	return p;
}

static id<MTLRenderPipelineState> rpso(const char *vs, int patches, BOOL icb)
{
	MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
	rd.vertexFunction = fn(vs);
	rd.fragmentFunction = fn("icb_fs");
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	rd.supportIndirectCommandBuffers = icb;
	if (patches) {
		rd.maxTessellationFactor = 16;
		rd.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
		rd.tessellationPartitionMode = MTLTessellationPartitionModePow2;
		rd.tessellationControlPointIndexType = patches == 2
		                                           ? MTLTessellationControlPointIndexTypeUInt32
		                                           : MTLTessellationControlPointIndexTypeNone;
	}
	NSError *e = nil;
	id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithDescriptor:rd error:&e];
	if (!p)
		printf("  ! render pipeline %s: %s\n", vs, e.description.UTF8String);
	return p;
}

static void setup(void)
{
	L = oracle_library(D, gArgv0, "oracle_icb");
	if (!L)
		exit(1);
	gRects = buf(32 * 16);
	float *r = gRects.contents;
	for (NSUInteger i = 0; i < 32; i++) {
		float x0 = -1.0f + 2.0f * (float)(i % kCells) / kCells;
		r[i * 4 + 0] = x0;
		r[i * 4 + 1] = -1.0f;
		r[i * 4 + 2] = x0 + 2.0f / kCells;
		r[i * 4 + 3] = 1.0f;
	}
	gColors = buf(32 * 16);
	float *c = gColors.contents;
	for (NSUInteger k = 0; k < 32; k++) {
		c[k * 4 + 0] = (float)(((k % 15) + 1) * 16) / 255.0f;
		c[k * 4 + 3] = 1.0f;
	}
	// MTLQuadTessellationFactorsHalf, every factor 1.0, 32 patches.
	gTf = buf(32 * 12);
	uint16_t *tf = gTf.contents;
	for (NSUInteger i = 0; i < 32 * 6; i++)
		tf[i] = 0x3c00;
	gIdx16 = buf(64);
	uint16_t *i16 = gIdx16.contents;
	for (int i = 0; i < 4; i++)
		i16[2 + i] = (uint16_t)i; // at offset 4
	gIdx32 = buf(64);
	uint32_t *i32 = gIdx32.contents;
	for (int i = 0; i < 4; i++)
		i32[i] = (uint32_t)i;
	gCpIdx = buf(32 * 4 * 4);
	uint32_t *cp = gCpIdx.contents;
	for (int i = 0; i < 32 * 4; i++)
		cp[i] = (uint32_t)(i % 4);
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:kCells * 8
	                                                      height:8
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModePrivate;
	gTarget = [D newTextureWithDescriptor:td];
	gReadback = buf(kCells * 8 * 4 * 8);
	gDraw = rpso("icb_vs", 0, YES);
	gPatch = rpso("icb_patch_vs", 1, YES);
	gPatchIdx = rpso("icb_patch_vs", 2, YES);
	gAdd = cpso("icb_add", YES);
	gMul = cpso("icb_mul", YES);
	gTg = cpso("icb_tg", YES);
	gSpin = cpso("icb_spin", NO);
}

static MTLIndirectCommandBufferDescriptor *cdesc_info(void)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
	d.maxKernelBufferBindCount = 2;
	return d;
}

static MTLIndirectCommandBufferDescriptor *rdesc(MTLIndirectCommandType types, BOOL inheritPso,
                                                 BOOL inheritBuffers)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = types;
	d.inheritPipelineState = inheritPso;
	d.inheritBuffers = inheritBuffers;
	d.maxVertexBufferBindCount = inheritBuffers ? 0 : 1;
	d.maxFragmentBufferBindCount = inheritBuffers ? 0 : 1;
	return d;
}

static id<MTLIndirectCommandBuffer> icb(MTLIndirectCommandBufferDescriptor *d, NSUInteger n)
{
	id<MTLIndirectCommandBuffer> b = [D newIndirectCommandBufferWithDescriptor:d
	                                                           maxCommandCount:n
	                                                                   options:0];
	if (!b)
		printf("  ! ICB of %lu commands not made\n", (unsigned long)n);
	return b;
}

// Command `i` of a render ICB draws cell `cell` in color `color` (no inheritance).
static void enc_draw(id<MTLIndirectCommandBuffer> b, NSUInteger i, NSUInteger cell,
                     NSUInteger color, BOOL setPso)
{
	id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
	if (setPso)
		[c setRenderPipelineState:gDraw];
	[c setVertexBuffer:gRects offset:cell * 16 atIndex:0];
	[c setFragmentBuffer:gColors offset:color * 16 atIndex:0];
	[c drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:1
	     baseInstance:0];
}

static id<MTLRenderCommandEncoder> begin_pass(id<MTLCommandBuffer> cb)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor new];
	rp.colorAttachments[0].texture = gTarget;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e useResource:gRects usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	[e useResource:gColors usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
	[e useResource:gTf usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	[e useResource:gIdx16 usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	[e useResource:gIdx32 usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	[e useResource:gCpIdx usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	return e;
}

static void end_pass(id<MTLCommandBuffer> cb, id<MTLRenderCommandEncoder> e)
{
	[e endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:gTarget
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(kCells * 8, 8, 1)
	                toBuffer:gReadback
	       destinationOffset:0
	  destinationBytesPerRow:kCells * 8 * 4
	destinationBytesPerImage:kCells * 8 * 4 * 8];
	[b endEncoding];
}

// The 8 cells' colors as a string.
static const char *cells(void)
{
	static char s[kCells + 1];
	const uint8_t *px = gReadback.contents;
	for (NSUInteger i = 0; i < kCells; i++) {
		const uint8_t *p = px + (4 * kCells * 8) * 4 + (i * 8 + 4) * 4; // row 4, cell centre
		if (p[0] == 0 && p[3] == 0)
			s[i] = '.';
		else if (p[0] % 16 == 0 && p[0] >= 16 && p[1] == 0 && p[2] == 0 && p[3] == 255)
			s[i] = "0123456789abcde"[p[0] / 16 - 1];
		else
			s[i] = '?';
	}
	s[kCells] = 0;
	return s;
}

static void run(id<MTLCommandBuffer> cb)
{
	oracle_commit_wait(cb, NULL, NULL);
}

static void line(const char *what, const char *expect)
{
	const char *got = cells();
	printf("  %-60s %s%s%s\n", what, got, expect ? (strcmp(got, expect) ? "  (expected " : "") : "",
	       expect && strcmp(got, expect) ? expect : "");
	if (expect && strcmp(got, expect))
		printf("    ^ differs\n");
}

// Executes `b`'s `range` (NSNotFound location: the whole buffer) in one pass, with `pso` set on
// the encoder where not nil, and the buffers set on the encoder where `encBuffers`.
static void exec_render(id<MTLIndirectCommandBuffer> b, NSRange range,
                        id<MTLRenderPipelineState> pso, BOOL encBuffers, NSUInteger encColor)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLRenderCommandEncoder> e = begin_pass(cb);
	if (pso)
		[e setRenderPipelineState:pso];
	if (encBuffers) {
		[e setVertexBuffer:gRects offset:0 atIndex:0];
		[e setFragmentBuffer:gColors offset:encColor * 16 atIndex:0];
	}
	if (range.location == NSNotFound)
		range = NSMakeRange(0, b.size);
	[e executeCommandsInBuffer:b withRange:range];
	end_pass(cb, e);
	run(cb);
}

// The members of `proto` (required and optional, instance) `obj` does not respond to.
static void responds(const char *what, id obj, Protocol *proto)
{
	printf("  %s (%s) lacks:", what, object_getClassName(obj));
	int missing = 0;
	for (int req = 1; req >= 0; req--) {
		unsigned n = 0;
		struct objc_method_description *m =
		    protocol_copyMethodDescriptionList(proto, (BOOL)req, YES, &n);
		for (unsigned i = 0; i < n; i++)
			if (![obj respondsToSelector:m[i].name]) {
				printf(" %s%s", req ? "" : "(opt)", sel_getName(m[i].name));
				missing++;
			}
		free(m);
	}
	printf("%s\n", missing ? "" : " nothing");
}

// ---- info ------------------------------------------------------------------------------------

static void case_info(void)
{
	printf("info:\n");
	printf("  argumentBuffersSupport %ld, maxBufferLength %lu, families:", (long)D.argumentBuffersSupport,
	       (unsigned long)D.maxBufferLength);
	const struct {
		MTLGPUFamily f;
		const char *n;
	} fams[] = {{MTLGPUFamilyMac2, "Mac2"},     {MTLGPUFamilyApple7, "Apple7"},
	            {MTLGPUFamilyApple8, "Apple8"}, {MTLGPUFamilyApple9, "Apple9"},
	            {MTLGPUFamilyMetal3, "Metal3"}, {MTLGPUFamilyCommon3, "Common3"}};
	for (size_t i = 0; i < sizeof fams / sizeof fams[0]; i++)
		if ([D supportsFamily:fams[i].f])
			printf(" %s", fams[i].n);
	printf("\n");
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	printf("  fresh descriptor: commandTypes %lu inheritPipelineState %s inheritBuffers %s "
	       "vertex %lu fragment %lu kernel %lu kernelTgMem %lu object %lu mesh %lu objectTgMem %lu "
	       "rayTracing %s dynamicStride %s colorMapping %s\n",
	       (unsigned long)d.commandTypes, yesno(d.inheritPipelineState), yesno(d.inheritBuffers),
	       (unsigned long)d.maxVertexBufferBindCount, (unsigned long)d.maxFragmentBufferBindCount,
	       (unsigned long)d.maxKernelBufferBindCount,
	       (unsigned long)d.maxKernelThreadgroupMemoryBindCount,
	       (unsigned long)d.maxObjectBufferBindCount, (unsigned long)d.maxMeshBufferBindCount,
	       (unsigned long)d.maxObjectThreadgroupMemoryBindCount, yesno(d.supportRayTracing),
	       yesno(d.supportDynamicAttributeStride), yesno(d.supportColorAttachmentMapping));
	printf("  fresh descriptor (26): depthStencil %s depthBias %s depthClip %s cull %s winding %s "
	       "fill %s\n",
	       yesno(d.inheritDepthStencilState), yesno(d.inheritDepthBias), yesno(d.inheritDepthClipMode),
	       yesno(d.inheritCullMode), yesno(d.inheritFrontFacingWinding), yesno(d.inheritTriangleFillMode));
	const struct {
		MTLIndirectCommandType t;
		MTLResourceOptions o;
		const char *n;
	} kinds[] = {
	    {MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed, 0, "render shared"},
	    {MTLIndirectCommandTypeDraw, MTLResourceStorageModePrivate, "render private"},
	    {MTLIndirectCommandTypeDraw, MTLResourceStorageModeManaged, "render managed"},
	    {MTLIndirectCommandTypeConcurrentDispatch, 0, "compute shared"},
	    {MTLIndirectCommandTypeDraw, MTLResourceHazardTrackingModeUntracked, "render untracked"},
	    {MTLIndirectCommandTypeDraw, MTLResourceCPUCacheModeWriteCombined, "render wc"},
	};
	for (size_t i = 0; i < sizeof kinds / sizeof kinds[0]; i++) {
		MTLIndirectCommandBufferDescriptor *k = [MTLIndirectCommandBufferDescriptor new];
		k.commandTypes = kinds[i].t;
		k.maxVertexBufferBindCount = 1;
		k.maxFragmentBufferBindCount = 1;
		k.maxKernelBufferBindCount = 2;
		id<MTLIndirectCommandBuffer> b = [D newIndirectCommandBufferWithDescriptor:k
		                                                           maxCommandCount:16
		                                                                   options:kinds[i].o];
		if (!b) {
			printf("  %s: nil\n", kinds[i].n);
			continue;
		}
		printf("  %s: class %s size %lu allocatedSize %lu gpuResourceID %llu storage %ld cpuCache "
		       "%ld hazard %ld options 0x%lx heap %s heapOffset %lu label '%s' aliasable %s "
		       "device-same %s\n",
		       kinds[i].n, object_getClassName(b), (unsigned long)b.size,
		       (unsigned long)b.allocatedSize, (unsigned long long)b.gpuResourceID._impl,
		       (long)b.storageMode, (long)b.cpuCacheMode, (long)b.hazardTrackingMode,
		       (unsigned long)b.resourceOptions, b.heap ? "yes" : "nil", (unsigned long)b.heapOffset,
		       b.label.UTF8String ?: "(nil)", yesno(b.isAliasable), yesno(b.device == D));
		MTLPurgeableState p0 = [b setPurgeableState:MTLPurgeableStateKeepCurrent];
		MTLPurgeableState p1 = [b setPurgeableState:MTLPurgeableStateVolatile];
		MTLPurgeableState p2 = [b setPurgeableState:MTLPurgeableStateNonVolatile];
		printf("    purgeable: current %ld, after volatile %ld, after nonvolatile %ld\n", (long)p0,
		       (long)p1, (long)p2);
		if (kinds[i].t & MTLIndirectCommandTypeConcurrentDispatch) {
			id<MTLIndirectComputeCommand> c0 = [b indirectComputeCommandAtIndex:0];
			id<MTLIndirectComputeCommand> c0b = [b indirectComputeCommandAtIndex:0];
			printf("    command class %s, same object twice %s\n", object_getClassName(c0),
			       yesno(c0 == c0b));
		} else {
			id<MTLIndirectRenderCommand> c0 = [b indirectRenderCommandAtIndex:0];
			id<MTLIndirectRenderCommand> c0b = [b indirectRenderCommandAtIndex:0];
			printf("    command class %s, same object twice %s\n", object_getClassName(c0),
			       yesno(c0 == c0b));
		}
	}
	// Resource ids of consecutive ICBs, beside a texture's and a pipeline's.
	printf("  gpuResourceIDs of 4 ICBs:");
	NSMutableArray *keep = [NSMutableArray new];
	for (int i = 0; i < 4; i++) {
		id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 4);
		[keep addObject:b];
		printf(" %llu", (unsigned long long)b.gpuResourceID._impl);
	}
	printf("; texture %llu; pipeline %llu\n", (unsigned long long)gTarget.gpuResourceID._impl,
	       (unsigned long long)gDraw.gpuResourceID._impl);
	{
		MTLIndirectCommandBufferDescriptor *k = [MTLIndirectCommandBufferDescriptor new];
		k.commandTypes = MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawMeshThreadgroups;
		k.maxVertexBufferBindCount = 1;
		k.maxFragmentBufferBindCount = 1;
		k.maxObjectBufferBindCount = 1;
		k.maxMeshBufferBindCount = 1;
		id<MTLIndirectCommandBuffer> rb = icb(k, 4);
		id<MTLIndirectCommandBuffer> cbuf = icb(cdesc_info(), 4);
		responds("ICB", rb, @protocol(MTLIndirectCommandBuffer));
		responds("render command", [rb indirectRenderCommandAtIndex:0],
		         @protocol(MTLIndirectRenderCommand));
		responds("compute command", [cbuf indirectComputeCommandAtIndex:0],
		         @protocol(MTLIndirectComputeCommand));
		printf("  descriptor SPI answers:");
		const char *spi[] = {"maxScissorRectCount", "maxViewportCount",
		                     "inheritStencilReferenceValues", "inheritDepthTestBounds",
		                     "inheritScissorRects", "inheritViewports", "inheritBlendColor",
		                     "allowOverrideRenderStates", "resourceIndex"};
		for (size_t i = 0; i < sizeof spi / sizeof spi[0]; i++) {
			SEL sel = sel_registerName(spi[i]);
			if ([k respondsToSelector:sel])
				printf(" %s=%ld", spi[i], (long)((NSInteger(*)(id, SEL))objc_msgSend)(k, sel));
			else
				printf(" %s=(none)", spi[i]);
		}
		printf("\n");
	}
	printf("  pipelines: draw supportICB %s, add supportICB %s, spin supportICB %s\n",
	       yesno(gDraw.supportIndirectCommandBuffers), yesno(gAdd.supportIndirectCommandBuffers),
	       yesno(gSpin.supportIndirectCommandBuffers));
}

// ---- types -----------------------------------------------------------------------------------

static void case_types(void)
{
	printf("types (maxCommandCount 4, bind counts 2):\n");
	const MTLIndirectCommandType all[] = {
	    MTLIndirectCommandTypeDraw,
	    MTLIndirectCommandTypeDrawIndexed,
	    MTLIndirectCommandTypeDrawPatches,
	    MTLIndirectCommandTypeDrawIndexedPatches,
	    MTLIndirectCommandTypeConcurrentDispatch,
	    MTLIndirectCommandTypeConcurrentDispatchThreads,
	    MTLIndirectCommandTypeDrawMeshThreadgroups,
	    MTLIndirectCommandTypeDrawMeshThreads,
	    MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed |
	        MTLIndirectCommandTypeDrawPatches | MTLIndirectCommandTypeDrawIndexedPatches,
	    MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads,
	    MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeConcurrentDispatch,
	    MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawMeshThreadgroups,
	    MTLIndirectCommandTypeDrawMeshThreadgroups | MTLIndirectCommandTypeDrawMeshThreads,
	    1 << 4,
	    1 << 9,
	};
	for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
		for (int inherit = 0; inherit < 2; inherit++) {
			MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
			d.commandTypes = all[i];
			d.inheritPipelineState = inherit;
			d.inheritBuffers = inherit;
			if (!inherit) {
				d.maxVertexBufferBindCount = 2;
				d.maxFragmentBufferBindCount = 2;
				d.maxKernelBufferBindCount = 2;
				if (all[i] & (MTLIndirectCommandTypeDrawMeshThreadgroups |
				              MTLIndirectCommandTypeDrawMeshThreads)) {
					d.maxObjectBufferBindCount = 2;
					d.maxMeshBufferBindCount = 2;
				}
			}
			id<MTLIndirectCommandBuffer> b = [D newIndirectCommandBufferWithDescriptor:d
			                                                           maxCommandCount:4
			                                                                   options:0];
			printf("  types 0x%03lx inherit %s: %s", (unsigned long)all[i], yesno(inherit),
			       b ? "made" : "nil");
			if (b)
				printf(" (size %lu, allocatedSize %lu)", (unsigned long)b.size,
				       (unsigned long)b.allocatedSize);
			printf("\n");
		}
	}
}

// ---- limits ----------------------------------------------------------------------------------

static void case_limits(int sub)
{
	const struct {
		MTLIndirectCommandType t;
		NSUInteger binds;
		const char *n;
	} kinds[] = {
	    {MTLIndirectCommandTypeDraw, 0, "draw inherit"},
	    {MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed, 31, "draw 31 binds"},
	    {MTLIndirectCommandTypeConcurrentDispatch, 31, "dispatch 31 binds"},
	    {MTLIndirectCommandTypeDrawPatches, 1, "patches 1 bind"},
	};
	if (sub >= (int)(sizeof kinds / sizeof kinds[0]))
		exit(3);
	printf("limits %s:\n", kinds[sub].n);
	unsigned long lastMade = 0;
	for (int s = 10; s <= 30; s++) {
		NSUInteger n = (NSUInteger)1 << s;
		MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
		d.commandTypes = kinds[sub].t;
		d.inheritPipelineState = kinds[sub].binds == 0;
		d.inheritBuffers = kinds[sub].binds == 0;
		d.maxVertexBufferBindCount = kinds[sub].binds;
		d.maxFragmentBufferBindCount = kinds[sub].binds;
		d.maxKernelBufferBindCount = kinds[sub].binds;
		@autoreleasepool {
			id<MTLIndirectCommandBuffer> b = [D newIndirectCommandBufferWithDescriptor:d
			                                                           maxCommandCount:n
			                                                                   options:0];
			printf("  maxCommandCount 2^%d: %s", s, b ? "made" : "nil");
			if (b) {
				printf(" size %lu allocatedSize %lu (%.1f B a command)", (unsigned long)b.size,
				       (unsigned long)b.allocatedSize, (double)b.allocatedSize / n);
				lastMade = n;
			}
			printf("\n");
			if (!b || b.allocatedSize > ((NSUInteger)1 << 30))
				break;
		}
	}
	// Between the last made and the first refused, bisect.
	if (lastMade) {
		NSUInteger lo = lastMade, hi = lastMade * 2;
		while (hi - lo > 1) {
			NSUInteger mid = lo + (hi - lo) / 2;
			MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
			d.commandTypes = kinds[sub].t;
			d.inheritPipelineState = kinds[sub].binds == 0;
			d.inheritBuffers = kinds[sub].binds == 0;
			d.maxVertexBufferBindCount = kinds[sub].binds;
			d.maxFragmentBufferBindCount = kinds[sub].binds;
			d.maxKernelBufferBindCount = kinds[sub].binds;
			@autoreleasepool {
				id<MTLIndirectCommandBuffer> b = [D newIndirectCommandBufferWithDescriptor:d
				                                                           maxCommandCount:mid
				                                                                   options:0];
				if (b && b.allocatedSize <= ((NSUInteger)1 << 30))
					lo = mid;
				else
					hi = mid;
			}
		}
		printf("  largest made (allocatedSize within 1 GiB): %lu\n", (unsigned long)lo);
	}
}

// ---- descriptor ------------------------------------------------------------------------------

static void case_descriptor(int sub)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = MTLIndirectCommandTypeDraw;
	d.maxVertexBufferBindCount = 1;
	d.maxFragmentBufferBindCount = 1;
	NSUInteger count = 4;
	MTLResourceOptions opts = 0;
	const char *what;
	switch (sub) {
	case 0: what = "baseline (draw, binds 1, count 4)"; break;
	case 1: what = "commandTypes 0"; d.commandTypes = 0; break;
	case 2: what = "maxCommandCount 0"; count = 0; break;
	case 3: what = "maxVertexBufferBindCount 31"; d.maxVertexBufferBindCount = 31; break;
	case 4: what = "maxVertexBufferBindCount 32"; d.maxVertexBufferBindCount = 32; break;
	case 5: what = "maxFragmentBufferBindCount 32"; d.maxFragmentBufferBindCount = 32; break;
	case 6: what = "maxVertexBufferBindCount 1000"; d.maxVertexBufferBindCount = 1000; break;
	case 7:
		what = "compute maxKernelBufferBindCount 32";
		d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
		d.maxKernelBufferBindCount = 32;
		break;
	case 8:
		what = "compute maxKernelThreadgroupMemoryBindCount 32";
		d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
		d.maxKernelBufferBindCount = 1;
		d.maxKernelThreadgroupMemoryBindCount = 32;
		break;
	case 9: what = "inheritBuffers with binds 1"; d.inheritBuffers = YES; break;
	case 10: what = "storage Private"; opts = MTLResourceStorageModePrivate; break;
	case 11: what = "storage Memoryless"; opts = MTLResourceStorageModeMemoryless; break;
	case 12: what = "storage Managed"; opts = MTLResourceStorageModeManaged; break;
	case 13:
		what = "compute inheritPipelineState";
		d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
		d.maxKernelBufferBindCount = 2;
		d.inheritPipelineState = YES;
		break;
	case 14:
		what = "compute inheritBuffers";
		d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
		d.inheritBuffers = YES;
		break;
	case 15: what = "supportDynamicAttributeStride"; d.supportDynamicAttributeStride = YES; break;
	case 16: what = "supportRayTracing"; d.supportRayTracing = YES; break;
	case 17: what = "inheritDepthStencilState NO"; d.inheritDepthStencilState = NO; break;
	case 18: what = "inheritCullMode NO"; d.inheritCullMode = NO; break;
	case 19: what = "inheritFrontFacingWinding NO"; d.inheritFrontFacingWinding = NO; break;
	case 20: what = "inheritTriangleFillMode NO"; d.inheritTriangleFillMode = NO; break;
	case 21: what = "inheritDepthBias NO"; d.inheritDepthBias = NO; break;
	case 22: what = "inheritDepthClipMode NO"; d.inheritDepthClipMode = NO; break;
	case 23: what = "supportColorAttachmentMapping"; d.supportColorAttachmentMapping = YES; break;
	case 24: what = "maxCommandCount 2^32"; count = (NSUInteger)1 << 32; break;
	case 25:
		what = "render with maxKernelBufferBindCount 40";
		d.maxKernelBufferBindCount = 40;
		break;
	case 26:
		what = "compute with maxVertexBufferBindCount 40";
		d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
		d.maxKernelBufferBindCount = 1;
		d.maxVertexBufferBindCount = 40;
		break;
	case 27:
		what = "options with hazard untracked + private";
		opts = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
		break;
	default: exit(3);
	}
	// The 26 render-state inherit properties default to YES; setting NO is the case.
	if (sub >= 17 && sub <= 22)
		;
	id<MTLIndirectCommandBuffer> b = [D newIndirectCommandBufferWithDescriptor:d
	                                                           maxCommandCount:count
	                                                                   options:opts];
	printf("  descriptor %2d %-48s %s", sub, what, b ? "made" : "nil");
	if (b)
		printf(" (size %lu storage %ld)", (unsigned long)b.size, (long)b.storageMode);
	printf("\n");
}

// ---- setters ---------------------------------------------------------------------------------

static void case_setters(int sub)
{
	MTLIndirectCommandBufferDescriptor *d = rdesc(MTLIndirectCommandTypeDraw, NO, NO);
	MTLIndirectCommandBufferDescriptor *dc = [MTLIndirectCommandBufferDescriptor new];
	dc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
	dc.maxKernelBufferBindCount = 2;
	const char *what;
	BOOL render = YES;
	id<MTLIndirectCommandBuffer> b = nil;
	switch (sub) {
	case 0:
		what = "baseline draw cell 0";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		break;
	case 1:
		what = "setVertexBuffer at index 1 (bind count 1)";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[[b indirectRenderCommandAtIndex:0] setVertexBuffer:gRects offset:16 atIndex:1];
		break;
	case 2:
		what = "setVertexBuffer at index 30 (bind count 1)";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[[b indirectRenderCommandAtIndex:0] setVertexBuffer:gRects offset:16 atIndex:30];
		break;
	case 3:
		what = "setVertexBuffer on inheritBuffers";
		b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, YES), 4);
		enc_draw(b, 0, 1, 1, YES);
		break;
	case 4:
		what = "setRenderPipelineState on inheritPipelineState";
		b = icb(rdesc(MTLIndirectCommandTypeDraw, YES, NO), 4);
		enc_draw(b, 0, 2, 2, YES);
		break;
	case 5:
		what = "indirectRenderCommandAtIndex: 4 of 4";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		printf("    command at 4: %s\n", object_getClassName([b indirectRenderCommandAtIndex:4]));
		break;
	case 6:
		what = "indirectComputeCommandAtIndex: of a render ICB";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		printf("    compute command: %s\n",
		       object_getClassName([b indirectComputeCommandAtIndex:0]));
		break;
	case 7:
		what = "drawIndexedPrimitives in a Draw-only ICB";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		{
			id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:1];
			[c setRenderPipelineState:gDraw];
			[c setVertexBuffer:gRects offset:16 atIndex:0];
			[c setFragmentBuffer:gColors offset:16 atIndex:0];
			[c drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip indexCount:4
			                   indexType:MTLIndexTypeUInt32 indexBuffer:gIdx32 indexBufferOffset:0
			               instanceCount:1 baseVertex:0 baseInstance:0];
		}
		break;
	case 8:
		what = "execute range past size ({2,4} of 4)";
		b = icb(d, 4);
		for (int i = 0; i < 4; i++)
			enc_draw(b, i, i, i, YES);
		exec_render(b, NSMakeRange(2, 4), nil, NO, 0);
		printf("  setters %2d %-52s %s\n", sub, what, cells());
		return;
	case 9:
		what = "drawPatches in a Draw-only ICB";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[[b indirectRenderCommandAtIndex:1] drawPatches:4 patchStart:1 patchCount:1
		                               patchIndexBuffer:nil patchIndexBufferOffset:0
		                                  instanceCount:1 baseInstance:0
		                       tessellationFactorBuffer:gTf tessellationFactorBufferOffset:0
		                tessellationFactorBufferInstanceStride:0];
		break;
	case 10:
		what = "setVertexBuffer:attributeStride: without dynamic stride";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[[b indirectRenderCommandAtIndex:0] setVertexBuffer:gRects offset:0 attributeStride:16
		                                            atIndex:0];
		break;
	case 11:
		what = "setCullMode (26) on a default ICB";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[[b indirectRenderCommandAtIndex:0] setCullMode:MTLCullModeBack];
		break;
	case 12:
		what = "compute: setKernelBuffer at index 2 (bind count 2)";
		render = NO;
		b = icb(dc, 4);
		[[b indirectComputeCommandAtIndex:0] setKernelBuffer:gRects offset:0 atIndex:2];
		break;
	case 13:
		what = "compute: setImageblockWidth:height:";
		render = NO;
		b = icb(dc, 4);
		[[b indirectComputeCommandAtIndex:0] setImageblockWidth:8 height:8];
		break;
	case 14:
		what = "compute: concurrentDispatchThreads in a threadgroups ICB";
		render = NO;
		b = icb(dc, 4);
		[[b indirectComputeCommandAtIndex:0] setComputePipelineState:gAdd];
		[[b indirectComputeCommandAtIndex:0] concurrentDispatchThreads:MTLSizeMake(64, 1, 1)
		                                         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		break;
	case 15:
		what = "compute: setComputePipelineState of a non-ICB pipeline";
		render = NO;
		b = icb(dc, 4);
		[[b indirectComputeCommandAtIndex:0] setComputePipelineState:gSpin];
		break;
	case 16:
		what = "render: setRenderPipelineState of a non-ICB pipeline";
		b = icb(d, 4);
		{
			id<MTLRenderPipelineState> p = rpso("icb_vs", 0, NO);
			[[b indirectRenderCommandAtIndex:0] setRenderPipelineState:p];
		}
		break;
	case 17:
		what = "resetWithRange: past size ({2,4} of 4)";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[b resetWithRange:NSMakeRange(2, 4)];
		break;
	case 18:
		what = "render: setBarrier (14)";
		b = icb(d, 4);
		enc_draw(b, 0, 0, 0, YES);
		[[b indirectRenderCommandAtIndex:0] setBarrier];
		break;
	default: exit(3);
	}
	if (render) {
		exec_render(b, NSMakeRange(0, 4), nil, NO, 0);
		printf("  setters %2d %-52s %s\n", sub, what, cells());
	} else {
		printf("  setters %2d %-52s survived\n", sub, what);
	}
}

// ---- render ----------------------------------------------------------------------------------

static void case_render(void)
{
	printf("render (CPU-encoded):\n");
	// No inheritance: each command sets its pipeline and both buffers.
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		enc_draw(b, i, i, i, YES);
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("no inheritance, 8 draws", "01234567");
	// Pipeline inherited.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, YES, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		enc_draw(b, i, 7 - i, i, NO);
	exec_render(b, NSMakeRange(NSNotFound, 0), gDraw, NO, 0);
	line("inheritPipelineState", "76543210");
	// Buffers inherited: the encoder's rects and color 5; commands draw cell i by base instance.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, YES), 8);
	for (NSUInteger i = 0; i < 8; i += 2) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gDraw];
		[c drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
		    instanceCount:1 baseInstance:i];
	}
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, YES, 5);
	line("inheritBuffers (color 5 on the encoder), base instance", "5.5.5.5.");
	// Both inherited.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, YES, YES), 8);
	for (NSUInteger i = 1; i < 8; i += 2)
		[[b indirectRenderCommandAtIndex:i] drawPrimitives:MTLPrimitiveTypeTriangleStrip
		                                       vertexStart:0
		                                       vertexCount:4
		                                     instanceCount:1
		                                      baseInstance:i];
	exec_render(b, NSMakeRange(NSNotFound, 0), gDraw, YES, 3);
	line("both inherited (color 3)", ".3.3.3.3");
	// Inherited buffers set after the execute? The encoder's binding at the execute counts.
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e setRenderPipelineState:gDraw];
		[e setVertexBuffer:gRects offset:0 atIndex:0];
		[e setFragmentBuffer:gColors offset:2 * 16 atIndex:0];
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 4)];
		[e setFragmentBuffer:gColors offset:6 * 16 atIndex:0];
		[e executeCommandsInBuffer:b withRange:NSMakeRange(4, 4)];
		end_pass(cb, e);
		run(cb);
		line("both inherited, color 2 then 6 between two executes", ".2.2.6.6");
	}
	// Indexed draws: 16-bit at offset 4, base instance; 32-bit with base vertex 4.
	b = icb(rdesc(MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed, NO, NO), 8);
	for (NSUInteger i = 0; i < 4; i++) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:gRects offset:0 atIndex:0];
		[c setFragmentBuffer:gColors offset:i * 16 atIndex:0];
		[c drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip indexCount:4
		               indexType:MTLIndexTypeUInt16 indexBuffer:gIdx16 indexBufferOffset:4
		           instanceCount:1 baseVertex:0 baseInstance:i];
	}
	for (NSUInteger i = 4; i < 8; i++) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:gRects offset:(i - 1) * 16 atIndex:0];
		[c setFragmentBuffer:gColors offset:i * 16 atIndex:0];
		[c drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip indexCount:4
		               indexType:MTLIndexTypeUInt32 indexBuffer:gIdx32 indexBufferOffset:0
		           instanceCount:1 baseVertex:4 baseInstance:0];
	}
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("indexed u16 (offset 4, base instance), u32 (base vertex 4)", "01234567");
	// Instanced: one command, 8 instances, color 9.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 1);
	{
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:0];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:gRects offset:0 atIndex:0];
		[c setFragmentBuffer:gColors offset:9 * 16 atIndex:0];
		[c drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
		    instanceCount:3 baseInstance:2];
	}
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("3 instances from base instance 2", "..999...");
	// Viewport and scissor on the encoder: inherited by every command.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		enc_draw(b, i, i, i, YES);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e setScissorRect:(MTLScissorRect){16, 0, 32, 8}];
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line("encoder scissor over cells 2..5", "..2345..");
	}
}

// ---- patches ---------------------------------------------------------------------------------

static void case_patches(void)
{
	printf("patches (CPU-encoded):\n");
	MTLIndirectCommandBufferDescriptor *d =
	    rdesc(MTLIndirectCommandTypeDrawPatches | MTLIndirectCommandTypeDrawIndexedPatches, NO, NO);
	id<MTLIndirectCommandBuffer> b = icb(d, 8);
	id<MTLBuffer> pidx = buf(64);
	((uint32_t *)pidx.contents)[3] = 6;
	for (NSUInteger i = 0; i < 4; i++) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gPatch];
		[c setVertexBuffer:gRects offset:0 atIndex:0];
		[c setFragmentBuffer:gColors offset:i * 16 atIndex:0];
		[c drawPatches:4 patchStart:i patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0
		                 instanceCount:1 baseInstance:0
		      tessellationFactorBuffer:gTf tessellationFactorBufferOffset:0
		    tessellationFactorBufferInstanceStride:0];
	}
	{
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:4];
		[c setRenderPipelineState:gPatch];
		[c setVertexBuffer:gRects offset:0 atIndex:0];
		[c setFragmentBuffer:gColors offset:6 * 16 atIndex:0];
		[c drawPatches:4 patchStart:3 patchCount:1 patchIndexBuffer:pidx
		        patchIndexBufferOffset:0 instanceCount:1 baseInstance:0
		      tessellationFactorBuffer:gTf tessellationFactorBufferOffset:0
		    tessellationFactorBufferInstanceStride:0];
	}
	for (NSUInteger i = 5; i < 8; i += 2) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gPatchIdx];
		[c setVertexBuffer:gRects offset:0 atIndex:0];
		[c setFragmentBuffer:gColors offset:i * 16 atIndex:0];
		[c drawIndexedPatches:4 patchStart:i patchCount:1 patchIndexBuffer:nil
		        patchIndexBufferOffset:0 controlPointIndexBuffer:gCpIdx
		    controlPointIndexBufferOffset:0 instanceCount:1 baseInstance:0
		      tessellationFactorBuffer:gTf tessellationFactorBufferOffset:0
		    tessellationFactorBufferInstanceStride:0];
	}
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLRenderCommandEncoder> e = begin_pass(cb);
	[e useResource:pidx usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 8)];
	end_pass(cb, e);
	run(cb);
	line("patches 0-3, patch index buffer -> 6, indexed patches 5 and 7", "0123.567");
}

// ---- ranges ----------------------------------------------------------------------------------

static void case_ranges(void)
{
	printf("ranges:\n");
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		enc_draw(b, i, i, i, YES);
	exec_render(b, NSMakeRange(2, 3), nil, NO, 0);
	line("withRange {2,3}", "..234...");
	exec_render(b, NSMakeRange(3, 0), nil, NO, 0);
	line("withRange {3,0}", "........");
	id<MTLBuffer> rb = buf(64);
	MTLIndirectCommandBufferExecutionRange *er = rb.contents;
	er[0] = MTLIndirectCommandBufferExecutionRangeMake(5, 2);
	er[1] = MTLIndirectCommandBufferExecutionRangeMake(0, 1);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e useResource:rb usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
		[e executeCommandsInBuffer:b indirectBuffer:rb indirectBufferOffset:0];
		[e executeCommandsInBuffer:b indirectBuffer:rb indirectBufferOffset:8];
		end_pass(cb, e);
		run(cb);
		line("indirect ranges {5,2} at 0 and {0,1} at 8", "0....56.");
	}
	// The range buffer written by the GPU in the same command buffer: a blit fills it.
	{
		id<MTLBuffer> rb2 = buf(64);
		id<MTLBuffer> src = buf(64);
		((MTLIndirectCommandBufferExecutionRange *)src.contents)[0] =
		    MTLIndirectCommandBufferExecutionRangeMake(1, 3);
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
		[bl copyFromBuffer:src sourceOffset:0 toBuffer:rb2 destinationOffset:0 size:8];
		[bl endEncoding];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e useResource:rb2 usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
		[e executeCommandsInBuffer:b indirectBuffer:rb2 indirectBufferOffset:0];
		end_pass(cb, e);
		run(cb);
		line("indirect range {1,3} blitted in the same command buffer", ".123....");
	}
	// Indirect range of length past the size.
	{
		er[2] = MTLIndirectCommandBufferExecutionRangeMake(6, 10);
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e useResource:rb usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
		[e executeCommandsInBuffer:b indirectBuffer:rb indirectBufferOffset:16];
		end_pass(cb, e);
		run(cb);
		line("indirect range {6,10} of 8", NULL);
	}
	// Commands never encoded in between.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		if (i != 1 && i != 6)
			enc_draw(b, i, i, i, YES);
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("commands 1 and 6 never encoded", "0.2345.7");
	// A command with only its state set, no draw.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	enc_draw(b, 0, 0, 0, YES);
	{
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:1];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:gRects offset:16 atIndex:0];
		[c setFragmentBuffer:gColors offset:16 atIndex:0];
	}
	enc_draw(b, 2, 2, 2, YES);
	exec_render(b, NSMakeRange(0, 3), nil, NO, 0);
	line("command 1 with state but no draw", "0.2.....");
	// Re-encoding a command's draw keeps its buffers.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	enc_draw(b, 0, 0, 0, YES);
	[[b indirectRenderCommandAtIndex:0] setFragmentBuffer:gColors offset:4 * 16 atIndex:0];
	exec_render(b, NSMakeRange(0, 1), nil, NO, 0);
	line("command 0's fragment buffer set again after its draw", "4.......");
	// Two executes in one encoder, and the same ICB in two encoders.
	b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		enc_draw(b, i, i, i, YES);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 2)];
		[e executeCommandsInBuffer:b withRange:NSMakeRange(6, 2)];
		end_pass(cb, e);
		run(cb);
		line("{0,2} and {6,2} in one encoder", "01....67");
	}
}

// ---- reset, copy, optimize -------------------------------------------------------------------

static id<MTLIndirectCommandBuffer> eight(NSUInteger colorBase)
{
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++)
		enc_draw(b, i, i, colorBase + i, YES);
	return b;
}

static void case_reset(void)
{
	printf("reset, copy, optimize:\n");
	id<MTLIndirectCommandBuffer> b = eight(0);
	[b resetWithRange:NSMakeRange(1, 2)];
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("resetWithRange {1,2}", "0..34567");
	[[b indirectRenderCommandAtIndex:5] reset];
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("then command 5's reset", "0..34.67");
	enc_draw(b, 1, 1, 9, YES);
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("then command 1 encoded again in color 9", "09.34.67");
	// A reset command whose draw alone is encoded again: is its state gone?
	[[b indirectRenderCommandAtIndex:3] reset];
	[[b indirectRenderCommandAtIndex:3] drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0
	                                       vertexCount:4 instanceCount:1 baseInstance:0];
	exec_render(b, NSMakeRange(3, 1), gDraw, NO, 0);
	line("command 3 reset, then only its draw (encoder has the pipeline)", NULL);
	// Blit reset in the command buffer that executes.
	b = eight(0);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
		[bl resetCommandsInBuffer:b withRange:NSMakeRange(0, 2)];
		[bl endEncoding];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line("blit resetCommandsInBuffer {0,2}, then execute", "..234567");
	}
	// Copy src {0,3} to dst at 4; dst's commands are colors 8+i.
	id<MTLIndirectCommandBuffer> src = eight(0), dst = eight(7);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
		[bl copyIndirectCommandBuffer:src sourceRange:NSMakeRange(0, 3) destination:dst
		             destinationIndex:4];
		[bl endEncoding];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:dst withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line("copy src {0,3} to dst at 4 (dst colors 7+i)", "012a...e");
	}
	// The copy is a copy: encoding src again later does not change dst.
	enc_draw(src, 0, 0, 0xe, YES);
	exec_render(dst, NSMakeRange(4, 1), nil, NO, 0);
	line("src command 0 encoded again; dst command 4", "0.......");
	// CPU encode of dst after a GPU copy into it.
	enc_draw(dst, 5, 5, 0xd, YES);
	exec_render(dst, NSMakeRange(4, 2), nil, NO, 0);
	line("dst command 5 encoded on the CPU after the copy", "0....d..");
	// Copy to an ICB with other inheritance.
	{
		id<MTLIndirectCommandBuffer> inh = icb(rdesc(MTLIndirectCommandTypeDraw, YES, NO), 8);
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
		[bl copyIndirectCommandBuffer:src sourceRange:NSMakeRange(1, 2) destination:inh
		             destinationIndex:0];
		[bl endEncoding];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e setRenderPipelineState:gDraw];
		[e executeCommandsInBuffer:inh withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line("copy into an ICB that inherits the pipeline (other descriptor)", NULL);
	}
	// Optimize, with reset commands in the range.
	b = eight(0);
	[b resetWithRange:NSMakeRange(2, 1)];
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
		[bl optimizeIndirectCommandBuffer:b withRange:NSMakeRange(0, 8)];
		[bl endEncoding];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line("optimize {0,8} (command 2 reset), then execute", "01.34567");
	}
	exec_render(b, NSMakeRange(3, 2), nil, NO, 0);
	line("after optimize, execute {3,2}", NULL);
	enc_draw(b, 0, 0, 0xa, YES);
	exec_render(b, NSMakeRange(0, 8), nil, NO, 0);
	line("after optimize, command 0 encoded again in color a", NULL);
}

// ---- order -----------------------------------------------------------------------------------

static void case_order(void)
{
	printf("order (CPU writes and executions):\n");
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	enc_draw(b, 0, 0, 1, YES);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 1)];
		end_pass(cb, e);
		enc_draw(b, 0, 0, 2, YES); // after the execute is encoded, before the commit
		enc_draw(b, 1, 1, 3, YES); // a command outside the range
		run(cb);
		line("re-encoded after the execute was encoded, before commit", "2.......");
	}
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 2)];
		end_pass(cb, e);
		[b resetWithRange:NSMakeRange(0, 1)]; // reset before commit
		run(cb);
		line("reset after the execute was encoded, before commit", ".3......");
	}
	// In flight: a long kernel first, then the execute; the CPU encodes while it runs. One
	// threadgroup, then 64, add the same 2^23 to one counter in all, tens of milliseconds of one
	// dispatch on a GPU that serializes the adds: 2^22 a thread of 64 groups ran 115 s on the
	// lane's RTX 5080, and a dispatch that long holds the GPU past a context switch's timeout.
	for (int spin = 0; spin < 2; spin++) {
		enc_draw(b, 0, 0, 4, YES);
		id<MTLBuffer> ctr = buf(16), n = buf(16);
		*(uint32_t *)n.contents = spin ? 1u << 11 : 1u << 17;
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:gSpin];
		[ce setBuffer:ctr offset:0 atIndex:0];
		[ce setBuffer:n offset:0 atIndex:1];
		[ce dispatchThreadgroups:MTLSizeMake(spin ? 64 : 1, 1, 1)
		    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		[ce endEncoding];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 1)];
		end_pass(cb, e);
		double t0 = oracle_now_ms();
		[cb commit];
		usleep(2000);
		enc_draw(b, 0, 0, 5, YES); // while the command buffer runs
		double t1 = oracle_now_ms();
		[cb waitUntilCompleted];
		double t2 = oracle_now_ms();
		char what[96];
		snprintf(what, sizeof what, "re-encoded %.1f ms after commit, done at %.1f ms", t1 - t0,
		         t2 - t0);
		line(what, NULL);
	}
}

// ---- compute ---------------------------------------------------------------------------------

static MTLIndirectCommandBufferDescriptor *cdesc(MTLIndirectCommandType t, BOOL inheritPso,
                                                 BOOL inheritBuffers)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = t;
	d.inheritPipelineState = inheritPso;
	d.inheritBuffers = inheritBuffers;
	d.maxKernelBufferBindCount = inheritBuffers ? 0 : 2;
	return d;
}

static void exec_compute(id<MTLIndirectCommandBuffer> b, NSRange r, id<MTLComputePipelineState> p,
                         id<MTLBuffer> out, id<MTLBuffer> vals, id<MTLBuffer> rangeBuf)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e useResource:out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
	if (vals)
		[e useResource:vals usage:MTLResourceUsageRead];
	if (p)
		[e setComputePipelineState:p];
	if (rangeBuf) {
		[e useResource:rangeBuf usage:MTLResourceUsageRead];
		[e executeCommandsInBuffer:b indirectBuffer:rangeBuf indirectBufferOffset:0];
	} else {
		[e executeCommandsInBuffer:b withRange:r];
	}
	[e endEncoding];
	run(cb);
}

static void print_out(const char *what, id<MTLBuffer> out, NSUInteger n, const uint32_t *expect)
{
	const uint32_t *o = out.contents;
	printf("  %-60s", what);
	BOOL same = YES;
	for (NSUInteger i = 0; i < n; i++) {
		printf(" %u", o[i * 64]);
		if (expect && o[i * 64] != expect[i])
			same = NO;
	}
	printf("%s\n", same ? "" : "  <- differs");
}

static void case_compute(void)
{
	printf("compute (CPU-encoded; out[i*64] per command):\n");
	id<MTLBuffer> out = buf(64 * 64 * 4), vals = buf(64 * 4);
	uint32_t *v = vals.contents;
	for (int i = 0; i < 64; i++)
		v[i] = (uint32_t)i + 1;
	id<MTLIndirectCommandBuffer> b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 4);
	for (NSUInteger i = 0; i < 4; i++) {
		id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:i];
		[c setComputePipelineState:gAdd];
		[c setKernelBuffer:out offset:i * 256 atIndex:0];
		[c setKernelBuffer:vals offset:i * 4 atIndex:1];
		[c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
	}
	exec_compute(b, NSMakeRange(0, 4), nil, out, vals, nil);
	const uint32_t e1[] = {1, 2, 3, 4};
	print_out("4 dispatches of add i+1", out, 4, e1);
	{
		const uint32_t *o = out.contents;
		BOOL all = YES;
		for (int i = 0; i < 4 * 64; i++)
			all &= o[i] == (uint32_t)(i / 64 + 1);
		printf("    every thread of each: %s\n", yesno(all));
	}
	exec_compute(b, NSMakeRange(1, 2), nil, out, vals, nil);
	const uint32_t e2[] = {1, 4, 6, 4};
	print_out("again, withRange {1,2}", out, 4, e2);
	id<MTLBuffer> rb = buf(16);
	*(MTLIndirectCommandBufferExecutionRange *)rb.contents =
	    MTLIndirectCommandBufferExecutionRangeMake(3, 1);
	exec_compute(b, NSMakeRange(0, 0), nil, out, vals, rb);
	const uint32_t e3[] = {1, 4, 6, 8};
	print_out("again, indirect range {3,1}", out, 4, e3);
	[b resetWithRange:NSMakeRange(0, 2)];
	exec_compute(b, NSMakeRange(0, 4), nil, out, vals, nil);
	const uint32_t e4[] = {1, 4, 9, 12};
	print_out("again, after resetWithRange {0,2}", out, 4, e4);
	// dispatchThreads: 60 threads of a 64 threadgroup.
	memset(out.contents, 0, out.length);
	b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatchThreads, NO, NO), 1);
	{
		id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:0];
		[c setComputePipelineState:gAdd];
		[c setKernelBuffer:out offset:0 atIndex:0];
		[c setKernelBuffer:vals offset:6 * 4 atIndex:1];
		[c concurrentDispatchThreads:MTLSizeMake(60, 1, 1)
		       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
	}
	exec_compute(b, NSMakeRange(0, 1), nil, out, vals, nil);
	{
		const uint32_t *o = out.contents;
		printf("  concurrentDispatchThreads 60 of 64: out[0] %u out[59] %u out[60] %u out[63] %u\n",
		       o[0], o[59], o[60], o[63]);
	}
	// Threadgroup memory.
	memset(out.contents, 0, out.length);
	b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 1);
	{
		id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:0];
		[c setComputePipelineState:gTg];
		[c setKernelBuffer:out offset:0 atIndex:0];
		[c setThreadgroupMemoryLength:256 atIndex:0];
		[c concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
		            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
	}
	exec_compute(b, NSMakeRange(0, 1), nil, out, vals, nil);
	{
		const uint32_t *o = out.contents;
		printf("  threadgroup memory 256 B: out[0] %u out[63] %u out[64] %u out[127] %u (expect "
		       "1063 1000 1127 1064)\n",
		       o[0], o[63], o[64], o[127]);
	}
	// Barriers: add 5 then multiply by 3, the second with and without a barrier, 50 runs each.
	for (int barrier = 0; barrier < 2; barrier++) {
		id<MTLBuffer> five = buf(16), three = buf(16);
		*(uint32_t *)five.contents = 5;
		*(uint32_t *)three.contents = 3;
		id<MTLBuffer> big = buf(64 * 1024 * 4);
		b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 2);
		id<MTLIndirectComputeCommand> c0 = [b indirectComputeCommandAtIndex:0];
		[c0 setComputePipelineState:gAdd];
		[c0 setKernelBuffer:big offset:0 atIndex:0];
		[c0 setKernelBuffer:five offset:0 atIndex:1];
		[c0 concurrentDispatchThreadgroups:MTLSizeMake(1024, 1, 1)
		             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		id<MTLIndirectComputeCommand> c1 = [b indirectComputeCommandAtIndex:1];
		[c1 setComputePipelineState:gMul];
		[c1 setKernelBuffer:big offset:0 atIndex:0];
		[c1 setKernelBuffer:three offset:0 atIndex:1];
		[c1 concurrentDispatchThreadgroups:MTLSizeMake(1024, 1, 1)
		             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		if (barrier)
			[c1 setBarrier];
		int wrong = 0;
		for (int r = 0; r < 50; r++) {
			memset(big.contents, 0, big.length);
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e useResource:big usage:MTLResourceUsageRead | MTLResourceUsageWrite];
			[e useResource:five usage:MTLResourceUsageRead];
			[e useResource:three usage:MTLResourceUsageRead];
			[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 2)];
			[e endEncoding];
			run(cb);
			const uint32_t *o = big.contents;
			for (int i = 0; i < 64 * 1024; i++)
				if (o[i] != 15) {
					wrong++;
					break;
				}
		}
		printf("  add 5 then mul 3 over 64Ki threads, %s barrier: %d of 50 runs not all 15\n",
		       barrier ? "with" : "without", wrong);
	}
	// Inheritance: pipeline from the encoder, buffers from the encoder.
	memset(out.contents, 0, out.length);
	b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, YES, NO), 2);
	for (NSUInteger i = 0; i < 2; i++) {
		id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:i];
		[c setKernelBuffer:out offset:i * 256 atIndex:0];
		[c setKernelBuffer:vals offset:(i + 4) * 4 atIndex:1];
		[c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
	}
	exec_compute(b, NSMakeRange(0, 2), gAdd, out, vals, nil);
	const uint32_t e5[] = {5, 6};
	print_out("inheritPipelineState (add on the encoder)", out, 2, e5);
	memset(out.contents, 0, out.length);
	b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, YES), 2);
	for (NSUInteger i = 0; i < 2; i++) {
		id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:i];
		[c setComputePipelineState:i ? gMul : gAdd];
		[c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		if (i)
			[c setBarrier];
	}
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setBuffer:out offset:0 atIndex:0];
		[e setBuffer:vals offset:6 * 4 atIndex:1];
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 2)];
		[e endEncoding];
		run(cb);
	}
	const uint32_t e6[] = {49};
	print_out("inheritBuffers (out, vals[6]=7): add 7 then mul 7 with barrier", out, 1, e6);
	// Mixed with the encoder's own dispatches: is the encoder's pipeline left as it was?
	memset(out.contents, 0, out.length);
	b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 1);
	{
		id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:0];
		[c setComputePipelineState:gMul];
		[c setKernelBuffer:out offset:0 atIndex:0];
		[c setKernelBuffer:vals offset:2 * 4 atIndex:1];
		[c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:gAdd];
		[e setBuffer:out offset:0 atIndex:0];
		[e setBuffer:vals offset:0 atIndex:1];
		[e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		[e useResource:out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		[e useResource:vals usage:MTLResourceUsageRead];
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 1)];
		[e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		[e endEncoding];
		run(cb);
	}
	const uint32_t e7[] = {4};
	print_out("encoder add 1, ICB mul 3, encoder add 1 (its state kept?)", out, 1, e7);
}

// ---- gpu -------------------------------------------------------------------------------------

static void case_gpu(void)
{
	printf("gpu (kernel-encoded):\n");
	id<MTLFunction> ef = fn("icb_encode_draw");
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = ef;
	NSError *err = nil;
	id<MTLComputePipelineState> enc = [D newComputePipelineStateWithDescriptor:cd
	                                                                   options:0
	                                                                reflection:nil
	                                                                     error:&err];
	if (!enc) {
		printf("  ! icb_encode_draw: %s\n", err.description.UTF8String);
		return;
	}
	id<MTLArgumentEncoder> ae = [ef newArgumentEncoderWithBufferIndex:0];
	printf("  argument encoder for {command_buffer, render_pipeline_state}: encodedLength %lu "
	       "alignment %lu\n",
	       (unsigned long)ae.encodedLength, (unsigned long)ae.alignment);
	id<MTLBuffer> modes = buf(64), idx = buf(64);
	for (int i = 0; i < 4; i++)
		((uint16_t *)idx.contents)[i] = (uint16_t)i;
	for (int pass = 0; pass < 5; pass++) {
		BOOL inheritPso = pass == 0 || pass == 2 || pass == 3;
		id<MTLIndirectCommandBuffer> b = icb(
		    rdesc(MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed, inheritPso, NO),
		    8);
		uint32_t *m = modes.contents;
		const char *what = "";
		const char *expect = "01234567";
		for (int i = 0; i < 8; i++)
			m[i] = inheritPso ? 0 : 16;
		if (pass == 2) {
			what = "draws, indexed on odd, pipeline inherited";
			for (int i = 1; i < 8; i += 2)
				m[i] = 1;
		} else if (pass == 3) {
			// CPU encodes colors 7+i first; the GPU then resets 2 and 6 and leaves odd ones.
			what = "CPU-encoded 7+i; GPU draws even, resets 2 and 6, skips odd";
			for (int i = 0; i < 8; i++)
				enc_draw(b, (NSUInteger)i, (NSUInteger)i, 7 + (NSUInteger)i, NO);
			for (int i = 1; i < 8; i += 2)
				m[i] = 3;
			m[2] = 2;
			m[6] = 2;
			expect = "08.a4c.e";
		} else if (pass == 0) {
			what = "draws, pipeline inherited";
		} else if (pass == 1) {
			what = "draws, pipeline from the argument buffer";
		} else {
			what = "draws, pipeline from the argument buffer, then copied by blit";
		}
		id<MTLBuffer> args = buf(ae.encodedLength);
		[ae setArgumentBuffer:args offset:0];
		[ae setIndirectCommandBuffer:b atIndex:0];
		if (!inheritPso)
			[ae setRenderPipelineState:gDraw atIndex:1];
		if (pass == 0) {
			const uint64_t *w = args.contents;
			printf("  argument buffer words: %llx %llx; icb gpuResourceID %llu, pipeline %llu\n",
			       (unsigned long long)w[0],
			       ae.encodedLength > 8 ? (unsigned long long)w[1] : 0ull,
			       (unsigned long long)b.gpuResourceID._impl,
			       (unsigned long long)gDraw.gpuResourceID._impl);
		}
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:enc];
		[ce setBuffer:args offset:0 atIndex:0];
		[ce setBuffer:gRects offset:0 atIndex:1];
		[ce setBuffer:gColors offset:0 atIndex:2];
		[ce setBuffer:modes offset:0 atIndex:3];
		[ce setBuffer:idx offset:0 atIndex:4];
		[ce useResource:b usage:MTLResourceUsageWrite];
		[ce dispatchThreads:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
		[ce endEncoding];
		id<MTLIndirectCommandBuffer> exe = b;
		if (pass == 4) {
			exe = icb(rdesc(MTLIndirectCommandTypeDraw | MTLIndirectCommandTypeDrawIndexed, NO,
			                NO),
			          8);
			id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
			[bl copyIndirectCommandBuffer:b sourceRange:NSMakeRange(0, 8) destination:exe
			             destinationIndex:0];
			[bl endEncoding];
		}
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e useResource:idx usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
		if (inheritPso)
			[e setRenderPipelineState:gDraw];
		[e executeCommandsInBuffer:exe withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line(what, expect);
		if (pass == 1) {
			// The CPU encodes over the GPU's commands afterwards.
			enc_draw(b, 0, 0, 0xc, YES);
			exec_render(b, NSMakeRange(0, 8), nil, NO, 0);
			line("then the CPU encodes command 0 in color c", "c1234567");
		}
	}
	// Compute commands encoded by a kernel.
	id<MTLFunction> df = fn("icb_encode_dispatch");
	cd.computeFunction = df;
	id<MTLComputePipelineState> denc = [D newComputePipelineStateWithDescriptor:cd
	                                                                    options:0
	                                                                 reflection:nil
	                                                                      error:&err];
	if (!denc) {
		printf("  ! icb_encode_dispatch: %s\n", err.description.UTF8String);
		return;
	}
	id<MTLArgumentEncoder> dae = [df newArgumentEncoderWithBufferIndex:0];
	id<MTLIndirectCommandBuffer> b = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 4);
	id<MTLBuffer> args = buf(dae.encodedLength);
	[dae setArgumentBuffer:args offset:0];
	[dae setIndirectCommandBuffer:b atIndex:0];
	[dae setComputePipelineState:gAdd atIndex:1];
	id<MTLBuffer> out = buf(4 * 64 * 4), vals = buf(16);
	uint32_t *v = vals.contents;
	v[0] = 3;
	v[1] = 0;
	v[2] = 7;
	v[3] = 9;
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
	[ce setComputePipelineState:denc];
	[ce setBuffer:args offset:0 atIndex:0];
	[ce setBuffer:out offset:0 atIndex:1];
	[ce setBuffer:vals offset:0 atIndex:2];
	[ce useResource:b usage:MTLResourceUsageWrite];
	[ce dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
	[ce endEncoding];
	ce = [cb computeCommandEncoder];
	[ce useResource:out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
	[ce useResource:vals usage:MTLResourceUsageRead];
	[ce executeCommandsInBuffer:b withRange:NSMakeRange(0, 4)];
	[ce endEncoding];
	run(cb);
	const uint32_t ex[] = {3, 0, 7, 9};
	print_out("kernel-encoded dispatches (add 3, reset, add 7, add 9)", out, 4, ex);
}

// ---- residency -------------------------------------------------------------------------------

static void case_residency(void)
{
	printf("residency:\n");
	// Fresh buffers the commands name, never made resident with useResource.
	id<MTLBuffer> rects = [D newBufferWithBytes:gRects.contents length:gRects.length
	                                    options:MTLResourceStorageModeShared];
	id<MTLBuffer> colors = [D newBufferWithBytes:gColors.contents length:gColors.length
	                                     options:MTLResourceStorageModeShared];
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	for (NSUInteger i = 0; i < 8; i++) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:rects offset:i * 16 atIndex:0];
		[c setFragmentBuffer:colors offset:i * 16 atIndex:0];
		[c drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
		    instanceCount:1 baseInstance:0];
	}
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor new];
	rp.colorAttachments[0].texture = gTarget;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 8)];
	end_pass(cb, e);
	run(cb);
	line("buffers never made resident", NULL);
	// Private buffers written by the GPU, never made resident.
	id<MTLBuffer> prects = [D newBufferWithLength:gRects.length options:MTLResourceStorageModePrivate];
	cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
	[bl copyFromBuffer:gRects sourceOffset:0 toBuffer:prects destinationOffset:0 size:gRects.length];
	[bl endEncoding];
	for (NSUInteger i = 0; i < 8; i++)
		[[b indirectRenderCommandAtIndex:i] setVertexBuffer:prects offset:i * 16 atIndex:0];
	e = [cb renderCommandEncoderWithDescriptor:rp];
	[e executeCommandsInBuffer:b withRange:NSMakeRange(0, 8)];
	end_pass(cb, e);
	run(cb);
	line("a Private vertex buffer blitted first, never made resident", NULL);
}

// ---- freed -----------------------------------------------------------------------------------

// The commands of `b` executed in one pass that makes `resident` resident (none past the null).
static void exec_resident(id<MTLIndirectCommandBuffer> b, id<MTLBuffer> __strong *resident, int n)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLRenderCommandEncoder> e = begin_pass(cb);
	for (int i = 0; i < n; i++)
		if (resident[i])
			[e useResource:resident[i]
			         usage:MTLResourceUsageRead
			        stages:MTLRenderStageVertex | MTLRenderStageFragment];
	[e executeCommandsInBuffer:b withRange:NSMakeRange(0, b.size)];
	end_pass(cb, e);
	run(cb);
}

// A buffer of `len` bytes, every 16-byte color in it color index `k`.
static id<MTLBuffer> colors_of(NSUInteger len, NSUInteger k)
{
	id<MTLBuffer> b = buf(len);
	float *c = b.contents;
	for (NSUInteger i = 0; i < len / 16; i++) {
		c[i * 4 + 0] = (float)(((k % 15) + 1) * 16) / 255.0f;
		c[i * 4 + 3] = 1.0f;
	}
	return b;
}

static void case_freed(void)
{
	printf("freed (a buffer the commands name released before the execute):\n");
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
	id<MTLBuffer> rects = [D newBufferWithBytes:gRects.contents length:gRects.length
	                                    options:MTLResourceStorageModeShared];
	id<MTLBuffer> colors = [D newBufferWithBytes:gColors.contents length:gColors.length
	                                     options:MTLResourceStorageModeShared];
	for (NSUInteger i = 0; i < 8; i++) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:rects offset:i * 16 atIndex:0];
		[c setFragmentBuffer:colors offset:i * 16 atIndex:0];
		[c drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
		    instanceCount:1 baseInstance:0];
	}
	{
		id<MTLBuffer> live[] = {rects, colors};
		exec_resident(b, live, 2);
		line("both buffers live and resident", "01234567");
	}
	@autoreleasepool {
		colors = nil;
	}
	{
		id<MTLBuffer> live[] = {rects};
		exec_resident(b, live, 1);
		line("the fragment buffer released", NULL);
	}
	{
		// Buffers of its size made since, in color 9, resident: does a command reach one?
		id<MTLBuffer> fresh[4];
		for (int i = 0; i < 4; i++)
			fresh[i] = colors_of(gColors.length, 9);
		id<MTLBuffer> live[] = {rects, fresh[0], fresh[1], fresh[2], fresh[3]};
		exec_resident(b, live, 5);
		line("then 4 buffers of its size made, in color 9, resident", NULL);
	}
	@autoreleasepool {
		rects = nil;
	}
	exec_resident(b, NULL, 0);
	line("the vertex buffer released too", NULL);
	// A heap's buffers: the heap made resident after its fragment buffer was released.
	{
		// Private: the x86 devices refuse a Shared heap (an assertion); a blit fills it.
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 64 * 1024;
		hd.storageMode = MTLStorageModePrivate;
		id<MTLHeap> heap = [D newHeapWithDescriptor:hd];
		id<MTLBuffer> hr = [heap newBufferWithLength:gRects.length options:MTLResourceStorageModePrivate];
		id<MTLBuffer> hc = [heap newBufferWithLength:gColors.length options:MTLResourceStorageModePrivate];
		{
			id<MTLCommandBuffer> fill = [Q commandBuffer];
			id<MTLBlitCommandEncoder> bl = [fill blitCommandEncoder];
			[bl copyFromBuffer:gRects sourceOffset:0 toBuffer:hr destinationOffset:0 size:gRects.length];
			[bl copyFromBuffer:gColors sourceOffset:0 toBuffer:hc destinationOffset:0 size:gColors.length];
			[bl endEncoding];
			run(fill);
		}
		id<MTLIndirectCommandBuffer> hb = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 8);
		for (NSUInteger i = 0; i < 8; i++) {
			id<MTLIndirectRenderCommand> c = [hb indirectRenderCommandAtIndex:i];
			[c setRenderPipelineState:gDraw];
			[c setVertexBuffer:hr offset:i * 16 atIndex:0];
			[c setFragmentBuffer:hc offset:i * 16 atIndex:0];
			[c drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
			    instanceCount:1 baseInstance:0];
		}
		@autoreleasepool {
			hc = nil;
		}
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e useHeap:heap stages:MTLRenderStageVertex | MTLRenderStageFragment];
		[e executeCommandsInBuffer:hb withRange:NSMakeRange(0, 8)];
		end_pass(cb, e);
		run(cb);
		line("a heap's fragment buffer released, the heap resident", NULL);
	}
	// Compute: a dispatch adds vals[0] (5) to out; vals released, then out.
	id<MTLBuffer> out = buf(64 * 4);
	id<MTLBuffer> vals = buf(16);
	*(uint32_t *)vals.contents = 5;
	id<MTLIndirectCommandBuffer> cbuf = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 1);
	{
		id<MTLIndirectComputeCommand> c = [cbuf indirectComputeCommandAtIndex:0];
		[c setComputePipelineState:gAdd];
		[c setKernelBuffer:out offset:0 atIndex:0];
		[c setKernelBuffer:vals offset:0 atIndex:1];
		[c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
	}
	exec_compute(cbuf, NSMakeRange(0, 1), nil, out, vals, nil);
	printf("  %-60s %u\n", "a dispatch adding vals[0] = 5, both live", ((uint32_t *)out.contents)[0]);
	@autoreleasepool {
		vals = nil;
	}
	exec_compute(cbuf, NSMakeRange(0, 1), nil, out, nil, nil);
	printf("  %-60s %u\n", "again, vals released", ((uint32_t *)out.contents)[0]);
	{
		id<MTLBuffer> fresh[4];
		for (int i = 0; i < 4; i++) {
			fresh[i] = buf(16);
			*(uint32_t *)fresh[i].contents = 100;
		}
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e useResource:out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		for (int i = 0; i < 4; i++)
			[e useResource:fresh[i] usage:MTLResourceUsageRead];
		[e executeCommandsInBuffer:cbuf withRange:NSMakeRange(0, 1)];
		[e endEncoding];
		run(cb);
		printf("  %-60s %u\n", "again, 4 buffers of 16 B made since (100), resident",
		       ((uint32_t *)out.contents)[0]);
	}
	// Released while a command buffer that executes the command waits behind a spin: a command
	// buffer of unretained references, so the release is the buffer's last.
	{
		id<MTLBuffer> o2 = buf(64 * 4), v2 = buf(16), n = buf(16);
		*(uint32_t *)v2.contents = 5;
		*(uint32_t *)n.contents = 1u << 20;
		id<MTLBuffer> counter = [D newBufferWithLength:16 options:MTLResourceStorageModePrivate];
		id<MTLIndirectCommandBuffer> c2 =
		    icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 1);
		id<MTLIndirectComputeCommand> k = [c2 indirectComputeCommandAtIndex:0];
		[k setComputePipelineState:gAdd];
		[k setKernelBuffer:o2 offset:0 atIndex:0];
		[k setKernelBuffer:v2 offset:0 atIndex:1];
		[k concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		id<MTLCommandBuffer> cb = [Q commandBufferWithUnretainedReferences];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:gSpin];
		[e setBuffer:counter offset:0 atIndex:0];
		[e setBuffer:n offset:0 atIndex:1];
		[e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		[e useResource:o2 usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		[e useResource:v2 usage:MTLResourceUsageRead];
		[e executeCommandsInBuffer:c2 withRange:NSMakeRange(0, 1)];
		[e endEncoding];
		[cb commit];
		@autoreleasepool {
			v2 = nil;
		}
		[cb waitUntilCompleted];
		printf("  %-60s %u%s\n", "vals released while the execute waits behind a spin",
		       ((uint32_t *)o2.contents)[0],
		       cb.status == MTLCommandBufferStatusCompleted ? "" : "  (command buffer failed)");
	}
	// A large buffer released (its own pages, no suballocation): what does the command read?
	for (NSUInteger mib = 1; mib <= 64; mib *= 8) {
		id<MTLBuffer> o3 = buf(64 * 4);
		id<MTLBuffer> big = [D newBufferWithLength:mib << 20 options:MTLResourceStorageModeShared];
		*(uint32_t *)big.contents = 5;
		id<MTLIndirectCommandBuffer> c3 =
		    icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 1);
		id<MTLIndirectComputeCommand> k = [c3 indirectComputeCommandAtIndex:0];
		[k setComputePipelineState:gAdd];
		[k setKernelBuffer:o3 offset:0 atIndex:0];
		[k setKernelBuffer:big offset:0 atIndex:1];
		[k concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		@autoreleasepool {
			big = nil;
		}
		exec_compute(c3, NSMakeRange(0, 1), nil, o3, nil, nil);
		char what[64];
		snprintf(what, sizeof what, "vals of %lu MiB released before the execute", (unsigned long)mib);
		printf("  %-60s %u\n", what, ((uint32_t *)o3.contents)[0]);
	}
	// The written buffer released: where do the writes go? A buffer made after it, resident.
	id<MTLBuffer> after = nil;
	@autoreleasepool {
		out = nil;
	}
	after = buf(64 * 4);
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e useResource:after usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		[e executeCommandsInBuffer:cbuf withRange:NSMakeRange(0, 1)];
		[e endEncoding];
		run(cb);
		printf("  %-60s %u\n", "out released too; a buffer of its size made since, resident",
		       ((uint32_t *)after.contents)[0]);
	}
}

// ---- resets ----------------------------------------------------------------------------------

// What a command's reset and a buffer's resetWithRange: keep: commands 0 and 1 are encoded whole,
// command 0 reset by its reset, command 1 by resetWithRange {1,1}; then each is given its
// buffers and its draw or dispatch again, no pipeline state, and executed with none on the encoder.
static void case_resets(void)
{
	printf("resets (what each reset keeps; buffers and draw encoded again, no pipeline):\n");
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), 2);
	for (NSUInteger i = 0; i < 2; i++)
		enc_draw(b, i, i, i + 1, YES);
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("render: both encoded", "12......");
	[[b indirectRenderCommandAtIndex:0] reset];
	[b resetWithRange:NSMakeRange(1, 1)];
	for (NSUInteger i = 0; i < 2; i++)
		enc_draw(b, i, i, i + 1, NO);
	exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
	line("render: 0 reset, 1 resetWithRange, then buffers and draw", NULL);
	id<MTLBuffer> out = buf(2 * 64 * 4), vals = buf(16);
	*(uint32_t *)vals.contents = 5;
	id<MTLIndirectCommandBuffer> c = icb(cdesc(MTLIndirectCommandTypeConcurrentDispatch, NO, NO), 2);
	for (int pass = 0; pass < 2; pass++) {
		for (NSUInteger i = 0; i < 2; i++) {
			id<MTLIndirectComputeCommand> k = [c indirectComputeCommandAtIndex:i];
			if (pass == 0)
				[k setComputePipelineState:gAdd];
			[k setKernelBuffer:out offset:i * 256 atIndex:0];
			[k setKernelBuffer:vals offset:0 atIndex:1];
			[k concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
			            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		}
		memset(out.contents, 0, out.length);
		exec_compute(c, NSMakeRange(0, 2), nil, out, vals, nil);
		const uint32_t *o = out.contents;
		printf("  %-60s %u %u\n",
		       pass ? "compute: 0 reset, 1 resetWithRange, then buffers and dispatch"
		            : "compute: both encoded (add 5)",
		       o[0], o[64]);
		if (pass == 0) {
			[[c indirectComputeCommandAtIndex:0] reset];
			[c resetWithRange:NSMakeRange(1, 1)];
		}
	}
}

// ---- timing ----------------------------------------------------------------------------------

// The wall time of committing an empty command buffer and waiting for it: what a commit after
// CPU encodes costs on top of an empty one's, where the device posts the encodes there.
static double commit_ms(void)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	double t0 = oracle_now_ms();
	[cb commit];
	[cb waitUntilCompleted];
	return oracle_now_ms() - t0;
}

static void case_timing(void)
{
	printf("timing:\n");
	const double empty = commit_ms();
	printf("  an empty command buffer's commit and wait: %.0f us\n", empty * 1e3);
	NSUInteger n = 1024;
	for (; n <= 16384; n *= 16) {
		id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), n);
		double t0 = oracle_now_ms();
		for (NSUInteger i = 0; i < n; i++)
			enc_draw(b, i, i % 8, i % 15, YES);
		double t1 = oracle_now_ms();
		printf("  %5lu: CPU encode, pipeline + 2 buffers + draw: %.0f ns a command, the commit "
		       "after it %.0f us\n",
		       (unsigned long)n, (t1 - t0) * 1e6 / n, commit_ms() * 1e3);
		id<MTLIndirectCommandBuffer> bi = icb(rdesc(MTLIndirectCommandTypeDraw, YES, YES), n);
		t0 = oracle_now_ms();
		for (NSUInteger i = 0; i < n; i++)
			[[bi indirectRenderCommandAtIndex:i] drawPrimitives:MTLPrimitiveTypeTriangleStrip
			                                        vertexStart:0
			                                        vertexCount:4
			                                      instanceCount:1
			                                       baseInstance:i % 8];
		t1 = oracle_now_ms();
		printf("  %5lu: CPU encode, draw alone (inherited): %.0f ns a command, the commit after "
		       "it %.0f us\n",
		       (unsigned long)n, (t1 - t0) * 1e6 / n, commit_ms() * 1e3);
		// What both encodes draw, executed once each after the timing: command i draws cell
		// i % 8 in color i % 15, so the last of each cell, 8 from the end, is color
		// (n - 8 + cell) % 15; the inherited draws take the encoder's color 7.
		char what[64];
		exec_render(b, NSMakeRange(NSNotFound, 0), nil, NO, 0);
		snprintf(what, sizeof what, "%5lu: pipeline + 2 buffers + draw, executed",
		         (unsigned long)n);
		line(what, "bcde0123");
		exec_render(bi, NSMakeRange(NSNotFound, 0), gDraw, YES, 7);
		snprintf(what, sizeof what, "%5lu: draw alone (inherited, color 7), executed",
		         (unsigned long)n);
		line(what, "77777777");
	}
	n = 16384;
	id<MTLIndirectCommandBuffer> b = icb(rdesc(MTLIndirectCommandTypeDraw, NO, NO), n);
	for (NSUInteger i = 0; i < n; i++)
		enc_draw(b, i, i % 8, i % 15, YES);
	double t0 = oracle_now_ms();
	[b resetWithRange:NSMakeRange(0, n)];
	double t1 = oracle_now_ms();
	printf("  resetWithRange of %lu: %.3f ms\n", (unsigned long)n, t1 - t0);
	for (NSUInteger i = 0; i < n; i++)
		enc_draw(b, i, i % 8, i % 15, YES);
	for (int k = 0; k < 3; k++) {
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLRenderCommandEncoder> e = begin_pass(cb);
		[e executeCommandsInBuffer:b withRange:NSMakeRange(0, n)];
		end_pass(cb, e);
		double wall, gpu;
		oracle_commit_wait(cb, &wall, &gpu);
		printf("  execute %lu draws: wall %.2f ms, GPU %.2f ms (%.0f ns a draw)\n", (unsigned long)n,
		       wall, gpu, gpu * 1e6 / n);
	}
	line("16384: encoded again after resetWithRange, executed", "bcde0123");
}

// ---- runner ----------------------------------------------------------------------------------

static int run_case(int argc, char **argv)
{
	if (argc < 1)
		return 1;
	setup();
	int sub = argc > 1 ? atoi(argv[1]) : 0;
	if (!strcmp(argv[0], "info"))
		case_info();
	else if (!strcmp(argv[0], "types"))
		case_types();
	else if (!strcmp(argv[0], "limits"))
		case_limits(sub);
	else if (!strcmp(argv[0], "descriptor"))
		case_descriptor(sub);
	else if (!strcmp(argv[0], "setters"))
		case_setters(sub);
	else if (!strcmp(argv[0], "render"))
		case_render();
	else if (!strcmp(argv[0], "patches"))
		case_patches();
	else if (!strcmp(argv[0], "ranges"))
		case_ranges();
	else if (!strcmp(argv[0], "reset"))
		case_reset();
	else if (!strcmp(argv[0], "order"))
		case_order();
	else if (!strcmp(argv[0], "compute"))
		case_compute();
	else if (!strcmp(argv[0], "gpu"))
		case_gpu();
	else if (!strcmp(argv[0], "residency"))
		case_residency();
	else if (!strcmp(argv[0], "timing"))
		case_timing();
	else if (!strcmp(argv[0], "freed"))
		case_freed();
	else if (!strcmp(argv[0], "resets"))
		case_resets();
	else
		return 2;
	return 0;
}

// Runs case `name` (subcase `sub` where >= 0) in a child, its stderr into our stdout; with
// `validate`, under MTL_DEBUG_LAYER=1. Answers the child's exit status (-signal if it died).
static int spawn_case(const char *dev, const char *name, int sub, BOOL validate)
{
	char subs[16];
	snprintf(subs, sizeof subs, "%d", sub);
	char *args[] = {(char *)gArgv0, (char *)dev, "--case", (char *)name, sub >= 0 ? subs : NULL,
	                NULL};
	char **env = environ;
	char **venv = NULL;
	if (validate) {
		size_t n = 0;
		while (environ[n])
			n++;
		venv = calloc(n + 2, sizeof(char *));
		memcpy(venv, environ, n * sizeof(char *));
		venv[n] = "MTL_DEBUG_LAYER=1";
		env = venv;
	}
	fflush(stdout);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, 1, 2);
	pid_t pid;
	int rc = posix_spawn(&pid, gArgv0, &fa, NULL, args, env);
	posix_spawn_file_actions_destroy(&fa);
	free(venv);
	if (rc)
		return 1;
	int st;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st)) {
		printf("  => %s %d%s died on signal %d\n", name, sub, validate ? " (validated)" : "",
		       WTERMSIG(st));
		return -WTERMSIG(st);
	}
	if (WEXITSTATUS(st) && WEXITSTATUS(st) != 3)
		printf("  => %s %d%s exited %d\n", name, sub, validate ? " (validated)" : "",
		       WEXITSTATUS(st));
	return WEXITSTATUS(st);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	gArgv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 2 && !strcmp(argv[2], "--case")) {
		Q = [D newCommandQueue];
		return run_case(argc - 3, argv + 3);
	}
	oracle_header("oracle_icb", D);
	D = nil;
	const BOOL validate = getenv("ORACLE_VALIDATE") && *getenv("ORACLE_VALIDATE") == '1';
	static const char *const all[] = {"info",   "types",   "limits", "descriptor", "setters",
	                                  "render", "patches", "ranges", "reset",      "order",
	                                  "compute", "gpu",    "residency", "timing", "freed",      "resets"};
	for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
		if (argc > 2 && strcmp(argv[2], all[i]))
			continue;
		const BOOL subcases = !strcmp(all[i], "limits") || !strcmp(all[i], "descriptor") ||
		                      !strcmp(all[i], "setters");
		if (!subcases) {
			spawn_case(argv[1], all[i], -1, NO);
			continue;
		}
		printf("%s:\n", all[i]);
		for (int v = 0; v < (validate && strcmp(all[i], "limits") ? 2 : 1); v++) {
			if (v)
				printf("%s (validated):\n", all[i]);
			for (int s = 0; s < 64; s++)
				if (spawn_case(argv[1], all[i], s, v) == 3)
					break;
		}
	}
	return 0;
}
