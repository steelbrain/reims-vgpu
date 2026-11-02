// oracle_caps: a device's limits, feature answers and per-pixel-format answers, one per line, to
// diff against what the reims-graphics device answers (metal_enum.m in a guest, and
// crates/reims-vgpu/src/graphics/device_info.rs).
//   oracle_caps DEVICE           the device's answers, then the formats
//   oracle_caps DEVICE device    the device's answers only
//   oracle_caps DEVICE formats   the formats only
// Device lines are `key=value`; a selector the device does not respond to prints n/r and is not
// sent. Format lines are `F <format> <step>=<answer>`, the steps in this order:
//   lin    minimumLinearTextureAlignmentForPixelFormat:
//   tbuf   minimumTextureBufferAlignmentForPixelFormat:
//   read   a 64 x 64 2D Private texture with usage ShaderRead (made / nil)
//   rt     the same with usage RenderTarget
//   write  the same with usage ShaderWrite
//   msaa4  a 4x 2D multisample Private texture with usage RenderTarget
// A step where Metal ends the process (an assertion) prints `died(signal)`; the formats run in
// a child that starts again past it.
// Build: see README.md.
#import "oracle_common.h"
#include <objc/message.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// The SDK's pixel formats, as core's SDK_PIXEL_FORMATS lists them.
static const unsigned kFormats[] = {
    1,   10,  11,  12,  13,  14,  20,  22,  23,  24,  25,  30,  31,  32,  33,  34,  40,  41,
    42,  43,  45,  46,  47,  48,  53,  54,  55,  60,  62,  63,  64,  65,  70,  71,  72,  73,
    74,  80,  81,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  103, 104, 105, 110, 112,
    113, 114, 115, 120, 121, 122, 123, 124, 125, 130, 131, 132, 133, 134, 135, 140, 141, 142,
    143, 150, 151, 152, 153, 160, 161, 162, 163, 164, 165, 166, 167, 170, 172, 174, 176, 178,
    179, 180, 181, 182, 183, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199,
    200, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 222, 223, 224,
    225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 240, 241, 250, 252, 253, 255, 260,
    261, 262, 263, 552, 553, 554, 555,
};
enum { kFormatCount = sizeof kFormats / sizeof kFormats[0], kSteps = 6 };
static const char *const kStep[kSteps] = {"lin", "tbuf", "read", "rt", "write", "msaa4"};

static void u64(id d, const char *sel_name)
{
	SEL sel = sel_registerName(sel_name);
	if (![d respondsToSelector:sel]) {
		printf("%s=n/r\n", sel_name);
		return;
	}
	@try {
		unsigned long long v = ((unsigned long long (*)(id, SEL))objc_msgSend)(d, sel);
		printf("%s=%llu\n", sel_name, v);
	} @catch (NSException *e) {
		printf("%s=threw %s\n", sel_name, e.name.UTF8String);
	}
}

static void flag(id d, const char *sel_name)
{
	SEL sel = sel_registerName(sel_name);
	if (![d respondsToSelector:sel]) {
		printf("%s=n/r\n", sel_name);
		return;
	}
	@try {
		BOOL v = ((BOOL (*)(id, SEL))objc_msgSend)(d, sel);
		printf("%s=%d\n", sel_name, (int)v);
	} @catch (NSException *e) {
		printf("%s=threw %s\n", sel_name, e.name.UTF8String);
	}
}

static void query(id d, const char *sel_name, long long arg)
{
	SEL sel = sel_registerName(sel_name);
	if (![d respondsToSelector:sel]) {
		printf("%s(%lld)=n/r\n", sel_name, arg);
		return;
	}
	@try {
		BOOL v = ((BOOL (*)(id, SEL, long long))objc_msgSend)(d, sel, arg);
		printf("%s(%lld)=%d\n", sel_name, arg, (int)v);
	} @catch (NSException *e) {
		printf("%s(%lld)=threw %s\n", sel_name, arg, e.name.UTF8String);
	}
}

static void device_answers(id<MTLDevice> d)
{
	id o = (id)d;
	printf("name=%s\n", d.name.UTF8String);
	printf("architecture.name=%s\n",
	       [o respondsToSelector:@selector(architecture)] ? d.architecture.name.UTF8String : "n/r");
	MTLSize t = d.maxThreadsPerThreadgroup;
	printf("maxThreadsPerThreadgroup=%lux%lux%lu\n", (unsigned long)t.width, (unsigned long)t.height,
	       (unsigned long)t.depth);
	static const char *const sizes[] = {
	    "registryID", "recommendedMaxWorkingSetSize", "location", "locationNumber",
	    "maxTransferRate", "peerGroupID", "peerIndex", "peerCount", "maxBufferLength",
	    "maxThreadgroupMemoryLength", "maxArgumentBufferSamplerCount", "argumentBuffersSupport",
	    "readWriteTextureSupport", "sparseTileSizeInBytes", "maxVertexAmplificationCount",
	    "maximumConcurrentCompilationTaskCount", "gpuCoreCount",
	    // MTLDeviceSPI, as core reads them (crates/reims-vgpu-metal/src/gpu/device.rs).
	    "featureProfile", "halfFPConfig", "singleFPConfig", "doubleFPConfig",
	    "deviceLinearTextureAlignmentBytes", "iosurfaceReadOnlyTextureAlignmentBytes",
	    "iosurfaceTextureAlignmentBytes", "linearTextureAlignmentBytes", "maxComputeBuffers",
	    "maxComputeLocalMemorySizes", "maxComputeSamplers", "maxComputeTextures",
	    "maxComputeThreadgroupMemory", "maxComputeThreadgroupMemoryAlignmentBytes",
	    "maxFragmentBuffers", "maxFragmentSamplers", "maxFragmentTextures",
	    "maxPredicatedNestingDepth", "maxRasterizationRateLayerCount", "maxTextureHeight2D",
	    "maxTextureLayers", "maxTextureWidth2D", "maxTileBuffers", "maxTileSamplers",
	    "maxTileTextures", "maxTotalComputeThreadsPerThreadgroup", "maxVertexBuffers",
	    "maxVertexSamplers", "maxVertexTextures", "maxViewportCount",
	};
	for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
		u64(o, sizes[i]);
	static const char *const flags[] = {
	    "isLowPower", "isHeadless", "isRemovable", "hasUnifiedMemory",
	    "isDepth24Stencil8PixelFormatSupported", "areRasterOrderGroupsSupported",
	    "supports32BitFloatFiltering", "supports32BitMSAA", "supportsQueryTextureLOD",
	    "supportsBCTextureCompression", "supportsPullModelInterpolation",
	    "areBarycentricCoordsSupported", "supportsShaderBarycentricCoordinates",
	    "areProgrammableSamplePositionsSupported", "supportsDynamicLibraries",
	    "supportsRenderDynamicLibraries", "supportsRaytracing", "supportsFunctionPointers",
	    "supportsFunctionPointersFromRender", "supportsRaytracingFromRender",
	    "supportsPrimitiveMotionBlur", "supportsPlacementSparse",
	    // MTLDeviceSPI flags, as core reads them.
	    "isFramebufferReadSupported", "isLargeMRTSupported", "isRGB10A2GammaSupported",
	    "supportsBufferWithAddressRanges", "supportsBufferWithIOSurface",
	    "supportsCommandBufferJump", "supportsComputeCompressedTextureWrite",
	    "supportsConditionalLoadStore", "supportsDynamicAttributeStride",
	    "supportsExtendedXR10Formats", "supportsFloat16BCubicFiltering",
	    "supportsHeapWithAddressRanges", "supportsImageBlocks", "supportsMemoryOrderAtomics",
	    "supportsNativeHardwareFP16", "supportsNonSquareTileShaders",
	    "supportsProgrammableSamplePositions", "supportsResourceDetachBacking",
	    "supportsSharedTextureHandles", "supportsSIMDReduction", "supportsSIMDShuffleAndFill",
	    "supportsTexture2DMultisampleArray", "supportsTileShaders",
	};
	for (size_t i = 0; i < sizeof flags / sizeof flags[0]; i++)
		flag(o, flags[i]);
	static const long long families[] = {
	    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 2001, 2002,
	    3001, 3002, 3003, 4001, 4002, 5001, 5002, -1009,
	};
	for (size_t i = 0; i < sizeof families / sizeof families[0]; i++)
		query(o, "supportsFamily:", families[i]);
	for (long long fs = 10000; fs <= 10005; fs++)
		query(o, "supportsFeatureSet:", fs);
	for (long long n = 1; n <= 8; n *= 2)
		query(o, "supportsTextureSampleCount:", n);
	for (long long n = 0; n < 10; n++)
		query(o, "supportsPrimitiveType:", n);
	for (long long n = 1; n <= 2; n++)
		query(o, "supportsTextureWriteRoundingMode:", n);
	for (long long n = 0; n <= 2; n++)
		query(o, "supportsRasterizationRateMapWithLayerCount:", n);
	for (long long n = 1; n <= 9; n++)
		query(o, "supportsVertexAmplificationCount:", n);
	if ([o respondsToSelector:@selector(supportsCounterSampling:)])
		for (long long p = 0; p < 5; p++)
			printf("supportsCounterSampling:(%lld)=%d\n", p,
			       (int)[d supportsCounterSampling:(MTLCounterSamplingPoint)p]);
	printf("counterSets=%lu\n", (unsigned long)d.counterSets.count);
	MTLSizeAndAlign sa = [d heapBufferSizeAndAlignWithLength:4096 options:MTLResourceStorageModePrivate];
	printf("heapBufferSizeAndAlign(4096,Private)=%lu/%lu\n", (unsigned long)sa.size,
	       (unsigned long)sa.align);
	sa = [d heapBufferSizeAndAlignWithLength:1 << 20 options:MTLResourceStorageModeShared];
	printf("heapBufferSizeAndAlign(1MiB,Shared)=%lu/%lu\n", (unsigned long)sa.size,
	       (unsigned long)sa.align);
	struct {
		MTLPixelFormat f;
		int w, h;
		MTLTextureUsage u;
	} ts[] = {{MTLPixelFormatRGBA8Unorm, 256, 256, MTLTextureUsageShaderRead},
	          {MTLPixelFormatBGRA8Unorm, 3840, 2400, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead},
	          {MTLPixelFormatDepth32Float, 1024, 1024, MTLTextureUsageRenderTarget}};
	for (size_t i = 0; i < 3; i++) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:ts[i].f
		                                                       width:ts[i].w
		                                                      height:ts[i].h
		                                                   mipmapped:NO];
		td.usage = ts[i].u;
		td.storageMode = MTLStorageModePrivate;
		sa = [d heapTextureSizeAndAlignWithDescriptor:td];
		printf("heapTextureSizeAndAlign(%lu,%dx%d,usage %#lx)=%lu/%lu\n", (unsigned long)ts[i].f,
		       ts[i].w, ts[i].h, (unsigned long)ts[i].u, (unsigned long)sa.size,
		       (unsigned long)sa.align);
	}
	if ([o respondsToSelector:@selector(sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:)]) {
		static const unsigned fmts[] = {80, 125};
		for (int i = 0; i < 2; i++)
			for (unsigned long n = 1; n <= 4; n *= 2)
				for (long page = 101; page <= 103; page++) {
					MTLSize z = [d sparseTileSizeWithTextureType:MTLTextureType2D
					                                 pixelFormat:(MTLPixelFormat)fmts[i]
					                                 sampleCount:n
					                              sparsePageSize:(MTLSparsePageSize)page];
					printf("sparseTileSize(%u,%lu,%ld)=%lux%lux%lu\n", fmts[i], n, page,
					       (unsigned long)z.width, (unsigned long)z.height,
					       (unsigned long)z.depth);
				}
	}
}

// The formats from index `start`, skipping the steps before `step` of the first.
static int formats_child(id<MTLDevice> d, int start, int step)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	for (int i = start; i < kFormatCount; i++) {
		MTLPixelFormat f = (MTLPixelFormat)kFormats[i];
		for (int s = i == start ? step : 0; s < kSteps; s++) {
			printf("@ %d %d\n", i, s);
			char out[64] = "?";
			@autoreleasepool {
				if (s == 0)
					snprintf(out, sizeof out, "%lu",
					         (unsigned long)[d minimumLinearTextureAlignmentForPixelFormat:f]);
				else if (s == 1)
					snprintf(out, sizeof out, "%lu",
					         (unsigned long)[d minimumTextureBufferAlignmentForPixelFormat:f]);
				else {
					MTLTextureDescriptor *td =
					    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
					                                                       width:64
					                                                      height:64
					                                                   mipmapped:NO];
					td.storageMode = MTLStorageModePrivate;
					td.usage = s == 2   ? MTLTextureUsageShaderRead
					           : s == 4 ? MTLTextureUsageShaderWrite
					                    : MTLTextureUsageRenderTarget;
					if (s == 5) {
						td.textureType = MTLTextureType2DMultisample;
						td.sampleCount = 4;
					}
					id<MTLTexture> t = [d newTextureWithDescriptor:td];
					snprintf(out, sizeof out, "%s", t ? "made" : "nil");
				}
			}
			printf("F %u %s=%s\n", kFormats[i], kStep[s], out);
		}
	}
	printf("@ done\n");
	return 0;
}

static void formats(const char *self, const char *dev)
{
	int start = 0, step = 0;
	while (start < kFormatCount) {
		int fds[2];
		pipe(fds);
		posix_spawn_file_actions_t fa;
		posix_spawn_file_actions_init(&fa);
		posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
		posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
		posix_spawn_file_actions_addclose(&fa, fds[0]);
		char a[16], b[16];
		snprintf(a, sizeof a, "%d", start);
		snprintf(b, sizeof b, "%d", step);
		char *args[] = {(char *)self, (char *)dev, "--formats", a, b, NULL};
		pid_t pid;
		posix_spawn(&pid, self, &fa, NULL, args, environ);
		close(fds[1]);
		FILE *in = fdopen(fds[0], "r");
		char line[512];
		int cur = start, cs = step, done = 0;
		while (fgets(line, sizeof line, in)) {
			if (!strncmp(line, "@ done", 6))
				done = 1;
			else if (line[0] == '@')
				sscanf(line, "@ %d %d", &cur, &cs);
			else if (line[0] == 'F')
				fputs(line, stdout);
			else if (strstr(line, "failed assertion") || strstr(line, "Validation"))
				printf("  # %s", line);
		}
		fclose(in);
		int st;
		waitpid(pid, &st, 0);
		if (done)
			return;
		printf("F %u %s=died(%d)\n", kFormats[cur], kStep[cs],
		       WIFSIGNALED(st) ? WTERMSIG(st) : -WEXITSTATUS(st));
		start = cur;
		step = cs + 1;
		if (step >= kSteps) {
			start++;
			step = 0;
		}
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [device|formats]\n", argv[0]);
		return 2;
	}
	id<MTLDevice> d = oracle_device(argv[1]);
	if (argc > 4 && !strcmp(argv[2], "--formats"))
		return formats_child(d, atoi(argv[3]), atoi(argv[4]));
	oracle_header("oracle_caps", d);
	if (argc < 3 || !strcmp(argv[2], "device"))
		device_answers(d);
	if (argc < 3 || !strcmp(argv[2], "formats")) {
		d = nil;
		formats(argv[0], argv[1]);
	}
	return 0;
}
