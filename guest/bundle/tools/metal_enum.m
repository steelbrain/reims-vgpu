// metal_enum: every Metal device's identity and capability answers, one per line, so a guest's
// answers can be compared line by line with a host's (the same source builds for both).
//   metal_enum          every device of MTLCopyAllDevices, then the system default's name
// Each line is `key=value`. A selector the device does not respond to prints `n/r`, and is
// not sent; one that raises prints `threw` and the exception's name. Families and feature sets
// are asked by raw value, so the build does not depend on
// which ones the SDK names.
// Build: xcrun clang -arch x86_64 -mmacosx-version-min=26.0 -fobjc-arc -framework Metal \
//        -framework Foundation metal_enum.m -o metal_enum
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/message.h>
#include <stdio.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

// A getter of an integer or BOOL answer, sent only where the device responds to it.
static void scalar(id<MTLDevice> d, const char *sel_name)
{
	SEL sel = sel_registerName(sel_name);
	if (![(id)d respondsToSelector:sel]) {
		printf("  %s=n/r\n", sel_name);
		return;
	}
	@try {
		unsigned long long v = ((unsigned long long (*)(id, SEL))objc_msgSend)((id)d, sel);
		printf("  %s=%llu (0x%llx)\n", sel_name, v, v);
	} @catch (NSException *e) {
		printf("  %s=threw %s\n", sel_name, e.name.UTF8String);
	}
}

// A BOOL getter, printed 0 or 1.
static void flag(id<MTLDevice> d, const char *sel_name)
{
	SEL sel = sel_registerName(sel_name);
	if (![(id)d respondsToSelector:sel]) {
		printf("  %s=n/r\n", sel_name);
		return;
	}
	@try {
		BOOL v = ((BOOL (*)(id, SEL))objc_msgSend)((id)d, sel);
		printf("  %s=%d\n", sel_name, (int)v);
	} @catch (NSException *e) {
		printf("  %s=threw %s\n", sel_name, e.name.UTF8String);
	}
}

// A one-argument BOOL query of an integer argument.
static void query(id<MTLDevice> d, const char *sel_name, const char *label, long long arg)
{
	SEL sel = sel_registerName(sel_name);
	if (![(id)d respondsToSelector:sel]) {
		printf("  %s(%s)=n/r\n", sel_name, label);
		return;
	}
	@try {
		BOOL v = ((BOOL (*)(id, SEL, long long))objc_msgSend)((id)d, sel, arg);
		printf("  %s(%s)=%d\n", sel_name, label, (int)v);
	} @catch (NSException *e) {
		printf("  %s(%s)=threw %s\n", sel_name, label, e.name.UTF8String);
	}
}

static void dump(id<MTLDevice> d)
{
	printf("device name=%s\n", d.name.UTF8String);
	printf("  class=%s\n", class_getName(object_getClass(d)));
	scalar(d, "registryID");
	if ([(id)d respondsToSelector:@selector(architecture)])
		printf("  architecture.name=%s\n", d.architecture.name.UTF8String);
	else
		printf("  architecture.name=n/r\n");
	MTLSize t = d.maxThreadsPerThreadgroup;
	printf("  maxThreadsPerThreadgroup=%lux%lux%lu\n", (unsigned long)t.width,
	       (unsigned long)t.height, (unsigned long)t.depth);
	static const char *const identity[] = {
	    "isLowPower", "isHeadless", "isRemovable", "hasUnifiedMemory",
	};
	for (size_t i = 0; i < sizeof(identity) / sizeof(identity[0]); i++)
		flag(d, identity[i]);
	static const char *const sizes[] = {
	    "recommendedMaxWorkingSetSize", "location", "locationNumber", "maxTransferRate",
	    "peerGroupID", "peerIndex", "peerCount", "maxBufferLength",
	    "maxThreadgroupMemoryLength", "maxArgumentBufferSamplerCount", "argumentBuffersSupport",
	    "readWriteTextureSupport", "sparseTileSizeInBytes", "maxVertexAmplificationCount",
	    "maximumConcurrentCompilationTaskCount", "gpuCoreCount", "currentAllocatedSize",
	};
	for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
		scalar(d, sizes[i]);
	static const char *const flags[] = {
	    "isDepth24Stencil8PixelFormatSupported", "areRasterOrderGroupsSupported",
	    "supports32BitFloatFiltering", "supports32BitMSAA", "supportsQueryTextureLOD",
	    "supportsBCTextureCompression", "supportsPullModelInterpolation",
	    "areBarycentricCoordsSupported", "supportsShaderBarycentricCoordinates",
	    "areProgrammableSamplePositionsSupported", "supportsDynamicLibraries",
	    "supportsRenderDynamicLibraries", "supportsRaytracing", "supportsFunctionPointers",
	    "supportsFunctionPointersFromRender", "supportsRaytracingFromRender",
	    "supportsPrimitiveMotionBlur", "supportsPlacementSparse",
	};
	for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++)
		flag(d, flags[i]);
	// The families of the macOS 27 SDK, and -1009, which names none.
	static const long long families[] = {
	    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 2001, 2002,
	    3001, 3002, 3003, 4001, 4002, 5001, 5002, -1009,
	};
	for (size_t i = 0; i < sizeof(families) / sizeof(families[0]); i++) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", families[i]);
		query(d, "supportsFamily:", label, families[i]);
	}
	// The feature sets of the SDK: iOS 0..16, macOS 10000..10005, tvOS 30000..30005.
	for (long long fs = 0; fs <= 16; fs++) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", fs);
		query(d, "supportsFeatureSet:", label, fs);
	}
	for (long long fs = 10000; fs <= 10005; fs++) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", fs);
		query(d, "supportsFeatureSet:", label, fs);
	}
	for (long long fs = 30000; fs <= 30005; fs++) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", fs);
		query(d, "supportsFeatureSet:", label, fs);
	}
	for (long long n = 1; n <= 8; n *= 2) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", n);
		query(d, "supportsTextureSampleCount:", label, n);
	}
	for (long long n = 0; n <= 3; n++) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", n);
		query(d, "supportsRasterizationRateMapWithLayerCount:", label, n);
	}
	for (long long n = 1; n <= 9; n += (n < 2 ? 1 : n == 8 ? 1 : 2)) {
		char label[16];
		snprintf(label, sizeof(label), "%lld", n);
		query(d, "supportsVertexAmplificationCount:", label, n);
	}
	// Per-format answers of two uncompressed formats (BGRA8Unorm, RGBA32Float), which Metal
	// answers on every device; it asserts on a compressed one, so none is asked.
	static const unsigned long formats[] = {80, 125};
	for (size_t i = 0; i < 2; i++) {
		MTLPixelFormat f = (MTLPixelFormat)formats[i];
		if ([(id)d respondsToSelector:@selector(minimumTextureBufferAlignmentForPixelFormat:)])
			printf("  minimumTextureBufferAlignmentForPixelFormat:(%lu)=%lu\n", formats[i],
			       (unsigned long)[d minimumTextureBufferAlignmentForPixelFormat:f]);
		else
			printf("  minimumTextureBufferAlignmentForPixelFormat:(%lu)=n/r\n", formats[i]);
		SEL tile = @selector(sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:);
		if (![(id)d respondsToSelector:tile]) {
			printf("  sparseTileSize(%lu)=n/r\n", formats[i]);
			continue;
		}
		for (unsigned long n = 1; n <= 4; n *= 2) {
			for (long page = 101; page <= 103; page++) {
				MTLSize z = [d sparseTileSizeWithTextureType:MTLTextureType2D
				                                 pixelFormat:f
				                                 sampleCount:n
				                              sparsePageSize:(MTLSparsePageSize)page];
				printf("  sparseTileSize(%lu,%lu samples,page %ld)=%lux%lux%lu\n", formats[i], n,
				       page, (unsigned long)z.width, (unsigned long)z.height,
				       (unsigned long)z.depth);
			}
		}
	}
	if ([(id)d respondsToSelector:@selector(supportsCounterSampling:)]) {
		static const char *const points[] = {"atStage", "atDraw", "atDispatch", "atTile",
		                                     "atBlit"};
		for (long long p = 0; p < 5; p++)
			printf("  supportsCounterSampling:(%s)=%d\n", points[p],
			       (int)[d supportsCounterSampling:(MTLCounterSamplingPoint)p]);
	} else {
		printf("  supportsCounterSampling:=n/r\n");
	}
	if ([(id)d respondsToSelector:@selector(counterSets)]) {
		NSArray *sets = d.counterSets;
		printf("  counterSets=%lu\n", (unsigned long)sets.count);
		for (id<MTLCounterSet> s in sets)
			printf("    counterSet=%s\n", s.name.UTF8String);
	}
}

int main(void)
{
	@autoreleasepool {
		NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
		printf("MTLCopyAllDevices count=%lu\n", (unsigned long)devs.count);
		for (id<MTLDevice> d in devs)
			dump(d);
		id<MTLDevice> def = MTLCreateSystemDefaultDevice();
		printf("MTLCreateSystemDefaultDevice=%s\n", def ? def.name.UTF8String : "(nil)");
	}
	return 0;
}
