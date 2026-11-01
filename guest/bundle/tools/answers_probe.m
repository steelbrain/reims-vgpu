// answers_probe: the default Metal device's answers to what WindowServer asks it as it starts,
// in the order WindowServer first asks (measured with dtrace on the reims-graphics device in a
// macOS 26 guest): the identity, limits and feature queries before its first composite, and
// supportsFamily: of every family. Selectors outside the public headers are sent only where the
// device responds to them, or where it is the reims-graphics device, whose forwarding answers
// the ones it lacks.
//   answers_probe   prints one line per query: the initialized device's answer, and, for the
//                   reims-graphics device, the answer of an instance of its class that was made
//                   (+alloc) but not initialized, which is what each query answers before the
//                   device has read the host's capabilities.
//
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        tools/answers_probe.m -o answers_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <objc/message.h>
#include <objc/runtime.h>
#include <stdio.h>
#include <string.h>

typedef enum { kBool, kInt, kString } Kind;

typedef struct {
	const char *selector;
	Kind kind;
} Query;

// WindowServer's queries before its first composite, in its order.
static const Query kQueries[] = {
	{"isRemovable", kBool},
	{"isLowPower", kBool},
	{"featureProfile", kInt},
	{"argumentBuffersSupport", kInt},
	{"maxBufferLength", kInt},
	{"supportsBCTextureCompression", kBool},
	{"supports32BitMSAA", kBool},
	{"supports32BitFloatFiltering", kBool},
	{"supportsQueryTextureLOD", kBool},
	{"supportsPullModelInterpolation", kBool},
	{"supportsFunctionPointers", kBool},
	{"supportsDynamicLibraries", kBool},
	{"supportsRenderDynamicLibraries", kBool},
	{"supportsFunctionPointersFromRender", kBool},
	{"supportsRaytracingFromRender", kBool},
	{"supportsPrimitiveMotionBlur", kBool},
	{"supportsShaderBarycentricCoordinates", kBool},
	{"registryID", kInt},
	{"isHeadless", kBool},
	{"name", kString},
	{"readWriteTextureSupport", kInt},
	{"maxFragmentTextures", kInt},
	{"recommendedMaxWorkingSetSize", kInt},
	{"peerGroupID", kInt},
};

static const struct {
	const char *name;
	NSInteger value;
} kFamilies[] = {
	{"Apple1", 1001}, {"Apple2", 1002}, {"Apple3", 1003}, {"Apple4", 1004}, {"Apple5", 1005},
	{"Apple6", 1006}, {"Apple7", 1007}, {"Apple8", 1008}, {"Apple9", 1009}, {"Apple10", 1010},
	{"Mac1", 2001},   {"Mac2", 2002},   {"Common1", 3001}, {"Common2", 3002}, {"Common3", 3003},
	{"MacCatalyst1", 4001}, {"MacCatalyst2", 4002}, {"Metal3", 5001}, {"Metal4", 5002},
};

// The answer of `object` to `sel` as text, or "-" where it is not sent.
static void answer(id object, const char *selector, Kind kind, BOOL sendUnanswered, char *out, size_t size)
{
	SEL sel = sel_registerName(selector);
	if (object == nil || (!sendUnanswered && ![object respondsToSelector:sel])) {
		snprintf(out, size, "-");
		return;
	}
	@try {
		if (kind == kString) {
			id s = ((id(*)(id, SEL))objc_msgSend)(object, sel);
			snprintf(out, size, "%s", s ? [[s description] UTF8String] : "nil");
		} else {
			uint64_t v = ((uint64_t(*)(id, SEL))objc_msgSend)(object, sel);
			if (kind == kBool)
				snprintf(out, size, "%s", (v & 0xff) ? "YES" : "NO");
			else
				snprintf(out, size, "%llu", (unsigned long long)v);
		}
	} @catch (NSException *e) {
		snprintf(out, size, "raised %s", e.name.UTF8String);
	}
}

int main(void)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			printf("no default Metal device\n");
			return 2;
		}
		const char *cls = class_getName([device class]);
		BOOL ours = strncmp(cls, "Rgfx", 4) == 0;
		// An instance made but not initialized: the state of the device's own fields while the
		// base class's init runs, before the device reads the host's capabilities.
		id early = ours ? [[device class] alloc] : nil;
		printf("device %s (%s)\n", device.name.UTF8String, cls);
		printf("%-38s %-24s %s\n", "query", "answer", ours ? "before capabilities" : "");
		char a[96], b[96];
		for (size_t i = 0; i < sizeof(kQueries) / sizeof(kQueries[0]); i++) {
			answer(device, kQueries[i].selector, kQueries[i].kind, ours, a, sizeof(a));
			answer(early, kQueries[i].selector, kQueries[i].kind, ours, b, sizeof(b));
			printf("%-38s %-24s %s\n", kQueries[i].selector, a, ours ? b : "");
		}
		for (size_t i = 0; i < sizeof(kFamilies) / sizeof(kFamilies[0]); i++) {
			BOOL now = [device supportsFamily:(MTLGPUFamily)kFamilies[i].value];
			BOOL before = early != nil && [early supportsFamily:(MTLGPUFamily)kFamilies[i].value];
			char label[48];
			snprintf(label, sizeof(label), "supportsFamily:%s", kFamilies[i].name);
			printf("%-38s %-24s %s\n", label, now ? "YES" : "NO", ours ? (before ? "YES" : "NO") : "");
		}
	}
	return 0;
}
