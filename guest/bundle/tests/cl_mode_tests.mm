// The OpenCL kernels' checks (tests/shell_tests.mm's main calls rgfxClModeTests()):
//   - what a kernel binds in OpenCL mode (src/RgfxClMode.*), from nodes written here in the shape
//     Metal's compiler writes for OpenCL C, and the refusals of a texture, a sampler and a binding
//     at no fixed index;
//   - the work-group size the AIR reader takes from a kernel's node (air.work_group_size), over
//     this Mac's compile of an OpenCL C program as the GL renderer compiles it;
//   - a kernel's function of OpenCL mode (src/RgfxFunction.mm, newFunctionWithPluginData:
//     bitcodeType:2), over that compile: its name, stage and plugin data, the plugin return data,
//     and its compute pipeline's reflection, each as the x86 Macs answer them (measured: macOS
//     26.7.1, the Intel 630 and the AMD 5300M alike, through the GL renderer), and none made where
//     the host makes no pipeline of OpenCL mode, or of another plugin data or bitcodeType.
// Nothing here runs on the GPU: tools/cl_probe.m runs the kernels on a device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#import "RgfxFunction.h"
#import "RgfxGLFunction.h"
#import "RgfxLibrary.h"
#import "RgfxPipelineReflection.h"

#include "RgfxAirArguments.hpp"
#include "RgfxClMode.hpp"
#include "RgfxMetallibWriter.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks
#include "RgfxSerializer.hpp"

void drainRing(TestRing &ring);                                                     // shell_tests.mm
NSData *clLibraryBytes(id<MTLDevice> metal, NSString *source, NSArray<NSString *> **names); // gl_tests.mm

// A device that answers whether the host makes pipelines of OpenCL mode, for a test's library.
@interface RgfxClModeTestsDevice : NSObject
@property(nonatomic) BOOL openCLMode;
- (BOOL)rgfxMakesOpenCLModePipelines;
@end

@implementation RgfxClModeTestsDevice
- (BOOL)rgfxMakesOpenCLModePipelines
{
	return self.openCLMode;
}
@end

namespace {

// The kernels of tools/cl_probe.m, as the x86 Macs were measured over them.
NSString *const kSource =
    @"typedef struct { float a; int b; } S;\n"
    @"__kernel void add(__global const float *a, __global const float *b, __global float *c) {\n"
    @"  size_t i = get_global_id(0); c[i] = a[i] + b[i]; }\n"
    @"__kernel void scale(__global float *b, float k, uint n) {\n"
    @"  size_t i = get_global_id(0); if (i < n) b[i] = b[i] * k; }\n"
    @"__kernel void lsum(__global const float *in, __global float *out, __local float *tmp) {\n"
    @"  size_t l = get_local_id(0); tmp[l] = in[get_global_id(0)]; barrier(CLK_LOCAL_MEM_FENCE);\n"
    @"  if (l == 0) { float s = 0; for (size_t j = 0; j < get_local_size(0); j++) s += tmp[j];\n"
    @"  out[get_group_id(0)] = s; } }\n"
    @"__kernel void cmul(__global float *o, __constant float *c) {\n"
    @"  size_t i = get_global_id(0); o[i] = o[i] * c[i & 3]; }\n"
    @"__kernel void st(__global float *o, S s) {\n"
    @"  size_t i = get_global_id(0); o[i] = s.a + (float)s.b; }\n"
    @"__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void wg(__global float *o) {\n"
    @"  size_t i = get_global_id(0); o[i] = (float)get_local_size(0); }\n"
    @"__kernel void items(__global uint *o, uint d) {\n"
    @"  size_t i = get_global_id(1) * get_global_size(0) + get_global_id(0);\n"
    @"  o[i] = get_local_id(0) + get_group_id(d) + get_work_dim(); }\n";

// What each kernel binds, as the x86 Macs answer it (the property list's lists and size).
struct Expected {
	const char *name;
	std::vector<uint32_t> buffers, byValue;
	bool workGroupSize;
};
const Expected kExpected[] = {
    {"__entry_add", {3, 4, 5}, {}, false},   {"__entry_scale", {3}, {4, 5}, false},
    {"__entry_lsum", {3, 4}, {}, false},     {"__entry_cmul", {3, 4}, {}, false},
    {"__entry_st", {3}, {4}, false},         {"__entry_wg", {3}, {}, true},
    {"__entry_items", {3}, {4}, false},
};

// One argument of a compute pipeline's reflection, as the x86 Macs answer it: its name, type,
// index and access, and a buffer's data size, alignment, data type and constant-ness, or
// threadgroup memory's alignment and size.
struct Argument {
	const char *name;
	NSUInteger type, index, access, size, alignment, dataType;
	bool constant;
};
const Argument kHidden[] = {{"__global_offset_and_num_dims", 0, 0, 0, 4, 4, 33, false},
                            {"__printf_buffer", 0, 1, 1, 1, 1, 45, false},
                            {"__image_order_data_type_table", 0, 2, 0, 4, 4, 33, false}};
const struct {
	const char *name;
	std::vector<Argument> arguments;
} kReflections[] = {
    {"__entry_add", {{"a", 0, 3, 0, 4, 4, 3, false}, {"b", 0, 4, 0, 4, 4, 3, false}, {"c", 0, 5, 1, 4, 4, 3, false}}},
    {"__entry_scale", {{"b", 0, 3, 1, 4, 4, 3, false}, {"k", 0, 4, 0, 4, 4, 3, false}, {"n", 0, 5, 0, 4, 4, 33, false}}},
    {"__entry_lsum", {{"in", 0, 3, 0, 4, 4, 3, false}, {"out", 0, 4, 1, 4, 4, 3, false}, {"tmp", 1, 0, 1, 4, 4, 0, false}}},
    {"__entry_cmul", {{"o", 0, 3, 1, 4, 4, 3, false}, {"c", 0, 4, 0, 4, 32, 3, true}}},
    {"__entry_st", {{"o", 0, 3, 1, 4, 4, 3, false}, {"s", 0, 4, 0, 8, 4, 1, false}}},
    {"__entry_wg", {{"o", 0, 3, 1, 4, 4, 3, false}}},
};

// A node of a kernel's argument, as Metal's compiler writes it for OpenCL C.
RgfxAirNode node(uint32_t position, const char *kind, uint32_t index, uint32_t addressSpace)
{
	RgfxAirNode n;
	n.hasPosition = true;
	n.position = position;
	n.kind = kind;
	n.hasLocation = true;
	n.index = index;
	n.count = 1;
	if (addressSpace != 0) {
		n.hasAddressSpace = true;
		n.addressSpace = addressSpace;
	}
	return n;
}

void test_cl_bindings_of()
{
	// A __global pointer, a value, a __constant pointer, a __local pointer and a struct by value,
	// and reqd_work_group_size(64, 2, 1).
	RgfxAirEntry k;
	k.name = "__entry_k";
	k.arguments = {node(0, "air.buffer", 0, 1), node(1, "air.constant", 1, 0), node(2, "air.buffer", 2, 2),
	               node(3, "air.buffer", 0, 3), node(4, "air.constant", 3, 0)};
	k.hasWorkGroupSize = true;
	k.workGroupSize[0] = 64;
	k.workGroupSize[1] = 2;
	RgfxClBindings b;
	std::string why;
	CHECK(RgfxClBindingsOf(k, b, why));
	CHECK(b.buffers == std::vector<uint32_t>({3, 5}) && b.byValue == std::vector<uint32_t>({4, 6}));
	CHECK(b.byValuePositions == std::vector<uint32_t>({1, 4}) && b.parameters == 5);
	CHECK(b.hasWorkGroupSize && b.workGroupSize[0] == 64 && b.workGroupSize[1] == 2 && b.workGroupSize[2] == 1);
	// A texture, a sampler, a binding at no fixed index, and a vertex function are refused.
	for (const char *kind : {"air.texture", "air.sampler", "air.indirect_buffer"}) {
		RgfxAirEntry t = k;
		t.arguments.push_back(node(5, kind, 0, 0));
		CHECK(!RgfxClBindingsOf(t, b, why) && b.buffers.empty() && !why.empty());
	}
	RgfxAirEntry g = k;
	g.arguments[0].indexIsGlobal = true;
	CHECK(!RgfxClBindingsOf(g, b, why));
	RgfxAirEntry v = k;
	v.stage = RgfxAirEntryStage::Vertex;
	CHECK(!RgfxClBindingsOf(v, b, why));
}

// The container this Mac's Metal compiles the probe's program to, and its kernels' .metallib.
bool programOfThisMac(id<MTLDevice> metal, NSData **container, std::vector<uint8_t> &metallib)
{
	NSArray<NSString *> *names = nil;
	*container = clLibraryBytes(metal, kSource, &names);
	if (*container == nil || !RgfxMetallibOfClLibrary(static_cast<const uint8_t *>((*container).bytes),
	                                                  (*container).length, metallib)) {
		fprintf(stderr, "cl_mode_tests: this Mac's Metal answers no OpenCL container: no kernels checked\n");
		return false;
	}
	return true;
}

void checkArgument(MTLArgument *a, const Argument &want)
{
	const bool named = strcmp(a.name.UTF8String, want.name) == 0;
	if (!named || a.type != want.type || a.index != want.index || a.access != want.access || !a.isActive)
		fprintf(stderr, "cl_mode_tests: argument %s: type %lu index %lu access %lu (want %s %lu %lu %lu)\n",
		        a.name.UTF8String, (unsigned long)a.type, (unsigned long)a.index, (unsigned long)a.access, want.name,
		        (unsigned long)want.type, (unsigned long)want.index, (unsigned long)want.access);
	CHECK(named && a.type == want.type && a.index == want.index && a.access == want.access && a.isActive);
	if (want.type == MTLArgumentTypeThreadgroupMemory) {
		CHECK(a.threadgroupMemoryAlignment == want.alignment && a.threadgroupMemoryDataSize == want.size);
		return;
	}
	if (a.bufferDataSize != want.size || a.bufferAlignment != want.alignment || a.bufferDataType != want.dataType)
		fprintf(stderr, "cl_mode_tests: argument %s: size %lu alignment %lu data type %lu\n", a.name.UTF8String,
		        (unsigned long)a.bufferDataSize, (unsigned long)a.bufferAlignment, (unsigned long)a.bufferDataType);
	CHECK(a.bufferDataSize == want.size && a.bufferAlignment == want.alignment && a.bufferDataType == want.dataType);
	id type = [a valueForKey:@"dataTypeDescription"];
	CHECK([type isKindOfClass:[MTLPointerType class]] && [[type valueForKey:@"isConstantBuffer"] boolValue] == want.constant);
}

void test_cl_function_of_opencl_mode()
{
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	NSData *container = nil;
	std::vector<uint8_t> metallib;
	if (!programOfThisMac(metal, &container, metallib))
		return;
	// The transport is never freed, as the shell tests' fixture's: a library may dealloc later.
	TestRing &ring = *new TestRing(0x40000);
	RgfxRingWriter &writer =
	    *new RgfxRingWriter(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(writer, ids);
	ring.setConsumerIdle(1);
	static RgfxClModeTestsDevice *device = [RgfxClModeTestsDevice new];
	device.openCLMode = YES;
	uint32_t status = 0;
	RgfxLibrary *program = [RgfxLibrary libraryWithData:metallib.data()
	                                             length:metallib.size()
	                                                key:nullptr
	                                             device:(id<MTLDevice>)device
	                                         serializer:&ser
	                                             status:&status];
	[program rgfxSetSource:dispatch_data_create(container.bytes, container.length, nil,
	                                            ^{
		                                            (void)container;
	                                            })
	           shadowMaker:^id<MTLLibrary>(dispatch_data_t) {
		           return nil;
	           }];
	drainRing(ring);
	CHECK(program != nil);
	// Each kernel's work-group size, as the AIR reader takes it from its module.
	for (const Expected &e : kExpected) {
		std::shared_ptr<const RgfxAirEntries> entries = [program rgfxAirEntryNamed:e.name];
		CHECK(entries != nullptr && entries->entries.size() == 1);
		if (entries == nullptr || entries->entries.size() != 1)
			continue;
		const RgfxAirEntry &k = entries->entries[0];
		CHECK(k.name == e.name && k.hasWorkGroupSize == e.workGroupSize);
		if (e.workGroupSize)
			CHECK(k.workGroupSize[0] == 64 && k.workGroupSize[1] == 1 && k.workGroupSize[2] == 1);
	}
	const uint8_t pluginBytes[32] = {2};
	NSData *plugin = [NSData dataWithBytes:pluginBytes length:sizeof pluginBytes];
	for (const Expected &e : kExpected) {
		RgfxFunction *kernel = (RgfxFunction *)[program newFunctionWithName:@(e.name)];
		CHECK(kernel != nil && [kernel rgfxClBindings] == nullptr && [(id)kernel valueForKey:@"pluginData"] == nil);
		id<MTLFunction> variant = [kernel newFunctionWithPluginData:plugin bitcodeType:2];
		CHECK([(id)variant isKindOfClass:[RgfxFunction class]] && variant != kernel);
		if (![(id)variant isKindOfClass:[RgfxFunction class]])
			continue;
		CHECK([variant.name isEqualToString:@(e.name)] && variant.functionType == MTLFunctionTypeKernel);
		CHECK([[(id)variant valueForKey:@"pluginData"] isEqual:plugin]);
		CHECK([(RgfxFunction *)variant rgfxClBindings] != nullptr);
		// The plugin return data, as Metal's.
		NSMutableDictionary *expected = [NSMutableDictionary dictionary];
		NSMutableArray *buffers = [NSMutableArray array], *values = [NSMutableArray array];
		for (uint32_t i : e.buffers)
			[buffers addObject:@(i)];
		for (uint32_t i : e.byValue)
			[values addObject:@(i)];
		expected[@"LinkerScript_buffer_bindings"] = buffers;
		if (values.count != 0)
			expected[@"LinkerScript_byVal_bindings"] = values;
		expected[@"LinkerScript_function_name"] = @(e.name);
		if (e.workGroupSize)
			expected[@"air.work_group_size"] = @[ @64, @1, @1 ];
		NSData *returned = RgfxGLPluginReturnData(variant);
		NSPropertyListFormat format = NSPropertyListXMLFormat_v1_0;
		id plist = returned != nil ? [NSPropertyListSerialization propertyListWithData:returned
		                                                                       options:0
		                                                                        format:&format
		                                                                         error:nil]
		                           : nil;
		if (![plist isEqual:expected])
			fprintf(stderr, "cl_mode_tests: %s: plugin return data %s\n", e.name, [plist description].UTF8String);
		CHECK([plist isEqual:expected] && format == NSPropertyListBinaryFormat_v1_0);
	}
	// The pipelines' reflection: the kernel's arguments at Metal's indices, then the buffers.
	for (const auto &want : kReflections) {
		id<MTLFunction> variant =
		    [(RgfxFunction *)[program newFunctionWithName:@(want.name)] newFunctionWithPluginData:plugin bitcodeType:2];
		id reflection = RgfxComputePipelineReflectionOf(variant, nil,
		                                                MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo);
		NSArray *arguments = [reflection valueForKey:@"arguments"];
		if (arguments.count != want.arguments.size() + 3)
			fprintf(stderr, "cl_mode_tests: %s: %lu arguments\n", want.name, (unsigned long)arguments.count);
		CHECK(arguments.count == want.arguments.size() + 3);
		if (arguments.count != want.arguments.size() + 3)
			continue;
		for (size_t i = 0; i < want.arguments.size(); i++)
			checkArgument(arguments[i], want.arguments[i]);
		for (size_t i = 0; i < 3; i++)
			checkArgument(arguments[want.arguments.size() + i], kHidden[i]);
	}
	// Another plugin data, another bitcodeType, and a host that makes no pipeline of OpenCL mode
	// make none.
	RgfxFunction *add = (RgfxFunction *)[program newFunctionWithName:@"__entry_add"];
	const uint8_t otherBytes[32] = {1};
	CHECK([add newFunctionWithPluginData:[NSData dataWithBytes:otherBytes length:sizeof otherBytes] bitcodeType:2] ==
	      nil);
	CHECK([add newFunctionWithPluginData:[NSData dataWithBytes:pluginBytes length:16] bitcodeType:2] == nil);
	CHECK([add newFunctionWithPluginData:plugin bitcodeType:1] == nil);
	device.openCLMode = NO;
	CHECK([add newFunctionWithPluginData:plugin bitcodeType:2] == nil);
	device.openCLMode = YES;
	drainRing(ring);
}

} // namespace

void rgfxClModeTests()
{
	@autoreleasepool {
		test_cl_bindings_of();
		test_cl_function_of_opencl_mode();
	}
}
