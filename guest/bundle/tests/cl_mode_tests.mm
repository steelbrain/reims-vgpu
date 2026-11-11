// The OpenCL kernels' checks (tests/shell_tests.mm's main calls rgfxClModeTests()):
//   - what a kernel binds in OpenCL mode (src/RgfxClMode.*), from nodes written here in the shape
//     Metal's compiler writes for OpenCL C: an image and a sampler at their own indices, and the
//     refusals of an array of images, an argument buffer and a binding at no fixed index;
//   - the work-group size the AIR reader takes from a kernel's node (air.work_group_size), over
//     this Mac's compile of an OpenCL C program as the GL renderer compiles it;
//   - a kernel's function of OpenCL mode (src/RgfxFunction.mm, newFunctionWithPluginData:
//     bitcodeType:2), over that compile: its name, stage and plugin data, the plugin return data,
//     and its compute pipeline's reflection, each as the x86 Macs answer them (measured: macOS
//     26.7.1, the Intel 630 and the AMD 5300M alike, through the GL renderer), and none made where
//     the host makes no pipeline of OpenCL mode, or of another plugin data or bitcodeType;
//   - the same of kernels of images and samplers, over this Mac's compile of a program of them, as
//     the x86 Macs answer them (measured: macOS 26.7.1, both Macs alike, Metal's OpenCL mode driven
//     as the renderer drives it).
// Nothing here runs on the GPU: tools/cl_probe.m runs the kernels on a device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <iterator>
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
// threadgroup memory's alignment and size; a texture's data type is its `dataType`.
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
	// An image and a sampler argument are at their own indices, in their order.
	RgfxAirEntry images = k;
	images.arguments.push_back(node(5, "air.texture", 1, 0));
	images.arguments.push_back(node(6, "air.sampler", 0, 0));
	images.arguments.push_back(node(7, "air.texture", 0, 0));
	CHECK(RgfxClBindingsOf(images, b, why));
	CHECK(b.textures == std::vector<uint32_t>({1, 0}) && b.samplers == std::vector<uint32_t>({0}) &&
	      b.buffers == std::vector<uint32_t>({3, 5}) && b.parameters == 8);
	// An array of images, an argument buffer, a binding at no fixed index, and a vertex function
	// are refused.
	RgfxAirEntry array = k;
	array.arguments.push_back(node(5, "air.texture", 0, 0));
	array.arguments.back().count = 2;
	CHECK(!RgfxClBindingsOf(array, b, why) && b.buffers.empty() && !why.empty());
	// A 1D image and a 1D array image are refused, until the host reads the 2D textures Apple's
	// OpenCL binds for them (RgfxClMode.hpp).
	for (const char *type : {"texture1d<unknown, read>", "texture1d_array<unknown, read>"}) {
		RgfxAirEntry oneD = images;
		oneD.arguments.push_back(node(8, "air.texture", 2, 0));
		oneD.arguments.back().typeName = type;
		CHECK(!RgfxClBindingsOf(oneD, b, why) && b.textures.empty() && why.find("1D image") != std::string::npos);
	}
	RgfxAirEntry t = k;
	t.arguments.push_back(node(5, "air.indirect_buffer", 0, 0));
	CHECK(!RgfxClBindingsOf(t, b, why) && b.buffers.empty() && !why.empty());
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

void checkArgument(MTLArgument *a, const Argument &want, NSUInteger textureType = 0, bool depth = false);

void checkTextureOrSampler(MTLArgument *a, const Argument &want, NSUInteger textureType, bool depth)
{
	const bool named = strcmp(a.name.UTF8String, want.name) == 0;
	const bool texture = want.type == MTLArgumentTypeTexture;
	const bool ok = named && a.type == want.type && a.index == want.index && a.access == want.access && a.isActive &&
	                (!texture || (a.textureType == textureType && a.textureDataType == want.dataType &&
	                              a.isDepthTexture == depth));
	if (!ok)
		fprintf(stderr, "cl_mode_tests: argument %s: type %lu index %lu access %lu texture type %lu data type %lu "
		                "(want %s %lu %lu %lu %lu %lu)\n",
		        a.name.UTF8String, (unsigned long)a.type, (unsigned long)a.index, (unsigned long)a.access,
		        texture ? (unsigned long)a.textureType : 0UL, texture ? (unsigned long)a.textureDataType : 0UL,
		        want.name, (unsigned long)want.type, (unsigned long)want.index, (unsigned long)want.access,
		        (unsigned long)textureType, (unsigned long)want.dataType);
	CHECK(ok);
}

void checkArgument(MTLArgument *a, const Argument &want, NSUInteger textureType, bool depth)
{
	if (want.type == MTLArgumentTypeTexture || want.type == MTLArgumentTypeSampler) {
		checkTextureOrSampler(a, want, textureType, depth);
		return;
	}
	const bool named = strcmp(a.name.UTF8String, want.name) == 0;
	if (!named || a.type != want.type || a.index != want.index || a.access != want.access || !a.isActive)
		fprintf(stderr, "cl_mode_tests: argument %s: type %lu index %lu access %lu (want %s %lu %lu %lu)\n",
		        a.name.UTF8String, (unsigned long)a.type, (unsigned long)a.index, (unsigned long)a.access, want.name,
		        (unsigned long)want.type, (unsigned long)want.index, (unsigned long)want.access);
	CHECK(named && a.type == want.type && a.index == want.index && a.access == want.access && a.isActive);
	if (want.type == MTLArgumentTypeThreadgroupMemory) {
		// A size of 0: its size and alignment were not measured.
		if (want.size != 0)
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

// The kernels of images and samplers the x86 Macs were measured over (an image read and one
// written; a buffer, an image, a sampler argument, a value and a buffer; an image's queries; two
// read images; a 3D image and a 1D buffer image; 1D, 1D array, 2D array and 3D images; samplers,
// images, a buffer, values and local memory interleaved; a sampler and images alone; a depth
// image), each a constant sampler's where it has no sampler argument.
NSString *const kImageSource =
    @"__constant sampler_t cs = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    @"__kernel void img_copy(__read_only image2d_t in, __write_only image2d_t out) {\n"
    @"  int2 p = (int2)(get_global_id(0), get_global_id(1));\n"
    @"  write_imagef(out, p, read_imagef(in, cs, p) * 2.0f); }\n"
    @"__kernel void img_mixed(__global float *pre, __read_only image2d_t in, sampler_t s, float k,\n"
    @"                        __global float4 *out) {\n"
    @"  int2 p = (int2)(get_global_id(0), get_global_id(1));\n"
    @"  out[p.y * get_global_size(0) + p.x] = read_imagef(in, s, p) * k + pre[0]; }\n"
    @"__kernel void img_query(__read_only image2d_t a, __write_only image2d_t b, __global int *o) {\n"
    @"  if (get_global_id(0) == 0 && get_global_id(1) == 0) {\n"
    @"    o[0] = get_image_width(a); o[1] = get_image_height(a);\n"
    @"    o[2] = get_image_channel_order(a); o[3] = get_image_channel_data_type(a);\n"
    @"    o[4] = get_image_width(b); o[5] = get_image_channel_order(b); o[6] = get_image_channel_data_type(b); } }\n"
    @"__kernel void img_two(__read_only image2d_t a, __read_only image2d_t b, __global uint4 *o) {\n"
    @"  int2 p = (int2)(get_global_id(0), get_global_id(1));\n"
    @"  o[p.y * get_global_size(0) + p.x] = read_imageui(a, cs, p) + read_imageui(b, cs, p); }\n"
    @"__kernel void img_shapes(__read_only image3d_t v, __read_only image1d_buffer_t lb, __global float4 *o) {\n"
    @"  int x = get_global_id(0), y = get_global_id(1);\n"
    @"  o[y * get_global_size(0) + x] = read_imagef(v, cs, (int4)(x, y, 1, 0)) + read_imagef(lb, x); }\n"
    @"__kernel void img_kinds(__read_only image1d_t a, __read_only image1d_array_t b,\n"
    @"                        __read_only image2d_array_t c, __write_only image3d_t d, __global float4 *o) {\n"
    @"  int x = get_global_id(0);\n"
    @"  o[x] = read_imagef(a, x) + read_imagef(b, (int2)(x, 0)) + read_imagef(c, (int4)(x, 0, 0, 0));\n"
    @"  write_imagef(d, (int4)(x, 0, 0, 0), o[x]); }\n"
    @"__kernel void img_order(sampler_t s1, __read_only image2d_t i1, __global float4 *b, uint n,\n"
    @"                        sampler_t s2, __local float4 *t, __read_only image2d_t i2, float v,\n"
    @"                        __write_only image2d_t w) {\n"
    @"  int2 p = (int2)(get_global_id(0), get_global_id(1));\n"
    @"  t[get_local_id(0)] = read_imagef(i1, s1, p) + read_imagef(i2, s2, p) * v;\n"
    @"  barrier(CLK_LOCAL_MEM_FENCE);\n"
    @"  if (p.x < n) b[p.x] = t[0];\n"
    @"  write_imagef(w, p, t[0]); }\n"
    @"__kernel void img_only_sampler(sampler_t s, __read_only image2d_t i, __write_only image2d_t w) {\n"
    @"  int2 p = (int2)(get_global_id(0), get_global_id(1));\n"
    @"  write_imagef(w, p, read_imagef(i, s, p)); }\n";
// A depth image's kernel (cl_khr_depth_images), a program of its own as it was measured.
NSString *const kDepthSource = @"#pragma OPENCL EXTENSION cl_khr_depth_images : enable\n"
                               @"__kernel void img_depth(__read_only image2d_depth_t z, __global float *o) {\n"
                               @"  int x = get_global_id(0);\n"
                               @"  o[x] = read_imagef(z, (int2)(x, 0)); }\n";

// Each kernel's plugin return data and reflection, as both x86 Macs answer them: the lists, then
// the arguments in Metal's order, before the three hidden buffers, each texture's texture type (0
// for another argument), and which are depth textures (none where it is empty).
struct ImageKernel {
	const char *name;
	std::vector<uint32_t> buffers, byValue, samplers, textures;
	std::vector<Argument> arguments;
	std::vector<NSUInteger> textureTypes;
	std::vector<bool> depth;
};
const ImageKernel kImageKernels[] = {
    {"__entry_img_copy", {}, {}, {}, {0, 1},
     {{"in", 2, 0, 0, 0, 0, 0, false}, {"out", 2, 1, 2, 0, 0, 0, false}}, {2, 2}, {}},
    {"__entry_img_mixed", {3, 5}, {4}, {0}, {0},
     {{"pre", 0, 3, 1, 4, 4, 3, false},
      {"in", 2, 0, 0, 0, 0, 0, false},
      {"s", 3, 0, 0, 0, 0, 0, false},
      {"out", 0, 5, 1, 16, 16, 6, false},
      {"k", 0, 4, 0, 4, 4, 3, false}},
     {0, 2, 0, 0, 0}, {}},
    {"__entry_img_query", {3}, {}, {}, {0, 1},
     {{"a", 2, 0, 0, 0, 0, 0, false}, {"b", 2, 1, 2, 0, 0, 0, false}, {"o", 0, 3, 1, 4, 4, 29, false}}, {2, 2, 0}, {}},
    {"__entry_img_two", {3}, {}, {}, {0, 1},
     {{"a", 2, 0, 0, 0, 0, 0, false}, {"b", 2, 1, 0, 0, 0, 0, false}, {"o", 0, 3, 1, 16, 16, 36, false}}, {2, 2, 0}, {}},
    {"__entry_img_shapes", {3}, {}, {}, {0, 1},
     {{"v", 2, 0, 0, 0, 0, 0, false}, {"lb", 2, 1, 0, 0, 0, 0, false}, {"o", 0, 3, 1, 16, 16, 6, false}}, {7, 9, 0}, {}},
    // A 1D image is a 2D texture, and a 1D array image a 2D array one.
    {"__entry_img_kinds", {3}, {}, {}, {0, 1, 2, 3},
     {{"a", 2, 0, 0, 0, 0, 0, false},
      {"b", 2, 1, 0, 0, 0, 0, false},
      {"c", 2, 2, 0, 0, 0, 0, false},
      {"d", 2, 3, 2, 0, 0, 0, false},
      {"o", 0, 3, 1, 16, 16, 6, false}},
     {2, 3, 3, 7, 0}, {}},
    {"__entry_img_order", {3}, {4, 5}, {0, 1}, {0, 1, 2},
     {{"s1", 3, 0, 0, 0, 0, 0, false},
      {"i1", 2, 0, 0, 0, 0, 0, false},
      {"b", 0, 3, 1, 16, 16, 6, false},
      {"s2", 3, 1, 0, 0, 0, 0, false},
      {"t", 1, 0, 1, 0, 0, 0, false},
      {"i2", 2, 1, 0, 0, 0, 0, false},
      {"w", 2, 2, 2, 0, 0, 0, false},
      {"n", 0, 4, 0, 4, 4, 33, false},
      {"v", 0, 5, 0, 4, 4, 3, false}},
     {0, 2, 0, 0, 0, 2, 2, 0, 0}, {}},
    {"__entry_img_only_sampler", {}, {}, {0}, {0, 1},
     {{"s", 3, 0, 0, 0, 0, 0, false}, {"i", 2, 0, 0, 0, 0, 0, false}, {"w", 2, 1, 2, 0, 0, 0, false}},
     {0, 2, 2}, {}},
};
// The kernels of a 1D image, which are not made (RgfxClMode.hpp): no function of OpenCL mode.
// Their entries above are the x86 Macs' answers, for when the host reads such images.
const char *const kRefusedImageKernels[] = {"__entry_img_kinds"};
const ImageKernel kDepthKernel = {"__entry_img_depth", {3}, {}, {}, {0},
                                  {{"z", 2, 0, 0, 0, 0, 0, false}, {"o", 0, 3, 1, 4, 4, 3, false}}, {2, 0},
                                  {true, false}};

// Each kernel of `source`, this Mac's compile of it, as `kernels` says the x86 Macs answer it.
void checkImageKernels(id<MTLDevice> metal, NSString *source, const ImageKernel *kernels, size_t count)
{
	NSArray<NSString *> *names = nil;
	NSData *container = clLibraryBytes(metal, source, &names);
	std::vector<uint8_t> metallib;
	if (container == nil || !RgfxMetallibOfClLibrary(static_cast<const uint8_t *>(container.bytes), container.length,
	                                                 metallib)) {
		fprintf(stderr, "cl_mode_tests: this Mac's Metal answers no OpenCL container: no image kernels checked\n");
		return;
	}
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
	const uint8_t pluginBytes[32] = {2};
	NSData *plugin = [NSData dataWithBytes:pluginBytes length:sizeof pluginBytes];
	auto numbers = [](const std::vector<uint32_t> &v) {
		NSMutableArray *a = [NSMutableArray array];
		for (uint32_t i : v)
			[a addObject:@(i)];
		return a;
	};
	for (size_t k = 0; k < count; k++) {
		const ImageKernel &want = kernels[k];
		RgfxFunction *kernel = (RgfxFunction *)[program newFunctionWithName:@(want.name)];
		id<MTLFunction> variant = [kernel newFunctionWithPluginData:plugin bitcodeType:2];
		bool refused = false;
		for (const char *name : kRefusedImageKernels)
			refused = refused || strcmp(name, want.name) == 0;
		if (refused) {
			CHECK(kernel != nil && variant == nil);
			continue;
		}
		if (![(id)variant isKindOfClass:[RgfxFunction class]])
			fprintf(stderr, "cl_mode_tests: %s: no function of OpenCL mode\n", want.name);
		CHECK([(id)variant isKindOfClass:[RgfxFunction class]]);
		NSMutableDictionary *expected = [NSMutableDictionary dictionary];
		if (!want.buffers.empty())
			expected[@"LinkerScript_buffer_bindings"] = numbers(want.buffers);
		if (!want.byValue.empty())
			expected[@"LinkerScript_byVal_bindings"] = numbers(want.byValue);
		if (!want.samplers.empty())
			expected[@"LinkerScript_sampler_bindings"] = numbers(want.samplers);
		if (!want.textures.empty())
			expected[@"LinkerScript_texture_bindings"] = numbers(want.textures);
		expected[@"LinkerScript_function_name"] = @(want.name);
		NSData *returned = RgfxGLPluginReturnData(variant);
		id plist = returned != nil ? [NSPropertyListSerialization propertyListWithData:returned
		                                                                       options:0
		                                                                        format:nil
		                                                                         error:nil]
		                           : nil;
		if (![plist isEqual:expected])
			fprintf(stderr, "cl_mode_tests: %s: plugin return data %s\n", want.name, [plist description].UTF8String);
		CHECK([plist isEqual:expected]);
		id reflection = RgfxComputePipelineReflectionOf(variant, nil,
		                                                MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo);
		NSArray *arguments = [reflection valueForKey:@"arguments"];
		if (arguments.count != want.arguments.size() + 3) {
			fprintf(stderr, "cl_mode_tests: %s: %lu arguments\n", want.name, (unsigned long)arguments.count);
			std::shared_ptr<const RgfxAirEntries> entries = [program rgfxAirEntryNamed:want.name];
			if (entries != nullptr && entries->entries.size() == 1)
				for (const RgfxAirNode &n : entries->entries[0].arguments)
					fprintf(stderr, "cl_mode_tests: %s: node %s %s index %u type %s\n", want.name, n.name.c_str(),
					        n.kind.c_str(), n.index, n.typeName.c_str());
		}
		CHECK(arguments.count == want.arguments.size() + 3);
		for (size_t i = 0; i < want.arguments.size(); i++)
			checkArgument(arguments[i], want.arguments[i], want.textureTypes[i], i < want.depth.size() && want.depth[i]);
		for (size_t i = 0; i < 3; i++)
			checkArgument(arguments[want.arguments.size() + i], kHidden[i]);
	}
	drainRing(ring);
}

void test_cl_images_of_opencl_mode()
{
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	checkImageKernels(metal, kImageSource, kImageKernels, std::size(kImageKernels));
	checkImageKernels(metal, kDepthSource, &kDepthKernel, 1);
}

} // namespace

void rgfxClModeTests()
{
	@autoreleasepool {
		test_cl_bindings_of();
		test_cl_function_of_opencl_mode();
		test_cl_images_of_opencl_mode();
	}
}
