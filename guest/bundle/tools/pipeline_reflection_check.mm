// pipeline_reflection_check: the reflection the bundle builds for render, mesh and compute pipelines
// (src/RgfxPipelineReflection.*, from the nodes src/RgfxAirArguments.* reads) against the
// reflection Metal answers for the same pipelines on the device it runs on.
//   pipeline_reflection_check DEVICE LIB.metallib [CASE] [-v]
// DEVICE is a substring of the device's name (AMD, Intel, Apple). LIB is
// tools/oracle/oracle_reflection.metal's library. For each case (oracle_reflection's, and the ones
// below), Metal's pipeline is made with options 0 to 3 and both reflections, ours from the same
// functions' AIR in LIB, are walked alike (tools/oracle/oracle_walk.h: every getter and every
// instance variable, recursively), the lines compared by their path:
//   k_arrays     a constant struct of arrays of scalars, vectors, matrices and structs
//   k_predicates arguments under kOn (true) and kOff (false), at plain indices
//   k_builtins   a kernel's further built-in arguments
//   d_more       k_stage_in by a descriptor, its stage input descriptor of buffers 1, 4 and 30, as
//                r_more's
//   r_more       v_more (more built-ins and outputs) and f_more, a vertex descriptor of buffers 1, 4
//                and 30, buffer 30 holding only an attribute the function declares and does not read
//   c_nested     an argument buffer of a struct of members, arrays of them, of samplers, buffers,
//                constants and textures, and a pointer to another argument buffer
//   c_tables     a visible function table, an intersection function table and acceleration
//                structures of both kinds
//   c_more_textures  textures of further shapes, sample types and access
//   c_imageblock an imageblock
//   c_atomic_texture  a texture of 64-bit atomics (its library, LIB, is
//                tools/oracle/oracle_reflection_31.metal's)
//   r_fetch      v_buffers and f_fetch, which reads its colour attachments (framebuffer fetch)
//   r_tables     v_buffers and f_tables, an acceleration structure and a function table in a
//                fragment function
//   c_specialised_on, c_specialised_off
//                k_specialised with kOn true, kOff false, kIndex 5 and kMode 1, kOpt undefined;
//                and with both false, kIndex 7, kMode 2 and kOpt true
//   r_specialised_one, r_specialised_two
//                v_specialised and f_plain, a vertex descriptor as r_stage_in's, kMode 1 and 2
// Prints each difference (with -v, every line both walks agree on), a case ours makes no reflection
// of with its reason and Metal's getters' answers (with -v its instance variables' too, to measure
// them), a case Metal makes no pipeline of with its error, and
//   pipeline_reflection_check: C cases, V values, D differ, N not made, U Metal made none
// Exits 1 where any differs. Metal's own answers are the contract the bundle's must equal; a case
// ours does not make answers nil, as the device did before it had any.
// Build: make pipeline-reflection-check (build/pipeline_reflection_check-x86_64; ARCH=arm64 for an
// Apple-silicon Mac), and the library: make build/oracle_reflection.metallib.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <map>
#include <string>
#include <vector>

#include "RgfxAirArguments.hpp"
#import "RgfxPipelineReflection.h"
#import "oracle/oracle_walk.h"

namespace {

id<MTLDevice> D;
id<MTLLibrary> L;
std::vector<uint8_t> Bytes; // LIB's bytes
bool Verbose;
unsigned Cases, Values, Differ, NotMade, Unmade;

// The nodes of LIB's entry point `name`.
bool entryOf(const std::string &name, RgfxAirEntries &module)
{
	const uint8_t *p = Bytes.data();
	size_t n = Bytes.size();
	rgfxMetallibAirSlice(p, n);
	const uint8_t *bitcode = nullptr;
	size_t size = 0;
	RgfxAirEntries all;
	if (!rgfxMetallibBitcode(p, n, name, &bitcode, &size) || !rgfxReadAirEntries(bitcode, size, all))
		return false;
	for (RgfxAirEntry &e : all.entries) {
		if (e.name == name) {
			module.entries = {e};
			module.constants = all.constants;
			module.code = all.code;
			return true;
		}
	}
	return false;
}

// One function of a case: its name and the constant values it is made with.
struct Fn {
	std::string name;
	std::vector<RgfxFunctionConstant> values;
};

bool stageOf(const Fn &f, NSString *prefix, const std::vector<RgfxDescriptorBuffer> &buffers, RgfxStageReflection &out,
             NSString *__strong *why)
{
	RgfxAirEntries module;
	if (!entryOf(f.name, module)) {
		*why = [NSString stringWithFormat:@"%s: its nodes were not read", f.name.c_str()];
		return false;
	}
	return RgfxStageReflectionOfEntry(module, module.entries[0], f.values.data(), static_cast<uint32_t>(f.values.size()),
	                                  prefix, buffers, out, why);
}

RgfxFunctionConstant constant(uint32_t index, MTLDataType type, int32_t value)
{
	RgfxFunctionConstant c = {};
	c.index = index;
	c.data_type = type;
	memcpy(c.value, &value, type == MTLDataTypeBool ? 1 : 4);
	return c;
}

id<MTLFunction> metalFunction(const Fn &f)
{
	NSError *error = nil;
	if (f.values.empty())
		return [L newFunctionWithName:@(f.name.c_str())];
	MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
	for (const RgfxFunctionConstant &c : f.values)
		[values setConstantValue:c.value type:static_cast<MTLDataType>(c.data_type) atIndex:c.index];
	return [L newFunctionWithName:@(f.name.c_str()) constantValues:values error:&error];
}

// The walk's lines of `o` keyed by path: `PATH = VALUE`, `PATH ivar NAME TYPE = VALUE`, `PATH class C`.
std::map<std::string, std::string> linesOf(id o, std::vector<std::string> &order)
{
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	walk(o, @"", 0, lines);
	resolve_ivars(lines);
	std::map<std::string, std::string> out;
	for (NSString *l in lines) {
		std::string s = l.UTF8String;
		size_t at = s.find(" = ");
		std::string key = at == std::string::npos ? s : s.substr(0, at);
		std::string value = at == std::string::npos ? "" : s.substr(at + 3);
		if (out.count(key) == 0)
			order.push_back(key);
		out[key] = value;
	}
	return out;
}

// Compares Metal's reflection with ours for one case and options.
void compare(const char *name, unsigned opts, id metal, id ours, NSString *why)
{
	Cases++;
	if (metal == nil || ours == nil) {
		if (metal != nil && ours == nil && why != nil) {
			NotMade++;
			printf("%s/%u: ours not made (%s)\n", name, opts, why.UTF8String);
			// Metal's own answers, to measure what ours lacks: its getters, and with -v its instance
			// variables.
			std::vector<std::string> order;
			auto m = linesOf(metal, order);
			for (const std::string &k : order)
				if (Verbose || k.find(" ivar ") == std::string::npos)
					printf("  metal %s = %s\n", k.c_str(), m[k].c_str());
			return;
		}
		Values++;
		if ((metal == nil) != (ours == nil)) {
			Differ++;
			printf("%s/%u: metal %s, ours %s\n", name, opts, metal ? "made" : "nil", ours ? "made" : "nil");
		}
		return;
	}
	std::vector<std::string> mo, oo;
	auto m = linesOf(metal, mo);
	auto o = linesOf(ours, oo);
	for (const std::string &k : mo) {
		Values++;
		auto it = o.find(k);
		if (it == o.end()) {
			Differ++;
			printf("%s/%u%s: metal %s, ours none\n", name, opts, k.c_str(), m[k].c_str());
		} else if (it->second != m[k]) {
			Differ++;
			printf("%s/%u%s: metal %s, ours %s\n", name, opts, k.c_str(), m[k].c_str(), it->second.c_str());
		} else if (Verbose) {
			printf("%s/%u%s = %s\n", name, opts, k.c_str(), m[k].c_str());
		}
	}
	for (const std::string &k : oo) {
		if (m.count(k) == 0) {
			Differ++;
			printf("%s/%u%s: metal none, ours %s\n", name, opts, k.c_str(), o[k].c_str());
		}
	}
}

void computeCase(const char *name, const Fn &kernel, MTLStageInputOutputDescriptor *stageIn, bool byDescriptor)
{
	id<MTLFunction> f = metalFunction(kernel);
	if (f == nil) {
		printf("%s: no function %s\n", name, kernel.name.c_str());
		Differ++;
		return;
	}
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedComputePipelineReflection metal = nil;
		NSError *error = nil;
		id state = nil;
		if (byDescriptor || stageIn != nil) {
			MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
			d.computeFunction = f;
			d.stageInputDescriptor = stageIn;
			state = [D newComputePipelineStateWithDescriptor:d options:opts reflection:&metal error:&error];
		} else {
			state = [D newComputePipelineStateWithFunction:f options:opts reflection:&metal error:&error];
		}
		if (state == nil) {
			printf("%s/%u: metal made no pipeline: %s\n", name, opts, error.localizedDescription.UTF8String);
			Unmade++;
			return;
		}
		id ours = nil;
		NSString *why = nil;
		RgfxStageReflection k;
		if (RgfxPipelineOptionsReflect(opts) &&
		    stageOf(kernel, @"stageInBuffer", RgfxStageInputBuffers(stageIn), k, &why))
			ours = RgfxComputePipelineReflectionMake(k, &why);
		compare(name, opts, metal, ours, why);
	}
}

void renderCase(const char *name, const Fn &vertex, const Fn *fragment, MTLVertexDescriptor *vd,
                MTLPixelFormat format0, MTLPixelFormat format1)
{
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = metalFunction(vertex);
	d.vertexDescriptor = vd;
	if (fragment != nullptr) {
		d.fragmentFunction = metalFunction(*fragment);
		d.colorAttachments[0].pixelFormat = format0;
		d.colorAttachments[1].pixelFormat = format1;
	} else {
		d.rasterizationEnabled = NO;
	}
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedRenderPipelineReflection metal = nil;
		NSError *error = nil;
		id state = [D newRenderPipelineStateWithDescriptor:d options:opts reflection:&metal error:&error];
		if (state == nil) {
			printf("%s/%u: metal made no pipeline: %s\n", name, opts, error.localizedDescription.UTF8String);
			Unmade++;
			return;
		}
		id ours = nil;
		NSString *why = nil;
		RgfxStageReflection v, f;
		if (RgfxPipelineOptionsReflect(opts) &&
		    stageOf(vertex, @"vertexBuffer", RgfxVertexDescriptorBuffers(vd), v, &why) &&
		    (fragment == nullptr || stageOf(*fragment, @"fragmentBuffer", {}, f, &why)))
			ours = RgfxRenderPipelineReflectionMake(&v, fragment ? &f : nullptr, nullptr, nullptr, &why);
		compare(name, opts, metal, ours, why);
	}
}

void meshCase(const char *name, const Fn &object, const Fn &mesh, const Fn &fragment)
{
	Class cls = NSClassFromString(@"MTLMeshRenderPipelineDescriptor");
	if (cls == Nil)
		return;
	MTLMeshRenderPipelineDescriptor *d = [cls new];
	d.objectFunction = metalFunction(object);
	d.meshFunction = metalFunction(mesh);
	d.fragmentFunction = metalFunction(fragment);
	d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedRenderPipelineReflection metal = nil;
		NSError *error = nil;
		[D newRenderPipelineStateWithMeshDescriptor:d options:opts reflection:&metal error:&error];
		id ours = nil;
		NSString *why = nil;
		RgfxStageReflection o, m, f;
		if (RgfxPipelineOptionsReflect(opts) && stageOf(object, @"objectBuffer", {}, o, &why) &&
		    stageOf(mesh, @"meshBuffer", {}, m, &why) && stageOf(fragment, @"fragmentBuffer", {}, f, &why))
			ours = RgfxRenderPipelineReflectionMake(nullptr, &f, &o, &m, &why);
		compare(name, opts, metal, ours, why);
	}
}

// A stage input descriptor: its attributes and strides at buffers 0, 1 and 2, or (`spread`) at
// buffers 4, 1 and 30, as vertexDescriptor's.
MTLStageInputOutputDescriptor *stageInput(bool spread)
{
	MTLStageInputOutputDescriptor *s = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
	const NSUInteger b0 = spread ? 4 : 0, b1 = 1, b2 = spread ? 30 : 2;
	const struct {
		NSUInteger attribute;
		MTLAttributeFormat format;
		NSUInteger offset, buffer;
	} kAttributes[] = {{0, MTLAttributeFormatFloat4, 0, b0},
	                   {1, MTLAttributeFormatFloat2, 0, b1},
	                   {2, MTLAttributeFormatFloat4, spread ? NSUInteger(0) : NSUInteger(16), spread ? b2 : b0},
	                   {5, MTLAttributeFormatFloat, 0, spread ? b0 : b2}};
	for (const auto &a : kAttributes) {
		s.attributes[a.attribute].format = a.format;
		s.attributes[a.attribute].offset = a.offset;
		s.attributes[a.attribute].bufferIndex = a.buffer;
	}
	const NSUInteger buffers[] = {b0, b1, b2}, strides[] = {32, 8, spread ? NSUInteger(16) : NSUInteger(4)};
	for (NSUInteger i = 0; i < 3; i++) {
		s.layouts[buffers[i]].stride = strides[i];
		s.layouts[buffers[i]].stepFunction = MTLStepFunctionThreadPositionInGridX;
	}
	return s;
}

// A vertex descriptor: d_stage_in's attributes and strides at buffers 0, 1 and 2, or (`spread`) at
// buffers 4, 1 and 30, attribute 2 alone in buffer 30.
MTLVertexDescriptor *vertexDescriptor(bool spread)
{
	MTLVertexDescriptor *v = [MTLVertexDescriptor vertexDescriptor];
	const NSUInteger b0 = spread ? 4 : 0, b1 = 1, b2 = spread ? 30 : 2;
	const struct {
		NSUInteger attribute;
		MTLVertexFormat format;
		NSUInteger offset, buffer;
	} kAttributes[] = {{0, MTLVertexFormatFloat4, 0, b0},
	                   {1, MTLVertexFormatFloat2, 0, b1},
	                   {2, MTLVertexFormatFloat4, spread ? NSUInteger(0) : NSUInteger(16), spread ? b2 : b0},
	                   {5, MTLVertexFormatFloat, 0, spread ? b0 : b2}};
	for (const auto &a : kAttributes) {
		v.attributes[a.attribute].format = a.format;
		v.attributes[a.attribute].offset = a.offset;
		v.attributes[a.attribute].bufferIndex = a.buffer;
	}
	v.layouts[b0].stride = 32;
	v.layouts[b1].stride = 8;
	v.layouts[b2].stride = spread ? 16 : 4;
	return v;
}

} // namespace

int main(int argc, char **argv)
{
	@autoreleasepool {
		const char *only = nullptr;
		std::vector<const char *> args;
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-v") == 0)
				Verbose = true;
			else
				args.push_back(argv[i]);
		}
		if (args.size() < 2) {
			fprintf(stderr, "usage: pipeline_reflection_check DEVICE LIB.metallib [CASE] [-v]\n");
			return 2;
		}
		if (args.size() > 2)
			only = args[2];
		setvbuf(stdout, nullptr, _IOLBF, 0);
		for (id<MTLDevice> d in MTLCopyAllDevices())
			if (strcasestr(d.name.UTF8String, args[0]))
				D = d;
		NSData *bytes = [NSData dataWithContentsOfFile:@(args[1])];
		NSError *error = nil;
		L = bytes ? [D newLibraryWithURL:[NSURL fileURLWithPath:@(args[1])] error:&error] : nil;
		if (D == nil || L == nil) {
			fprintf(stderr, "pipeline_reflection_check: no device %s or library %s\n", args[0], args[1]);
			return 2;
		}
		Bytes.assign(static_cast<const uint8_t *>(bytes.bytes), static_cast<const uint8_t *>(bytes.bytes) + bytes.length);
		printf("# pipeline_reflection_check on \"%s\", %s\n", D.name.UTF8String,
		       NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
		walk_init();
		Ivars = YES;
#define CASE(n) (only == nullptr || strcmp(only, n) == 0)
		const struct {
			const char *name, *kernel;
		} kKernels[] = {{"c_basic", "k_basic"},         {"c_unused", "k_unused"},
		                {"c_types", "k_types"},         {"c_more", "k_more"},
		                {"c_builtins_only", "k_builtins_only"}, {"c_nothing", "k_nothing"},
		                {"c_argument_buffer", "k_argument_buffer"}, {"k_arrays", "k_arrays"},
		                {"k_builtins", "k_builtins"}, {"c_nested", "k_nested"},
		                {"c_tables", "k_tables"}, {"c_more_textures", "k_more_textures"},
		                {"c_imageblock", "k_imageblock"}};
		for (const auto &k : kKernels)
			if (CASE(k.name))
				computeCase(k.name, Fn{k.kernel, {}}, nil, false);
		const std::vector<RgfxFunctionConstant> values = {constant(0, MTLDataTypeBool, 1),
		                                                  constant(1, MTLDataTypeBool, 0),
		                                                  constant(2, MTLDataTypeInt, 5)};
		if (CASE("c_constants"))
			computeCase("c_constants", Fn{"k_constants", values}, nil, false);
		if (only != nullptr && strcmp(only, "c_atomic_texture") == 0)
			computeCase("c_atomic_texture", Fn{"k_atomic_texture", {}}, nil, false);
		const std::vector<RgfxFunctionConstant> on = {constant(0, MTLDataTypeBool, 1), constant(1, MTLDataTypeBool, 0),
		                                              constant(2, MTLDataTypeInt, 5), constant(3, MTLDataTypeInt, 1)};
		const std::vector<RgfxFunctionConstant> off = {constant(0, MTLDataTypeBool, 0), constant(1, MTLDataTypeBool, 0),
		                                               constant(2, MTLDataTypeInt, 7), constant(3, MTLDataTypeInt, 2),
		                                               constant(4, MTLDataTypeBool, 1)};
		if (CASE("c_specialised_on"))
			computeCase("c_specialised_on", Fn{"k_specialised", on}, nil, false);
		if (CASE("c_specialised_off"))
			computeCase("c_specialised_off", Fn{"k_specialised", off}, nil, false);
		if (CASE("k_predicates"))
			computeCase("k_predicates", Fn{"k_predicates", {values[0], values[1]}}, nil, false);
		if (CASE("d_basic"))
			computeCase("d_basic", Fn{"k_basic", {}}, nil, true);
		if (CASE("d_stage_in"))
			computeCase("d_stage_in", Fn{"k_stage_in", {}}, stageInput(false), true);
		if (CASE("d_more"))
			computeCase("d_more", Fn{"k_stage_in", {}}, stageInput(true), true);
		const Fn textures{"f_textures", {}}, plain{"f_plain", {}}, targets{"f_targets", {}}, more{"f_more", {}};
		if (CASE("r_buffers"))
			renderCase("r_buffers", Fn{"v_buffers", {}}, &textures, nil, MTLPixelFormatBGRA8Unorm, MTLPixelFormatInvalid);
		if (CASE("r_stage_in"))
			renderCase("r_stage_in", Fn{"v_stage_in", {}}, &plain, vertexDescriptor(false), MTLPixelFormatBGRA8Unorm,
			           MTLPixelFormatInvalid);
		if (CASE("r_vertex"))
			renderCase("r_vertex", Fn{"v_only", {}}, nullptr, nil, MTLPixelFormatInvalid, MTLPixelFormatInvalid);
		if (CASE("r_targets"))
			renderCase("r_targets", Fn{"v_buffers", {}}, &targets, nil, MTLPixelFormatRGBA32Float,
			           MTLPixelFormatRGBA16Float);
		if (CASE("r_more"))
			renderCase("r_more", Fn{"v_more", {}}, &more, vertexDescriptor(true), MTLPixelFormatBGRA8Unorm,
			           MTLPixelFormatInvalid);
		const Fn fetch{"f_fetch", {}}, tables{"f_tables", {}};
		if (CASE("r_fetch"))
			renderCase("r_fetch", Fn{"v_buffers", {}}, &fetch, nil, MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float);
		if (CASE("r_tables"))
			renderCase("r_tables", Fn{"v_buffers", {}}, &tables, nil, MTLPixelFormatBGRA8Unorm, MTLPixelFormatInvalid);
		if (CASE("r_specialised_one"))
			renderCase("r_specialised_one", Fn{"v_specialised", on}, &plain, vertexDescriptor(false),
			           MTLPixelFormatBGRA8Unorm, MTLPixelFormatInvalid);
		if (CASE("r_specialised_two"))
			renderCase("r_specialised_two", Fn{"v_specialised", off}, &plain, vertexDescriptor(false),
			           MTLPixelFormatBGRA8Unorm, MTLPixelFormatInvalid);
		if (CASE("r_mesh"))
			meshCase("r_mesh", Fn{"o_payload", {}}, Fn{"m_payload", {}}, Fn{"f_mesh", {}});
		printf("pipeline_reflection_check: %u cases, %u values, %u differ, %u not made, %u Metal made none\n", Cases,
		       Values, Differ, NotMade, Unmade);
		return Differ != 0 ? 1 : 0;
	}
}
