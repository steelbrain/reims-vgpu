// RgfxPipelineReflectionOf: the pipeline reflection of our own functions (RgfxPipelineReflection.h,
// RgfxStageReflectionOf and the creators' RgfxComputePipelineReflectionOf,
// RgfxRenderPipelineReflectionOf and RgfxMeshPipelineReflectionOf): the entry point's nodes come
// from its library (RgfxLibrary, rgfxAirEntryNamed:), its constants' values from the function.
// Apart from RgfxPipelineReflection.mm, which tools/pipeline_reflection_check.mm builds alone.
#import "RgfxPipelineReflection.h"

#import "RgfxFunction.h"
#import "RgfxGLFunction.h"
#import "RgfxLibrary.h"
#import "RgfxTexture.h" // rgfxAppleGPU
#include "RgfxLog.hpp"

#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace {

// Logs `why` for a pipeline of `what`, once per reason.
void logOnce(NSString *what, NSString *why)
{
	static std::mutex lock;
	static std::set<std::string> said;
	std::lock_guard<std::mutex> hold(lock);
	if (said.insert(why.UTF8String).second)
		rgfxLogLine(@"RGFX: %@: no reflection (%@); answered nil", what, why);
}

// A kernel's entry as its pipeline of OpenCL mode binds it (RgfxClMode.hpp; measured: macOS
// 26.7.1, the Intel 630 and the AMD 5300M alike, through the GL renderer, and driving OpenCL mode
// directly for images and samplers): each __global and __constant pointer at its index plus 3;
// each value passed by value a buffer at its index plus 3, read only and no constant buffer, of
// the value's type, size and alignment; each __local pointer the threadgroup memory at its own
// index; each image and sampler at its own. The arguments are in the kernel's order, the values
// after all the others (img_mixed's `pre, in, s, k, out` is answered `pre, in, s, out, k`); then
// the three buffers the renderer binds.
RgfxAirEntry clModeEntry(const RgfxAirEntry &entry, const RgfxClBindings &cl)
{
	RgfxAirEntry out = entry;
	std::vector<RgfxAirNode> values;
	out.arguments.clear();
	for (RgfxAirNode n : entry.arguments) {
		if (n.kind == "air.constant") {
			n.kind = "air.buffer";
			n.hasAddressSpace = true;
			n.addressSpace = 1;
			n.access = RgfxAirAccess::Read;
			n.index += kRgfxClHiddenBuffers;
			values.push_back(std::move(n));
			continue;
		}
		if (n.kind == "air.buffer" && n.hasAddressSpace && n.addressSpace != 3)
			n.index += kRgfxClHiddenBuffers;
		out.arguments.push_back(std::move(n));
	}
	for (RgfxAirNode &n : values)
		out.arguments.push_back(std::move(n));
	static const struct {
		const char *name, *type;
		RgfxAirAccess access;
		uint32_t size;
	} kHidden[kRgfxClHiddenBuffers] = {{"__global_offset_and_num_dims", "uint", RgfxAirAccess::Read, 4},
	                                   {"__printf_buffer", "char", RgfxAirAccess::ReadWrite, 1},
	                                   {"__image_order_data_type_table", "uint", RgfxAirAccess::Read, 4}};
	for (uint32_t i = 0; i < kRgfxClHiddenBuffers; i++) {
		RgfxAirNode h;
		h.hasPosition = true;
		h.position = cl.parameters + i;
		h.kind = "air.buffer";
		h.hasLocation = true;
		h.index = i;
		h.count = 1;
		h.access = kHidden[i].access;
		h.hasAddressSpace = true;
		h.addressSpace = 1;
		h.hasTypeSize = h.hasTypeAlign = true;
		h.typeSize = h.typeAlign = kHidden[i].size;
		h.typeName = kHidden[i].type;
		h.name = kHidden[i].name;
		out.arguments.push_back(std::move(h));
	}
	return out;
}

} // namespace

bool RgfxStageReflectionOf(id<MTLFunction> function, NSString *prefix, const std::vector<RgfxDescriptorBuffer> &buffers,
                           RgfxStageReflection &out, NSString *__strong *why, bool openCLMode)
{
	if (![(id)function isKindOfClass:[RgfxFunction class]]) {
		*why = @"a function no library of the device made";
		return false;
	}
	RgfxFunction *f = (RgfxFunction *)function;
	RgfxResource *library = [f rgfxLibrary];
	if (![library isKindOfClass:[RgfxLibrary class]]) {
		*why = @"a function of no library";
		return false;
	}
	std::shared_ptr<const RgfxAirEntries> module = [(RgfxLibrary *)library rgfxAirEntryNamed:[f rgfxEntryPointUTF8]];
	if (module == nullptr || module->entries.size() != 1) {
		*why = @"its function's AIR was not read";
		return false;
	}
	std::shared_ptr<const RgfxClBindings> cl = openCLMode ? [f rgfxClBindings] : nullptr;
	const RgfxAirEntry entry = cl != nullptr ? clModeEntry(module->entries[0], *cl) : module->entries[0];
	if (!RgfxStageReflectionOfEntry(*module, entry, [f rgfxConstants], [f rgfxConstantCount], prefix, buffers, out,
	                                why, rgfxAppleGPU(f.device)))
		return false;
	out.pluginReturnData = RgfxGLPluginReturnData(function);
	return true;
}

id RgfxComputePipelineReflectionOf(id<MTLFunction> kernel, MTLStageInputOutputDescriptor *stageInput,
                                   MTLPipelineOption options)
{
	if (!RgfxPipelineOptionsReflect(options) || kernel == nil)
		return nil;
	RgfxStageReflection k;
	NSString *why = nil;
	id r = RgfxStageReflectionOf(kernel, @"stageInBuffer", RgfxStageInputBuffers(stageInput), k, &why, true)
	           ? RgfxComputePipelineReflectionMake(k, &why)
	           : nil;
	if (r == nil)
		logOnce([NSString stringWithFormat:@"compute pipeline of %@", kernel.name], why);
	return r;
}

id RgfxFunctionReflectionOf(id<MTLFunction> function)
{
	RgfxStageReflection stage;
	NSString *why = nil;
	id r = RgfxStageReflectionOf(function, @"stageInBuffer", {}, stage, &why) ? RgfxFunctionReflectionMake(stage, &why)
	                                                                          : nil;
	if (r == nil)
		logOnce([NSString stringWithFormat:@"function %@", function.name], why);
	return r;
}

// The stage input attributes Metal's own function declares, where `f`'s library was given Metal's
// own library of the compile, link or stitching that made it as its shadow: each attribute's name,
// index, type, use and patch flags, for a vertex function from vertexAttributes and for a kernel
// from stageInputAttributes. The function is Metal's own of the same name and constant values
// (rgfxShadowOrMake), so `specialised` is set where Metal specialised them; where `f` does not hold
// the values it was specialised by, Metal's own unspecialised function, for its AIR to specialise.
// None where its library was given no shadow or the shadow holds no such function.
// measured: on x86-64 macOS 11.7.11 the metadata of oracle_fnreflect's library compiled from
// source declares no function's attributes (the bundle reads no VATT or VATY tag of it), and
// Metal's own unspecialised function there leaves out an attribute under a function constant,
// where on 26.7.2 the metadata declares them and Metal's function answers that one too.
static void shadowStageAttributes(RgfxFunction *f, std::vector<RgfxStageAttributeDecl> &out, bool &specialised)
{
	RgfxResource *library = [f rgfxLibrary];
	if (![library isKindOfClass:[RgfxLibrary class]] || [(RgfxLibrary *)library rgfxAdoptedShadow] == nil)
		return;
	id<MTLFunction> genuine = [f rgfxShadowOrMake];
	specialised = genuine != nil && [f rgfxConstantCount] != 0;
	if (genuine == nil && f.name != nil)
		genuine = [[(RgfxLibrary *)library rgfxAdoptedShadow] newFunctionWithName:f.name];
	NSArray *list = f.functionType == MTLFunctionTypeVertex ? (NSArray *)genuine.vertexAttributes
	                                                        : (NSArray *)genuine.stageInputAttributes;
	for (MTLAttribute *a in list) {
		RgfxStageAttributeDecl d;
		d.name = a.name.UTF8String ?: "";
		d.index = static_cast<uint32_t>(a.attributeIndex);
		d.type = static_cast<uint32_t>(a.attributeType);
		d.active = a.active;
		d.patchControlPoint = a.patchControlPointData;
		d.patchData = a.patchData;
		out.push_back(std::move(d));
	}
}

NSArray *RgfxStageAttributesOf(id<MTLFunction> function)
{
	if (![(id)function isKindOfClass:[RgfxFunction class]])
		return nil;
	RgfxFunction *f = (RgfxFunction *)function;
	std::vector<RgfxStageAttributeDecl> attributes;
	NSString *why = nil;
	const bool read = [f rgfxDeclaredStageAttributes:attributes];
	// Where the metadata declares none, Metal's own function's, where its library has Metal's own
	// library (a function of no stage input declares none there either).
	bool specialised = false;
	if (read && attributes.empty())
		shadowStageAttributes(f, attributes, specialised);
	if (!read) {
		why = @"its library's metadata of it was not read";
	} else if (attributes.empty()) {
		return nil;
	} else if ([f rgfxConstantCount] != 0 && !specialised) {
		RgfxResource *library = [f rgfxLibrary];
		std::shared_ptr<const RgfxAirEntries> module =
		    [library isKindOfClass:[RgfxLibrary class]]
		        ? [(RgfxLibrary *)library rgfxAirEntryNamed:[f rgfxEntryPointUTF8]]
		        : nullptr;
		if (module == nullptr || module->entries.size() != 1)
			why = @"its function's AIR was not read";
		else
			RgfxSpecialiseStageAttributes(*module, module->entries[0], [f rgfxConstants], [f rgfxConstantCount],
			                              attributes, &why);
	}
	NSArray *list = why == nil ? RgfxStageAttributesMake(attributes, f.functionType == MTLFunctionTypeVertex, &why) : nil;
	if (list == nil)
		logOnce([NSString stringWithFormat:@"stage input attributes of %@", f.name], why);
	return list;
}

id RgfxRenderPipelineReflectionOf(id<MTLFunction> vertex, MTLVertexDescriptor *vertexDescriptor,
                                  id<MTLFunction> fragment, MTLPipelineOption options)
{
	if (!RgfxPipelineOptionsReflect(options) || vertex == nil)
		return nil;
	RgfxStageReflection v, f;
	NSString *why = nil;
	bool ok = RgfxStageReflectionOf(vertex, @"vertexBuffer", RgfxVertexDescriptorBuffers(vertexDescriptor), v, &why);
	ok = ok && (fragment == nil || RgfxStageReflectionOf(fragment, @"fragmentBuffer", {}, f, &why));
	id r = ok ? RgfxRenderPipelineReflectionMake(&v, fragment != nil ? &f : nullptr, nullptr, nullptr, &why) : nil;
	if (r == nil)
		logOnce([NSString stringWithFormat:@"render pipeline of %@", vertex.name], why);
	return r;
}

id RgfxMeshPipelineReflectionOf(id<MTLFunction> object, id<MTLFunction> mesh, id<MTLFunction> fragment,
                                MTLPipelineOption options)
{
	if (!RgfxPipelineOptionsReflect(options) || mesh == nil)
		return nil;
	RgfxStageReflection o, ms, f;
	NSString *why = nil;
	bool ok = object == nil || RgfxStageReflectionOf(object, @"objectBuffer", {}, o, &why);
	ok = ok && RgfxStageReflectionOf(mesh, @"meshBuffer", {}, ms, &why);
	ok = ok && (fragment == nil || RgfxStageReflectionOf(fragment, @"fragmentBuffer", {}, f, &why));
	id r = ok ? RgfxRenderPipelineReflectionMake(nullptr, fragment != nil ? &f : nullptr, object != nil ? &o : nullptr,
	                                             &ms, &why)
	          : nil;
	if (r == nil)
		logOnce([NSString stringWithFormat:@"mesh pipeline of %@", mesh.name], why);
	return r;
}
