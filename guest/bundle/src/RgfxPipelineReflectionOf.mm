// RgfxPipelineReflectionOf: the pipeline reflection of our own functions (RgfxPipelineReflection.h,
// RgfxStageReflectionOf and the creators' RgfxComputePipelineReflectionOf,
// RgfxRenderPipelineReflectionOf and RgfxMeshPipelineReflectionOf): the entry point's nodes come
// from its library (RgfxLibrary, rgfxAirEntryNamed:), its constants' values from the function.
// Apart from RgfxPipelineReflection.mm, which tools/pipeline_reflection_check.mm builds alone.
#import "RgfxPipelineReflection.h"

#import "RgfxFunction.h"
#import "RgfxGLFunction.h"
#import "RgfxLibrary.h"

#include <mutex>
#include <set>
#include <string>

namespace {

// Logs `why` for a pipeline of `what`, once per reason.
void logOnce(NSString *what, NSString *why)
{
	static std::mutex lock;
	static std::set<std::string> said;
	std::lock_guard<std::mutex> hold(lock);
	if (said.insert(why.UTF8String).second)
		NSLog(@"RGFX: %@: no reflection (%@); answered nil", what, why);
}

// A kernel's entry as its pipeline of OpenCL mode binds it (RgfxClMode.hpp; measured: macOS
// 26.7.1, the Intel 630 and the AMD 5300M alike, through the GL renderer): each __global and
// __constant pointer at its index plus 3; each value passed by value a buffer at its index plus 3,
// read only and no constant buffer, of the value's type, size and alignment; each __local pointer
// the threadgroup memory at its own index; then the three buffers the renderer binds, after the
// kernel's own arguments.
RgfxAirEntry clModeEntry(const RgfxAirEntry &entry, const RgfxClBindings &cl)
{
	RgfxAirEntry out = entry;
	for (RgfxAirNode &n : out.arguments) {
		if (n.kind == "air.constant") {
			n.kind = "air.buffer";
			n.hasAddressSpace = true;
			n.addressSpace = 1;
			n.access = RgfxAirAccess::Read;
			n.index += kRgfxClHiddenBuffers;
		} else if (n.kind == "air.buffer" && n.hasAddressSpace && n.addressSpace != 3) {
			n.index += kRgfxClHiddenBuffers;
		}
	}
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
	                                why))
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
