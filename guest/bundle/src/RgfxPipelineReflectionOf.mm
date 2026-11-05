// RgfxPipelineReflectionOf: the pipeline reflection of our own functions (RgfxPipelineReflection.h,
// RgfxStageReflectionOf and the creators' RgfxComputePipelineReflectionOf,
// RgfxRenderPipelineReflectionOf and RgfxMeshPipelineReflectionOf): the entry point's nodes come
// from its library (RgfxLibrary, rgfxAirEntryNamed:), its constants' values from the function.
// Apart from RgfxPipelineReflection.mm, which tools/pipeline_reflection_check.mm builds alone.
#import "RgfxPipelineReflection.h"

#import "RgfxFunction.h"
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

} // namespace

bool RgfxStageReflectionOf(id<MTLFunction> function, NSString *prefix, const std::vector<RgfxDescriptorBuffer> &buffers,
                           RgfxStageReflection &out, NSString *__strong *why)
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
	return RgfxStageReflectionOfEntry(*module, module->entries[0], [f rgfxConstants], [f rgfxConstantCount], prefix,
	                                  buffers, out, why);
}

id RgfxComputePipelineReflectionOf(id<MTLFunction> kernel, MTLStageInputOutputDescriptor *stageInput,
                                   MTLPipelineOption options)
{
	if (!RgfxPipelineOptionsReflect(options) || kernel == nil)
		return nil;
	RgfxStageReflection k;
	NSString *why = nil;
	id r = RgfxStageReflectionOf(kernel, @"stageInBuffer", RgfxStageInputBuffers(stageInput), k, &why)
	           ? RgfxComputePipelineReflectionMake(k, &why)
	           : nil;
	if (r == nil)
		logOnce([NSString stringWithFormat:@"compute pipeline of %@", kernel.name], why);
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
