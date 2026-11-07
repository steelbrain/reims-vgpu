// RgfxPipelineReflection: the reflection Metal answers for a render, mesh or compute pipeline (the
// `reflection:` out-parameter of the pipeline creators, and their completion handlers' second
// argument), built in the guest from the AIR metadata of the pipeline's functions
// (RgfxAirArguments.hpp, rgfxReadAirEntries) and the pipeline's descriptor.
//
// Metal's own device makes it with its compiler, from what the compiler reports of each function
// it compiled; ours has none, and the host sends none (the wire's RgfxPipelineReflection carries
// the pipeline state's own limits). The answers are an x86 Mac's Metal's (measured: macOS 26.7.1,
// an AMD Radeon Pro 5300M and an Intel UHD Graphics 630 alike, over
// tools/oracle/oracle_reflection.metal: tools/oracle/oracle_reflection.m prints them, and
// tools/pipeline_reflection_check.mm compares ours with the device's own, getter by getter and
// instance variable by instance variable):
//   - options of neither MTLPipelineOptionBindingInfo nor MTLPipelineOptionBufferTypeInfo answer
//     no reflection (nil); either answers the whole of it;
//   - each stage's bindings are its function's arguments of a buffer, an argument buffer,
//     threadgroup memory, a texture, a sampler, an object's payload, a function table or an
//     acceleration structure, in the order the function declares them, after the buffers its
//     vertex or stage input descriptor reads; its built-in arguments (thread_position_in_grid,
//     vertex_id, a mesh, ...) are listed apart; a fragment function's, and the colour attachments
//     it reads, are not listed; an empty list is none, which `-...Bindings` answers as an empty
//     array and `-...Arguments` as nil;
//   - an argument buffer's members are its textures, buffers, samplers, constants and pointers to
//     argument buffers of their own, metal::arrays of textures and samplers, C arrays of any of
//     those, and structs of members and C arrays of them, as RgfxPipelineReflection.mm's
//     argumentMember says;
//   - the function is specialised by its constants' values first (RgfxAirArguments.hpp, "An entry
//     point's specialisation"): an argument its predicate leaves out (`[[function_constant(k)]]`,
//     of a constant or of an expression of them) is left out, one at an index or of a count
//     constants compute is at that index and of that count, and one the AIR marks
//     `air.arg_unused`, or the specialised code does not use, is not used (isUsed and isActive
//     0);
//   - a constant buffer's alignment is 32, a device or threadgroup buffer's its pointee's;
//   - a descriptor's buffer is named `vertexBuffer.N` (`stageInBuffer.N` for a compute
//     pipeline's), N its place among the descriptor's buffers in buffer index order (buffers 1, 4
//     and 30 are `vertexBuffer.0`, `.1` and `.2`), at its buffer index, of its layout's stride,
//     alignment 4 and data type struct,
//     of no struct type, used where the function reads an attribute it holds, and no argument
//     (isArgument 0) where every other binding is one;
//   - a vertex function's `postVertexDumpStride` is its outputs' size and 12 (the AMD GPU's; the
//     Intel GPU's is 8 more than the outputs), a mesh pipeline's 0.
// The objects are Metal's own reflection classes, made through their initialisers, with the
// instance variables those leave unset set by name, each class, initialiser and instance variable
// looked up at run time: what is missing makes no reflection.
//
// What is left out, logged once per reason, answering nil (no reflection, as the device did
// before):
//   - a function no library of ours made, or whose library kept no bytes (Core Image's link,
//     stitching), or whose module the reader does not follow;
//   - an argument at a binding index or of a count function constants compute, or under a
//     predicate, where the function has no value of a constant it needs, or the module's code is
//     not one the reader follows;
//   - a binding, member, built-in or type of a kind no x86 Mac's answer was measured for, an
//     imageblock among them (neither x86 Mac makes a pipeline of one).
// Where the module's code is not one the reader follows, an argument the AIR marks used whose only
// reads are in code the function's constants remove is answered used, where Metal answers it
// unused (it reflects the specialised code).
#ifndef RGFX_PIPELINE_REFLECTION_H_
#define RGFX_PIPELINE_REFLECTION_H_

#import <Metal/Metal.h>

#ifdef __cplusplus
#include <cstdint>
#include <vector>

#include "RgfxAirArguments.hpp"
#include "reims_graphics_wire.h" // RgfxFunctionConstant

NS_ASSUME_NONNULL_BEGIN

// Whether a pipeline made with `options` answers a reflection.
static inline bool RgfxPipelineOptionsReflect(MTLPipelineOption options)
{
	return (options & (MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo)) != 0;
}

// One buffer a vertex or stage input descriptor's attributes read: its index, its layout's stride
// and the attributes that read it.
struct RgfxDescriptorBuffer {
	uint32_t index = 0;
	uint64_t stride = 0;
	std::vector<uint32_t> attributes;
};

// The buffers a vertex or stage input descriptor's attributes of a format read, in buffer index
// order.
std::vector<RgfxDescriptorBuffer> RgfxVertexDescriptorBuffers(MTLVertexDescriptor *_Nullable descriptor);
std::vector<RgfxDescriptorBuffer> RgfxStageInputBuffers(MTLStageInputOutputDescriptor *_Nullable descriptor);

// One stage's answers.
struct RgfxStageReflection {
	NSArray *_Nullable bindings = nil; // nil where there are none
	NSArray *_Nullable builtIns = nil; // nil where there are none
	uint64_t outputBytes = 0;          // a vertex function's outputs' size
	NSData *_Nullable pluginReturnData = nil; // a GL shader variant's (RgfxGLPluginReturnData)
};

// The answers of the entry point `entry` of the module `module` (rgfxReadAirEntries), specialised by
// the `count` function constant values at `values`, the descriptor's `buffers` (in buffer index
// order) first, named `prefix.0` on: false where they are not made (the file's comment), with
// `*why` saying why.
bool RgfxStageReflectionOfEntry(const RgfxAirEntries &module, const RgfxAirEntry &entry,
                                const RgfxFunctionConstant *_Nullable values, uint32_t count, NSString *prefix,
                                const std::vector<RgfxDescriptorBuffer> &buffers, RgfxStageReflection &out,
                                NSString *__strong _Nullable *_Nonnull why);

// As RgfxStageReflectionOfEntry, of `function`, one of our libraries' (RgfxFunction). This and the
// creators' three below are RgfxPipelineReflectionOf.mm's.
bool RgfxStageReflectionOf(id<MTLFunction> function, NSString *prefix, const std::vector<RgfxDescriptorBuffer> &buffers,
                           RgfxStageReflection &out, NSString *__strong _Nullable *_Nonnull why);

// Metal's compute pipeline reflection of a kernel's answers; nil, with `*why`, where Metal lacks a
// class or instance variable it needs.
id _Nullable RgfxComputePipelineReflectionMake(const RgfxStageReflection &kernel, NSString *__strong _Nullable *_Nonnull why);

// Metal's reflection of a function by itself (MTLFunctionReflectionInternal, which
// -[_MTLFunction reflectionWithOptions:] answers): its arguments and bindings, the same objects, as
// a pipeline's of the function lists them (measured: macOS 26.7.1, the AMD 5300M and the Intel 630
// alike, OpenCL C and Metal kernels). nil, with `*why`, as RgfxComputePipelineReflectionMake.
id _Nullable RgfxFunctionReflectionMake(const RgfxStageReflection &stage, NSString *__strong _Nullable *_Nonnull why);

// Metal's render pipeline reflection of its stages' answers, null for a stage the pipeline does not
// have: a vertex pipeline's (`vertex`, `fragment`), or a mesh pipeline's (`object`, `mesh`,
// `fragment`, with `mesh` not null). nil, with `*why`, as RgfxComputePipelineReflectionMake.
id _Nullable RgfxRenderPipelineReflectionMake(const RgfxStageReflection *_Nullable vertex,
                                              const RgfxStageReflection *_Nullable fragment,
                                              const RgfxStageReflection *_Nullable object,
                                              const RgfxStageReflection *_Nullable mesh,
                                              NSString *__strong _Nullable *_Nonnull why);

// The reflection the creators answer for a pipeline of these functions and descriptor, made with
// `options`: nil where the options ask for none, and nil, logged once per reason, where it is not
// made.
id _Nullable RgfxComputePipelineReflectionOf(id<MTLFunction> _Nullable kernel,
                                             MTLStageInputOutputDescriptor *_Nullable stageInput,
                                             MTLPipelineOption options);
id _Nullable RgfxRenderPipelineReflectionOf(id<MTLFunction> _Nullable vertex,
                                            MTLVertexDescriptor *_Nullable vertexDescriptor,
                                            id<MTLFunction> _Nullable fragment, MTLPipelineOption options);
// The reflection a function answers by itself (RgfxFunctionReflectionMake): nil, logged once per
// reason, where it is not made.
id _Nullable RgfxFunctionReflectionOf(id<MTLFunction> function);
id _Nullable RgfxMeshPipelineReflectionOf(id<MTLFunction> _Nullable object, id<MTLFunction> _Nullable mesh,
                                          id<MTLFunction> _Nullable fragment, MTLPipelineOption options);

NS_ASSUME_NONNULL_END
#endif

#endif // RGFX_PIPELINE_REFLECTION_H_
