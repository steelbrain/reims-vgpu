/*
 * RgfxArgumentEncoder: see RgfxArgumentEncoder.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxArgumentEncoder.h"

#import "RgfxFunction.h"
#import "RgfxUnsupported.h"

#include <cstring>
#include <mutex>
#include <vector>

#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {

/// The first block the host's answer is asked into: a layout of 80 members.
constexpr uint64_t kFirstReplyBlock = 4096;
/// The largest answer asked for: a layout of about 87,000 members.
constexpr uint64_t kMaxReplyBlock = 4u << 20;

/// The host's layout of one query, posted by `post` into a block of `region` (`post` answers the
/// device's status): asked again, once, into a block of the size the host says it needs where
/// the first is too small. Null where no layout comes back.
template <typename Post>
std::shared_ptr<const RgfxArgLayout> hostLayout(RgfxRegion *region, Post post, const char **why)
{
	uint64_t size = kFirstReplyBlock;
	for (int attempt = 0; attempt < 2; attempt++) {
		RgfxRegion::Block block = region->allocate(size);
		if (!block.ok) {
			*why = "no region block for the answer";
			return nullptr;
		}
		const uint32_t status = post(block.offset, size);
		std::vector<uint8_t> bytes;
		if (status == RGFX_STATUS_OK || status == RGFX_STATUS_TOO_SMALL) {
			const uint64_t n = status == RGFX_STATUS_OK ? size : sizeof(uint64_t);
			bytes.resize(static_cast<size_t>(n));
			for (uint64_t i = 0; i < n; i++)
				bytes[static_cast<size_t>(i)] = block.ptr[i];
		}
		region->release(block.offset);
		if (status == RGFX_STATUS_OK) {
			auto layout = RgfxArgLayout::parse(bytes.data(), bytes.size());
			if (!layout)
				*why = "the device's answer is not whole";
			return layout;
		}
		if (status != RGFX_STATUS_TOO_SMALL) {
			*why = status == kRgfxNoAnswer ? "no answer came back" : "the device refused it";
			return nullptr;
		}
		uint64_t needed = 0;
		std::memcpy(&needed, bytes.data(), sizeof(needed));
		if (needed <= size || needed > kMaxReplyBlock) {
			*why = "the device's answer does not fit any block";
			return nullptr;
		}
		size = (needed + 4095) & ~uint64_t(4095);
	}
	*why = "the device's answer does not fit the block it asked for";
	return nullptr;
}

/// The wire's descriptor of `d`.
RgfxArgumentDescriptor wireDescriptor(MTLArgumentDescriptor *d)
{
	RgfxArgumentDescriptor w = {};
	w.data_type = static_cast<uint32_t>(d.dataType);
	w.index = static_cast<uint32_t>(d.index);
	w.array_length = static_cast<uint32_t>(d.arrayLength);
	w.access = static_cast<uint32_t>(d.access);
	w.texture_type = static_cast<uint32_t>(d.textureType);
	w.constant_block_alignment = static_cast<uint32_t>(d.constantBlockAlignment);
	return w;
}

uint64_t resourceId(MTLResourceID id)
{
	return id._impl;
}

/// The functions the device's reflections of buffer bindings were noted for, by binding, each
/// held while its binding lives.
std::mutex g_bindingsLock;
NSMapTable<id, RgfxFunction *> *g_bindings;

RgfxFunction *notedFunction(id binding)
{
	std::lock_guard<std::mutex> hold(g_bindingsLock);
	return g_bindings != nil ? [g_bindings objectForKey:binding] : nil;
}

} // namespace

void RgfxNoteArgumentBinding(id<MTLBinding> binding, RgfxFunction *function)
{
	if (binding == nil || function == nil)
		return;
	std::lock_guard<std::mutex> hold(g_bindingsLock);
	if (g_bindings == nil)
		g_bindings = [NSMapTable weakToStrongObjectsMapTable];
	[g_bindings setObject:function forKey:binding];
}

NSArray<MTLArgumentDescriptor *> *RgfxDescriptorsOfArguments(MTLStructType *type)
{
	NSMutableArray<MTLArgumentDescriptor *> *out = [NSMutableArray array];
	for (MTLStructMember *m in type.members) {
		MTLArgumentDescriptor *d = [MTLArgumentDescriptor argumentDescriptor];
		d.index = m.argumentIndex;
		MTLDataType t = m.dataType;
		MTLTextureReferenceType *texture = m.textureReferenceType;
		MTLPointerType *pointer = m.pointerType;
		if (t == MTLDataTypeArray) {
			MTLArrayType *a = m.arrayType;
			d.arrayLength = a.arrayLength;
			t = a.elementType;
			texture = a.elementTextureReferenceType;
			pointer = a.elementPointerType;
		}
		if (t == MTLDataTypeStruct || t == MTLDataTypeArray)
			return nil;
		d.dataType = t;
		if (t == MTLDataTypeTexture && texture != nil) {
			d.textureType = texture.textureType;
			d.access = texture.access;
		} else if (t == MTLDataTypePointer && pointer != nil) {
			d.access = pointer.access;
		}
		[out addObject:d];
	}
	return out;
}

@implementation RgfxArgumentEncoder {
	std::shared_ptr<const RgfxArgLayout> _layout;
	id<MTLBuffer> _argumentBuffer; // kept while set
	uint64_t _offset;              // the arguments' offset in it
}

- (instancetype)initWithLayout:(std::shared_ptr<const RgfxArgLayout>)layout
                        device:(id<MTLDevice>)device
                    serializer:(RgfxSerializer *)serializer
{
	// No host handle: the layout is all the host gives an encoder (RGFX_OBJ_NONE).
	self = [super initWithObjId:RGFX_OBJ_NONE device:device serializer:serializer];
	if (self) {
		_layout = layout ? std::move(layout) : std::make_shared<RgfxArgLayout>();
	}
	return self;
}

+ (instancetype)encoderOfArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                            region:(RgfxRegion *)region
                            device:(id<MTLDevice>)device
                        serializer:(RgfxSerializer *)serializer
{
	if (region == nullptr || serializer == nullptr || arguments.count > UINT32_MAX) {
		NSLog(@"RGFX: newArgumentEncoderWithArguments: no encoder: no device region");
		return nil;
	}
	std::vector<RgfxArgumentDescriptor> wire;
	wire.reserve(arguments.count);
	for (MTLArgumentDescriptor *d in arguments)
		wire.push_back(wireDescriptor(d));
	const char *why = "";
	auto layout = hostLayout(
	    region,
	    [&](uint64_t offset, uint64_t length) {
		    return serializer
		    ->argumentLayoutOfDescriptors(wire.data(), static_cast<uint32_t>(wire.size()), offset,
		                                  length);
	    },
	    &why);
	if (!layout) {
		NSLog(@"RGFX: newArgumentEncoderWithArguments: (%lu arguments): no encoder: %s",
		      static_cast<unsigned long>(arguments.count), why);
		return nil;
	}
	return [[self alloc] initWithLayout:layout device:device serializer:serializer];
}

+ (instancetype)encoderOfFunction:(RgfxFunction *)function
                      bufferIndex:(NSUInteger)bufferIndex
                           region:(RgfxRegion *)region
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
{
	if (region == nullptr || serializer == nullptr || bufferIndex > UINT32_MAX ||
	    function.rgfxLibraryObjId == RGFX_OBJ_NONE) {
		NSLog(@"RGFX: newArgumentEncoderWithBufferIndex:%lu: no encoder: no device region or "
		      @"library",
		      static_cast<unsigned long>(bufferIndex));
		return nil;
	}
	const char *why = "";
	auto layout = hostLayout(
	    region,
	    [&](uint64_t offset, uint64_t length) {
		    return serializer
		    ->argumentLayoutOfFunction(function.rgfxLibraryObjId, function.rgfxEntryPointUTF8,
		                               function.rgfxConstants, function.rgfxConstantCount,
		                               static_cast<uint32_t>(bufferIndex), offset, length);
	    },
	    &why);
	if (!layout) {
		NSLog(@"RGFX: newArgumentEncoderWithBufferIndex:%lu of %s: no encoder: %s",
		      static_cast<unsigned long>(bufferIndex), function.rgfxEntryPointUTF8, why);
		return nil;
	}
	return [[self alloc] initWithLayout:layout device:device serializer:serializer];
}

+ (instancetype)encoderOfBinding:(id<MTLBufferBinding>)binding
                          region:(RgfxRegion *)region
                          device:(id<MTLDevice>)device
                      serializer:(RgfxSerializer *)serializer
{
	if (binding == nil || binding.type != MTLBindingTypeBuffer ||
	    ![(id)binding conformsToProtocol:@protocol(MTLBufferBinding)] ||
	    binding.bufferDataType != MTLDataTypeStruct || !binding.bufferPointerType.elementIsArgumentBuffer)
		return nil; // no argument buffer: the AMD 5300M answers nil (oracle_bundle argbuf.binding.plain)
	if (RgfxFunction *function = notedFunction(binding))
		return [self encoderOfFunction:function
		                   bufferIndex:binding.index
		                        region:region
		                        device:device
		                    serializer:serializer];
	NSArray<MTLArgumentDescriptor *> *arguments = RgfxDescriptorsOfArguments(binding.bufferStructType);
	if (arguments == nil) {
		NSLog(@"RGFX: newArgumentEncoderWithBufferBinding: %@: no encoder: a member is a struct, "
		      @"which no argument descriptor names",
		      binding.name);
		return nil;
	}
	return [self encoderOfArguments:arguments region:region device:device serializer:serializer];
}

- (std::shared_ptr<const RgfxArgLayout>)rgfxLayout
{
	return _layout;
}

- (NSUInteger)encodedLength
{
	return static_cast<NSUInteger>(_layout->encodedLength);
}

- (NSUInteger)alignment
{
	return static_cast<NSUInteger>(_layout->alignment);
}

// --- the argument buffer -----------------------------------------------------------

- (void)setArgumentBuffer:(id<MTLBuffer>)argumentBuffer offset:(NSUInteger)offset
{
	_argumentBuffer = argumentBuffer;
	_offset = offset;
}

- (void)setArgumentBuffer:(id<MTLBuffer>)argumentBuffer
              startOffset:(NSUInteger)startOffset
             arrayElement:(NSUInteger)arrayElement
{
	_argumentBuffer = argumentBuffer;
	_offset = uint64_t(startOffset) + uint64_t(arrayElement) * _layout->encodedLength;
}

/// Where argument `index` lies in the argument buffer's contents, `size` bytes of it inside the
/// buffer: its place, or the arguments' offset 0 for an index no argument has. Null, logged
/// against the guest's selector `cmd`, where no argument buffer is set (plain Metal's process
/// ends there), it has no contents (Private), or the bytes pass its end.
- (uint8_t *)rgfxAt:(NSUInteger)index size:(uint64_t)size cmd:(SEL)cmd
{
	if (_argumentBuffer == nil) {
		NSLog(@"RGFX: -[MTLArgumentEncoder %s]: no argument buffer set; nothing written",
		      sel_getName(cmd));
		return nullptr;
	}
	uint8_t *contents = static_cast<uint8_t *>(_argumentBuffer.contents);
	RgfxArgPlace place = {};
	const uint64_t at = index <= UINT32_MAX && _layout->place(static_cast<uint32_t>(index), &place)
	                        ? place.offset
	                        : 0;
	const uint64_t start = _offset + at;
	if (contents == nullptr || start < _offset || start + size < start ||
	    start + size > _argumentBuffer.length) {
		NSLog(@"RGFX: -[MTLArgumentEncoder %s]: index %lu lies past the argument buffer, or "
		      @"in one the CPU cannot write; nothing written",
		      sel_getName(cmd), static_cast<unsigned long>(index));
		return nullptr;
	}
	return contents + start;
}

- (void)rgfxWord:(uint64_t)word atIndex:(NSUInteger)index cmd:(SEL)cmd
{
	if (uint8_t *at = [self rgfxAt:index size:sizeof(word) cmd:cmd])
		std::memcpy(at, &word, sizeof(word));
}

// --- the arguments -------------------------------------------------------------------

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	[self rgfxWord:buffer != nil ? buffer.gpuAddress + offset : 0 atIndex:index cmd:_cmd];
}

- (void)setBuffers:(const id<MTLBuffer> __nullable[__nonnull])buffers
           offsets:(const NSUInteger[__nonnull])offsets
         withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}

- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	[self rgfxWord:texture != nil ? resourceId(texture.gpuResourceID) : 0 atIndex:index cmd:_cmd];
}

- (void)setTextures:(const id<MTLTexture> __nullable[__nonnull])textures withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setTexture:textures[i] atIndex:range.location + i];
}

- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	[self rgfxWord:sampler != nil ? resourceId(sampler.gpuResourceID) : 0 atIndex:index cmd:_cmd];
}

- (void)setSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
               withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setSamplerState:samplers[i] atIndex:range.location + i];
}

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipeline atIndex:(NSUInteger)index
{
	[self rgfxWord:pipeline != nil ? resourceId(pipeline.gpuResourceID) : 0 atIndex:index cmd:_cmd];
}

- (void)setRenderPipelineStates:(const id<MTLRenderPipelineState> __nullable[__nonnull])pipelines
                      withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setRenderPipelineState:pipelines[i] atIndex:range.location + i];
}

- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline atIndex:(NSUInteger)index
{
	[self rgfxWord:pipeline != nil ? resourceId(pipeline.gpuResourceID) : 0 atIndex:index cmd:_cmd];
}

- (void)setComputePipelineStates:(const id<MTLComputePipelineState> __nullable[__nonnull])pipelines
                       withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setComputePipelineState:pipelines[i] atIndex:range.location + i];
}

// A depth/stencil state writes its gpuResourceID, which raises an unrecognized selector, as the
// Intel 630's encoder does under macOS 26.7.1 (oracle_bundle argbuf.ds.set and .mixed): no x86
// Mac holds a state in an argument buffer, and the AMD 5300M ends the process earlier, at an
// encoder of one (argbuf.ds.encoder). nil writes nothing.
- (void)setDepthStencilState:(id<MTLDepthStencilState>)depthStencilState atIndex:(NSUInteger)index
{
	if (depthStencilState != nil)
		[self rgfxWord:resourceId(depthStencilState.gpuResourceID) atIndex:index cmd:_cmd];
}

- (void)setDepthStencilStates:(const id<MTLDepthStencilState> __nullable[__nonnull])depthStencilStates
                    withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setDepthStencilState:depthStencilStates[i] atIndex:range.location + i];
}

// The buffer's gpuResourceID, the id the host's device numbered it by, as a pipeline's; 0 for nil.
- (void)setIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                         atIndex:(NSUInteger)index
{
	[self rgfxWord:indirectCommandBuffer != nil ? resourceId(indirectCommandBuffer.gpuResourceID) : 0
	       atIndex:index
	           cmd:_cmd];
}

- (void)setIndirectCommandBuffers:(const id<MTLIndirectCommandBuffer> __nullable[__nonnull])buffers
                        withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setIndirectCommandBuffer:buffers[i] atIndex:range.location + i];
}

// The device makes no acceleration structure or function table: only nil reaches these, which
// writes 0.
- (void)setAccelerationStructure:(id<MTLAccelerationStructure>)accelerationStructure
                         atIndex:(NSUInteger)index
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
	[self rgfxWord:0 atIndex:index cmd:_cmd];
}

- (void)setIntersectionFunctionTable:(id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                             atIndex:(NSUInteger)index
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
	[self rgfxWord:0 atIndex:index cmd:_cmd];
}

- (void)setIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                            withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setIntersectionFunctionTable:intersectionFunctionTables[i]
		                           atIndex:range.location + i];
}

- (void)setVisibleFunctionTable:(id<MTLVisibleFunctionTable>)visibleFunctionTable
                        atIndex:(NSUInteger)index
{
	if (visibleFunctionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
	[self rgfxWord:0 atIndex:index cmd:_cmd];
}

- (void)setVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])visibleFunctionTables
                       withRange:(NSRange)range
{
	for (NSUInteger i = 0; i < range.length; i++)
		[self setVisibleFunctionTable:visibleFunctionTables[i] atIndex:range.location + i];
}

// Metal's encoder answers these names too, and writes with them the bytes its atIndex: and
// withRange: names write (measured: macOS 27, M3 Max).
- (void)setIntersectionFunctionTable:(id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                       atBufferIndex:(NSUInteger)index
{
	[self setIntersectionFunctionTable:intersectionFunctionTable atIndex:index];
}

- (void)setIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                      withBufferRange:(NSRange)range
{
	[self setIntersectionFunctionTables:intersectionFunctionTables withRange:range];
}

- (void)setVisibleFunctionTable:(id<MTLVisibleFunctionTable>)visibleFunctionTable
                  atBufferIndex:(NSUInteger)index
{
	[self setVisibleFunctionTable:visibleFunctionTable atIndex:index];
}

- (void)setVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])visibleFunctionTables
                 withBufferRange:(NSRange)range
{
	[self setVisibleFunctionTables:visibleFunctionTables withRange:range];
}

// --- constants and nested argument buffers --------------------------------------------

- (void *)constantDataAtIndex:(NSUInteger)index
{
	if (_argumentBuffer == nil || _argumentBuffer.contents == nullptr)
		return nullptr; // as plain Metal answers with no argument buffer set
	RgfxArgPlace place = {};
	const uint64_t at = index <= UINT32_MAX && _layout->place(static_cast<uint32_t>(index), &place)
	                        ? place.offset
	                        : 0;
	return static_cast<uint8_t *>(_argumentBuffer.contents) + _offset + at;
}

- (id<MTLArgumentEncoder>)newArgumentEncoderForBufferAtIndex:(NSUInteger)index
{
	RgfxArgPlace place = {};
	if (index > UINT32_MAX || !_layout->place(static_cast<uint32_t>(index), &place) ||
	    place.kind != RGFX_ARGUMENT_BUFFER || !place.nested)
		return nil; // no argument buffer at that index, as Metal answers
	return [[RgfxArgumentEncoder alloc] initWithLayout:place.nested
	                                            device:self.device
	                                        serializer:[self serializer]];
}

@end
