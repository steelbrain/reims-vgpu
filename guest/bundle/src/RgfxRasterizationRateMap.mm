/*
 * RgfxRasterizationRateMap: see RgfxRasterizationRateMap.h. Our own ObjC++.
 */

#import "RgfxRasterizationRateMap.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <objc/message.h>

#import "RgfxBuffer.h"
#include "RgfxLogOnce.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxLog.hpp"

namespace {

// Whether `d` answers the private mutability (MTLRasterizationRateMapDescriptor's SPI): Metal's
// descriptors are of its private subclass, MTLRasterizationRateMapDescriptorInternal, which
// alone declares it (macOS 26.7.1 and 26.7.2, guest/bundle/tools/oracle/oracle_ratemap).
bool answersMutability(MTLRasterizationRateMapDescriptor *d)
{
	return [d respondsToSelector:@selector(mutability)];
}

MTLSize sizeOf(uint64_t width, uint64_t height, uint64_t depth)
{
	return MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height),
	                   static_cast<NSUInteger>(depth));
}

float floatOf(uint64_t bits)
{
	const uint32_t low = static_cast<uint32_t>(bits);
	float value;
	std::memcpy(&value, &low, sizeof(value));
	return value;
}

// Ends the process at a call of `call` that the host refused where Metal's map ends the process:
// every Mac's at a copy into a Private buffer (SIGSEGV on the AMD Radeon Pro 5300M under macOS
// 26.7.1, guest/bundle/tools/oracle/oracle_ratemap; an assertion on the M3 Max and the M4, as the
// host's runtime records), and the Apple GPUs' at a coordinate map of a map of no layers or of an
// empty screen and at the physical-to-screen map of a Mutable map not yet reset, which the host's
// runtime answers as they do: the 5300M ends the process at a map of no layers already at its
// creation, and compiles a Mutable map at its creation, so no answer of its is the host's.
[[noreturn]] void endsTheProcess(const char *call)
{
	rgfxLogDeathLine(@"RGFX: -[RgfxRasterizationRateMap %s] ends the process, as Metal's map does", call);
	abort();
}

void noteNoAnswer(const char *call, uint32_t status)
{
	static char site;
	if (rgfxLogFirst(&site, rgfxLogKindOf(call) ^ status))
		rgfxLogLine(@"RGFX: rasterization rate map %s: the host answered nothing (status 0x%x)", call,
		            status);
}

} // namespace

RgfxRateMapShape RgfxRateMapShapeOf(MTLRasterizationRateMapDescriptor *d)
{
	RgfxRateMapShape shape;
	const MTLSize screen = d.screenSize;
	shape.screen_width = screen.width;
	shape.screen_height = screen.height;
	shape.screen_depth = screen.depth;
	if (answersMutability(d)) {
		shape.mutability_set = true;
		shape.mutability = static_cast<uint32_t>(
		    ((NSUInteger(*)(id, SEL))objc_msgSend)(d, @selector(mutability)));
	}
	// The layers Metal makes the map of: from index 0 up to the first without samples on an
	// axis, which its descriptor holds as nil (layerCount).
	const NSUInteger count = d.layerCount;
	for (NSUInteger i = 0; i < count; i++) {
		MTLRasterizationRateLayerDescriptor *layer = [d layerAtIndex:i];
		if (layer == nil)
			break;
		const MTLSize samples = layer.sampleCount;
		RgfxRateMapShape::Layer out;
		out.horizontal_count = static_cast<uint32_t>(samples.width);
		out.vertical_count = static_cast<uint32_t>(samples.height);
		const float *h = layer.horizontalSampleStorage;
		const float *v = layer.verticalSampleStorage;
		out.samples.reserve(samples.width + samples.height);
		for (NSUInteger k = 0; k < samples.width; k++)
			out.samples.push_back(h ? h[k] : 0.0f);
		for (NSUInteger k = 0; k < samples.height; k++)
			out.samples.push_back(v ? v[k] : 0.0f);
		shape.layers.push_back(std::move(out));
	}
	return shape;
}

float RgfxRateMapMinFactorOf(MTLRasterizationRateMapDescriptor *d)
{
	static const bool answers =
	    [MTLRasterizationRateMapDescriptor instancesRespondToSelector:@selector(minFactor)];
	if (!answers || ![d respondsToSelector:@selector(minFactor)])
		return 0.0f;
	return ((float (*)(id, SEL))objc_msgSend)(d, @selector(minFactor));
}

@implementation RgfxRasterizationRateMap {
	std::mutex _lock; // the state below, which a reset replaces
	RgfxRateMapState _state;
	MTLMutability _mutability;
	float _minFactor;
	id<MTLBuffer> _scratch; // the copy's Shared buffer for a buffer the host sees no contents of
}

- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)rateMapWithDescriptor:(MTLRasterizationRateMapDescriptor *)descriptor
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil)
		return nil;
	const RgfxRateMapShape shape = RgfxRateMapShapeOf(descriptor);
	RgfxRateMapState state = {};
	RgfxCreated created = serializer->createRateMap(shape, &state);
	if (serializer->refused(created) || created.status != RGFX_STATUS_OK) {
		static char site;
		if (rgfxLogFirst(&site, created.status))
			rgfxLogLine(@"RGFX: newRasterizationRateMapWithDescriptor: no map (status 0x%x)",
			            created.status);
		return nil;
	}
	RgfxRasterizationRateMap *map = [[RgfxRasterizationRateMap alloc] initWithObjId:created.obj_id
	                                                                         device:device
	                                                                     serializer:serializer];
	if (map) {
		map->_state = state;
		map->_mutability = static_cast<MTLMutability>(state.mutability);
		map->_minFactor = RgfxRateMapMinFactorOf(descriptor);
		// The descriptor's label, nil for an empty one, as Metal's objects take theirs.
		if (descriptor.label.length != 0)
			map.label = descriptor.label;
	}
	return map;
}

- (RgfxRateMapState)rgfxState
{
	std::lock_guard<std::mutex> guard(_lock);
	return _state;
}

// --- MTLRasterizationRateMap ---------------------------------------------------------------

- (MTLSize)screenSize
{
	const RgfxRateMapState s = [self rgfxState];
	return sizeOf(s.screen_width, s.screen_height, s.screen_depth);
}

- (MTLSize)physicalGranularity
{
	const RgfxRateMapState s = [self rgfxState];
	return sizeOf(s.granularity_width, s.granularity_height, s.granularity_depth);
}

- (NSUInteger)layerCount
{
	return static_cast<NSUInteger>([self rgfxState].layer_count);
}

- (MTLSizeAndAlign)parameterBufferSizeAndAlign
{
	const RgfxRateMapState s = [self rgfxState];
	MTLSizeAndAlign out;
	out.size = static_cast<NSUInteger>(s.parameter_size);
	out.align = static_cast<NSUInteger>(s.parameter_align);
	return out;
}

- (MTLSize)physicalSizeForLayer:(NSUInteger)layerIndex
{
	uint64_t values[3];
	const uint32_t status = _serializer->queryRateMap(
	    _objId, RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE, layerIndex, RGFX_OBJ_NONE, 0, 0, values);
	if (status != RGFX_STATUS_OK)
		noteNoAnswer("physicalSizeForLayer:", status);
	return sizeOf(values[0], values[1], values[2]);
}

- (MTLCoordinate2D)rgfxMap:(uint32_t)query
                        at:(MTLCoordinate2D)coordinate
                     layer:(NSUInteger)layerIndex
                      call:(const char *)call
{
	uint64_t values[3];
	const uint32_t status = _serializer->queryRateMap(_objId, query, layerIndex, RGFX_OBJ_NONE,
	                                                  coordinate.x, coordinate.y, values);
	if (status == RGFX_STATUS_REFUSED)
		endsTheProcess(call);
	if (status != RGFX_STATUS_OK)
		noteNoAnswer(call, status);
	return MTLCoordinate2DMake(floatOf(values[0]), floatOf(values[1]));
}

- (MTLCoordinate2D)mapScreenToPhysicalCoordinates:(MTLCoordinate2D)screenCoordinates
                                         forLayer:(NSUInteger)layerIndex
{
	return [self rgfxMap:RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL
	                  at:screenCoordinates
	               layer:layerIndex
	                call:"mapScreenToPhysicalCoordinates:forLayer:"];
}

- (MTLCoordinate2D)mapPhysicalToScreenCoordinates:(MTLCoordinate2D)physicalCoordinates
                                         forLayer:(NSUInteger)layerIndex
{
	return [self rgfxMap:RGFX_RATE_MAP_QUERY_PHYSICAL_TO_SCREEN
	                  at:physicalCoordinates
	               layer:layerIndex
	                call:"mapPhysicalToScreenCoordinates:forLayer:"];
}

// The host writes the bytes in place where it sees the buffer's contents: a Shared buffer of
// ours with no copy of app memory. For any other the bytes go through the map's scratch Shared
// buffer, and from there to the contents, which a Managed buffer's app tells with didModifyRange:
// as on Metal, and which a buffer over app memory takes at its next use. A Private buffer, whose
// contents the CPU never reaches, ends the process, as every Mac's copy does (endsTheProcess); a
// copy past the buffer's end writes nothing, where Metal's writes past it.
- (void)copyParameterDataToBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset
{
	const uint64_t size = [self rgfxState].parameter_size;
	const uint64_t bufferId = RgfxResourceId(buffer);
	if (bufferId != 0 && buffer.storageMode == MTLStorageModePrivate)
		endsTheProcess("copyParameterDataToBuffer:offset:");
	const bool inPlace = bufferId != 0 && [buffer isKindOfClass:[RgfxBuffer class]] &&
	                     buffer.storageMode == MTLStorageModeShared &&
	                     [(RgfxBuffer *)buffer rgfxShadow] == nullptr;
	uint64_t values[3];
	if (inPlace || bufferId == 0) {
		const uint32_t status = _serializer->queryRateMap(
		    _objId, RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA, offset, bufferId, 0, 0, values);
		if (status != RGFX_STATUS_OK)
			noteNoAnswer("copyParameterDataToBuffer:offset:", status);
		return;
	}
	void *contents = buffer.contents;
	if (contents == nullptr || offset + size > buffer.length) {
		// A copy past the buffer's end, which the host refuses too
		// (RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA).
		noteNoAnswer("copyParameterDataToBuffer:offset:", RGFX_STATUS_REFUSED);
		return;
	}
	id<MTLBuffer> scratch = nil;
	{
		std::lock_guard<std::mutex> guard(_lock);
		if (_scratch == nil || _scratch.length < size)
			_scratch = [self.device newBufferWithLength:static_cast<NSUInteger>(size)
			                                    options:MTLResourceStorageModeShared];
		scratch = _scratch;
	}
	if (scratch == nil || RgfxResourceId(scratch) == 0) {
		noteNoAnswer("copyParameterDataToBuffer:offset:", kRgfxNoAnswer);
		return;
	}
	const uint32_t status = _serializer->queryRateMap(
	    _objId, RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA, 0, RgfxResourceId(scratch), 0, 0, values);
	if (status != RGFX_STATUS_OK) {
		noteNoAnswer("copyParameterDataToBuffer:offset:", status);
		return;
	}
	std::memcpy(static_cast<uint8_t *>(contents) + offset, scratch.contents,
	            static_cast<size_t>(size));
}

// --- MTLRasterizationRateMapSPI -------------------------------------------------------------

- (MTLMutability)mutability
{
	std::lock_guard<std::mutex> guard(_lock);
	return _mutability;
}

- (float)minFactor
{
	std::lock_guard<std::mutex> guard(_lock);
	return _minFactor;
}

// As Metal's base map words it (measured: macOS 26.7.1, the AMD Radeon Pro 5300M's map, which
// does not override it): the object's description, then its label, screen size, layer count,
// granularity and parameter size and alignment, each on a line of its own.
- (NSString *)formattedDescription:(NSUInteger)indent
{
	const RgfxRateMapState s = [self rgfxState];
	NSString *pad = [@"\n" stringByPaddingToLength:indent + 4 withString:@" " startingAtIndex:0];
	NSArray *fields = @[
		pad, @"label =", self.label ?: @"<none>", pad, @"screenSize.width =", @(s.screen_width), pad,
		@"screenSize.height =", @(s.screen_height), pad, @"layerCount =", @(s.layer_count), pad,
		@"physicalGranularity.width =", @(s.granularity_width), pad,
		@"physicalGranularity.height =", @(s.granularity_height), pad, @"parameterBuffer.size =",
		@(s.parameter_size), pad, @"parameterBuffer.align =", @(s.parameter_align)
	];
	return [NSString stringWithFormat:@"%@%@", [self description], [fields componentsJoinedByString:@" "]];
}

// A reset keeps the map's mutability and label, and its minimum factor, which Metal's reset
// leaves as the map's own.
- (void)resetUsingDescriptor:(MTLRasterizationRateMapDescriptor *)descriptor
{
	if (descriptor == nil || _serializer == nullptr)
		return;
	RgfxRateMapState state = {};
	const uint32_t status =
	    _serializer->resetRateMap(_objId, RgfxRateMapShapeOf(descriptor), &state);
	if (status != RGFX_STATUS_OK) {
		noteNoAnswer("resetUsingDescriptor:", status);
		return;
	}
	std::lock_guard<std::mutex> guard(_lock);
	_state = state;
}

@end
