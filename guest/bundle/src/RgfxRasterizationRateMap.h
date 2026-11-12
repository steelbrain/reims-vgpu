/*
 * RgfxRasterizationRateMap: the reims-graphics bundle's MTLRasterizationRateMap wrapper (the
 * wrapper convention is RgfxResource.h).
 *
 * Made by `+rateMapWithDescriptor:device:serializer:`, which carries the descriptor's screen
 * size, its private mutability and its layers' samples in an `RGFX_CMD_RATE_MAP_CREATE` and
 * waits for the host's answer, the map's state (RATE_MAP_STATE): the host's platform compiles
 * the map, and its state is what the map answers from then on (screenSize, physicalGranularity,
 * layerCount, parameterBufferSizeAndAlign). The answers that take an argument
 * (physicalSizeForLayer:, both coordinate maps) are the host's, asked each call
 * (RGFX_CMD_RATE_MAP_QUERY). A map the host refuses, or whose create is not answered, is nil, as
 * Metal answers a map it cannot make; and so is every map where the host makes none
 * (RgfxSerializer::carriesRateMaps).
 *
 * copyParameterDataToBuffer:offset: writes the map's parameter data, the bytes a shader's
 * rasterization_rate_map_decoder reads, where the buffer's contents lie, as Metal's CPU copy
 * does: the host writes them in place into a Shared buffer it sees in place, and into a scratch
 * Shared buffer of the map's for any other buffer (a Managed one's CPU copy, app memory the GPU
 * sees through a copy), from which the bundle copies them to the contents.
 *
 * The label is the descriptor's (nil for an empty one), and a render pass names the map
 * (MTLRenderPassDescriptor's rasterizationRateMap) by its objId, which the command buffer keeps
 * until it completes.
 *
 * Our own ObjC++: conforms to MTLRasterizationRateMap but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_RASTERIZATION_RATE_MAP_H_
#define RGFX_RASTERIZATION_RATE_MAP_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include "RgfxSerializer.hpp" // RgfxRateMapShape
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxRasterizationRateMap : RgfxResource <MTLRasterizationRateMap>

/// Makes a map of `descriptor` through `serializer` (which picks the objId), labelled with the
/// descriptor's label. Nil where `serializer` is null, `descriptor` is nil, or the host makes
/// none. `device` is the owning shell (nil in a unit test).
+ (nullable instancetype)rateMapWithDescriptor:(MTLRasterizationRateMapDescriptor *)descriptor
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

/// The map's private mutability (MTLMutability), its descriptor's (MTLRasterizationRateMapSPI).
@property(nonatomic, readonly) MTLMutability mutability;

/// The map's private minimum factor, its descriptor's (MTLRasterizationRateMapSPI): it changes
/// nothing the host's map answers or draws, so it stays in the guest.
@property(nonatomic, readonly) float minFactor;

/// The map's description and its fields, a line each, indented by `indent` + 4 spaces, as
/// Metal's base map words them (MTLRasterizationRateMapSPI).
- (NSString *)formattedDescription:(NSUInteger)indent;

/// Compiles the map again from `descriptor` (MTLRasterizationRateMapSPI), keeping its mutability
/// and label: the host's platform compiles it, and the map answers its new state.
- (void)resetUsingDescriptor:(MTLRasterizationRateMapDescriptor *)descriptor;

@end

#ifdef __cplusplus
/// `descriptor` as RATE_MAP_CREATE carries it: its screen size, its mutability where its class
/// answers one, and its layers from index 0 (its layerCount), each its sampleCount and samples.
RgfxRateMapShape RgfxRateMapShapeOf(MTLRasterizationRateMapDescriptor *descriptor);

/// `descriptor`'s private minimum factor, 0 where its class answers none (Metal's default: a
/// fresh descriptor answers 0 on the AMD Radeon Pro 5300M, macOS 26.7.1).
float RgfxRateMapMinFactorOf(MTLRasterizationRateMapDescriptor *descriptor);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_RASTERIZATION_RATE_MAP_H_
