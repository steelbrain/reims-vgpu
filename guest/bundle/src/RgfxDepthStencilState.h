/*
 * RgfxDepthStencilState: the reims-graphics bundle's MTLDepthStencilState wrapper
 * (resource slice; the wrapper convention is RgfxResource.h).
 *
 * Created by `+depthStencilStateWithDescriptor:device:serializer:`, which maps the
 * MTLDepthStencilDescriptor (the depth test and write, and the two faces' stencil
 * state) into the serialiser's fields and routes an `RGFX_CMD_DEPTH_STENCIL_CREATE`
 * (picking the objId). The device category `newDepthStencilStateWithDescriptor:`
 * is a thin forwarder.
 *
 * MTLDepthStencilState exposes only `label` and `device` (and `gpuResourceID` on
 * macOS 26+); its state lives on the wire, so the wrapper holds only its objId. It also answers
 * the four queries the system's GL renderer sends a state it binds, `readsDepth`, `writesDepth`,
 * `readsStencil` and `writesStencil`, YES for every state, as the x86 Macs' states answer them
 * (RgfxDepthStencilState.mm).
 *
 * Our own ObjC++: conforms to MTLDepthStencilState but subclasses only NSObject
 * (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_DEPTH_STENCIL_STATE_H_
#define RGFX_DEPTH_STENCIL_STATE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include "RgfxSerializer.hpp" // RgfxStencilDescriptor
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxDepthStencilState : RgfxResource <MTLDepthStencilState>

/// Creates a depth/stencil state of shape `descriptor`, serialising an
/// `RGFX_CMD_DEPTH_STENCIL_CREATE` through `serializer` (which picks the objId), labelled with
/// the descriptor's label (nil for an empty one, as Metal's). A new object each call: the
/// device answers equal descriptors with one through its RgfxStateCache. `device` is the owning
/// shell (nil in a unit test). Returns nil if `serializer` is null or `descriptor` is nil, or
/// where the device never got the create.
+ (nullable instancetype)depthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
                                                  device:(nullable id<MTLDevice>)device
                                              serializer:(nullable RgfxSerializer *)serializer;

/// The GL renderer's queries: YES for every state.
- (BOOL)readsDepth;
- (BOOL)writesDepth;
- (BOOL)readsStencil;
- (BOOL)writesStencil;

@end

#ifdef __cplusplus
/// One face of a depth/stencil descriptor as DEPTH_STENCIL_CREATE carries it, each enum its
/// Metal raw value. A nil face is the zero record (MTLCompareFunctionNever, the keep
/// operations, no mask bits); MTLDepthStencilDescriptor's faces are never nil (null_resettable:
/// a face set to nil reads back as a new MTLStencilDescriptor).
RgfxStencilDescriptor RgfxStencilOf(MTLStencilDescriptor *_Nullable face);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_DEPTH_STENCIL_STATE_H_
