/*
 * RgfxComputePipelineState: the reims-graphics bundle's MTLComputePipelineState
 * wrapper. The factories map an id<MTLFunction> (or an MTLComputePipelineDescriptor)
 * into the serialiser's RgfxComputePipelineParams and route an
 * RGFX_CMD_COMPUTE_PIPELINE_CREATE; the guest picks the objId. As with the render
 * pipeline, the host's RgfxPipelineReflection reply is read off the completion ring
 * by a later transport piece, so the wrapper holds objId + seqno and its reflection
 * getters (threadExecutionWidth, maxTotalThreadsPerThreadgroup, gpuResourceID, …)
 * answer documented placeholders until that reader lands.
 *
 * Our own ObjC++: it conforms to MTLComputePipelineState but subclasses only
 * NSObject (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_COMPUTE_PIPELINE_STATE_H_
#define RGFX_COMPUTE_PIPELINE_STATE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxComputePipelineState : RgfxResource <MTLComputePipelineState>

/// Creates a compute pipeline state from the compute `function`, serialising an
/// RGFX_CMD_COMPUTE_PIPELINE_CREATE. `device` is the owning shell (nil in a unit
/// test). On failure returns nil and, when `error` is non-null, sets it.
+ (nullable instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                                  device:(nullable id<MTLDevice>)device
                                              serializer:(nullable RgfxSerializer *)serializer
                                                   error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                             error;

/// Creates a compute pipeline state from `descriptor` (its computeFunction, ICB
/// support, maxTotalThreadsPerThreadgroup and the thread-execution-width flag).
+ (nullable instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                                     device:(nullable id<MTLDevice>)device
                                                 serializer:(nullable RgfxSerializer *)serializer
                                                      error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                                error;

/// Designated initialiser: wraps an already-posted create.
- (instancetype)initWithObjId:(uint64_t)objId
                        seqno:(uint64_t)seqno
                        label:(nullable NSString *)label
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

/// The create's ring seqno, the key the host's RgfxPipelineReflection reply is
/// matched by.
@property(nonatomic, readonly) uint64_t rgfxSeqno;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_COMPUTE_PIPELINE_STATE_H_
