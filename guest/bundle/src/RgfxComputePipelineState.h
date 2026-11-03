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
#include <memory>

#include "RgfxAirArguments.hpp"

class RgfxSerializer;
class RgfxReplyReader;
class RgfxKnownPipelines;
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

#ifdef __cplusplus
/// As the two factories above, posted against `known`, the device's known pipelines (null: none),
/// as RgfxRenderPipelineState's factory with `known:` is.
+ (nullable instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                                  device:(nullable id<MTLDevice>)device
                                              serializer:(nullable RgfxSerializer *)serializer
                                                   known:(std::shared_ptr<RgfxKnownPipelines>)known
                                                   error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                             error;
+ (nullable instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                                     device:(nullable id<MTLDevice>)device
                                                 serializer:(nullable RgfxSerializer *)serializer
                                                      known:(std::shared_ptr<RgfxKnownPipelines>)known
                                                      error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                                error;
#endif

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

/// Registers with `reader` for the RgfxPipelineReflection the host posts for this
/// create's seqno; after it arrives the reflection getters report the host's values
/// instead of placeholders. The Sync/Pipelines device slice calls this after the
/// create; a unit test calls it directly before feeding the reply.
- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader;

/// Whether the host answered the create with a refusal (RGFX_PIPELINE_REFUSED): it made no
/// pipeline, so the device's creators answer nil and an NSError, as Metal's do. NO for a create
/// posted without waiting (rgfxDeferred), whose answer comes after.
@property(nonatomic, readonly) BOOL rgfxRefused;

/// Whether the create was posted without waiting for its answer, its descriptor equal to one the
/// device accepted for this process.
@property(nonatomic, readonly) BOOL rgfxDeferred;

#ifdef __cplusplus
/// The arguments of the pipeline's function, as its library read them (RgfxFunction's
/// rgfxAirFunction): which textures a dispatch of it writes. Null where they are unknown.
- (std::shared_ptr<const RgfxAirFunction>)rgfxAirFunction;
#endif

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_COMPUTE_PIPELINE_STATE_H_
