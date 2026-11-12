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

/// Whose words and order a compute pipeline's threadgroup fields are checked in
/// (RgfxComputeThreadsVerdictOf): the x86 Macs' (the AMD Radeon Pro 5300M's, where the Intel UHD
/// Graphics 630's differ), which the x86-64 guest's device answers, or Apple silicon's (the M3
/// Max's), which the arm64 guest's does.
typedef NS_ENUM(NSInteger, RgfxThreadsRules) {
	RgfxThreadsRulesX86,
	RgfxThreadsRulesApple,
};

/// The rules of this build's guest: Apple silicon's in an arm64 guest, else the x86 Macs'.
static inline RgfxThreadsRules RgfxGuestThreadsRules(void)
{
#if defined(__arm64__)
	return RgfxThreadsRulesApple;
#else
	return RgfxThreadsRulesX86;
#endif
}

/// What Metal makes of a compute pipeline's threadgroup fields: the process ended at a failed
/// assertion of its validation (`assertion` its words, from `assertionFunction`), the pipeline
/// refused (`error`), or made (both nil), then of `maxTotalThreads` threads per threadgroup where its
/// required size sets them, 0 where the made pipeline answers the device's own limit.
@interface RgfxThreadsVerdict : NSObject
@property(nonatomic, readonly, nullable) NSString *assertion;
@property(nonatomic, readonly, nullable) const char *assertionFunction;
@property(nonatomic, readonly, nullable) NSError *error;
@property(nonatomic, readonly) NSUInteger maxTotalThreads;
@end

/// The verdict on a compute pipeline of the kernel `kernelName`, whose
/// [[max_total_threads_per_threadgroup(n)]] is `kernelMax` (0: none, or not read), made with
/// `required` threads per threadgroup ({0, 0, 0}: none) and `maxTotal` (0: none), on a device of
/// `deviceLimit` threads per threadgroup (0: not known), as `rules`' Metal answers it
/// (RgfxComputePipelineState.mm).
RgfxThreadsVerdict *RgfxComputeThreadsVerdictOf(RgfxThreadsRules rules, MTLSize required, NSUInteger maxTotal,
                                                NSUInteger kernelMax, NSString *kernelName, NSUInteger deviceLimit);

/// Metal's refusal of a compute pipeline of macOS 27's `forwardProgressUsage` `progress` and
/// `contentionRelief` `relief` (`hasProgress`, `hasRelief` where the descriptor has them), nil
/// where it makes it (RgfxComputePipelineState.mm).
NSError *_Nullable RgfxComputeProgressError(BOOL hasProgress, NSInteger progress, BOOL hasRelief, NSInteger relief);

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
