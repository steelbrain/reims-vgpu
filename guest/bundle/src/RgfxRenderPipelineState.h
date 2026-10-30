/*
 * RgfxRenderPipelineState: the reims-graphics bundle's MTLRenderPipelineState
 * wrapper. The factory maps an MTLRenderPipelineDescriptor into the serialiser's
 * RgfxRenderPipelineParams (functions, colour attachments, vertex descriptor,
 * flags, formats) and routes an RGFX_CMD_RENDER_PIPELINE_CREATE; the guest picks
 * the objId. The create is answered by the host with an RgfxPipelineReflection on
 * the queue's completion ring, matched by the create's `seqno` — but the reply-ring
 * reader is a LATER transport piece, so the wrapper holds objId + seqno and its
 * reflection getters (maxTotalThreadsPerThreadgroup, gpuResourceID, …) answer
 * documented placeholders until that reader lands.
 *
 * Our own ObjC++: it conforms to MTLRenderPipelineState but subclasses only NSObject
 * (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_RENDER_PIPELINE_STATE_H_
#define RGFX_RENDER_PIPELINE_STATE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
class RgfxReplyReader;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxRenderPipelineState : RgfxResource <MTLRenderPipelineState>

/// Creates a render pipeline state from `descriptor`, mapping it into
/// RgfxRenderPipelineParams and serialising an RGFX_CMD_RENDER_PIPELINE_CREATE.
/// `device` is the owning shell (nil in a unit test). On failure returns nil and,
/// when `error` is non-null, sets it: `serializer` null (no route), the create did
/// not post (did not fit the ring), or the descriptor's functions are missing.
+ (nullable instancetype)renderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                                    device:(nullable id<MTLDevice>)device
                                                serializer:(nullable RgfxSerializer *)serializer
                                                     error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                               error;

/// Designated initialiser: wraps an already-posted create. `objId` and `seqno` are
/// the create's; `label` is kept locally (diagnostics).
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
/// create's seqno: when it arrives, the reflection getters below stop answering
/// placeholders and report the host's values. The device's Pipelines slice calls this
/// after the create; a unit test calls it directly before feeding the reply.
- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader;

/// Whether the host answered the create with a refusal (RGFX_PIPELINE_REFUSED): it made no
/// pipeline, so the device's creators answer nil and an NSError, as Metal's do.
@property(nonatomic, readonly) BOOL rgfxRefused;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_RENDER_PIPELINE_STATE_H_
