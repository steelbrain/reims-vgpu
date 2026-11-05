/*
 * RgfxRenderPipelineState: the reims-graphics bundle's MTLRenderPipelineState
 * wrapper. The factory maps an MTLRenderPipelineDescriptor into the serialiser's
 * RgfxRenderPipelineParams (functions, colour attachments, vertex descriptor,
 * flags, formats) and routes an RGFX_CMD_RENDER_PIPELINE_CREATE; the guest picks
 * the objId. The create is answered by the host with an RgfxPipelineReflection on
 * the queue's completion ring, matched by the create's `seqno` — but the reply-ring
 * reader is a LATER transport piece, so the wrapper holds objId + seqno and its
 * reflection getters (maxTotalThreadsPerThreadgroup, gpuResourceID, …) answer
 * documented placeholders until that reader lands. A mesh render pipeline
 * (MTLMeshRenderPipelineDescriptor, RGFX_CMD_MESH_RENDER_PIPELINE_CREATE) is one of these too:
 * its answer carries its object and mesh stages' limits after the reflection.
 *
 * Our own ObjC++: it conforms to MTLRenderPipelineState but subclasses only NSObject
 * (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_RENDER_PIPELINE_STATE_H_
#define RGFX_RENDER_PIPELINE_STATE_H_

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

#ifdef __cplusplus
/// As renderPipelineStateWithDescriptor:device:serializer:error:, posted against `known`, the
/// device's known pipelines (null: none). A descriptor equal to one the device accepted for this
/// process is posted without waiting for its answer: the state takes that one's reflection at
/// once, and its gpuResourceID waits at its first read for its own answer. The first create of
/// each descriptor is answered as any, and its answer kept in `known` once the device accepted it
/// (rgfxRegisterWithReplyReader:).
+ (nullable instancetype)renderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                                    device:(nullable id<MTLDevice>)device
                                                serializer:(nullable RgfxSerializer *)serializer
                                                     known:(std::shared_ptr<RgfxKnownPipelines>)known
                                                     error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                               error;

/// Ends the process where Metal's validation of a mesh render pipeline's descriptor does, with its
/// words: no mesh function, no fragment function while rasterization is enabled, and
/// supportIndirectCommandBuffers, which the x86 Macs' Metal makes no mesh pipeline of.
/// meshRenderPipelineStateWithDescriptor: calls it first; the device's creators, before they ask
/// whether the host makes mesh pipelines.
+ (void)rgfxEndAtInvalidMeshDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor;

/// Creates a mesh render pipeline state from `descriptor`, mapping it into
/// RgfxMeshRenderPipelineParams and serialising an RGFX_CMD_MESH_RENDER_PIPELINE_CREATE, whose
/// answer it waits for. `maxTotalThreads` is the device's most threads in a threadgroup. On
/// failure returns nil and, when `error` is non-null, sets it: as
/// renderPipelineStateWithDescriptor:device:serializer:error:, and Metal's refusal of a stage
/// function of the wrong type, of a maximum above `maxTotalThreads` and of a payload above 16 KiB.
/// It ends the process first where Metal's validation does (rgfxEndAtInvalidMeshDescriptor:).
+ (nullable instancetype)meshRenderPipelineStateWithDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                                        device:(nullable id<MTLDevice>)device
                                                    serializer:(nullable RgfxSerializer *)serializer
                                               maxTotalThreads:(NSUInteger)maxTotalThreads
                                                         error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                                   error;
#endif

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
/// pipeline, so the device's creators answer nil and an NSError, as Metal's do. NO for a create
/// posted without waiting (rgfxDeferred), whose answer comes after.
@property(nonatomic, readonly) BOOL rgfxRefused;

/// Whether the create was posted without waiting for its answer, its descriptor equal to one the
/// device accepted for this process.
@property(nonatomic, readonly) BOOL rgfxDeferred;

#ifdef __cplusplus
/// The arguments of the pipeline's vertex and fragment functions, as their library read them
/// (RgfxFunction's rgfxAirFunction): which textures a draw writes at each stage. A stage with no
/// function has none; null where they are unknown.
- (std::shared_ptr<const RgfxAirFunction>)rgfxVertexAirFunction;
- (std::shared_ptr<const RgfxAirFunction>)rgfxFragmentAirFunction;
/// The same, of a mesh render pipeline's object and mesh functions: none for a vertex and
/// fragment pipeline, which has neither stage.
- (std::shared_ptr<const RgfxAirFunction>)rgfxObjectAirFunction;
- (std::shared_ptr<const RgfxAirFunction>)rgfxMeshAirFunction;
#endif

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_RENDER_PIPELINE_STATE_H_
