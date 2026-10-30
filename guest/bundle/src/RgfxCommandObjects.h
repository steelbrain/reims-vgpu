/*
 * RgfxCommandObjects: the reims-graphics bundle's command-submission object graph —
 * the command queue, the command buffer, and the render/compute/blit (and parallel
 * render) command encoders — slice 3 of the Metal shell.
 *
 * THE OBJECT GRAPH (Metal app -> op stream):
 *   RgfxCommandQueue        vends command buffers; its create routes an
 *                           RGFX_CMD_QUEUE_CREATE through the serialiser, and it
 *                           owns the submission ring its command buffers post on.
 *   RgfxCmdBuffer       owns ONE RgfxEncoder (RgfxEncoder.hpp), the workhorse
 *                           that serialises the whole command-buffer op stream; it
 *                           vends the encoders, which all drive that one RgfxEncoder,
 *                           and on `commit` it posts the built command buffer on the
 *                           queue's submission ring.
 *   RgfxRenderCommandEncoder / RgfxComputeCommandEncoder / RgfxBlitCommandEncoder /
 *   RgfxParallelRenderCommandEncoder  translate each Metal encoder selector into the
 *                           matching RgfxEncoder op, extracting each argument's host
 *                           handle with `[(RgfxResource *)x rgfxObjId]` (0 for nil).
 *                           They never serialise wire themselves — RgfxEncoder does.
 *
 * WHY THESE WRAPPERS ARE HERE (and the device category is in
 * RgfxMetalDevice+CommandBuffer.mm): like RgfxBuffer, every class here subclasses
 * only NSObject (the queue and buffer through RgfxResource, the encoders directly),
 * so they instantiate and are unit-tested on the Mac in user space, driven by a test
 * serialiser + submission ring and a nil device. The `RgfxMetalDevice (CommandBuffer)`
 * category that vends a queue from the real device needs the Apple-private device
 * base and so lives in its own bundle-only file; these wrappers do not, and are built
 * into both the unit test and the bundle.
 *
 * LIFECYCLE BITS THAT WAIT ON THE RING TRANSPORT (documented stubs, as slice 0's
 * device does): the completion ring's reader is a later transport piece, so
 * `addCompletedHandler:`/`addScheduledHandler:` hold their blocks and `commit`
 * records the submission seqno; the handlers fire and `waitUntilCompleted`/
 * `waitUntilScheduled` return once that reader lands. `presentDrawable:` and the
 * command-buffer-scope event ops wait on their own later pieces (see the .mm).
 *
 * Our own ObjC++; the only Apple names are the real Metal protocols these conform to.
 */

#ifndef RGFX_COMMAND_OBJECTS_H_
#define RGFX_COMMAND_OBJECTS_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
class RgfxRingWriter;
class RgfxIdSpace;
class RgfxEncoder;
class RgfxReplyReader;
#endif

NS_ASSUME_NONNULL_BEGIN

@class RgfxCmdBuffer;

// --- the command queue -------------------------------------------------------

/// A command queue: an RGFX_CMD_QUEUE_CREATE routed through the serialiser, plus the
/// submission ring its command buffers post on. Vends RgfxCmdBuffer.
@interface RgfxCommandQueue : RgfxResource <MTLCommandQueue>

/// Creates a queue, serialising an RGFX_CMD_QUEUE_CREATE through `serializer` (which
/// picks the objId). `submitRing` is the queue's submission ring (guest -> host),
/// borrowed; `submitOffset`, `completeOffset` and `completeDataSize` are the ring
/// placement the region allocator (RgfxRegion) picked, carried in the QUEUE_CREATE so
/// the host can map the rings (the submission ring's data size is `submitRing`'s);
/// `ids` is the device's id space, borrowed, from which command-buffer ids are minted;
/// `maxCount` is `maxCommandBufferCount` (0 means unbounded). `device` is the owning
/// shell (nil in a unit test). Returns nil if `serializer`, `ids` or `submitRing` is
/// null (a queue must have a route and a ring).
+ (nullable instancetype)queueWithSubmitRing:(nullable RgfxRingWriter *)submitRing
                                submitOffset:(uint64_t)submitOffset
                              completeOffset:(uint64_t)completeOffset
                            completeDataSize:(uint32_t)completeDataSize
                                         ids:(nullable RgfxIdSpace *)ids
                                    maxCount:(NSUInteger)maxCount
                                      device:(nullable id<MTLDevice>)device
                                  serializer:(nullable RgfxSerializer *)serializer;

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

/// The reader of the completion ring this queue's command buffers wait on (borrowed by
/// default). The queue's command buffers register their submission seqno with it on
/// commit so their completion handlers fire and waitUntilCompleted wakes. Null in a unit
/// test that drives the queue without a reader (the command buffer then holds its seqno,
/// as before the reader landed). For a queue with its own per-queue completion ring, the
/// device hands it an owned reader with -adoptReplyReader: (below).
@property(nonatomic, nullable) RgfxReplyReader *replyReader;

/// Adopts `reader` as this queue's completion-ring reader and takes ownership of it: the
/// queue deletes it on dealloc and sets `replyReader` to it. The host posts a queue's
/// command-buffer completions on the queue's own completion ring (the ring loop), so the
/// device creates a reader over that ring and hands it here.
- (void)adoptReplyReader:(nullable RgfxReplyReader *)reader;

@end

// --- the command buffer ------------------------------------------------------

/// A command buffer: owns one RgfxEncoder building the op stream, vends the encoders,
/// and on `commit` posts the command buffer on its queue's submission ring. Its objId
/// is the cb_obj_id the op-stream header and the completion carry.
///
/// Named RgfxCmdBuffer, not RgfxCommandBuffer, because the wire header already defines
/// a C struct `RgfxCommandBuffer` (the command-buffer submission message RgfxEncoder
/// builds); the two names would collide in ObjC++.
@interface RgfxCmdBuffer : RgfxResource <MTLCommandBuffer>

/// Made by the queue (see RgfxCommandQueue): binds the buffer to its queue, its
/// guest-picked `cbObjId`, the queue's submission `ring`, and whether it retains its
/// references. Constructs the RgfxEncoder over `ring` for (queue, cb).
+ (instancetype)commandBufferForQueue:(RgfxCommandQueue *)queue
                                 ring:(RgfxRingWriter *)ring
                              cbObjId:(uint64_t)cbObjId
                             retained:(BOOL)retained
                               device:(nullable id<MTLDevice>)device
                           serializer:(nullable RgfxSerializer *)serializer;

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

@end

// --- the encoders ------------------------------------------------------------

/// A render command encoder: each selector forwards to the matching RgfxEncoder
/// render op. The pass framing (RGFX_OP_ENCODER_BEGIN_RENDER) is written by the
/// command buffer when it vends the encoder; `endEncoding` writes RGFX_OP_ENCODER_END
/// — unless this encoder belongs to a parallel render encoder, which owns the framing.
@interface RgfxRenderCommandEncoder : NSObject <MTLRenderCommandEncoder>
/// `enc` is the command buffer's RgfxEncoder (borrowed); `emitEnd` is NO for a
/// sub-encoder of a parallel render encoder (the parallel encoder writes the END).
- (instancetype)initWithEncoder:(RgfxEncoder *)enc
                        emitEnd:(BOOL)emitEnd
                         device:(nullable id<MTLDevice>)device NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// A compute command encoder: each selector forwards to the matching RgfxEncoder
/// compute op.
@interface RgfxComputeCommandEncoder : NSObject <MTLComputeCommandEncoder>
- (instancetype)initWithEncoder:(RgfxEncoder *)enc
                         device:(nullable id<MTLDevice>)device NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// A blit command encoder: each selector forwards to the matching RgfxEncoder blit op.
@interface RgfxBlitCommandEncoder : NSObject <MTLBlitCommandEncoder>
- (instancetype)initWithEncoder:(RgfxEncoder *)enc
                         device:(nullable id<MTLDevice>)device NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// A parallel render command encoder: one render pass whose sub-encoders encode into
/// the same RgfxEncoder. It writes the pass framing; each `renderCommandEncoder` it
/// vends forwards ops to the shared RgfxEncoder without its own begin/end, and this
/// encoder's `endEncoding` writes the single RGFX_OP_ENCODER_END.
@interface RgfxParallelRenderCommandEncoder : NSObject <MTLParallelRenderCommandEncoder>
- (instancetype)initWithEncoder:(RgfxEncoder *)enc
                         device:(nullable id<MTLDevice>)device NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

NS_ASSUME_NONNULL_END

#endif // RGFX_COMMAND_OBJECTS_H_
