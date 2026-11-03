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
 * COMPLETION. `commit` registers the buffer's submission seqno with its queue's
 * completion-ring reader (RgfxReplyReader). The process's completion thread
 * (RgfxCompletionThread) takes the host's RGFX_EVT_CB_SCHEDULED and RGFX_EVT_CB_COMPLETE
 * off the ring, whether or not anything waits for the buffer: there the buffer is
 * submitted, and at its completion what the host is done with goes back (the queue's
 * slot, its message's block, its inline bytes' chunks). Its record then runs on the
 * queue's completion queue, the dispatch queue setCompletionQueue: gave or a serial queue
 * of the queue's own, behind the queue's earlier records, as Metal runs its handlers:
 * the write-backs, the status, the scheduled then completed handlers, the waiters. A buffer
 * with no handler and nothing to write back, behind none of its queue's records still to
 * run, has its records run on the thread that took them off instead, which wakes its waiters.
 * `waitUntilCompleted` blocks until the record has run the handlers, as Metal's does, and
 * takes the ring itself only where no thread drives it (a unit test, or its thread has
 * ended), where the taker runs the record too.
 * `presentDrawable:` and its variants present the drawable from a scheduled handler,
 * unless the buffer failed by then (see the .mm).
 *
 * Our own ObjC++; the only Apple names are the real Metal protocols these conform to.
 */

#ifndef RGFX_COMMAND_OBJECTS_H_
#define RGFX_COMMAND_OBJECTS_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include <functional>
#include <memory>

class RgfxSerializer;
class RgfxRingWriter;
class RgfxIdSpace;
class RgfxEncoder;
class RgfxReplyReader;
class RgfxRegion;
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

/// The device region (borrowed) a command buffer too large for the queue's submission ring
/// puts its message in, a block of the process's own the ring then names
/// (COMMAND_BUFFER_IN_REGION), kept until the buffer completes. Null in a unit test without
/// one: such a buffer then completes with an out-of-memory error.
@property(nonatomic, nullable) RgfxRegion *region;

/// Whether a command buffer of it that commitAndWaitUntilSubmitted commits asks the host for its
/// submission (RGFX_OP_SUBMITTED), where the device has RGFX_FEATURE_CB_SUBMITTED: the call then
/// returns at the host's RGFX_EVT_CB_SUBMITTED, else at the buffer's CB_SCHEDULED. NO by default.
@property(nonatomic) BOOL asksSubmitted;

/// The bytes the queue guesses a new command buffer's message takes (RgfxEncoder): the
/// largest of the queue's recent messages, as its command buffers committed them. Each smaller
/// one takes an eighth off, so one large buffer does not hold it. A message has room for it
/// before it grows, up to the size its ring takes; past that, the block of the queue's pool it
/// moves to has room for it. 0 before the first commit.
@property(nonatomic, readonly) uint32_t rgfxMessageReserve;

/// The most command buffers of this queue alive and not completed at once:
/// `maxCommandBufferCount`, 64 where the queue was made with 0, as Metal's default.
/// -commandBuffer blocks while that many are, as Metal's does, where the queue has a reader
/// whose completions free them.
@property(nonatomic, readonly) NSUInteger maxCommandBufferCount;

/// Adopts `reader` as this queue's completion-ring reader and takes ownership of it: the
/// queue shares it with its in-flight command buffers and the completion thread
/// (sharedReplyReader), and sets `replyReader` to it. The host posts a queue's
/// command-buffer completions on the queue's own completion ring (the ring loop), so the
/// device creates a reader over that ring and hands it here.
- (void)adoptReplyReader:(nullable RgfxReplyReader *)reader;

#ifdef __cplusplus
/// The reader adoptReplyReader: took, shared: the completion thread drives it, and each
/// command buffer committed on the queue holds it until its completion. Empty for a
/// borrowed reader (the `replyReader` a unit test sets).
- (std::shared_ptr<RgfxReplyReader>)sharedReplyReader;

/// What the queue's end gives back once the device has ended the queue (its rings' blocks,
/// the writer over its submission ring). The queue ends as it goes, once its command
/// buffers have: each keeps it alive, as Metal's do. The device
/// stops reading its completion ring, then the serialiser ends it (destroyQueue), then this
/// runs.
- (void)rgfxSetRelease:(std::function<void()>)release;
#endif

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

/// The error options the buffer's descriptor asked for (errorOptions answers them).
- (void)rgfxSetErrorOptions:(MTLCommandBufferErrorOption)options;

/// Encodes the buffer's one encoder, a video encoder on the session `session`, of `codec`
/// (RGFX_VIDEO_CODEC_*), holding one decode (RGFX_OP_VIDEO_DECODE_H264 or _H265): the access unit
/// that is `bitstream`'s first `bytes` bytes, after the `size` bytes of parameter sets at
/// `parameterSets`, into the surface `luma` and `chroma` lie over, which the buffer writes. The
/// video driver's (RgfxVideoChannel.hpp).
- (void)rgfxEncodeVideoSession:(uint64_t)session
                         codec:(uint32_t)codec
                 parameterSets:(nullable const uint8_t *)parameterSets
                          size:(uint32_t)size
                     bitstream:(id<MTLBuffer>)bitstream
                       ofBytes:(uint64_t)bytes
                          luma:(id<MTLTexture>)luma
                        chroma:(id<MTLTexture>)chroma;

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
