/*
 * RgfxCommandObjects: see RgfxCommandObjects.h for the object graph and conventions.
 * Each encoder selector is a thin forwarder to the matching RgfxEncoder op; the op
 * stream, its framing and the command-buffer header are RgfxEncoder's (RgfxEncoder.hpp),
 * which mirrors the host decoder crates/reims-vgpu/src/graphics/decode/{cmdbuf,render,
 * compute,blit}.rs. Our own ObjC++; no Apple private SPI.
 *
 * ARGUMENT HANDLES. Every resource/pipeline/state/event argument Metal hands an
 * encoder is one of our RgfxResource wrappers; `rgfxId` reads its host handle (0 for
 * nil), as the convention requires.
 *
 * SINGLE vs ARRAY ops. Where RgfxEncoder has only the array form of a binding op
 * (e.g. renderSetVertexBuffers but no single vertex-buffer op), the single Metal
 * selector (setVertexBuffer:offset:atIndex:) forwards a one-element array at the given
 * index, exactly as the host reads it.
 *
 * INLINE BYTES. The set*Bytes:length:atIndex: setters have no op of their own: the bytes go
 * to an upload chunk of the command buffer's in the device region (RgfxUploads.h), bound as a
 * buffer at their offset there.
 *
 * NOT MAPPED HERE (no RgfxEncoder op yet — follow-ups, no wire invented): the render SPI-only
 * ops that have no public selector (provoking-vertex, alpha-test ref, line width, point size,
 * viewport-transform, primitive-restart, depth/stencil "cleared", resolve textures,
 * tile-dispatch in-region, vertex-amplification mode); the compute indirect-command-encoding
 * ops (encodeStartIf/While/Else/DoWhile, compressed-texture flush) and
 * computeDispatchThreadsIndirect; the blit fill-texture, fill-buffer-pattern4 and
 * invalidate-compressed ops. The public members of features the device does not support
 * (function tables, acceleration structures, mesh and object stages, counters, tensors,
 * queue-stage barriers, depth bounds, colour attachment maps, depth store-action options) do
 * nothing on the wire and are logged (RgfxUnsupported.h).
 */

#import "RgfxCommandObjects.h"
#import "RgfxIOSurface.h"
#import "RgfxUnsupported.h"
#import "RgfxUploads.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mach/mach_time.h>
#include <vector>

#include "RgfxCompletionThread.hpp"
#include "RgfxEncoder.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRetirement.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"

namespace {

// The guest's host time in seconds, the clock CACurrentMediaTime reads (mach_absolute_time).
CFTimeInterval rgfxHostTime()
{
	static mach_timebase_info_data_t timebase = [] {
		mach_timebase_info_data_t t = {};
		mach_timebase_info(&t);
		return t;
	}();
	return static_cast<CFTimeInterval>(mach_absolute_time()) * timebase.numer / timebase.denom /
	       1e9;
}

} // namespace

// The logs of a command buffer: none, as the device runs no shader that logs.
@interface RgfxLogContainer : NSObject <MTLLogContainer>
@end

@implementation RgfxLogContainer
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state
                                  objects:(id __unsafe_unretained _Nullable[_Nonnull])buffer
                                    count:(NSUInteger)len
{
	return 0;
}
@end

namespace {

/// How many command buffers' completions this thread is delivering (their handlers run
/// inside): a waitUntilCompleted there takes the ring itself, as the thread that would
/// otherwise deliver may be this one.
thread_local int t_delivering = 0;

/// Whether the `n`th (from 0) unhonoured command-buffer event op is logged: the first 16,
/// then every power of two, so a frame loop's ops stay visible without flooding the log.
bool logEventOp(std::atomic<uint64_t> &count, uint64_t *n)
{
	*n = count.fetch_add(1, std::memory_order_relaxed);
	return *n < 16 || (*n & (*n - 1)) == 0;
}

std::atomic<uint64_t> g_eventWaits{0};
std::atomic<uint64_t> g_eventSignals{0};
std::atomic<uint64_t> g_bytesDropped{0};

/// Puts an encoder's inline bytes (set*Bytes:length:atIndex:) in its command buffer's upload
/// chunks (RgfxUploads.h), answering the buffer and offset they lie at to bind; false, said in
/// the log, where no chunk could be had (the region has no room), and the binding is not made.
bool rgfxPutBytes(RgfxUploadArena *uploads, const void *bytes, NSUInteger length,
                  id<MTLBuffer> __strong *buffer, uint64_t *offset)
{
	if (uploads != nullptr && uploads->put(bytes, length, buffer, offset)) {
		return true;
	}
	uint64_t n = 0;
	if (logEventOp(g_bytesDropped, &n)) {
		NSLog(@"RGFX: %lu inline bytes have no upload chunk (no region room): their binding is "
		      @"not made (so far: %llu)",
		      (unsigned long)length, (unsigned long long)(n + 1));
	}
	return false;
}

/// The host handle naming `obj` (0 = nil). Every Metal object the guest sees is one
/// of our RgfxResource wrappers.
uint64_t rgfxId(id obj)
{
	if (obj != nil && [obj isKindOfClass:[RgfxResource class]]) {
		return [(RgfxResource *)obj rgfxObjId];
	}
	return 0;
}

/// The RGFX_RESOURCE_KIND_* of a Metal resource, for the resource-reference ops.
uint32_t rgfxResourceKind(id<MTLResource> res)
{
	if ([res conformsToProtocol:@protocol(MTLBuffer)])
		return RGFX_RESOURCE_KIND_BUFFER;
	if ([res conformsToProtocol:@protocol(MTLTexture)])
		return RGFX_RESOURCE_KIND_TEXTURE;
	if ([res conformsToProtocol:@protocol(MTLIndirectCommandBuffer)])
		return RGFX_RESOURCE_KIND_ICB;
	return RGFX_RESOURCE_KIND_NONE;
}

/// The level of `t` whose size is `other`'s first level's, in `*level`; false where none is.
bool rgfxLevelMatching(id<MTLTexture> t, id<MTLTexture> other, NSUInteger *level)
{
	for (NSUInteger m = 0; m < t.mipmapLevelCount; m++) {
		if (std::max<NSUInteger>(1, t.width >> m) == other.width &&
		    std::max<NSUInteger>(1, t.height >> m) == other.height &&
		    std::max<NSUInteger>(1, t.depth >> m) == other.depth) {
			*level = m;
			return true;
		}
	}
	return false;
}

/// The slices a blit's whole-surface copy counts in `t`: its array length, six faces a cube.
NSUInteger rgfxSurfaceSlices(id<MTLTexture> t)
{
	NSUInteger slices = t.arrayLength != 0 ? t.arrayLength : 1;
	if (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray)
		slices *= 6;
	return slices;
}

/// The f64 bits of `v`, as a clear-colour/viewport field travels on the wire.
uint64_t f64bits(double v)
{
	uint64_t bits;
	std::memcpy(&bits, &v, sizeof(bits));
	return bits;
}

/// Fills a colour attachment record from a Metal descriptor.
void fillColor(RgfxRenderPassColorAttachment *c, MTLRenderPassColorAttachmentDescriptor *d)
{
	*c = {};
	c->texture_obj_id = rgfxId(d.texture);
	c->resolve_texture_obj_id = rgfxId(d.resolveTexture);
	MTLClearColor cc = d.clearColor;
	c->clear_color_bits[0] = f64bits(cc.red);
	c->clear_color_bits[1] = f64bits(cc.green);
	c->clear_color_bits[2] = f64bits(cc.blue);
	c->clear_color_bits[3] = f64bits(cc.alpha);
	c->load_action = static_cast<uint32_t>(d.loadAction);
	c->store_action = static_cast<uint32_t>(d.storeAction);
	c->store_action_options = static_cast<uint32_t>(d.storeActionOptions);
	c->level = static_cast<uint32_t>(d.level);
	c->slice = static_cast<uint32_t>(d.slice);
	c->depth_plane = static_cast<uint32_t>(d.depthPlane);
	c->resolve_level = static_cast<uint32_t>(d.resolveLevel);
	c->resolve_slice = static_cast<uint32_t>(d.resolveSlice);
	c->resolve_depth_plane = static_cast<uint32_t>(d.resolveDepthPlane);
}

/// Translates an MTLRenderPassDescriptor into the wire pass descriptor and its colour
/// attachments, returning the colour count (highest set attachment index + 1, so
/// gaps stay as absent attachments). `colors` is filled with `count` records.
/// Records on `enc` every IOSurface texture the pass `d` writes: its attachments and
/// their resolve targets.
void noteRenderPassWrites(RgfxEncoder *enc, MTLRenderPassDescriptor *d)
{
	for (uint32_t i = 0; i < 8; i++) {
		rgfxIOSurfaceNoteWrite(enc, d.colorAttachments[i].texture);
		rgfxIOSurfaceNoteWrite(enc, d.colorAttachments[i].resolveTexture);
	}
	rgfxIOSurfaceNoteWrite(enc, d.depthAttachment.texture);
	rgfxIOSurfaceNoteWrite(enc, d.depthAttachment.resolveTexture);
	rgfxIOSurfaceNoteWrite(enc, d.stencilAttachment.texture);
	rgfxIOSurfaceNoteWrite(enc, d.stencilAttachment.resolveTexture);
}

uint32_t fillRenderPass(MTLRenderPassDescriptor *d, RgfxRenderPassDesc *desc,
                        std::vector<RgfxRenderPassColorAttachment> *colors)
{
	*desc = {};

	uint32_t count = 0;
	for (uint32_t i = 0; i < 8; i++) {
		if (rgfxId(d.colorAttachments[i].texture) != 0) {
			count = i + 1;
		}
	}
	colors->resize(count);
	for (uint32_t i = 0; i < count; i++) {
		fillColor(&(*colors)[i], d.colorAttachments[i]);
	}

	uint32_t flags = 0;
	if (d.renderTargetWidth != 0 || d.renderTargetHeight != 0) {
		flags |= RGFX_PASS_FLAG_RT_SIZE;
		desc->render_target_width = static_cast<uint32_t>(d.renderTargetWidth);
		desc->render_target_height = static_cast<uint32_t>(d.renderTargetHeight);
	}
	if (d.renderTargetArrayLength != 0) {
		flags |= RGFX_PASS_FLAG_RT_ARRAY_LENGTH;
		desc->render_target_array_length = static_cast<uint32_t>(d.renderTargetArrayLength);
	}
	if (d.defaultRasterSampleCount != 0) {
		flags |= RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT;
		desc->default_raster_sample_count = static_cast<uint32_t>(d.defaultRasterSampleCount);
	}
	desc->flags = flags;

	MTLRenderPassDepthAttachmentDescriptor *dep = d.depthAttachment;
	if (rgfxId(dep.texture) != 0) {
		desc->depth.texture_obj_id = rgfxId(dep.texture);
		desc->depth.resolve_texture_obj_id = rgfxId(dep.resolveTexture);
		desc->depth.clear_depth_bits = f64bits(dep.clearDepth);
		desc->depth.load_action = static_cast<uint32_t>(dep.loadAction);
		desc->depth.store_action = static_cast<uint32_t>(dep.storeAction);
		desc->depth.store_action_options = static_cast<uint32_t>(dep.storeActionOptions);
		desc->depth.level = static_cast<uint32_t>(dep.level);
		desc->depth.slice = static_cast<uint32_t>(dep.slice);
		desc->depth.depth_plane = static_cast<uint32_t>(dep.depthPlane);
		desc->depth.resolve_level = static_cast<uint32_t>(dep.resolveLevel);
		desc->depth.resolve_slice = static_cast<uint32_t>(dep.resolveSlice);
		desc->depth.resolve_depth_plane = static_cast<uint32_t>(dep.resolveDepthPlane);
		desc->depth.depth_resolve_filter = static_cast<uint32_t>(dep.depthResolveFilter);
	}

	MTLRenderPassStencilAttachmentDescriptor *st = d.stencilAttachment;
	if (rgfxId(st.texture) != 0) {
		desc->stencil.texture_obj_id = rgfxId(st.texture);
		desc->stencil.resolve_texture_obj_id = rgfxId(st.resolveTexture);
		desc->stencil.load_action = static_cast<uint32_t>(st.loadAction);
		desc->stencil.store_action = static_cast<uint32_t>(st.storeAction);
		desc->stencil.store_action_options = static_cast<uint32_t>(st.storeActionOptions);
		desc->stencil.level = static_cast<uint32_t>(st.level);
		desc->stencil.slice = static_cast<uint32_t>(st.slice);
		desc->stencil.depth_plane = static_cast<uint32_t>(st.depthPlane);
		desc->stencil.resolve_level = static_cast<uint32_t>(st.resolveLevel);
		desc->stencil.resolve_slice = static_cast<uint32_t>(st.resolveSlice);
		desc->stencil.resolve_depth_plane = static_cast<uint32_t>(st.resolveDepthPlane);
		desc->stencil.stencil_resolve_filter = static_cast<uint32_t>(st.stencilResolveFilter);
		desc->stencil.clear_stencil = static_cast<uint32_t>(st.clearStencil);
	}

	return count;
}

} // namespace

// =============================================================================
// RgfxCommandQueue
// =============================================================================

@interface RgfxCmdBuffer ()
/// The buffer holds one of its queue's `slots` (maxCommandBufferCount), given back once it
/// completes, or at its dealloc where it never does.
- (void)rgfxHoldSlot:(dispatch_semaphore_t)slots;
@end

@interface RgfxCommandQueue ()
- (std::shared_ptr<RgfxUploadPool>)rgfxUploadPool;
@end

@implementation RgfxCommandQueue {
	RgfxRingWriter *_submitRing;   // borrowed: the queue's submission ring
	RgfxIdSpace *_ids;             // borrowed: the device id space, for cb ids
	NSUInteger _maxCount;
	// maxCommandBufferCount slots, one taken by each command buffer made while the queue has a
	// reader, given back as it completes.
	dispatch_semaphore_t _slots;
	NSString *_queueLabel;
	std::shared_ptr<RgfxReplyReader> _ownedReader; // adoptReplyReader:'s per-queue completion reader
	std::function<void()> _release;                // rgfxSetRelease:'s, run at the queue's end
	dispatch_queue_t _submissionQueue; // kept for -submissionQueue (setSubmissionQueue: SPI)
	dispatch_queue_t _completionQueue; // kept for -completionQueue (setCompletionQueue: SPI)
	NSUInteger _gpuPriority;           // kept for -GPUPriority (setGPUPriority: SPI)
	unsigned short _gpuPriorityOffset; // and the offsets and background priority with it
	NSUInteger _backgroundGpuPriority;
	unsigned short _backgroundGpuPriorityOffset;
	// The chunks its command buffers' inline bytes go to (RgfxUploads.h), made with the first
	// command buffer, over the queue's region.
	std::shared_ptr<RgfxUploadPool> _uploadPool;
}

+ (instancetype)queueWithSubmitRing:(RgfxRingWriter *)submitRing
                       submitOffset:(uint64_t)submitOffset
                     completeOffset:(uint64_t)completeOffset
                   completeDataSize:(uint32_t)completeDataSize
                                ids:(RgfxIdSpace *)ids
                           maxCount:(NSUInteger)maxCount
                             device:(id<MTLDevice>)device
                         serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || ids == nullptr || submitRing == nullptr) {
		return nil; // a queue must have a route, an id space and a ring
	}
	// The ring offsets and the completion-ring size are the region allocator's
	// (RgfxRegion::allocateRingPair); the submission ring's size is the ring we hold,
	// and the queue's own objId is picked by the create. The priority/vector are the
	// host's scheduling knobs, 0 (normal) here.
	RgfxCreated created = serializer->createQueue(submitOffset, completeOffset,
	                                              submitRing->dataSize(), completeDataSize,
	                                              /*priority*/ 0, /*vector*/ 0);
	RgfxCommandQueue *q = [[RgfxCommandQueue alloc] initWithObjId:created.obj_id
	                                                       device:device
	                                                   serializer:serializer];
	if (q != nil) {
		q->_submitRing = submitRing;
		q->_ids = ids;
		q->_maxCount = maxCount != 0 ? maxCount : 64;
		q->_slots = dispatch_semaphore_create(static_cast<long>(q->_maxCount));
	}
	return q;
}

// The device ends the queue once every command buffer in flight now has completed (its own
// all have: each keeps the queue alive), and only then do its rings go back: the completion
// ring is read no more first, as the completion thread may still hold its reader.
- (void)dealloc
{
	RgfxSerializer *serializer = _serializer;
	uint64_t objId = _objId;
	std::shared_ptr<RgfxReplyReader> reader = _ownedReader;
	std::function<void()> release = std::move(_release);
	RgfxRetirement::process().retire([serializer, objId, reader, release] {
		if (reader != nullptr)
			reader->close();
		if (serializer != nullptr)
			serializer->destroyQueue(objId);
		if (release)
			release();
	});
}

- (void)rgfxSetRelease:(std::function<void()>)release
{
	_release = std::move(release);
}

- (NSUInteger)maxCommandBufferCount
{
	return _maxCount;
}

// Takes one of the queue's slots, blocking while every one is held, as Metal's -commandBuffer
// does. The completion thread gives them back as it delivers; where it does not drive the ring
// (a unit test, or its thread has ended), or this is the delivering thread (a handler making a
// buffer), the ring is taken here meanwhile.
- (void)rgfxTakeSlot
{
	if (dispatch_semaphore_wait(_slots, DISPATCH_TIME_NOW) == 0)
		return;
	RgfxReplyReader *reader = self.replyReader;
	bool delivering = t_delivering > 0 || RgfxCompletionThread::onCurrentThread();
	for (;;) {
		if (reader->driven() && !delivering) {
			if (dispatch_semaphore_wait(_slots, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC)) == 0)
				return;
			continue;
		}
		reader->pump();
		if (dispatch_semaphore_wait(_slots, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_MSEC)) == 0)
			return;
	}
}

// --- MTLCommandQueue ---------------------------------------------------------

- (nullable id<MTLCommandBuffer>)commandBuffer
{
	return [self rgfxCommandBufferRetained:YES];
}

- (nullable id<MTLCommandBuffer>)commandBufferWithUnretainedReferences
{
	return [self rgfxCommandBufferRetained:NO];
}

- (nullable id<MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)descriptor
{
	BOOL retained = descriptor == nil || descriptor.retainedReferences;
	RgfxCmdBuffer *cb = [self rgfxCommandBufferRetained:retained];
	// The error options are diagnostics: kept for the buffer to answer, they change nothing.
	if (descriptor != nil)
		[cb rgfxSetErrorOptions:descriptor.errorOptions];
	return cb;
}

// The queue's upload pool, made over its region on first use.
- (std::shared_ptr<RgfxUploadPool>)rgfxUploadPool
{
	@synchronized(self) {
		if (_uploadPool == nullptr) {
			_uploadPool = std::make_shared<RgfxUploadPool>(self.region, [self serializer],
			                                               self.device);
		}
		return _uploadPool;
	}
}

- (RgfxCmdBuffer *)rgfxCommandBufferRetained:(BOOL)retained
{
	// Only completions give a slot back: a queue without a reader (a unit test) takes none.
	bool slotted = self.replyReader != nullptr;
	if (slotted)
		[self rgfxTakeSlot];
	uint64_t cbObjId = _ids->allocObjId();
	RgfxCmdBuffer *cb = [RgfxCmdBuffer commandBufferForQueue:self
	                                                    ring:_submitRing
	                                                 cbObjId:cbObjId
	                                                retained:retained
	                                                  device:self.device
	                                              serializer:[self serializer]];
	if (slotted) {
		if (cb != nil)
			[cb rgfxHoldSlot:_slots];
		else
			dispatch_semaphore_signal(_slots);
	}
	return cb;
}

- (NSString *)label
{
	return _queueLabel;
}
- (void)setLabel:(NSString *)label
{
	_queueLabel = [label copy];
}

- (void)adoptReplyReader:(RgfxReplyReader *)reader
{
	if (_ownedReader.get() != reader)
		_ownedReader.reset(reader);
	self.replyReader = reader;
}

- (std::shared_ptr<RgfxReplyReader>)sharedReplyReader
{
	return _ownedReader;
}

// --- dispatch queues and priority (MTLCommandQueue SPI WindowServer sets) -----------
// WindowServer hands its queues the dispatch queues Metal is to run submission and
// completion work on, and a GPU priority. Our queue submits on the caller's thread and
// completes on the process's completion thread, and the host schedules every queue alike,
// so each is kept and answered back, and changes nothing the queue does.

- (void)setSubmissionQueue:(dispatch_queue_t)queue
{
	_submissionQueue = queue;
}

- (dispatch_queue_t)submissionQueue
{
	return _submissionQueue;
}

- (void)setCompletionQueue:(dispatch_queue_t)queue
{
	_completionQueue = queue;
}

- (dispatch_queue_t)completionQueue
{
	return _completionQueue;
}

// The runtime's setter answers whether the priority was taken (`c24@0:8Q16`): it always is.
- (BOOL)setGPUPriority:(NSUInteger)priority
{
	_gpuPriority = priority;
	return YES;
}

- (NSUInteger)GPUPriority
{
	return _gpuPriority;
}

// RenderBox reads the priority back as it sets up its queue (`Q16@0:8`), and the background
// priority with it; the rest of the family is the runtime's too (MTLCommandQueueSPI).
- (NSUInteger)getGPUPriority
{
	return _gpuPriority;
}

- (BOOL)setGPUPriority:(NSUInteger)priority offset:(unsigned short)offset
{
	_gpuPriority = priority;
	_gpuPriorityOffset = offset;
	return YES;
}

- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority
{
	_backgroundGpuPriority = priority;
	return YES;
}

- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(unsigned short)offset
{
	_backgroundGpuPriority = priority;
	_backgroundGpuPriorityOffset = offset;
	return YES;
}

- (NSUInteger)getBackgroundGPUPriority
{
	return _backgroundGpuPriority;
}

@end

// =============================================================================
// RgfxCmdBuffer
// =============================================================================

// The completion-delivery entry the reader's handler calls (defined below); declared
// here so the lambda in `commit` sees its signature.
@interface RgfxCmdBuffer ()
- (void)rgfxDeliverCompletion:(const RgfxCbComplete *)msg;
@end

// The encoders' inline bytes go to their command buffer's upload arena, handed over as the
// command buffer vends them.
@interface RgfxRenderCommandEncoder ()
- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads;
// The tile size its pass descriptor asked for (0: the device's choice).
- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height;
@end
@interface RgfxComputeCommandEncoder ()
- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads;
// The dispatch type its pass was opened with.
- (void)rgfxSetDispatchType:(MTLDispatchType)dispatchType;
@end
@interface RgfxParallelRenderCommandEncoder ()
- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads;
- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height;
@end

@implementation RgfxCmdBuffer {
	RgfxCommandQueue *_queue; // kept alive, as Metal's command buffers keep their queue
	RgfxEncoder *_encoder; // owned: built incrementally, posted on commit
	BOOL _retained;
	BOOL _committed;
	BOOL _enqueued;
	uint64_t _seqno;
	// Written by the thread that delivers the completion, read by any: atomic.
	std::atomic<MTLCommandBufferStatus> _status;
	NSString *_cbLabel;
	// What the client asked of the buffer's content protection (setProtectionOptions:).
	uint64_t _protectionOptions;
	// The error options its descriptor asked for (commandBufferWithDescriptor:).
	MTLCommandBufferErrorOption _errorOptions;
	// Its times, in the guest's host time (rgfxHostTime), 0 until known: the CPU's
	// scheduling from commit until the buffer is posted to the device, and the GPU's run
	// (rgfxDeliverCompletion:). Each is written once, before the status that makes it known.
	std::atomic<CFTimeInterval> _kernelStart;
	std::atomic<CFTimeInterval> _kernelEnd;
	std::atomic<CFTimeInterval> _gpuStart;
	std::atomic<CFTimeInterval> _gpuEnd;
	NSError *_cbError;
	NSMutableArray *_completedHandlers;
	NSMutableArray *_scheduledHandlers;
	// The queue's completion-ring reader, taken at commit; null when the queue has none
	// (a unit test driving the queue without one), in which case the buffer holds its
	// seqno. `_readerHold` keeps a reader the queue owns alive while the buffer is in
	// flight, though the queue may go first.
	RgfxReplyReader *_reader;
	std::shared_ptr<RgfxReplyReader> _readerHold;
	// Entered at commit with a reader, left once the completion's handlers have run:
	// what waitUntilCompleted waits on while the completion thread delivers. `_left`
	// says it was left, so a buffer whose completion never came leaves it at dealloc.
	dispatch_group_t _done;
	std::atomic<bool> _left;
	// The queue's slot this buffer holds (rgfxHoldSlot:), and whether it still does.
	dispatch_semaphore_t _slots;
	std::atomic<bool> _slotHeld;
	// A buffer too large for its ring: the region block its message lies in until it completes.
	RgfxRegion *_bodyRegion;
	uint64_t _bodyOffset;
	// The process's retirement ticket a buffer that will complete holds from its making to its
	// completion (RgfxRetirement), 0 for none: objects let go meanwhile stay on the host, as
	// Metal's buffer keeps what it encodes from the encode on.
	std::atomic<uint64_t> _ticket;
	// The IOSurface copies this buffer writes, written back to their surfaces' pages
	// when it completes (RgfxIOSurface.h).
	std::vector<std::shared_ptr<void>> _surfaceWrites;
	// Where its encoders' inline bytes go: chunks of the queue's upload pool, given back once
	// the host has completed the buffer (RgfxUploads.h). The encoders share it.
	std::shared_ptr<RgfxUploadArena> _uploads;
}

+ (instancetype)commandBufferForQueue:(RgfxCommandQueue *)queue
                                 ring:(RgfxRingWriter *)ring
                              cbObjId:(uint64_t)cbObjId
                             retained:(BOOL)retained
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	RgfxCmdBuffer *cb = [[RgfxCmdBuffer alloc] initWithObjId:cbObjId
	                                                          device:device
	                                                      serializer:serializer];
	if (cb != nil) {
		cb->_queue = queue;
		cb->_retained = retained;
		cb->_status.store(MTLCommandBufferStatusNotEnqueued);
		cb->_completedHandlers = [NSMutableArray array];
		cb->_scheduledHandlers = [NSMutableArray array];
		cb->_encoder = new RgfxEncoder(*ring, queue.rgfxObjId, cbObjId);
		// One that will complete holds back the releases from now on: an object it encodes
		// may be let go before its commit.
		if (queue.replyReader != nullptr)
			cb->_ticket.store(RgfxRetirement::process().begin());
		cb->_uploads = std::make_shared<RgfxUploadArena>([queue rgfxUploadPool]);
	}
	return cb;
}

- (void)dealloc
{
	// A group released while entered ends the process: one whose completion never came
	// (its reader went first) is left here.
	if (_done != nil && !_left.exchange(true))
		dispatch_group_leave(_done);
	// A buffer never committed, or whose completion never came, gives its slot back, and its
	// ticket.
	[self rgfxGiveSlotBack];
	if (uint64_t ticket = _ticket.exchange(0))
		RgfxRetirement::process().end(ticket);
	delete _encoder;
}

- (void)rgfxHoldSlot:(dispatch_semaphore_t)slots
{
	_slots = slots;
	_slotHeld.store(true);
}

- (void)rgfxGiveSlotBack
{
	if (_slots != nil && _slotHeld.exchange(false))
		dispatch_semaphore_signal(_slots);
}

// --- vending the encoders ----------------------------------------------------

- (nullable id<MTLRenderCommandEncoder>)renderCommandEncoderWithDescriptor:
    (MTLRenderPassDescriptor *)renderPassDescriptor
{
	RgfxRenderPassDesc desc = {};
	std::vector<RgfxRenderPassColorAttachment> colors;
	uint32_t count = fillRenderPass(renderPassDescriptor, &desc, &colors);
	noteRenderPassWrites(_encoder, renderPassDescriptor);
	_encoder->beginRenderPass(desc, colors.empty() ? nullptr : colors.data(), count);
	RgfxRenderCommandEncoder *enc = [[RgfxRenderCommandEncoder alloc] initWithEncoder:_encoder
	                                                                          emitEnd:YES
	                                                                           device:self.device];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetTileWidth:renderPassDescriptor.tileWidth height:renderPassDescriptor.tileHeight];
	return enc;
}

- (nullable id<MTLComputeCommandEncoder>)computeCommandEncoder
{
	_encoder->beginComputePass();
	RgfxComputeCommandEncoder *enc = [[RgfxComputeCommandEncoder alloc] initWithEncoder:_encoder
	                                                                             device:self.device];
	[enc rgfxSetUploads:_uploads];
	return enc;
}

- (nullable id<MTLComputeCommandEncoder>)computeCommandEncoderWithDescriptor:
    (MTLComputePassDescriptor *)computePassDescriptor
{
	return [self computeCommandEncoderWithDispatchType:computePassDescriptor.dispatchType];
}

// The pass carries the dispatch type, which the encoder answers (dispatchType).
- (nullable id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:
    (MTLDispatchType)dispatchType
{
	_encoder->beginComputePass(static_cast<uint32_t>(dispatchType));
	RgfxComputeCommandEncoder *enc = [[RgfxComputeCommandEncoder alloc] initWithEncoder:_encoder
	                                                                             device:self.device];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetDispatchType:dispatchType];
	return enc;
}

- (nullable id<MTLBlitCommandEncoder>)blitCommandEncoder
{
	_encoder->beginBlitPass();
	return [[RgfxBlitCommandEncoder alloc] initWithEncoder:_encoder device:self.device];
}

- (nullable id<MTLBlitCommandEncoder>)blitCommandEncoderWithDescriptor:
    (MTLBlitPassDescriptor *)blitPassDescriptor
{
	// The blit pass descriptor carries only sample-buffer attachments (diagnostics);
	// the op stream's blit pass has no body.
	_encoder->beginBlitPass();
	return [[RgfxBlitCommandEncoder alloc] initWithEncoder:_encoder device:self.device];
}

- (nullable id<MTLParallelRenderCommandEncoder>)parallelRenderCommandEncoderWithDescriptor:
    (MTLRenderPassDescriptor *)renderPassDescriptor
{
	RgfxRenderPassDesc desc = {};
	std::vector<RgfxRenderPassColorAttachment> colors;
	uint32_t count = fillRenderPass(renderPassDescriptor, &desc, &colors);
	noteRenderPassWrites(_encoder, renderPassDescriptor);
	_encoder->beginRenderPass(desc, colors.empty() ? nullptr : colors.data(), count);
	RgfxParallelRenderCommandEncoder *enc =
	    [[RgfxParallelRenderCommandEncoder alloc] initWithEncoder:_encoder device:self.device];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetTileWidth:renderPassDescriptor.tileWidth height:renderPassDescriptor.tileHeight];
	return enc;
}

// --- submission --------------------------------------------------------------

- (void)enqueue
{
	// Ordering across command buffers is a later transport piece; record the intent.
	_enqueued = YES;
	MTLCommandBufferStatus expected = MTLCommandBufferStatusNotEnqueued;
	_status.compare_exchange_strong(expected, MTLCommandBufferStatusEnqueued);
}

- (void)commit
{
	if (_committed) {
		return; // commit is once
	}
	_committed = YES;
	_kernelStart.store(rgfxHostTime());
	// The IOSurface copies the CPU changed since they last synced take the surfaces'
	// pages before the GPU reads them; the copies this buffer writes go back at its
	// completion.
	rgfxIOSurfaceSyncBeforeCommit();
	_surfaceWrites = _encoder->takeSurfaceWrites();
	// Committed before the host can see it, so a completion never finds an older status.
	_status.store(MTLCommandBufferStatusCommitted);

	// With a reader, the RGFX_EVT_CB_COMPLETE for the buffer's seqno fires its handlers,
	// flips its status and wakes a waiter. It is expected under the ring's lock, before the
	// host can see the buffer, so the completion thread never takes the completion first.
	// The handler captures `self` strongly, so a committed buffer lives until its completion
	// lands (as Metal retains an in-flight command buffer); the reader removes the one-shot
	// handler when it fires. Without a reader the status stays Committed and the handlers
	// are held.
	RgfxCommandQueue *queue = _queue;
	_reader = queue.replyReader;
	std::function<void(uint64_t)> expect;
	std::function<void()> whileWaiting;
	if (_reader != nullptr) {
		// Before the host can see it: an object let go from now on outlives it on the host
		// (the ticket taken as the buffer was made, or one taken here).
		if (_ticket.load() == 0)
			_ticket.store(RgfxRetirement::process().begin());
		_readerHold = [queue sharedReplyReader];
		_done = dispatch_group_create();
		dispatch_group_enter(_done);
		RgfxCmdBuffer *strongSelf = self;
		RgfxReplyReader *reader = _reader;
		expect = [reader, strongSelf](uint64_t seqno) {
			reader->expectCompletion(seqno, [strongSelf](const RgfxCbComplete &m) {
				@autoreleasepool {
					[strongSelf rgfxDeliverCompletion:&m];
				}
			});
		};
		// While the ring is full: where nothing else takes the queue's completions (this is
		// the delivering thread, or no thread drives the ring), take them here, as the host
		// may be waiting for room on the completion ring before it takes more.
		if (t_delivering > 0 || RgfxCompletionThread::onCurrentThread() || !reader->driven())
			whileWaiting = [reader] { reader->pump(); };
	}

	// Post the command buffer on the queue's submission ring, waiting for room; one too
	// large for the ring goes in a block of the process's own, which the ring names.
	bool posted = false;
	if (_encoder->fitsRing()) {
		posted = _encoder->finish(&_seqno, whileWaiting, expect);
	} else if (RgfxRegion *region = queue.region) {
		RgfxRegion::Block block = region->allocate(_encoder->messageSize());
		if (block.ok) {
			_bodyRegion = region;
			_bodyOffset = block.offset;
			posted = _encoder->finishInRegion(block.ptr, block.offset, &_seqno, whileWaiting,
			                                  expect);
		}
	}
	// The CPU's part ends here, posted or not; the completion may already have come.
	CFTimeInterval unset = 0;
	_kernelEnd.compare_exchange_strong(unset, rgfxHostTime());
	if (!posted) {
		// No block for it: the buffer fails as Metal's does when memory runs out, and its
		// handlers run off the committing thread.
		NSLog(@"RGFX: command buffer 0x%llx: no region block for its %llu bytes: it fails "
		      @"with MTLCommandBufferErrorOutOfMemory",
		      (unsigned long long)[self rgfxObjId],
		      (unsigned long long)_encoder->messageSize());
		RgfxCbComplete failed = {};
		failed.header.type = RGFX_EVT_CB_COMPLETE;
		failed.cb_obj_id = [self rgfxObjId];
		failed.status = RGFX_CB_INTERNAL;
		failed.error_code = MTLCommandBufferErrorOutOfMemory;
		RgfxCmdBuffer *strongSelf = self;
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
			@autoreleasepool {
				[strongSelf rgfxDeliverCompletion:&failed];
			}
		});
	}
}

// Delivers the host's completion for this command buffer, on the thread that took it off
// the ring (the completion thread, or a waiter where none drives the ring): records any
// error, flips the status, then fires the scheduled then the completed handlers (in the
// order added) with this buffer, and lets the waiters go. We have no separate scheduled
// signal, so the scheduled handlers fire here too (a completed buffer was scheduled).
- (void)rgfxDeliverCompletion:(const RgfxCbComplete *)msg
{
	t_delivering++;
	// The surfaces this buffer wrote hold its results before anyone sees it complete.
	rgfxIOSurfaceWriteBack(_surfaceWrites);
	_surfaceWrites.clear();
	// The host is done with the inline bytes: their chunks go back to the queue's pool.
	if (_uploads != nullptr) {
		_uploads->giveBack();
	}
	// The host read the message out of its block before it ran the buffer.
	if (_bodyRegion != nullptr) {
		_bodyRegion->release(_bodyOffset);
		_bodyRegion = nullptr;
	}
	// The queue's slot goes back before the handlers run, so one that makes the next buffer
	// does not wait for its own.
	[self rgfxGiveSlotBack];
	[self rgfxNoteGPUTimes:msg];
	if (msg->status == RGFX_CB_OK) {
		_status.store(MTLCommandBufferStatusCompleted);
	} else {
		// The host preserved the raw MTLCommandBufferError code behind the status; it
		// travels back in the native command-buffer error domain for the guest.
		_cbError = [NSError errorWithDomain:MTLCommandBufferErrorDomain
		                               code:static_cast<NSInteger>(msg->error_code)
		                           userInfo:nil];
		_status.store(MTLCommandBufferStatusError);
	}
	for (MTLCommandBufferHandler block in _scheduledHandlers) {
		block(self);
	}
	[_scheduledHandlers removeAllObjects];
	for (MTLCommandBufferHandler block in _completedHandlers) {
		block(self);
	}
	[_completedHandlers removeAllObjects];
	t_delivering--;
	if (_done != nil && !_left.exchange(true))
		dispatch_group_leave(_done);
	// The objects let go while it ran may go from the host now, after its handlers, which
	// may still name them.
	if (uint64_t ticket = _ticket.exchange(0))
		RgfxRetirement::process().end(ticket);
}

// The GPU's times, in the guest's host time. The device's clock is the host's, not the
// guest's, so the guest's own time anchors them: the run ends when the completion is
// delivered (later than the GPU finished by the delivery's latency), and starts the run's
// length as the device measured it (gpu_end_ns - gpu_start_ns) before that, never before the
// CPU posted the buffer. With no length known (a failed buffer), the run starts as it ends.
- (void)rgfxNoteGPUTimes:(const RgfxCbComplete *)msg
{
	CFTimeInterval end = rgfxHostTime();
	CFTimeInterval unset = 0;
	_kernelEnd.compare_exchange_strong(unset, end);
	CFTimeInterval start = end;
	if (msg->gpu_start_ns != 0 && msg->gpu_end_ns >= msg->gpu_start_ns) {
		start = end - static_cast<CFTimeInterval>(msg->gpu_end_ns - msg->gpu_start_ns) / 1e9;
		start = std::max(start, _kernelEnd.load());
	}
	_gpuStart.store(start);
	_gpuEnd.store(end);
}

- (BOOL)rgfxFinished
{
	MTLCommandBufferStatus status = _status.load();
	return status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError;
}

- (void)addCompletedHandler:(MTLCommandBufferHandler)block
{
	// Invoked with this command buffer when its completion is delivered, in the order
	// added (rgfxDeliverCompletion:).
	if (block != nil) {
		[_completedHandlers addObject:[block copy]];
	}
}

- (void)addScheduledHandler:(MTLCommandBufferHandler)block
{
	if (block != nil) {
		[_scheduledHandlers addObject:[block copy]];
	}
}

// The command-buffer SPI VideoToolbox calls: commits, then waits until the buffer is
// submitted, which here is once it is scheduled (waitUntilScheduled); YES unless it failed.
- (BOOL)commitAndWaitUntilSubmitted
{
	[self commit];
	[self waitUntilScheduled];
	return self.status != MTLCommandBufferStatusError;
}

- (void)waitUntilScheduled
{
	// We have no separate scheduled signal: a buffer is taken as scheduled once its
	// completion lands (its scheduled handlers fire there), so this waits as
	// waitUntilCompleted does.
	[self waitUntilCompleted];
}

- (void)waitUntilCompleted
{
	// Blocks until this buffer's RGFX_EVT_CB_COMPLETE has been delivered: its status
	// flipped and its handlers run, as Metal's does. Without a reader (a unit test
	// without one, or before commit) there is nothing to wait on, so this returns at once.
	if (_reader == nullptr) {
		return;
	}
	// Inside a completion's handlers, the delivering thread is this one: the group would
	// never be left, so the ring is taken here.
	bool delivering = t_delivering > 0 || RgfxCompletionThread::onCurrentThread();
	while (![self rgfxFinished]) {
		if (_reader->driven() && !delivering) {
			// The completion thread takes the ring. A thread that ends (the process's
			// context gone) leaves the reader undriven, so look again each second.
			if (dispatch_group_wait(_done, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC)) == 0)
				return;
			continue;
		}
		_reader->pump();
	}
	// Finished, perhaps on another thread that is still running the handlers.
	if (!delivering)
		dispatch_group_wait(_done, DISPATCH_TIME_FOREVER);
}

- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value
{
	// Command-buffer-scope event waits are not carried yet: no op names one, and no
	// command buffer signals an event (encodeSignalEvent: below), so the buffer runs
	// without waiting. Said in the log, so work that relied on the wait is seen.
	uint64_t n = 0;
	if (logEventOp(g_eventWaits, &n)) {
		NSLog(@"RGFX: command buffer 0x%llx: encodeWaitForEvent 0x%llx value %llu is not "
		      @"carried (no command-buffer event op yet); the buffer runs without the wait "
		      @"(unhonoured waits so far: %llu)",
		      (unsigned long long)[self rgfxObjId], (unsigned long long)rgfxId(event),
		      (unsigned long long)value, (unsigned long long)(n + 1));
	}
}

- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value
{
	// Not carried yet either: a listener or a wait for this value never fires. Logged.
	uint64_t n = 0;
	if (logEventOp(g_eventSignals, &n)) {
		NSLog(@"RGFX: command buffer 0x%llx: encodeSignalEvent 0x%llx value %llu is not "
		      @"carried (no command-buffer event op yet); nothing waiting for it wakes "
		      @"(unhonoured signals so far: %llu)",
		      (unsigned long long)[self rgfxObjId], (unsigned long long)rgfxId(event),
		      (unsigned long long)value, (unsigned long long)(n + 1));
	}
}

- (void)presentDrawable:(id<MTLDrawable>)drawable
{
	// The present path is a later piece (documented stub).
	(void)drawable;
}

- (void)presentDrawable:(id<MTLDrawable>)drawable atTime:(CFTimeInterval)presentationTime
{
	(void)drawable;
	(void)presentationTime;
}

- (void)presentDrawable:(id<MTLDrawable>)drawable afterMinimumDuration:(CFTimeInterval)duration
{
	(void)drawable;
	(void)duration;
}

// --- MTLCommandBuffer state --------------------------------------------------

- (id<MTLCommandQueue>)commandQueue
{
	return _queue;
}
- (BOOL)retainedReferences
{
	return _retained;
}
- (MTLCommandBufferStatus)status
{
	return _status.load();
}
- (NSError *)error
{
	return _cbError;
}
- (MTLCommandBufferErrorOption)errorOptions
{
	return _errorOptions;
}
- (void)rgfxSetErrorOptions:(MTLCommandBufferErrorOption)options
{
	_errorOptions = options;
}
- (id<MTLLogContainer>)logs
{
	return [[RgfxLogContainer alloc] init];
}
// The times (rgfxNoteGPUTimes:), 0 until the buffer has them.
- (CFTimeInterval)kernelStartTime
{
	return _kernelStart.load();
}
- (CFTimeInterval)kernelEndTime
{
	return _kernelEnd.load();
}
- (CFTimeInterval)GPUStartTime
{
	return _gpuStart.load();
}
- (CFTimeInterval)GPUEndTime
{
	return _gpuEnd.load();
}
- (NSString *)label
{
	return _cbLabel;
}
- (void)setLabel:(NSString *)label
{
	_cbLabel = [label copy];
}

// What the client asked of the buffer's content protection, as set.
- (uint64_t)protectionOptions
{
	return _protectionOptions;
}

// The command-buffer SPI WindowServer's CoreDisplay calls before it fills a display surface.
// The device protects no content, so the options are kept and change nothing; options asking
// for protection are logged once.
- (void)setProtectionOptions:(uint64_t)options
{
	_protectionOptions = options;
	static std::atomic<bool> logged{false};
	if (options != 0 && !logged.exchange(true)) {
		NSLog(@"RGFX: a command buffer asks for protection options 0x%llx; the device protects "
		      @"no content",
		      static_cast<unsigned long long>(options));
	}
}

- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}

@end

// =============================================================================
// RgfxRenderCommandEncoder
// =============================================================================

@implementation RgfxRenderCommandEncoder {
	RgfxEncoder *_enc; // borrowed: the command buffer's encoder
	BOOL _emitEnd;
	__weak id<MTLDevice> _device;
	NSString *_encLabel;
	std::shared_ptr<RgfxUploadArena> _uploads; // the command buffer's, for inline bytes
	NSUInteger _tileWidth, _tileHeight;          // the pass descriptor's tile size
}

- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads
{
	_uploads = std::move(uploads);
}

- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height
{
	_tileWidth = width;
	_tileHeight = height;
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc
                        emitEnd:(BOOL)emitEnd
                         device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_emitEnd = emitEnd;
		_device = device;
	}
	return self;
}

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}

- (void)endEncoding
{
	if (_emitEnd) {
		_enc->endEncoder();
	}
}

// --- state -------------------------------------------------------------------

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipelineState
{
	_enc->renderSetRenderPipelineState(rgfxId(pipelineState));
}
- (void)setDepthStencilState:(id<MTLDepthStencilState>)depthStencilState
{
	_enc->renderSetDepthStencilState(rgfxId(depthStencilState));
}
- (void)setViewport:(MTLViewport)viewport
{
	_enc->renderSetViewport(viewport.originX, viewport.originY, viewport.width, viewport.height,
	                        viewport.znear, viewport.zfar);
}
- (void)setViewports:(const MTLViewport *)viewports count:(NSUInteger)count
{
	std::vector<RgfxViewportElem> elems(count);
	for (NSUInteger i = 0; i < count; i++) {
		elems[i].origin_x_bits = f64bits(viewports[i].originX);
		elems[i].origin_y_bits = f64bits(viewports[i].originY);
		elems[i].width_bits = f64bits(viewports[i].width);
		elems[i].height_bits = f64bits(viewports[i].height);
		elems[i].znear_bits = f64bits(viewports[i].znear);
		elems[i].zfar_bits = f64bits(viewports[i].zfar);
	}
	_enc->renderSetViewports(elems.empty() ? nullptr : elems.data(), static_cast<uint32_t>(count));
}
- (void)setScissorRect:(MTLScissorRect)rect
{
	_enc->renderSetScissorRect(rect.x, rect.y, rect.width, rect.height);
}
- (void)setScissorRects:(const MTLScissorRect *)scissorRects count:(NSUInteger)count
{
	std::vector<RgfxScissorRectElem> elems(count);
	for (NSUInteger i = 0; i < count; i++) {
		elems[i].x = scissorRects[i].x;
		elems[i].y = scissorRects[i].y;
		elems[i].width = scissorRects[i].width;
		elems[i].height = scissorRects[i].height;
	}
	_enc->renderSetScissorRects(elems.empty() ? nullptr : elems.data(),
	                            static_cast<uint32_t>(count));
}
- (void)setCullMode:(MTLCullMode)cullMode
{
	_enc->renderSetCullMode(static_cast<uint64_t>(cullMode));
}
- (void)setFrontFacingWinding:(MTLWinding)frontFacingWinding
{
	_enc->renderSetFrontFacingWinding(static_cast<uint64_t>(frontFacingWinding));
}
- (void)setTriangleFillMode:(MTLTriangleFillMode)fillMode
{
	_enc->renderSetTriangleFillMode(static_cast<uint64_t>(fillMode));
}
- (void)setDepthClipMode:(MTLDepthClipMode)depthClipMode
{
	_enc->renderSetDepthClipModeSpi(static_cast<uint64_t>(depthClipMode));
}
- (void)setDepthBias:(float)depthBias slopeScale:(float)slopeScale clamp:(float)clamp
{
	_enc->renderSetDepthBias(depthBias, slopeScale, clamp);
}
- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha
{
	_enc->renderSetBlendColor(red, green, blue, alpha);
}
- (void)setStencilReferenceValue:(uint32_t)referenceValue
{
	_enc->renderSetStencilReferenceValues(referenceValue, referenceValue);
}
- (void)setStencilFrontReferenceValue:(uint32_t)frontReferenceValue
                   backReferenceValue:(uint32_t)backReferenceValue
{
	_enc->renderSetStencilReferenceValues(frontReferenceValue, backReferenceValue);
}
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset
{
	_enc->renderSetVisibilityResultMode(static_cast<uint64_t>(mode), offset);
}
- (void)setVertexAmplificationCount:(NSUInteger)count
                       viewMappings:(const MTLVertexAmplificationViewMapping *)viewMappings
{
	uint32_t n = viewMappings != nullptr ? static_cast<uint32_t>(count) : 0;
	std::vector<RgfxViewMapping> elems(n);
	for (uint32_t i = 0; i < n; i++) {
		elems[i].viewport_array_index_offset = viewMappings[i].viewportArrayIndexOffset;
		elems[i].render_target_array_index_offset = viewMappings[i].renderTargetArrayIndexOffset;
	}
	_enc->renderSetVertexAmplificationCount(elems.empty() ? nullptr : elems.data(), n);
}

// --- store actions -----------------------------------------------------------

- (void)setColorStoreAction:(MTLStoreAction)storeAction atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreAction(static_cast<uint64_t>(storeAction), colorAttachmentIndex);
}
- (void)setDepthStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetDepthStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setStencilStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetStencilStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setColorStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
                           atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreActionOptions(static_cast<uint64_t>(storeActionOptions),
	                                       colorAttachmentIndex);
}
- (void)setStencilStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	_enc->renderSetStencilStoreActionOptions(static_cast<uint64_t>(storeActionOptions));
}

// --- vertex bindings ---------------------------------------------------------

- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetVertexBuffer(rgfxId(buffer), offset, index);
}
- (void)setVertexBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetVertexBufferOffset(offset, index);
}
- (void)setVertexBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
                 offsets:(const NSUInteger *)offsets
               withRange:(NSRange)range
{
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxId(buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetVertexBuffers(range.location, b.empty() ? nullptr : b.data(),
	                             static_cast<uint32_t>(range.length));
}
- (void)setVertexBuffer:(id<MTLBuffer>)buffer
                 offset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	RgfxBufferStrideBinding e = {};
	e.buffer_obj_id = rgfxId(buffer);
	e.offset = offset;
	e.stride = stride;
	_enc->renderSetVertexBuffersAttributeStrides(index, &e, 1);
}
- (void)setVertexBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
                 offsets:(const NSUInteger *)offsets
        attributeStrides:(const NSUInteger *)strides
               withRange:(NSRange)range
{
	std::vector<RgfxBufferStrideBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxId(buffers[i]);
		b[i].offset = offsets[i];
		b[i].stride = strides[i];
	}
	_enc->renderSetVertexBuffersAttributeStrides(range.location, b.empty() ? nullptr : b.data(),
	                                             static_cast<uint32_t>(range.length));
}
- (void)setVertexBufferOffset:(NSUInteger)offset
              attributeStride:(NSUInteger)stride
                      atIndex:(NSUInteger)index
{
	_enc->renderSetVertexBufferOffsetAttributeStride(offset, stride, index);
}
- (void)setVertexTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	RgfxTextureBinding e = {};
	e.texture_obj_id = rgfxId(texture);
	_enc->renderSetVertexTextures(index, &e, 1);
}
- (void)setVertexTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		t[i].texture_obj_id = rgfxId(textures[i]);
	}
	_enc->renderSetVertexTextures(range.location, t.empty() ? nullptr : t.data(),
	                              static_cast<uint32_t>(range.length));
}
- (void)setVertexSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	_enc->renderSetVertexSamplerStates(index, &e, 1);
}
- (void)setVertexSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                     withRange:(NSRange)range
{
	std::vector<RgfxSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
	}
	_enc->renderSetVertexSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                                   static_cast<uint32_t>(range.length));
}
- (void)setVertexSamplerState:(id<MTLSamplerState>)sampler
                  lodMinClamp:(float)lodMinClamp
                  lodMaxClamp:(float)lodMaxClamp
                      atIndex:(NSUInteger)index
{
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetVertexSamplerStatesLodClamps(index, &e, 1);
}
- (void)setVertexSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                  lodMinClamps:(const float *)lodMinClamps
                  lodMaxClamps:(const float *)lodMaxClamps
                     withRange:(NSRange)range
{
	std::vector<RgfxSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetVertexSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                            static_cast<uint32_t>(range.length));
}

// --- fragment bindings -------------------------------------------------------

- (void)setFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetFragmentBuffer(rgfxId(buffer), offset, index);
}
- (void)setFragmentBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetFragmentBufferOffset(offset, index);
}
- (void)setFragmentBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
                   offsets:(const NSUInteger *)offsets
                 withRange:(NSRange)range
{
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxId(buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetFragmentBuffers(range.location, b.empty() ? nullptr : b.data(),
	                               static_cast<uint32_t>(range.length));
}
- (void)setFragmentTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	_enc->renderSetFragmentTexture(rgfxId(texture), index);
}
- (void)setFragmentTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		t[i].texture_obj_id = rgfxId(textures[i]);
	}
	_enc->renderSetFragmentTextures(range.location, t.empty() ? nullptr : t.data(),
	                                static_cast<uint32_t>(range.length));
}
- (void)setFragmentSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	_enc->renderSetFragmentSamplerStates(index, &e, 1);
}
- (void)setFragmentSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                       withRange:(NSRange)range
{
	std::vector<RgfxSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
	}
	_enc->renderSetFragmentSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                                     static_cast<uint32_t>(range.length));
}
- (void)setFragmentSamplerState:(id<MTLSamplerState>)sampler
                    lodMinClamp:(float)lodMinClamp
                    lodMaxClamp:(float)lodMaxClamp
                        atIndex:(NSUInteger)index
{
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetFragmentSamplerStatesLodClamps(index, &e, 1);
}
- (void)setFragmentSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                    lodMinClamps:(const float *)lodMinClamps
                    lodMaxClamps:(const float *)lodMaxClamps
                       withRange:(NSRange)range
{
	std::vector<RgfxSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetFragmentSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                             static_cast<uint32_t>(range.length));
}

// --- tile bindings -----------------------------------------------------------

- (void)setTileBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RgfxBufferBinding e = {};
	e.buffer_obj_id = rgfxId(buffer);
	e.offset = offset;
	_enc->renderSetTileBuffers(index, &e, 1);
}
- (void)setTileBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetTileBufferOffset(offset, index);
}
- (void)setTileBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
               offsets:(const NSUInteger *)offsets
             withRange:(NSRange)range
{
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxId(buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetTileBuffers(range.location, b.empty() ? nullptr : b.data(),
	                           static_cast<uint32_t>(range.length));
}
- (void)setTileTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	RgfxTextureBinding e = {};
	e.texture_obj_id = rgfxId(texture);
	_enc->renderSetTileTextures(index, &e, 1);
}
- (void)setTileTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		t[i].texture_obj_id = rgfxId(textures[i]);
	}
	_enc->renderSetTileTextures(range.location, t.empty() ? nullptr : t.data(),
	                            static_cast<uint32_t>(range.length));
}
- (void)setTileSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	_enc->renderSetTileSamplerStates(index, &e, 1);
}
- (void)setTileSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers withRange:(NSRange)range
{
	std::vector<RgfxSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
	}
	_enc->renderSetTileSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                                 static_cast<uint32_t>(range.length));
}
- (void)setTileSamplerState:(id<MTLSamplerState>)sampler
                lodMinClamp:(float)lodMinClamp
                lodMaxClamp:(float)lodMaxClamp
                    atIndex:(NSUInteger)index
{
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetTileSamplerStatesLodClamps(index, &e, 1);
}
- (void)setTileSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                lodMinClamps:(const float *)lodMinClamps
                lodMaxClamps:(const float *)lodMaxClamps
                   withRange:(NSRange)range
{
	std::vector<RgfxSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetTileSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                         static_cast<uint32_t>(range.length));
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length
                            offset:(NSUInteger)offset
                           atIndex:(NSUInteger)index
{
	_enc->renderSetThreadgroupMemoryLength(length, offset, index);
}
- (void)dispatchThreadsPerTile:(MTLSize)threadsPerTile
{
	_enc->renderDispatchThreadsPerTile(threadsPerTile.width, threadsPerTile.height,
	                                   threadsPerTile.depth);
}

// --- tessellation ------------------------------------------------------------

- (void)setTessellationFactorBuffer:(id<MTLBuffer>)buffer
                             offset:(NSUInteger)offset
                     instanceStride:(NSUInteger)instanceStride
{
	_enc->renderSetTessellationFactorBuffer(rgfxId(buffer), offset, instanceStride);
}
- (void)setTessellationFactorScale:(float)scale
{
	_enc->renderSetTessellationFactorScale(scale);
}

// --- draws -------------------------------------------------------------------

- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
{
	_enc->renderDrawPrimitives(static_cast<uint64_t>(primitiveType), vertexStart, vertexCount);
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
         instanceCount:(NSUInteger)instanceCount
{
	_enc->renderDrawPrimitivesInstances(static_cast<uint64_t>(primitiveType), vertexStart,
	                                    vertexCount, instanceCount);
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
         instanceCount:(NSUInteger)instanceCount
          baseInstance:(NSUInteger)baseInstance
{
	_enc->renderDrawPrimitivesBase(static_cast<uint64_t>(primitiveType), vertexStart, vertexCount,
	                               instanceCount, baseInstance);
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
        indirectBuffer:(id<MTLBuffer>)indirectBuffer
  indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->renderDrawPrimitivesIndirect(static_cast<uint64_t>(primitiveType),
	                                   rgfxId(indirectBuffer), indirectBufferOffset);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
{
	_enc->renderDrawIndexedPrimitives(static_cast<uint64_t>(primitiveType), indexCount,
	                                  static_cast<uint64_t>(indexType), rgfxId(indexBuffer),
	                                  indexBufferOffset);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
                instanceCount:(NSUInteger)instanceCount
{
	_enc->renderDrawIndexedPrimitivesInstances(static_cast<uint64_t>(primitiveType), indexCount,
	                                           static_cast<uint64_t>(indexType), rgfxId(indexBuffer),
	                                           indexBufferOffset, instanceCount);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
                instanceCount:(NSUInteger)instanceCount
                   baseVertex:(NSInteger)baseVertex
                 baseInstance:(NSUInteger)baseInstance
{
	_enc->renderDrawIndexedPrimitivesBase(static_cast<uint64_t>(primitiveType), indexCount,
	                                      static_cast<uint64_t>(indexType), rgfxId(indexBuffer),
	                                      indexBufferOffset, instanceCount,
	                                      static_cast<int64_t>(baseVertex), baseInstance);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
               indirectBuffer:(id<MTLBuffer>)indirectBuffer
         indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->renderDrawIndexedPrimitivesIndirect(static_cast<uint64_t>(primitiveType),
	                                          static_cast<uint64_t>(indexType), rgfxId(indexBuffer),
	                                          indexBufferOffset, rgfxId(indirectBuffer),
	                                          indirectBufferOffset);
}

// --- patches -----------------------------------------------------------------

- (void)drawPatches:(NSUInteger)numberOfPatchControlPoints
                patchStart:(NSUInteger)patchStart
                patchCount:(NSUInteger)patchCount
          patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
    patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
             instanceCount:(NSUInteger)instanceCount
              baseInstance:(NSUInteger)baseInstance
{
	_enc->renderDrawPatches(numberOfPatchControlPoints, patchStart, patchCount,
	                        rgfxId(patchIndexBuffer), patchIndexBufferOffset, instanceCount,
	                        baseInstance);
}
- (void)drawPatches:(NSUInteger)numberOfPatchControlPoints
          patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
    patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
            indirectBuffer:(id<MTLBuffer>)indirectBuffer
      indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->renderDrawPatchesIndirect(numberOfPatchControlPoints, rgfxId(patchIndexBuffer),
	                                patchIndexBufferOffset, rgfxId(indirectBuffer),
	                                indirectBufferOffset);
}
- (void)drawIndexedPatches:(NSUInteger)numberOfPatchControlPoints
                     patchStart:(NSUInteger)patchStart
                     patchCount:(NSUInteger)patchCount
               patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
         patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
        controlPointIndexBuffer:(id<MTLBuffer>)controlPointIndexBuffer
  controlPointIndexBufferOffset:(NSUInteger)controlPointIndexBufferOffset
                  instanceCount:(NSUInteger)instanceCount
                   baseInstance:(NSUInteger)baseInstance
{
	_enc->renderDrawIndexedPatches(numberOfPatchControlPoints, patchStart, patchCount,
	                               rgfxId(patchIndexBuffer), patchIndexBufferOffset,
	                               rgfxId(controlPointIndexBuffer),
	                               controlPointIndexBufferOffset, instanceCount, baseInstance);
}
- (void)drawIndexedPatches:(NSUInteger)numberOfPatchControlPoints
               patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
         patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
        controlPointIndexBuffer:(id<MTLBuffer>)controlPointIndexBuffer
  controlPointIndexBufferOffset:(NSUInteger)controlPointIndexBufferOffset
                 indirectBuffer:(id<MTLBuffer>)indirectBuffer
           indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->renderDrawIndexedPatchesIndirect(numberOfPatchControlPoints, rgfxId(patchIndexBuffer),
	                                       patchIndexBufferOffset, rgfxId(controlPointIndexBuffer),
	                                       controlPointIndexBufferOffset, rgfxId(indirectBuffer),
	                                       indirectBufferOffset);
}

// --- fences / residency / ICB ------------------------------------------------

- (void)updateFence:(id<MTLFence>)fence afterStages:(MTLRenderStages)stages
{
	_enc->renderUpdateFence(rgfxId(fence), static_cast<uint64_t>(stages));
}
- (void)waitForFence:(id<MTLFence>)fence beforeStages:(MTLRenderStages)stages
{
	_enc->renderWaitForFence(rgfxId(fence), static_cast<uint64_t>(stages));
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage
{
	RgfxResourceBinding e = {};
	e.obj_id = rgfxId(resource);
	e.kind = rgfxResourceKind(resource);
	_enc->renderUseResources(&e, 1, static_cast<uint64_t>(usage));
}
- (void)useResources:(const id<MTLResource> __nonnull[__nonnull])resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
{
	std::vector<RgfxResourceBinding> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxId(resources[i]);
		r[i].kind = rgfxResourceKind(resources[i]);
	}
	_enc->renderUseResources(r.empty() ? nullptr : r.data(), static_cast<uint32_t>(count),
	                         static_cast<uint64_t>(usage));
}
- (void)useResource:(id<MTLResource>)resource
              usage:(MTLResourceUsage)usage
             stages:(MTLRenderStages)stages
{
	RgfxResourceBinding e = {};
	e.obj_id = rgfxId(resource);
	e.kind = rgfxResourceKind(resource);
	_enc->renderUseResourcesStages(&e, 1, static_cast<uint64_t>(usage),
	                               static_cast<uint64_t>(stages));
}
- (void)useResources:(const id<MTLResource> __nonnull[__nonnull])resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
              stages:(MTLRenderStages)stages
{
	std::vector<RgfxResourceBinding> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxId(resources[i]);
		r[i].kind = rgfxResourceKind(resources[i]);
	}
	_enc->renderUseResourcesStages(r.empty() ? nullptr : r.data(), static_cast<uint32_t>(count),
	                               static_cast<uint64_t>(usage), static_cast<uint64_t>(stages));
}
- (void)useHeap:(id<MTLHeap>)heap
{
	RgfxHeapBinding e = {};
	e.heap_obj_id = rgfxId(heap);
	_enc->renderUseHeaps(&e, 1);
}
- (void)useHeaps:(const id<MTLHeap> __nonnull[__nonnull])heaps count:(NSUInteger)count
{
	std::vector<RgfxHeapBinding> h(count);
	for (NSUInteger i = 0; i < count; i++) {
		h[i].heap_obj_id = rgfxId(heaps[i]);
	}
	_enc->renderUseHeaps(h.empty() ? nullptr : h.data(), static_cast<uint32_t>(count));
}
- (void)useHeap:(id<MTLHeap>)heap stages:(MTLRenderStages)stages
{
	RgfxHeapBinding e = {};
	e.heap_obj_id = rgfxId(heap);
	_enc->renderUseHeapsStages(&e, 1, static_cast<uint64_t>(stages));
}
- (void)useHeaps:(const id<MTLHeap> __nonnull[__nonnull])heaps
           count:(NSUInteger)count
          stages:(MTLRenderStages)stages
{
	std::vector<RgfxHeapBinding> h(count);
	for (NSUInteger i = 0; i < count; i++) {
		h[i].heap_obj_id = rgfxId(heaps[i]);
	}
	_enc->renderUseHeapsStages(h.empty() ? nullptr : h.data(), static_cast<uint32_t>(count),
	                           static_cast<uint64_t>(stages));
}
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                      withRange:(NSRange)executionRange
{
	_enc->renderExecuteCommandsInBuffer(rgfxId(indirectCommandBuffer), executionRange.location,
	                                    executionRange.length);
}
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandbuffer
                 indirectBuffer:(id<MTLBuffer>)indirectRangeBuffer
           indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->renderExecuteCommandsIndirect(rgfxId(indirectCommandbuffer), rgfxId(indirectRangeBuffer),
	                                    indirectBufferOffset);
}

// --- inline bytes ------------------------------------------------------------
// The wire carries no inline constant data: the bytes go to an upload chunk of the command
// buffer's (RgfxUploads.h), which is bound as a buffer at their offset there.

- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setVertexBuffer:buffer offset:offset atIndex:index];
}
- (void)setVertexBytes:(const void *)bytes
                length:(NSUInteger)length
       attributeStride:(NSUInteger)stride
               atIndex:(NSUInteger)index
{
	id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setVertexBuffer:buffer offset:offset attributeStride:stride atIndex:index];
}
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setFragmentBuffer:buffer offset:offset atIndex:index];
}
- (void)setTileBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setTileBuffer:buffer offset:offset atIndex:index];
}

// --- barriers ------------------------------------------------------------------

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope
                   afterStages:(MTLRenderStages)after
                  beforeStages:(MTLRenderStages)before
{
	_enc->renderMemoryBarrierWithScope(static_cast<uint64_t>(scope), static_cast<uint64_t>(after),
	                                   static_cast<uint64_t>(before));
}

// The wire's render barrier names scopes, not resources: the barrier covers the scope of every
// kind among `resources` (buffers; textures, which also covers them as render targets), a
// superset of the resources, so everything that waited on them still waits.
- (void)memoryBarrierWithResources:(const id<MTLResource> __nonnull[__nonnull])resources
                             count:(NSUInteger)count
                       afterStages:(MTLRenderStages)after
                      beforeStages:(MTLRenderStages)before
{
	MTLBarrierScope scope = 0;
	for (NSUInteger i = 0; i < count; i++) {
		if ([resources[i] conformsToProtocol:@protocol(MTLTexture)])
			scope |= MTLBarrierScopeTextures | MTLBarrierScopeRenderTargets;
		else
			scope |= MTLBarrierScopeBuffers;
	}
	if (scope != 0)
		[self memoryBarrierWithScope:scope afterStages:after beforeStages:before];
}

// What the deprecated texture barrier is: the render targets' writes before later fragment reads.
- (void)textureBarrier
{
	[self memoryBarrierWithScope:MTLBarrierScopeRenderTargets
	                 afterStages:MTLRenderStageFragment
	                beforeStages:MTLRenderStageFragment];
}

// --- the pass's tile size --------------------------------------------------------
// The pass descriptor's, where it named one; where it left the choice to the device, which
// size the host GPU picks is not on the wire: answered 0 and logged.

- (NSUInteger)tileWidth
{
	if (_tileWidth == 0)
		RGFX_UNSUPPORTED("the device's own tile size (no host query on the wire)");
	return _tileWidth;
}
- (NSUInteger)tileHeight
{
	if (_tileHeight == 0)
		RGFX_UNSUPPORTED("the device's own tile size (no host query on the wire)");
	return _tileHeight;
}

// --- unsupported: the device answers NO for these features -----------------------
// A nil or default argument asks for nothing, so it is exact; any other is logged and dropped.

- (void)setDepthStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	if (storeActionOptions != MTLStoreActionOptionNone)
		RGFX_UNSUPPORTED("depth store action options (no op on the wire)");
}
- (void)setDepthTestMinBound:(float)minBound maxBound:(float)maxBound
{
	if (minBound != 0.0f || maxBound != 1.0f)
		RGFX_UNSUPPORTED("depth bounds testing (no op on the wire)");
}
- (void)setColorAttachmentMap:(nullable MTLLogicalToPhysicalColorAttachmentMap *)mapping
{
	if (mapping != nil)
		RGFX_UNSUPPORTED("color attachment mapping (no op on the wire)");
}
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	RGFX_UNSUPPORTED("queue-stage barriers (no op on the wire)");
}
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
	RGFX_UNSUPPORTED("counter sampling");
}

// Mesh shading: no mesh pipeline can be made (newRenderPipelineStateWithMeshDescriptor:), so
// these draw nothing.
- (void)drawMeshThreadgroups:(MTLSize)threadgroupsPerGrid
    threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
      threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	RGFX_UNSUPPORTED("mesh shading");
}
- (void)drawMeshThreads:(MTLSize)threadsPerGrid
    threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
      threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	RGFX_UNSUPPORTED("mesh shading");
}
- (void)drawMeshThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)indirectBufferOffset
                   threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
                     threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	RGFX_UNSUPPORTED("mesh shading");
}

- (void)setObjectBuffer:(nullable id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (buffer != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectBuffers:(const id<MTLBuffer> __nullable[__nonnull])buffers
                offsets:(const NSUInteger[__nonnull])offsets
              withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectSamplerState:(nullable id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	if (sampler != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectSamplerState:(nullable id<MTLSamplerState>)sampler
                  lodMinClamp:(float)lodMinClamp
                  lodMaxClamp:(float)lodMaxClamp
                      atIndex:(NSUInteger)index
{
	if (sampler != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                  lodMinClamps:(const float[__nonnull])lodMinClamps
                  lodMaxClamps:(const float[__nonnull])lodMaxClamps
                     withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                     withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectTexture:(nullable id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	if (texture != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setObjectTextures:(const id<MTLTexture> __nullable[__nonnull])textures withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshBuffer:(nullable id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (buffer != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshBuffers:(const id<MTLBuffer> __nullable[__nonnull])buffers
                offsets:(const NSUInteger[__nonnull])offsets
              withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshSamplerState:(nullable id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	if (sampler != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshSamplerState:(nullable id<MTLSamplerState>)sampler
                  lodMinClamp:(float)lodMinClamp
                  lodMaxClamp:(float)lodMaxClamp
                      atIndex:(NSUInteger)index
{
	if (sampler != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                  lodMinClamps:(const float[__nonnull])lodMinClamps
                  lodMaxClamps:(const float[__nonnull])lodMaxClamps
                     withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                     withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshTexture:(nullable id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	if (texture != nil)
		RGFX_UNSUPPORTED("mesh shading");
}
- (void)setMeshTextures:(const id<MTLTexture> __nullable[__nonnull])textures withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("mesh shading");
}

- (void)setObjectThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("mesh shading");
}

// Ray tracing and function tables: the device makes no acceleration structure, intersection
// or visible function table, so only nil reaches these.
- (void)setVertexAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVertexIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVertexIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVertexVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setVertexVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setFragmentAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setFragmentIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setFragmentIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setFragmentVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setFragmentVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setTileAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setTileIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setTileIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setTileVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setTileVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}

@end

// =============================================================================
// RgfxComputeCommandEncoder
// =============================================================================

@implementation RgfxComputeCommandEncoder {
	RgfxEncoder *_enc;
	__weak id<MTLDevice> _device;
	NSString *_encLabel;
	std::shared_ptr<RgfxUploadArena> _uploads; // the command buffer's, for inline bytes
	MTLDispatchType _dispatchType;              // the pass's; Serial, Metal's default, unless set
}

- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads
{
	_uploads = std::move(uploads);
}

- (void)rgfxSetDispatchType:(MTLDispatchType)dispatchType
{
	_dispatchType = dispatchType;
}

- (MTLDispatchType)dispatchType
{
	return _dispatchType;
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_device = device;
	}
	return self;
}

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}
- (void)endEncoding
{
	_enc->endEncoder();
}

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state
{
	_enc->computeSetComputePipelineState(rgfxId(state));
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RgfxComputeBufferBinding e = {};
	e.buffer_obj_id = rgfxId(buffer);
	e.offset = offset;
	_enc->computeSetBuffers(index, &e, 1);
}
- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->computeSetBufferOffset(offset, index);
}
- (void)setBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
           offsets:(const NSUInteger *)offsets
         withRange:(NSRange)range
{
	std::vector<RgfxComputeBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxId(buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->computeSetBuffers(range.location, b.empty() ? nullptr : b.data(),
	                        static_cast<uint32_t>(range.length));
}
- (void)setBuffer:(id<MTLBuffer>)buffer
          offset:(NSUInteger)offset
 attributeStride:(NSUInteger)stride
         atIndex:(NSUInteger)index
{
	RgfxComputeBufferStrideBinding e = {};
	e.buffer_obj_id = rgfxId(buffer);
	e.offset = offset;
	e.stride = stride;
	_enc->computeSetBuffersAttributeStrides(index, &e, 1);
}
- (void)setBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
           offsets:(const NSUInteger *)offsets
  attributeStrides:(const NSUInteger *)strides
         withRange:(NSRange)range
{
	std::vector<RgfxComputeBufferStrideBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxId(buffers[i]);
		b[i].offset = offsets[i];
		b[i].stride = strides[i];
	}
	_enc->computeSetBuffersAttributeStrides(range.location, b.empty() ? nullptr : b.data(),
	                                        static_cast<uint32_t>(range.length));
}
- (void)setBufferOffset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	_enc->computeSetBufferOffsetAttributeStride(offset, stride, index);
}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	rgfxIOSurfaceNoteShaderWrite(_enc, texture);
	_enc->computeSetTexture(rgfxId(texture), index);
}
- (void)setTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxComputeTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		rgfxIOSurfaceNoteShaderWrite(_enc, textures[i]);
		t[i].texture_obj_id = rgfxId(textures[i]);
	}
	_enc->computeSetTextures(range.location, t.empty() ? nullptr : t.data(),
	                         static_cast<uint32_t>(range.length));
}
- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxComputeSamplerBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	_enc->computeSetSamplerStates(index, &e, 1);
}
- (void)setSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers withRange:(NSRange)range
{
	std::vector<RgfxComputeSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
	}
	_enc->computeSetSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                              static_cast<uint32_t>(range.length));
}
- (void)setSamplerState:(id<MTLSamplerState>)sampler
            lodMinClamp:(float)lodMinClamp
            lodMaxClamp:(float)lodMaxClamp
                atIndex:(NSUInteger)index
{
	RgfxComputeSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxId(sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->computeSetSamplerStatesLodClamps(index, &e, 1);
}
- (void)setSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
            lodMinClamps:(const float *)lodMinClamps
            lodMaxClamps:(const float *)lodMaxClamps
               withRange:(NSRange)range
{
	std::vector<RgfxComputeSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxId(samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->computeSetSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                       static_cast<uint32_t>(range.length));
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	_enc->computeSetThreadgroupMemoryLength(length, index);
}
- (void)setImageblockWidth:(NSUInteger)width height:(NSUInteger)height
{
	_enc->computeSetImageblockWidth(width, height);
}
- (void)setStageInRegion:(MTLRegion)region
{
	_enc->computeSetStageInRegion(region.origin.x, region.origin.y, region.origin.z,
	                              region.size.width, region.size.height, region.size.depth);
}
- (void)setStageInRegionWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                      indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->computeSetStageInRegionIndirect(rgfxId(indirectBuffer), indirectBufferOffset);
}
- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid
       threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	_enc->computeDispatchThreadgroups(threadgroupsPerGrid.width, threadgroupsPerGrid.height,
	                                  threadgroupsPerGrid.depth, threadsPerThreadgroup.width,
	                                  threadsPerThreadgroup.height, threadsPerThreadgroup.depth);
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)indirectBufferOffset
                         threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	_enc->computeDispatchThreadgroupsIndirect(rgfxId(indirectBuffer), indirectBufferOffset,
	                                          threadsPerThreadgroup.width,
	                                          threadsPerThreadgroup.height,
	                                          threadsPerThreadgroup.depth);
}
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	_enc->computeDispatchThreads(threadsPerGrid.width, threadsPerGrid.height, threadsPerGrid.depth,
	                             threadsPerThreadgroup.width, threadsPerThreadgroup.height,
	                             threadsPerThreadgroup.depth);
}
- (void)updateFence:(id<MTLFence>)fence
{
	_enc->computeUpdateFence(rgfxId(fence));
}
- (void)waitForFence:(id<MTLFence>)fence
{
	_enc->computeWaitForFence(rgfxId(fence));
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage
{
	RgfxComputeResourceRef e = {};
	e.obj_id = rgfxId(resource);
	e.kind = rgfxResourceKind(resource);
	_enc->computeUseResources(&e, 1, static_cast<uint64_t>(usage));
}
- (void)useResources:(const id<MTLResource> __nonnull[__nonnull])resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
{
	std::vector<RgfxComputeResourceRef> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxId(resources[i]);
		r[i].kind = rgfxResourceKind(resources[i]);
	}
	_enc->computeUseResources(r.empty() ? nullptr : r.data(), static_cast<uint32_t>(count),
	                          static_cast<uint64_t>(usage));
}
- (void)useHeap:(id<MTLHeap>)heap
{
	RgfxComputeHeapBinding e = {};
	e.heap_obj_id = rgfxId(heap);
	_enc->computeUseHeaps(&e, 1);
}
- (void)useHeaps:(const id<MTLHeap> __nonnull[__nonnull])heaps count:(NSUInteger)count
{
	std::vector<RgfxComputeHeapBinding> h(count);
	for (NSUInteger i = 0; i < count; i++) {
		h[i].heap_obj_id = rgfxId(heaps[i]);
	}
	_enc->computeUseHeaps(h.empty() ? nullptr : h.data(), static_cast<uint32_t>(count));
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope
{
	_enc->computeMemoryBarrierScope(static_cast<uint64_t>(scope));
}
- (void)memoryBarrierWithResources:(const id<MTLResource> __nonnull[__nonnull])resources
                             count:(NSUInteger)count
{
	std::vector<RgfxComputeResourceRef> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxId(resources[i]);
		r[i].kind = rgfxResourceKind(resources[i]);
	}
	_enc->computeMemoryBarrierResources(r.empty() ? nullptr : r.data(),
	                                    static_cast<uint32_t>(count));
}
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                      withRange:(NSRange)executionRange
{
	_enc->computeExecuteCommandsInBuffer(rgfxId(indirectCommandBuffer), executionRange.location,
	                                     executionRange.length);
}
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandbuffer
                 indirectBuffer:(id<MTLBuffer>)indirectRangeBuffer
           indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->computeExecuteCommandsIndirect(rgfxId(indirectCommandbuffer), rgfxId(indirectRangeBuffer),
	                                     indirectBufferOffset);
}

// --- inline bytes ------------------------------------------------------------
// As the render encoder's: an upload chunk of the command buffer's, bound at its offset.

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setBuffer:buffer offset:offset atIndex:index];
}
- (void)setBytes:(const void *)bytes
             length:(NSUInteger)length
    attributeStride:(NSUInteger)stride
            atIndex:(NSUInteger)index
{
	id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setBuffer:buffer offset:offset attributeStride:stride atIndex:index];
}

// --- unsupported: the device answers NO for these features -----------------------
// The device makes no acceleration structure or function table, so only nil reaches these.

- (void)setAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                   atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setIntersectionFunctionTable:(nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                      withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)visibleFunctionTable
                  atBufferIndex:(NSUInteger)bufferIndex
{
	if (visibleFunctionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])visibleFunctionTables
                 withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
	RGFX_UNSUPPORTED("counter sampling");
}
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	RGFX_UNSUPPORTED("queue-stage barriers (no op on the wire)");
}

@end

// =============================================================================
// RgfxBlitCommandEncoder
// =============================================================================

@implementation RgfxBlitCommandEncoder {
	RgfxEncoder *_enc;
	__weak id<MTLDevice> _device;
	NSString *_encLabel;
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_device = device;
	}
	return self;
}

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}
- (void)endEncoding
{
	_enc->endEncoder();
}

- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer
          sourceOffset:(NSUInteger)sourceOffset
              toBuffer:(id<MTLBuffer>)destinationBuffer
     destinationOffset:(NSUInteger)destinationOffset
                  size:(NSUInteger)size
{
	_enc->blitCopyBufferToBuffer(rgfxId(sourceBuffer), rgfxId(destinationBuffer), sourceOffset,
	                             destinationOffset, size);
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer
          sourceOffset:(NSUInteger)sourceOffset
     sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
   sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
            sourceSize:(MTLSize)sourceSize
             toTexture:(id<MTLTexture>)destinationTexture
      destinationSlice:(NSUInteger)destinationSlice
      destinationLevel:(NSUInteger)destinationLevel
     destinationOrigin:(MTLOrigin)destinationOrigin
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyBufferToTexture(rgfxId(sourceBuffer), rgfxId(destinationTexture), sourceOffset,
	                              sourceBytesPerRow, sourceBytesPerImage, sourceSize.width,
	                              sourceSize.height, sourceSize.depth, destinationSlice,
	                              destinationLevel, destinationOrigin.x, destinationOrigin.y,
	                              destinationOrigin.z);
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer
          sourceOffset:(NSUInteger)sourceOffset
     sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
   sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
            sourceSize:(MTLSize)sourceSize
             toTexture:(id<MTLTexture>)destinationTexture
      destinationSlice:(NSUInteger)destinationSlice
      destinationLevel:(NSUInteger)destinationLevel
     destinationOrigin:(MTLOrigin)destinationOrigin
               options:(MTLBlitOption)options
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyBufferToTextureOptions(rgfxId(sourceBuffer), rgfxId(destinationTexture),
	                                     sourceOffset, sourceBytesPerRow, sourceBytesPerImage,
	                                     sourceSize.width, sourceSize.height, sourceSize.depth,
	                                     destinationSlice, destinationLevel, destinationOrigin.x,
	                                     destinationOrigin.y, destinationOrigin.z,
	                                     static_cast<uint64_t>(options));
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
           sourceOrigin:(MTLOrigin)sourceOrigin
             sourceSize:(MTLSize)sourceSize
               toBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
 destinationBytesPerRow:(NSUInteger)destinationBytesPerRow
destinationBytesPerImage:(NSUInteger)destinationBytesPerImage
{
	_enc->blitCopyTextureToBuffer(rgfxId(sourceTexture), rgfxId(destinationBuffer), sourceSlice,
	                              sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
	                              sourceSize.width, sourceSize.height, sourceSize.depth,
	                              destinationOffset, destinationBytesPerRow,
	                              destinationBytesPerImage);
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
           sourceOrigin:(MTLOrigin)sourceOrigin
             sourceSize:(MTLSize)sourceSize
               toBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
 destinationBytesPerRow:(NSUInteger)destinationBytesPerRow
destinationBytesPerImage:(NSUInteger)destinationBytesPerImage
                options:(MTLBlitOption)options
{
	_enc->blitCopyTextureToBufferOptions(rgfxId(sourceTexture), rgfxId(destinationBuffer),
	                                     sourceSlice, sourceLevel, sourceOrigin.x, sourceOrigin.y,
	                                     sourceOrigin.z, sourceSize.width, sourceSize.height,
	                                     sourceSize.depth, destinationOffset, destinationBytesPerRow,
	                                     destinationBytesPerImage, static_cast<uint64_t>(options));
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
           sourceOrigin:(MTLOrigin)sourceOrigin
             sourceSize:(MTLSize)sourceSize
              toTexture:(id<MTLTexture>)destinationTexture
       destinationSlice:(NSUInteger)destinationSlice
       destinationLevel:(NSUInteger)destinationLevel
      destinationOrigin:(MTLOrigin)destinationOrigin
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyTextureRegion(rgfxId(sourceTexture), rgfxId(destinationTexture), sourceSlice,
	                            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
	                            sourceSize.width, sourceSize.height, sourceSize.depth,
	                            destinationSlice, destinationLevel, destinationOrigin.x,
	                            destinationOrigin.y, destinationOrigin.z);
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
              toTexture:(id<MTLTexture>)destinationTexture
       destinationSlice:(NSUInteger)destinationSlice
       destinationLevel:(NSUInteger)destinationLevel
             sliceCount:(NSUInteger)sliceCount
             levelCount:(NSUInteger)levelCount
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyTextureSlices(rgfxId(sourceTexture), rgfxId(destinationTexture), sourceSlice,
	                            sourceLevel, destinationSlice, destinationLevel, sliceCount,
	                            levelCount);
}
// As many whole surfaces as both textures have, as Metal computes them: the level of one
// texture whose size is the other's first level's lines the two up; then the levels both have
// from there, and the slices both have (a cube's six faces each).
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture
{
	NSUInteger sourceLevel = 0, destinationLevel = 0;
	if (!rgfxLevelMatching(sourceTexture, destinationTexture, &sourceLevel)) {
		if (!rgfxLevelMatching(destinationTexture, sourceTexture, &destinationLevel))
			return; // no level of one is the other's size: Metal's validation refuses the copy
	}
	NSUInteger levels = std::min(sourceTexture.mipmapLevelCount - sourceLevel,
	                             destinationTexture.mipmapLevelCount - destinationLevel);
	NSUInteger slices = std::min(rgfxSurfaceSlices(sourceTexture), rgfxSurfaceSlices(destinationTexture));
	[self copyFromTexture:sourceTexture
	          sourceSlice:0
	          sourceLevel:sourceLevel
	            toTexture:destinationTexture
	     destinationSlice:0
	     destinationLevel:destinationLevel
	           sliceCount:slices
	           levelCount:levels];
}
- (void)generateMipmapsForTexture:(id<MTLTexture>)texture
{
	rgfxIOSurfaceNoteWrite(_enc, texture);
	_enc->blitGenerateMipmaps(rgfxId(texture));
}
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value
{
	_enc->blitFillBuffer(rgfxId(buffer), range.location, range.length, value);
}
- (void)updateFence:(id<MTLFence>)fence
{
	_enc->blitUpdateFence(rgfxId(fence));
}
- (void)waitForFence:(id<MTLFence>)fence
{
	_enc->blitWaitForFence(rgfxId(fence));
}
- (void)optimizeContentsForGPUAccess:(id<MTLTexture>)texture
{
	_enc->blitOptimizeForGpu(rgfxId(texture));
}
- (void)optimizeContentsForGPUAccess:(id<MTLTexture>)texture
                               slice:(NSUInteger)slice
                               level:(NSUInteger)level
{
	_enc->blitOptimizeForGpuSlice(rgfxId(texture), slice, level);
}
- (void)optimizeContentsForCPUAccess:(id<MTLTexture>)texture
{
	_enc->blitOptimizeForCpu(rgfxId(texture));
}
- (void)optimizeContentsForCPUAccess:(id<MTLTexture>)texture
                               slice:(NSUInteger)slice
                               level:(NSUInteger)level
{
	_enc->blitOptimizeForCpuSlice(rgfxId(texture), slice, level);
}
- (void)resetCommandsInBuffer:(id<MTLIndirectCommandBuffer>)buffer withRange:(NSRange)range
{
	_enc->blitResetCommandsInBuffer(rgfxId(buffer), range.location, range.length);
}
- (void)copyIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)source
                      sourceRange:(NSRange)sourceRange
                      destination:(id<MTLIndirectCommandBuffer>)destination
                 destinationIndex:(NSUInteger)destinationIndex
{
	_enc->blitCopyIndirectCommandBuffer(rgfxId(source), rgfxId(destination), sourceRange.location,
	                                    sourceRange.length, destinationIndex);
}
- (void)optimizeIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                            withRange:(NSRange)range
{
	_enc->blitOptimizeIndirectCommandBuffer(rgfxId(indirectCommandBuffer), range.location,
	                                        range.length);
}
- (void)synchronizeResource:(id<MTLResource>)resource
{
	_enc->blitSynchronizeResource(rgfxId(resource), rgfxResourceKind(resource));
}
- (void)synchronizeTexture:(id<MTLTexture>)texture slice:(NSUInteger)slice level:(NSUInteger)level
{
	_enc->blitSynchronizeTexture(rgfxId(texture), slice, level);
}

// --- unsupported: the device answers NO for these features -----------------------

- (void)resolveCounters:(id<MTLCounterSampleBuffer>)sampleBuffer
                inRange:(NSRange)range
      destinationBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
{
	RGFX_UNSUPPORTED("counter sampling");
}
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
	RGFX_UNSUPPORTED("counter sampling");
}
// The device makes no tensor (newTensorWithDescriptor:error:), so these copy nothing.
- (void)copyFromTensor:(id<MTLTensor>)sourceTensor
              sourceOrigin:(MTLTensorExtents *)sourceOrigin
          sourceDimensions:(MTLTensorExtents *)sourceDimensions
                  toTensor:(id<MTLTensor>)destinationTensor
         destinationOrigin:(MTLTensorExtents *)destinationOrigin
     destinationDimensions:(MTLTensorExtents *)destinationDimensions
{
	RGFX_UNSUPPORTED("tensors");
}
- (void)copyFromTensor:(id<MTLTensor>)sourceTensor
              sourceOrigin:(MTLTensorExtents *)sourceOrigin
          sourceDimensions:(MTLTensorExtents *)sourceDimensions
               sourcePlane:(MTLTensorPlaneType)sourcePlane
                  toTensor:(id<MTLTensor>)destinationTensor
         destinationOrigin:(MTLTensorExtents *)destinationOrigin
     destinationDimensions:(MTLTensorExtents *)destinationDimensions
          destinationPlane:(MTLTensorPlaneType)destinationPlane
{
	RGFX_UNSUPPORTED("tensors");
}
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	RGFX_UNSUPPORTED("queue-stage barriers (no op on the wire)");
}

@end

// =============================================================================
// RgfxParallelRenderCommandEncoder
// =============================================================================

@implementation RgfxParallelRenderCommandEncoder {
	RgfxEncoder *_enc;
	__weak id<MTLDevice> _device;
	NSString *_encLabel;
	BOOL _ended;
	std::shared_ptr<RgfxUploadArena> _uploads; // the command buffer's, for its sub-encoders
	NSUInteger _tileWidth, _tileHeight;
}

- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads
{
	_uploads = std::move(uploads);
}

- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height
{
	_tileWidth = width;
	_tileHeight = height;
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_device = device;
	}
	return self;
}

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}

- (nullable id<MTLRenderCommandEncoder>)renderCommandEncoder
{
	// Sub-encoders of a parallel pass share the one open render pass; they forward
	// ops to the same RgfxEncoder and do NOT write their own begin/end framing. This
	// encoder writes the single RGFX_OP_ENCODER_END on its own endEncoding.
	RgfxRenderCommandEncoder *sub = [[RgfxRenderCommandEncoder alloc] initWithEncoder:_enc
	                                                                          emitEnd:NO
	                                                                           device:_device];
	[sub rgfxSetUploads:_uploads];
	[sub rgfxSetTileWidth:_tileWidth height:_tileHeight];
	return sub;
}

- (void)endEncoding
{
	if (!_ended) {
		_ended = YES;
		_enc->endEncoder();
	}
}

- (void)setColorStoreAction:(MTLStoreAction)storeAction atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreAction(static_cast<uint64_t>(storeAction), colorAttachmentIndex);
}
- (void)setDepthStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetDepthStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setStencilStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetStencilStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setColorStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
                           atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreActionOptions(static_cast<uint64_t>(storeActionOptions),
	                                       colorAttachmentIndex);
}
- (void)setStencilStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	_enc->renderSetStencilStoreActionOptions(static_cast<uint64_t>(storeActionOptions));
}
- (void)setDepthStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	if (storeActionOptions != MTLStoreActionOptionNone)
		RGFX_UNSUPPORTED("depth store action options (no op on the wire)");
}
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	RGFX_UNSUPPORTED("queue-stage barriers (no op on the wire)");
}

@end
