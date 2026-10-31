/*
 * RgfxAcceleratorUserClient's context selectors (RgfxAccelerator.hpp names them): the Metal
 * process's context, its region blocks, its chunks of texture resource ids and its command
 * queues, all released when the client closes or its process dies (releaseContext): its queues
 * are closed, then its context (the device drops its objects), then its chunks and blocks are
 * freed, so no block or id is handed out again while the device still holds what used it.
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAcceleratorUserClient.hpp"
#include "ReimsGraphics.hpp"
#include "RgfxRegionSpans.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>

namespace {
// How long a pipeline create waits for its reflection: the host compiles it first, which may
// take seconds cold, behind whatever was posted before it.
constexpr UInt32 kReflectionMs = 60000;
// A lifecycle message's obj_id follows its header.
constexpr UInt32 kObjIdOffset = sizeof(RgfxMsgHeader);
} // namespace

IOReturn RgfxAcceleratorUserClient::contextMethod(uint32_t selector,
                                                  IOExternalMethodArguments *a)
{
	if (!owner)
		return kIOReturnNotReady;
	const uint64_t *in = a->scalarInput;
	uint64_t *out = a->scalarOutput;
	switch (selector) {
	case kRgfxAccelMethodOpenContext: {
		if (a->scalarOutputCount != 3)
			return kIOReturnBadArgument;
		IOLockLock(queueLock);
		if (!hasContext) {
			IOReturn r = owner->openContext(&context, &slot);
			if (r != kIOReturnSuccess) {
				IOLockUnlock(queueLock);
				return r;
			}
			hasContext = true;
		}
		out[0] = context;
		out[1] = static_cast<uint64_t>(context) << RGFX_CONTEXT_OBJ_ID_SHIFT;
		out[2] = RGFX_CONTEXT_VA_BASE + static_cast<uint64_t>(slot) * RGFX_CONTEXT_VA_SPAN;
		IOLockUnlock(queueLock);
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodResourceIds: {
		if (a->scalarOutputCount != 1)
			return kIOReturnBadArgument;
		// Under the queue lock, so a chunk is never taken for a context releaseContext has
		// closed: no grant follows its CONTEXT_CLOSE, and no chunk outlives its freeIdChunks.
		IOLockLock(queueLock);
		UInt64 first = 0;
		IOReturn r = hasContext ? owner->takeIdChunk(this, context, &first)
		                        : kIOReturnBadArgument;
		IOLockUnlock(queueLock);
		out[0] = first;
		return r;
	}
	case kRgfxAccelMethodAlloc: {
		if (a->scalarInputCount != 2 || a->scalarOutputCount != 1 || !hasContext)
			return kIOReturnBadArgument;
		UInt64 offset = 0;
		IOReturn r = owner->allocBlock(this, context, in[0], in[1], &offset);
		out[0] = offset;
		return r;
	}
	case kRgfxAccelMethodFree:
		if (a->scalarInputCount != 1)
			return kIOReturnBadArgument;
		return owner->freeBlock(this, in[0]);
	case kRgfxAccelMethodQueueCreate: {
		if (a->scalarInputCount != 6 || a->scalarOutputCount != 1 || !hasContext)
			return kIOReturnBadArgument;
		RgfxQueueCreate open = {};
		open.obj_id = in[0];
		open.submit_offset = in[1];
		open.complete_offset = in[2];
		open.submit_data_size = static_cast<uint32_t>(in[3]);
		open.complete_data_size = static_cast<uint32_t>(in[4]);
		open.priority = static_cast<uint32_t>(in[5]);
		open.vector = RGFX_CONTROL_VECTOR;
		// The queue's id is in the context's range, and each of its rings lies in one of the
		// client's own blocks, so a process names nothing of another's. The blocks stay pinned
		// until the device has answered the create, so neither is freed under it.
		if ((open.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) != context)
			return kIOReturnBadArgument;
		RgfxBlockPin pins[2];
		if (!owner->pinRange(this, open.submit_offset,
		                     RGFX_RING_HEADER_SIZE + static_cast<uint64_t>(open.submit_data_size),
		                     &pins[0]))
			return kIOReturnBadArgument;
		if (!owner->pinRange(this, open.complete_offset,
		                     RGFX_RING_HEADER_SIZE + static_cast<uint64_t>(open.complete_data_size),
		                     &pins[1])) {
			owner->unpinBlock(pins[0]);
			return kIOReturnBadArgument;
		}
		auto unpin = [&] {
			owner->unpinBlock(pins[0]);
			owner->unpinBlock(pins[1]);
		};
		IOLockLock(queueLock);
		if (queueCount == kMaxQueues) {
			IOLockUnlock(queueLock);
			unpin();
			return kIOReturnNoResources;
		}
		// Watched before the device has it, so no completion comes before the watch.
		IOReturn watch = owner->watchQueue(this, open.obj_id, open.complete_offset);
		if (watch != kIOReturnSuccess) {
			IOLockUnlock(queueLock);
			unpin();
			return watch;
		}
		UInt32 status = owner->openQueue(open);
		if (status == RGFX_STATUS_OK)
			queues[queueCount++] = open.obj_id;
		else
			owner->unwatchQueue(this, open.obj_id);
		IOLockUnlock(queueLock);
		unpin();
		out[0] = status;
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodQueueDestroy: {
		if (a->scalarInputCount != 1)
			return kIOReturnBadArgument;
		IOLockLock(queueLock);
		for (UInt32 i = 0; i < queueCount; i++) {
			if (queues[i] == in[0]) {
				owner->unwatchQueue(this, in[0]);
				owner->closeQueue(in[0]);
				queues[i] = queues[--queueCount];
				break;
			}
		}
		IOLockUnlock(queueLock);
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodSubmit:
		return submitMethod(a);
	case kRgfxAccelMethodWaitCompletions:
		return waitCompletionsMethod(a);
	case kRgfxAccelMethodSurfaceScanout: {
		if (a->scalarInputCount != 2 || a->scalarOutputCount != 5 || !hasContext)
			return kIOReturnBadArgument;
		// The surface's one buffer.
		if (in[0] == 0 || in[0] > UINT32_MAX || in[1] != 0)
			return kIOReturnBadArgument;
		RgfxAccelerator::Scanout scanout = {};
		IOReturn r = owner->grantSurfaceScanout(this, context, task, static_cast<UInt32>(in[0]),
		                                        &scanout);
		if (r != kIOReturnSuccess)
			return r;
		a->scalarOutput[0] = scanout.offset;
		a->scalarOutput[1] = scanout.length;
		a->scalarOutput[2] = scanout.width;
		a->scalarOutput[3] = scanout.height;
		a->scalarOutput[4] = scanout.rowBytes;
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodSurfaceBlock: {
		if (a->scalarInputCount != 3 || a->scalarOutputCount != 4)
			return kIOReturnBadArgument;
		// Under the queue lock, so no hold is made for a context releaseContext has closed:
		// every hold of the client's goes in its dropSurfaceBlocks.
		IOLockLock(queueLock);
		RgfxAccelerator::SurfaceBlock block = {};
		IOReturn r = hasContext ? owner->acquireSurfaceBlock(this, context, task, in[0], in[1],
		                                                     in[2], &block)
		                        : kIOReturnBadArgument;
		IOLockUnlock(queueLock);
		if (r != kIOReturnSuccess)
			return r;
		out[0] = block.offset;
		out[1] = block.size;
		out[2] = block.serial;
		out[3] = block.fresh ? 1 : 0;
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodSurfaceBlockRelease:
		if (a->scalarInputCount != 1)
			return kIOReturnBadArgument;
		return owner->releaseSurfaceBlock(this, in[0]);
	default:
		return kIOReturnUnsupported;
	}
}

// Whether a LIBRARY_FROM_REGION message is its library's last chunk, the one the device answers:
// its end reaches the library's length, or overflows. A message too short to be one is not.
static bool lastLibraryChunk(const UInt8 *message, UInt64 size)
{
	RgfxLibraryFromRegion chunk;
	if (size < sizeof(chunk))
		return false;
	memcpy(&chunk, message, sizeof(chunk));
	const UInt64 end = chunk.chunk_offset + chunk.byte_length;
	return end < chunk.chunk_offset || end >= chunk.total_length;
}

// kRgfxAccelMethodSubmit (RgfxAccelerator.hpp): one create or destroy of the client's onto the
// control ring, and the device's answer back where the create has one: a pipeline's reflection,
// a library create's or lookup's COMPLETE.
IOReturn RgfxAcceleratorUserClient::submitMethod(IOExternalMethodArguments *a)
{
	if (!hasContext || a->scalarOutputCount != 1)
		return kIOReturnBadArgument;
	IOMemoryDescriptor *desc = a->structureInputDescriptor;
	UInt64 size = desc ? desc->getLength() : a->structureInputSize;
	if (size < kObjIdOffset + sizeof(UInt64) || size % 8 ||
	    size > ReimsGraphics::kLifecycleMessageMax)
		return kIOReturnBadArgument;
	UInt8 *message = static_cast<UInt8 *>(IOMalloc(size));
	if (!message)
		return kIOReturnNoMemory;
	IOReturn r = kIOReturnSuccess;
	if (desc) {
		r = desc->prepare(kIODirectionOut);
		if (r == kIOReturnSuccess) {
			if (desc->readBytes(0, message, size) != size)
				r = kIOReturnBadArgument;
			desc->complete(kIODirectionOut);
		}
	} else {
		memcpy(message, a->structureInput, size);
	}
	UInt32 type = 0;
	UInt64 objId = 0;
	memcpy(&type, message, sizeof(type));
	memcpy(&objId, message + kObjIdOffset, sizeof(objId));
	// Creates and destroys of the client's own objects only: QUEUE_CREATE goes through its own
	// selector (its rings are checked), CONTEXT_CLOSE is the kext's.
	if (r == kIOReturnSuccess &&
	    (type <= RGFX_CMD_QUEUE_CREATE || type >= RGFX_CMD_CONTEXT_CLOSE ||
	     (objId >> RGFX_CONTEXT_OBJ_ID_SHIFT) != context))
		r = kIOReturnBadArgument;
	// The region the message names lies in the client's own blocks: a process places its heaps,
	// buffers and library chunks in no one else's memory, nor in the rings or the scanout. Each
	// block stays pinned until the message is posted and answered, so none is freed, and given
	// to another process, between this check and the device carrying the message out.
	RgfxRegionSpan spans[kRgfxMaxRegionSpans];
	RgfxBlockPin pins[kRgfxMaxRegionSpans];
	int pinned = 0;
	int spanCount = r == kIOReturnSuccess ? rgfxRegionSpans(type, message, size, spans) : 0;
	if (spanCount < 0)
		r = kIOReturnBadArgument;
	for (int i = 0; i < spanCount && r == kIOReturnSuccess; i++) {
		if (owner->pinRange(this, spans[i].offset, spans[i].length, &pins[pinned]))
			pinned++;
		else
			r = kIOReturnBadArgument;
	}
	// The answers the client waits for: a pipeline create's reflection, and a library create's
	// or lookup's COMPLETE (a library's last region chunk is its create).
	UInt32 replyType = 0;
	if (type == RGFX_CMD_RENDER_PIPELINE_CREATE || type == RGFX_CMD_COMPUTE_PIPELINE_CREATE)
		replyType = RGFX_EVT_PIPELINE_REFLECTION;
	else if (type == RGFX_CMD_LIBRARY_CREATE || type == RGFX_CMD_LIBRARY_LOOKUP ||
	         (type == RGFX_CMD_LIBRARY_FROM_REGION && lastLibraryChunk(message, size)))
		replyType = RGFX_EVT_COMPLETE;
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 replyLength = 0;
	UInt64 seqno = 0;
	if (r == kIOReturnSuccess)
		r = owner->postLifecycle(type, message, static_cast<UInt32>(size), replyType, reply,
		                         sizeof(reply), &replyLength, &seqno, kReflectionMs);
	IOFree(message, size);
	if (r == kIOReturnTimeout) {
		// Posted, but not answered in time: the client takes the create as failed, and may make
		// it again under a new id. Destroy this one, so the device keeps nothing of it: the
		// destroy follows the create on the ring, and cancels it where it is still being made.
		RgfxObjectDestroy destroy = {};
		destroy.header.type = RGFX_CMD_OBJECT_DESTROY;
		destroy.obj_id = objId;
		owner->postLifecycle(RGFX_CMD_OBJECT_DESTROY, &destroy, sizeof(destroy), 0, nullptr, 0,
		                     nullptr, nullptr, 0);
	}
	// A free of a block from here posts its release behind the message (and its destroy).
	for (int i = 0; i < pinned; i++)
		owner->unpinBlock(pins[i]);
	if (r != kIOReturnSuccess)
		return r;
	a->scalarOutput[0] = seqno;
	UInt32 n = replyLength < a->structureOutputSize ? replyLength : a->structureOutputSize;
	if (n && a->structureOutput)
		memcpy(a->structureOutput, reply, n);
	a->structureOutputSize = n;
	return kIOReturnSuccess;
}

// kRgfxAccelMethodWaitCompletions (RgfxAccelerator.hpp): the bundle's completion thread
// sleeps here, interruptibly, until a completion ring of the client's holds records for it.
IOReturn RgfxAcceleratorUserClient::waitCompletionsMethod(IOExternalMethodArguments *a)
{
	if (a->scalarInputCount != 1 || a->scalarOutputCount != 1)
		return kIOReturnBadArgument;
	IOReturn r = kIOReturnSuccess;
	IOLockLock(completionLock);
	while (completionCount == a->scalarInput[0] && !completionsClosed) {
		if (IOLockSleep(completionLock, &completionCount, THREAD_ABORTSAFE) != THREAD_AWAKENED) {
			r = kIOReturnAborted;
			break;
		}
	}
	if (completionsClosed)
		r = kIOReturnNotOpen;
	a->scalarOutput[0] = completionCount;
	IOLockUnlock(completionLock);
	return r;
}

void RgfxAcceleratorUserClient::wakeCompletions()
{
	IOLockLock(completionLock);
	completionCount++;
	IOLockWakeup(completionLock, &completionCount, false);
	IOLockUnlock(completionLock);
}

void RgfxAcceleratorUserClient::releaseContext()
{
	if (!queueLock)
		return;
	// The completion waiters return, and no ring of the client's wakes it any more.
	IOLockLock(completionLock);
	completionsClosed = true;
	IOLockWakeup(completionLock, &completionCount, false);
	IOLockUnlock(completionLock);
	IOLockLock(queueLock);
	RgfxAccelerator *accel = owner;
	// Under the queue lock, which a QUEUE_CREATE watches its ring under, so none is left.
	if (accel)
		accel->unwatchClient(this);
	if (!hasContext || !accel) {
		IOLockUnlock(queueLock);
		return;
	}
	for (UInt32 i = 0; i < queueCount; i++)
		accel->closeQueue(queues[i]);
	queueCount = 0;
	// The blocks go only once the device has answered the close: it then holds nothing of
	// them, and the context's command buffers have finished.
	bool closed = accel->closeContext(context, slot);
	accel->freeIdChunks(this);
	if (closed)
		accel->freeBlocks(this);
	else
		accel->keepBlocks(this);
	accel->dropSurfaceBlocks(this, closed);
	accel->revokeGrants(this);
	hasContext = false;
	IOLockUnlock(queueLock);
}
