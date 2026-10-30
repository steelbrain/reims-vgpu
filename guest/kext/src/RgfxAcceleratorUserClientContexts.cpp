/*
 * RgfxAcceleratorUserClient's context selectors (RgfxAccelerator.hpp names them): the Metal
 * process's context, its region blocks and its command queues, all released when the client
 * closes or its process dies (releaseContext): its queues are closed, then its context (the
 * device drops its objects), then its blocks are freed, so no block is handed out again while the
 * device still reads a ring in it.
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAcceleratorUserClient.hpp"
#include "ReimsGraphics.hpp"

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
		if (a->scalarOutputCount != 4)
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
		out[3] = 1 + static_cast<uint64_t>(slot) * RGFX_CONTEXT_RESOURCE_IDS;
		IOLockUnlock(queueLock);
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodAlloc: {
		if (a->scalarInputCount != 2 || a->scalarOutputCount != 1 || !hasContext)
			return kIOReturnBadArgument;
		UInt64 offset = 0;
		IOReturn r = owner->allocBlock(this, in[0], in[1], &offset);
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
		// client's own blocks, so a process names nothing of another's.
		if ((open.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) != context ||
		    !owner->ownsRange(this, open.submit_offset,
		                      RGFX_RING_HEADER_SIZE + static_cast<uint64_t>(open.submit_data_size)) ||
		    !owner->ownsRange(this, open.complete_offset,
		                      RGFX_RING_HEADER_SIZE +
		                          static_cast<uint64_t>(open.complete_data_size)))
			return kIOReturnBadArgument;
		IOLockLock(queueLock);
		if (queueCount == kMaxQueues) {
			IOLockUnlock(queueLock);
			return kIOReturnNoResources;
		}
		UInt32 status = owner->openQueue(open);
		if (status == RGFX_STATUS_OK)
			queues[queueCount++] = open.obj_id;
		IOLockUnlock(queueLock);
		out[0] = status;
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodQueueDestroy: {
		if (a->scalarInputCount != 1)
			return kIOReturnBadArgument;
		IOLockLock(queueLock);
		for (UInt32 i = 0; i < queueCount; i++) {
			if (queues[i] == in[0]) {
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
	default:
		return kIOReturnUnsupported;
	}
}

// kRgfxAccelMethodSubmit (RgfxAccelerator.hpp): one create or destroy of the client's onto the
// control ring, and a pipeline create's reflection back.
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
	UInt32 replyType = 0;
	if (type == RGFX_CMD_RENDER_PIPELINE_CREATE || type == RGFX_CMD_COMPUTE_PIPELINE_CREATE)
		replyType = RGFX_EVT_PIPELINE_REFLECTION;
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 replyLength = 0;
	UInt64 seqno = 0;
	if (r == kIOReturnSuccess)
		r = owner->postLifecycle(type, message, static_cast<UInt32>(size), replyType, reply,
		                         sizeof(reply), &replyLength, &seqno, kReflectionMs);
	IOFree(message, size);
	if (r != kIOReturnSuccess)
		return r;
	a->scalarOutput[0] = seqno;
	UInt32 n = replyLength < a->structureOutputSize ? replyLength : a->structureOutputSize;
	if (n && a->structureOutput)
		memcpy(a->structureOutput, reply, n);
	a->structureOutputSize = n;
	return kIOReturnSuccess;
}

void RgfxAcceleratorUserClient::releaseContext()
{
	if (!queueLock)
		return;
	IOLockLock(queueLock);
	RgfxAccelerator *accel = owner;
	if (!hasContext || !accel) {
		IOLockUnlock(queueLock);
		return;
	}
	for (UInt32 i = 0; i < queueCount; i++)
		accel->closeQueue(queues[i]);
	queueCount = 0;
	accel->closeContext(context, slot);
	accel->freeBlocks(this);
	hasContext = false;
	IOLockUnlock(queueLock);
}
