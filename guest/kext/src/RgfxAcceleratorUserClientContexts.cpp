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

#include <IOKit/IOLib.h>

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
		if (a->scalarInputCount != 7 || a->scalarOutputCount != 1 || !hasContext)
			return kIOReturnBadArgument;
		RgfxQueueOpen open = {};
		open.obj_id = in[0];
		open.submit_offset = in[1];
		open.complete_offset = in[2];
		open.submit_data_size = static_cast<uint32_t>(in[3]);
		open.complete_data_size = static_cast<uint32_t>(in[4]);
		open.priority = static_cast<uint32_t>(in[5]);
		open.vector = RGFX_CONTROL_VECTOR;
		open.context = context;
		open.flags = static_cast<uint32_t>(in[6]) & RGFX_QUEUE_FLAG_DEVICE;
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
	default:
		return kIOReturnUnsupported;
	}
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
