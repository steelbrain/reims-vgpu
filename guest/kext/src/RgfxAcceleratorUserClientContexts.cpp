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
#include "RgfxSubmitGate.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <sys/proc.h>

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
		if (a->scalarOutputCount != 5)
			return kIOReturnBadArgument;
		IOLockLock(queueLock);
		if (!hasContext) {
			// The window first: a context with no window is never opened.
			IOReturn r = openWindow();
			if (r == kIOReturnSuccess) {
				r = owner->openContext(&context, &slot);
				if (r != kIOReturnSuccess)
					closeWindow();
			}
			if (r != kIOReturnSuccess) {
				IOLockUnlock(queueLock);
				return r;
			}
			hasContext = true;
		}
		out[0] = context;
		out[1] = static_cast<uint64_t>(context) << RGFX_CONTEXT_OBJ_ID_SHIFT;
		out[2] = RGFX_CONTEXT_VA_BASE + static_cast<uint64_t>(slot) * RGFX_CONTEXT_VA_SPAN;
		IOLockLock(windowLock);
		out[3] = windowBase;
		out[4] = windowSize;
		IOLockUnlock(windowLock);
		IOLockUnlock(queueLock);
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodDeviceInfo:
		return deviceInfoMethod(a);
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
		destroyQueueIfTracked(in[0]);
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
		// Three inputs ask for a copy block; a fourth of 1, for one in place; a fifth and a
		// sixth name the surface and whether it may be cached. A fifth output says whether it
		// is.
		const UInt32 inputs = a->scalarInputCount;
		if ((inputs < 3 || inputs > 6 || inputs == 5) ||
		    (a->scalarOutputCount != 4 && a->scalarOutputCount != 5))
			return kIOReturnBadArgument;
		bool inPlace = inputs >= 4 && in[3] != 0;
		if (inputs >= 4 && in[3] > 1)
			return kIOReturnBadArgument;
		if (inputs == 6 && (in[4] > UINT32_MAX || in[5] > 1))
			return kIOReturnBadArgument;
		const uint32_t surfaceId = inputs == 6 ? static_cast<uint32_t>(in[4]) : 0;
		const bool cacheable = inputs == 6 && in[5] != 0;
		// Under the queue lock, so no hold is made for a context releaseContext has closed:
		// every hold of the client's goes in its dropSurfaceBlocks.
		IOLockLock(queueLock);
		RgfxAccelerator::SurfaceBlock block = {};
		IOReturn r = hasContext ? owner->acquireSurfaceBlock(this, context, task, in[0], in[1],
		                                                     in[2], inPlace, surfaceId,
		                                                     cacheable, &block)
		                        : kIOReturnBadArgument;
		IOLockUnlock(queueLock);
		if (r != kIOReturnSuccess)
			return r;
		out[0] = block.offset;
		out[1] = block.size;
		out[2] = block.serial;
		out[3] = block.fresh ? 1 : 0;
		if (a->scalarOutputCount == 5)
			out[4] = block.cached ? 1 : 0;
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodSurfaceBlockRelease:
		if (a->scalarInputCount != 1)
			return kIOReturnBadArgument;
		return owner->releaseSurfaceBlock(this, in[0]);
	case kRgfxAccelMethodSurfaceWrites:
		return surfaceWritesMethod(a);
	case kRgfxAccelMethodSurfaceRetire:
		if (a->scalarInputCount != 2 || !ownsQueue(in[0]))
			return kIOReturnBadArgument;
		owner->retireSurfaceWrites(in[0], in[1]);
		return kIOReturnSuccess;
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
// a library create's or lookup's COMPLETE, an argument layout's COMPLETE.
IOReturn RgfxAcceleratorUserClient::submitMethod(IOExternalMethodArguments *a)
{
	// The context as it is now, read under the queue lock that releaseContext closes it under,
	// to tell after the post whether it closed meanwhile (rgfxSubmitDestroysBehind).
	IOLockLock(queueLock);
	const bool open = hasContext;
	const UInt32 ctx = context;
	IOLockUnlock(queueLock);
	if (!open || a->scalarOutputCount != 1)
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
	// selector (its rings are checked), CONTEXT_CLOSE is the kext's (RgfxSubmitGate.hpp).
	if (r == kIOReturnSuccess &&
	    (!rgfxSubmitAdmitted(type) || (objId >> RGFX_CONTEXT_OBJ_ID_SHIFT) != ctx))
		r = kIOReturnBadArgument;
	// An OBJECT_DESTROY of one of the client's command queues is the one accepted message that
	// changes the kext's own tracking: the kext watches the queue's completion ring and lists it.
	// Route it the way QUEUE_DESTROY goes (stop watching, drop it, post the destroy once), so the
	// kext never keeps watching a queue the host has torn down, nor scans a completion ring whose
	// block the client then frees and another process takes. A destroy of any other object names
	// one only the host tracks, and is posted as it is.
	if (r == kIOReturnSuccess && rgfxSubmitRoutesQueueDestroy(type) && destroyQueueIfTracked(objId)) {
		IOFree(message, size);
		a->scalarOutput[0] = 0;
		a->structureOutputSize = 0;
		return kIOReturnSuccess;
	}
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
	// The answers the client waits for: a pipeline create's reflection, a heap texture size
	// query's SIZE_AND_ALIGN, a sampler create's SAMPLER_STATE (its resource id), and the
	// COMPLETE of a library create or lookup (a library's last region chunk is its create) or of
	// an argument layout, whose layout the device writes into the client's block before it.
	UInt32 replyType = 0;
	if (type == RGFX_CMD_RENDER_PIPELINE_CREATE || type == RGFX_CMD_COMPUTE_PIPELINE_CREATE)
		replyType = RGFX_EVT_PIPELINE_REFLECTION;
	else if (type == RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN)
		replyType = RGFX_EVT_SIZE_AND_ALIGN;
	else if (type == RGFX_CMD_SAMPLER_CREATE)
		replyType = RGFX_EVT_SAMPLER_STATE;
	else if (type == RGFX_CMD_LIBRARY_CREATE || type == RGFX_CMD_LIBRARY_LOOKUP ||
	         type == RGFX_CMD_ARGUMENT_LAYOUT ||
	         (type == RGFX_CMD_LIBRARY_FROM_REGION && lastLibraryChunk(message, size)))
		replyType = RGFX_EVT_COMPLETE;
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 replyLength = 0;
	UInt64 seqno = 0;
	const bool checked = r == kIOReturnSuccess;
	if (checked)
		r = owner->postLifecycle(type, message, static_cast<UInt32>(size), replyType, reply,
		                         sizeof(reply), &replyLength, &seqno, kReflectionMs);
	IOFree(message, size);
	// Posted where postLifecycle answered or timed out waiting for the answer. Whether the context
	// closed since the check: its CONTEXT_CLOSE may then have gone before this create, which the
	// device's close then never dropped.
	const bool posted = checked && (r == kIOReturnSuccess || r == kIOReturnTimeout);
	IOLockLock(queueLock);
	const bool closedSince = !hasContext || context != ctx;
	IOLockUnlock(queueLock);
	if (rgfxSubmitDestroysBehind(type, posted, r == kIOReturnTimeout, closedSince)) {
		// Not answered in time (the client takes the create as failed, and may make it again
		// under a new id), or made for a context that has closed: destroy it, so the device keeps
		// nothing of it. The destroy follows the create on the ring, and cancels it where it is
		// still being made. A query or a write into an object makes nothing, so nothing follows.
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

// Whether `queue` is one of the client's open queues.
bool RgfxAcceleratorUserClient::ownsQueue(UInt64 queue)
{
	IOLockLock(queueLock);
	bool owns = false;
	for (UInt32 i = 0; i < queueCount && !owns; i++)
		owns = queues[i] == queue;
	IOLockUnlock(queueLock);
	return owns;
}

bool RgfxAcceleratorUserClient::destroyQueueIfTracked(UInt64 objId)
{
	bool found = false;
	// The whole find, unwatch, close and remove under the queue lock, which QUEUE_CREATE watches
	// and lists a queue under, so a create of this id cannot slip between the check and the post.
	IOLockLock(queueLock);
	for (UInt32 i = 0; i < queueCount; i++) {
		if (queues[i] == objId) {
			owner->unwatchQueue(this, objId);
			owner->closeQueue(objId);
			queues[i] = queues[--queueCount];
			found = true;
			break;
		}
	}
	IOLockUnlock(queueLock);
	return found;
}

// kRgfxAccelMethodSurfaceWrites (RgfxAccelerator.hpp): the surfaces a command buffer of one of the
// client's queues writes, whose CPU locks wait for it from here.
IOReturn RgfxAcceleratorUserClient::surfaceWritesMethod(IOExternalMethodArguments *a)
{
	if (!hasContext || a->scalarInputCount != 0 || a->scalarOutputCount != 0)
		return kIOReturnBadArgument;
	IOMemoryDescriptor *desc = a->structureInputDescriptor;
	UInt64 size = desc ? desc->getLength() : a->structureInputSize;
	if (size < 3 * sizeof(UInt64) || size % sizeof(UInt64) || size > kMaxSurfaceWritesBytes)
		return kIOReturnBadArgument;
	UInt64 *words = static_cast<UInt64 *>(IOMalloc(size));
	if (!words)
		return kIOReturnNoMemory;
	IOReturn r = kIOReturnSuccess;
	if (desc) {
		r = desc->prepare(kIODirectionOut);
		if (r == kIOReturnSuccess) {
			if (desc->readBytes(0, words, size) != size)
				r = kIOReturnBadArgument;
			desc->complete(kIODirectionOut);
		}
	} else {
		memcpy(words, a->structureInput, size);
	}
	const UInt64 count = r == kIOReturnSuccess ? words[2] : 0;
	if (r == kIOReturnSuccess && (count != (size / sizeof(UInt64) - 3) / 2 ||
	                              size != (3 + 2 * count) * sizeof(UInt64) || !ownsQueue(words[0])))
		r = kIOReturnBadArgument;
	if (r == kIOReturnSuccess) {
		owner->noteSurfaceWrites(this, task, words[0], words[1], words + 3,
		                         static_cast<UInt32>(count));
	} else {
		// A refused naming leaves its surfaces' locks waiting for nothing: the first few say so.
		static UInt32 refusalsLogged;
		if (__atomic_fetch_add(&refusalsLogged, 1, __ATOMIC_RELAXED) < 8) {
			char name[32] = {};
			proc_name(proc_selfpid(), name, sizeof(name));
			IOLog("RgfxAccelerator: %s[%d]'s naming of a command buffer's surface writes "
			      "(%llu bytes) is refused (0x%x): their locks do not wait for them\n",
			      name, proc_selfpid(), size, r);
		}
	}
	IOFree(words, size);
	return r;
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
	// The process reaches nothing of the region from here on: its window goes before anything
	// it held is freed, and given to another process.
	closeWindow();
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
	// The device dropped this context's objects at its close: a block whose release it refused
	// because an object lay in it may now be free to go.
	if (closed)
		accel->retryRefusedBlocks();
}

// kRgfxAccelMethodDeviceInfo (RgfxAccelerator.hpp): the device info the host wrote, read from
// the region for the process, which maps no page of it; with the video caps after it
// (RGFX_VIDEO_CAPS_OFFSET) where the output has room for both.
IOReturn RgfxAcceleratorUserClient::deviceInfoMethod(IOExternalMethodArguments *a)
{
	// Larger than a message's inline bytes, it comes out through a descriptor, which IOKit
	// sizes apart from the inline output.
	const UInt64 outSize = a->structureOutputDescriptor ? a->structureOutputDescriptorSize
	                                                    : a->structureOutputSize;
	static_assert(RGFX_VIDEO_CAPS_OFFSET == RGFX_DEVICE_INFO_OFFSET + sizeof(RgfxDeviceInfo),
	              "the video caps follow the device info");
	const UInt64 size = sizeof(RgfxDeviceInfo) + sizeof(RgfxVideoCaps);
	if (a->scalarInputCount != 0 || a->scalarOutputCount != 0 || a->structureInputSize != 0 ||
	    (outSize != sizeof(RgfxDeviceInfo) && outSize != size))
		return kIOReturnBadArgument;
	IODeviceMemory *region = owner->copyRegion();
	if (!region)
		return kIOReturnNotReady;
	// Its pages, mapped into the kernel write-back as the framebuffer maps the rings: the
	// region is BAR memory, which a physical copy does not read.
	const UInt64 page = RGFX_DEVICE_INFO_OFFSET & ~0xfffull;
	const UInt64 length = ((RGFX_DEVICE_INFO_OFFSET + outSize + 0xfffull) & ~0xfffull) - page;
	IOMemoryMap *map = region->createMappingInTask(
	    kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache | kIOMapReadOnly, page, length);
	region->release();
	if (!map)
		return kIOReturnNoResources;
	const void *info =
	    reinterpret_cast<const UInt8 *>(map->getVirtualAddress()) + (RGFX_DEVICE_INFO_OFFSET - page);
	// The descriptor where there is one: the inline buffer is there too, but holds less.
	IOReturn r = kIOReturnSuccess;
	if (IOMemoryDescriptor *desc = a->structureOutputDescriptor) {
		r = desc->prepare(kIODirectionIn);
		if (r == kIOReturnSuccess) {
			if (desc->writeBytes(0, info, outSize) != outSize)
				r = kIOReturnBadArgument;
			desc->complete(kIODirectionIn);
		}
	} else if (a->structureOutput && a->structureOutputSize == outSize) {
		memcpy(a->structureOutput, info, outSize);
	} else {
		r = kIOReturnBadArgument;
	}
	map->release();
	return r;
}
