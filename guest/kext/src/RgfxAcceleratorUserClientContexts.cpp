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
#include "RgfxContextSlots.hpp"
#include "RgfxFeatures.hpp"
#include "RgfxRegionSpans.hpp"
#include "RgfxSharedEventPorts.hpp"
#include "RgfxSharedEventUserClient.hpp"
#include "RgfxSubmitGate.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <kern/clock.h>
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
		// Five outputs, or a sixth of the span (kRgfxKextFeatureContextSpan). A bundle that asks
		// for five picks through RGFX_CONTEXT_VA_SPAN, so its context holds that span.
		const bool spanAsked = a->scalarOutputCount == 6;
		if (a->scalarOutputCount != 5 && !spanAsked)
			return kIOReturnBadArgument;
		IOLockLock(queueLock);
		// None once the client's close has begun.
		IOLockLock(listLock);
		const bool closing = asks.closing;
		IOLockUnlock(listLock);
		if (closing) {
			IOLockUnlock(queueLock);
			return kIOReturnNotOpen;
		}
		if (!hasContext) {
			// The window first: a context with no window is never opened.
			UInt32 opened = 0, openedFirst = 0, openedUnits = 0;
			IOReturn r = openWindow();
			if (r == kIOReturnSuccess) {
				r = owner->openContext(&opened, !spanAsked, windowServer, &openedFirst,
				                       &openedUnits);
				if (r != kIOReturnSuccess)
					closeWindow();
			}
			if (r != kIOReturnSuccess) {
				IOLockUnlock(queueLock);
				return r;
			}
			IOLockLock(listLock);
			context = opened;
			firstUnit = openedFirst;
			units = openedUnits;
			hasContext = true;
			IOLockUnlock(listLock);
		}
		const uint64_t span = static_cast<uint64_t>(units) * RGFX_CONTEXT_VA_UNIT;
		// A context opened with a span asked for, asked again for five: its span is less than
		// the one the asker would pick through.
		if (!spanAsked && span < RGFX_CONTEXT_VA_SPAN) {
			IOLockUnlock(queueLock);
			return kIOReturnBadArgument;
		}
		out[0] = context;
		out[1] = static_cast<uint64_t>(context) << RGFX_CONTEXT_OBJ_ID_SHIFT;
		out[2] = rgfxContextBase(firstUnit);
		IOLockLock(windowLock);
		out[3] = windowBase;
		out[4] = windowSize;
		IOLockUnlock(windowLock);
		if (spanAsked)
			out[5] = span;
		IOLockUnlock(queueLock);
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodDeviceInfo:
		return deviceInfoMethod(a);
	case kRgfxAccelMethodResourceIds: {
		if (a->scalarOutputCount != 1)
			return kIOReturnBadArgument;
		// An ask, so a chunk is never taken for a context releaseContext has closed: no grant
		// follows its CONTEXT_CLOSE, and no chunk outlives its freeIdChunks.
		UInt32 ctx = 0;
		Ask ask = {this, beginAsk(&ctx)};
		UInt64 first = 0;
		IOReturn r = ask.begun ? owner->takeIdChunk(this, ctx, &first) : kIOReturnBadArgument;
		out[0] = first;
		return r;
	}
	case kRgfxAccelMethodAlloc: {
		// Two inputs, or a third of the block's write-combined part (kRgfxKextFeatureWriteCombined).
		if ((a->scalarInputCount != 2 && a->scalarInputCount != 3) || a->scalarOutputCount != 1)
			return kIOReturnBadArgument;
		// An ask, so no block is made for a context releaseContext has closed.
		UInt32 ctx = 0;
		Ask ask = {this, beginAsk(&ctx)};
		if (!ask.begun)
			return kIOReturnBadArgument;
		UInt64 offset = 0;
		const UInt64 writeCombinedFrom = a->scalarInputCount == 3 ? in[2] : UINT64_MAX;
		IOReturn r = owner->allocBlock(this, ctx, in[0], in[1], writeCombinedFrom, &offset);
		out[0] = offset;
		return r;
	}
	case kRgfxAccelMethodFree:
		if (a->scalarInputCount != 1)
			return kIOReturnBadArgument;
		return owner->freeBlock(this, in[0]);
	case kRgfxAccelMethodQueueCreate: {
		// Six inputs, or a seventh of the flags the process wants with a second output of the
		// flags granted. A queue of WindowServer's is also made unshared where the host took the
		// feature (rgfxQueueFlagsSent), whatever the process asked: the second output answers only
		// what it asked for and was granted.
		const bool flagged = a->scalarInputCount == 7;
		if ((a->scalarInputCount != 6 && !flagged) || a->scalarOutputCount != (flagged ? 2u : 1u))
			return kIOReturnBadArgument;
		if (flagged && in[6] > UINT32_MAX)
			return kIOReturnBadArgument;
		// An ask, so no queue is made or watched for a context releaseContext has closed.
		UInt32 ctx = 0;
		Ask ask = {this, beginAsk(&ctx)};
		if (!ask.begun)
			return kIOReturnBadArgument;
		const uint64_t word = owner->featureWord();
		const UInt32 granted = flagged ? rgfxQueueFlagsGranted(static_cast<uint32_t>(in[6]), word) : 0;
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
		if ((open.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) != ctx)
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
		// Under the queue lock among the client's queue creates and destroys: the list's room is
		// taken and filled under one hold, and a destroy of this id waits for the create.
		IOLockLock(queueLock);
		if (!rgfxQueueAdmitted(queueCount)) {
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
		// Named in the making, so a destroy of its id meanwhile waits for the create
		// (destroyQueueIfTracked).
		IOLockLock(listLock);
		creatingQueue = open.obj_id;
		IOLockUnlock(listLock);
		UInt32 status = owner->openQueue(open, rgfxQueueFlagsSent(granted, windowServer, word));
		IOLockLock(listLock);
		if (status == RGFX_STATUS_OK)
			queues[queueCount++] = open.obj_id;
		creatingQueue = 0;
		IOLockUnlock(listLock);
		if (status != RGFX_STATUS_OK)
			owner->unwatchQueue(this, open.obj_id);
		IOLockUnlock(queueLock);
		unpin();
		out[0] = status;
		if (flagged)
			out[1] = granted;
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
	case kRgfxAccelMethodTakeAnswers:
		return takeAnswersMethod(a);
	case kRgfxAccelMethodWaitCompletions:
		return waitCompletionsMethod(a);
	case kRgfxAccelMethodSurfaceScanout: {
		if (a->scalarInputCount != 2 || a->scalarOutputCount != 5)
			return kIOReturnBadArgument;
		// The surface's one buffer.
		if (in[0] == 0 || in[0] > UINT32_MAX || in[1] != 0)
			return kIOReturnBadArgument;
		// An ask, so no grant outlives the close's revokeGrants.
		UInt32 ctx = 0;
		Ask ask = {this, beginAsk(&ctx)};
		if (!ask.begun)
			return kIOReturnBadArgument;
		RgfxAccelerator::Scanout scanout = {};
		IOReturn r = owner->grantSurfaceScanout(this, ctx, task, static_cast<UInt32>(in[0]),
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
		// sixth name the surface and whether it may be cached, and a struct input with them is
		// the surface's layout. A fifth output says whether it is.
		const UInt32 inputs = a->scalarInputCount;
		if ((inputs < 3 || inputs > 6 || inputs == 5) ||
		    (a->scalarOutputCount != 4 && a->scalarOutputCount != 5))
			return kIOReturnBadArgument;
		RgfxSurfacePagesLayout given = {};
		const bool laidOut = a->structureInputSize != 0 || a->structureInputDescriptor;
		if (laidOut && (inputs != 6 || a->structureInputDescriptor || !a->structureInput ||
		                a->structureInputSize != sizeof(given)))
			return kIOReturnBadArgument;
		if (laidOut)
			memcpy(&given, a->structureInput, sizeof(given));
		bool inPlace = inputs >= 4 && in[3] != 0;
		if (inputs >= 4 && in[3] > 1)
			return kIOReturnBadArgument;
		if (inputs == 6 && (in[4] > UINT32_MAX || in[5] > 1))
			return kIOReturnBadArgument;
		const uint32_t surfaceId = inputs == 6 ? static_cast<uint32_t>(in[4]) : 0;
		const bool cacheable = inputs == 6 && in[5] != 0;
		// An ask, so no hold is made for a context releaseContext has closed: every hold of the
		// client's goes in its dropSurfaceBlocks.
		UInt32 ctx = 0;
		Ask ask = {this, beginAsk(&ctx)};
		RgfxAccelerator::SurfaceBlock block = {};
		IOReturn r = ask.begun ? owner->acquireSurfaceBlock(this, ctx, task, in[0], in[1], in[2],
		                                                    inPlace, surfaceId, cacheable,
		                                                    laidOut ? &given : nullptr, &block)
		                       : kIOReturnBadArgument;
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
	case kRgfxAccelMethodPagesBlock: {
		if (a->scalarInputCount != 3 || a->scalarOutputCount != 2 || in[2] > 1)
			return kIOReturnBadArgument;
		// An ask, so no pages block is made for a context releaseContext has closed: every block
		// of the client's goes after its close.
		UInt32 ctx = 0;
		Ask ask = {this, beginAsk(&ctx)};
		UInt64 offset = 0, size = 0;
		IOReturn r = ask.begun ? owner->pagesBlock(this, ctx, task, in[0], in[1], in[2] != 0,
		                                           &offset, &size)
		                       : kIOReturnBadArgument;
		if (r != kIOReturnSuccess)
			return r;
		out[0] = offset;
		out[1] = size;
		return kIOReturnSuccess;
	}
	case kRgfxAccelMethodSurfaceWrites:
		return surfaceWritesMethod(a);
	case kRgfxAccelMethodSurfaceRetire:
		if (a->scalarInputCount != 2 || !ownsQueue(in[0]))
			return kIOReturnBadArgument;
		owner->retireSurfaceWrites(in[0], in[1]);
		return kIOReturnSuccess;
	case kRgfxAccelMethodFeatures:
		if (a->scalarInputCount != 0 || a->scalarOutputCount != 2)
			return kIOReturnBadArgument;
		out[0] = owner->featureWord();
		out[1] = kRgfxKextFeatureWord;
		return kIOReturnSuccess;
	default:
		return kIOReturnUnsupported;
	}
}

// MARK: - Shared events' ports

namespace {
// How long a shared event's name or import waits for the device: the host carries it out on the
// caller's context's lane, behind what the context posted before it, a pipeline's compile among
// them.
constexpr UInt32 kShareMs = 60000;
} // namespace

IOReturn RgfxAcceleratorUserClient::connectClient(IOUserClient *client)
{
	RgfxSharedEventUserClient *port = OSDynamicCast(RgfxSharedEventUserClient, client);
	const bool share = ucType == kRgfxAccelUserClientType && port &&
	                   rgfxPortIsShareOf(port->acceleratorOf(), owner);
	return share ? kIOReturnSuccess : kIOReturnBadArgument;
}

bool RgfxAcceleratorUserClient::holdsContext(task_t caller, UInt64 ctx)
{
	if (ucType != kRgfxAccelUserClientType)
		return false;
	IOLockLock(listLock);
	const bool holds = !asks.closing && rgfxPortCallerHolds(task, caller, hasContext, context, ctx);
	IOLockUnlock(listLock);
	return holds;
}

IOReturn RgfxAcceleratorUserClient::exportSharedEvent(UInt64 ctx, UInt64 eventId, UInt64 listen,
                                                      UInt64 name, UInt32 *status, bool *posted)
{
	*posted = false;
	// An ask, so the name is posted before the context's CONTEXT_CLOSE, never after it.
	UInt32 asked = 0;
	Ask ask = {this, beginAsk(&asked)};
	if (!ask.begun || asked != ctx || !ownsQueue(listen))
		return kIOReturnBadArgument;
	RgfxSharedEventName m = rgfxSharedEventNameMessage(eventId, name, listen);
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 length = 0;
	UInt64 seqno = 0;
	IOReturn r = owner->postLifecycle(RGFX_CMD_SHARED_EVENT_NAME, &m, sizeof(m), RGFX_EVT_COMPLETE,
	                                  reply, sizeof(reply), &length, &seqno, kShareMs);
	*posted = r == kIOReturnSuccess || r == kIOReturnTimeout;
	if (r != kIOReturnSuccess)
		return r;
	RgfxComplete complete = {};
	if (length < sizeof(complete))
		return kIOReturnInternalError;
	memcpy(&complete, reply, sizeof(complete));
	*status = complete.status;
	return kIOReturnSuccess;
}

IOReturn RgfxAcceleratorUserClient::importSharedEvent(UInt64 ctx, UInt64 eventId, UInt64 listen,
                                                      UInt64 name, UInt32 *status, UInt64 *value)
{
	// An ask, so the new event is made before the context's CONTEXT_CLOSE, never after it.
	UInt32 asked = 0;
	Ask ask = {this, beginAsk(&asked)};
	if (!ask.begun || asked != ctx || !ownsQueue(listen))
		return kIOReturnBadArgument;
	RgfxSharedEventImport m = rgfxSharedEventImportMessage(eventId, name, listen);
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 length = 0;
	UInt64 seqno = 0;
	IOReturn r = owner->postLifecycle(RGFX_CMD_SHARED_EVENT_IMPORT, &m, sizeof(m),
	                                  RGFX_EVT_SHARED_EVENT_VALUE, reply, sizeof(reply), &length,
	                                  &seqno, kShareMs);
	if (r == kIOReturnTimeout) {
		// Not answered in time: the process takes the import as failed, and the device keeps
		// nothing of it, the destroy following the import on the ring.
		RgfxObjectDestroy destroy = {};
		destroy.header.type = RGFX_CMD_OBJECT_DESTROY;
		destroy.obj_id = eventId;
		owner->postLifecycle(RGFX_CMD_OBJECT_DESTROY, &destroy, sizeof(destroy), 0, nullptr, 0,
		                     nullptr, nullptr, 0);
	}
	if (r != kIOReturnSuccess)
		return r;
	RgfxSharedEventValue answer = {};
	if (length < sizeof(answer))
		return kIOReturnInternalError;
	memcpy(&answer, reply, sizeof(answer));
	*status = answer.status;
	*value = answer.value;
	return kIOReturnSuccess;
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

// kRgfxAccelMethodSubmit (RgfxUserClientABI.hpp): one create or destroy of the client's onto the
// control ring, and the device's answer back where the create has one: a pipeline's reflection,
// a sampler's or an indirect command buffer's state, a library create's or lookup's COMPLETE, an
// argument layout's COMPLETE. A pipeline create with kRgfxSubmitDeferred is posted without
// waiting where a deferred slot is free (postDeferred), its answer then taken through
// kRgfxAccelMethodTakeAnswers.
IOReturn RgfxAcceleratorUserClient::submitMethod(IOExternalMethodArguments *a)
{
	// The context as it is now, read under the list lock that releaseContext closes it under,
	// to tell after the post whether it closed meanwhile (rgfxSubmitDestroysBehind).
	IOLockLock(listLock);
	const bool open = hasContext;
	const UInt32 ctx = context;
	IOLockUnlock(listLock);
	if (!open || a->scalarOutputCount != 1 || a->scalarInputCount > 1 ||
	    (a->scalarInputCount == 1 && (a->scalarInput[0] & ~uint64_t{kRgfxSubmitDeferred}) != 0))
		return kIOReturnBadArgument;
	const bool deferAsked = a->scalarInputCount == 1 && (a->scalarInput[0] & kRgfxSubmitDeferred);
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
	const bool pipeline = rgfxSubmitIsPipelineCreate(type);
	if (r == kIOReturnSuccess &&
	    (!rgfxSubmitAdmitted(type) || (objId >> RGFX_CONTEXT_OBJ_ID_SHIFT) != ctx ||
	     (deferAsked && !pipeline)))
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
	// query's SIZE_AND_ALIGN, a sampler create's SAMPLER_STATE and an indirect command buffer
	// create's ICB_STATE (each its resource id), and the
	// COMPLETE of a library create or lookup (a library's last region chunk is its create) or of
	// an argument layout, whose layout the device writes into the client's block before it.
	UInt32 replyType = 0;
	if (pipeline)
		replyType = RGFX_EVT_PIPELINE_REFLECTION;
	else if (type == RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN)
		replyType = RGFX_EVT_SIZE_AND_ALIGN;
	else if (type == RGFX_CMD_SAMPLER_CREATE)
		replyType = RGFX_EVT_SAMPLER_STATE;
	else if (type == RGFX_CMD_ICB_CREATE)
		replyType = RGFX_EVT_ICB_STATE;
	else if (type == RGFX_CMD_LIBRARY_CREATE || type == RGFX_CMD_LIBRARY_LOOKUP ||
	         type == RGFX_CMD_ARGUMENT_LAYOUT ||
	         (type == RGFX_CMD_LIBRARY_FROM_REGION && lastLibraryChunk(message, size)))
		replyType = RGFX_EVT_COMPLETE;
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 replyLength = 0;
	UInt64 seqno = 0;
	const bool checked = r == kIOReturnSuccess;
	// A deferred create, where a slot is free; with none it is posted and waited for as any.
	bool deferred = false;
	if (checked && deferAsked) {
		r = owner->postDeferred(ctx, type, message, static_cast<UInt32>(size), &seqno);
		deferred = r == kIOReturnSuccess;
		if (r == kIOReturnNoResources)
			r = kIOReturnSuccess;
	}
	if (checked && !deferred && r == kIOReturnSuccess)
		r = owner->postLifecycle(type, message, static_cast<UInt32>(size), replyType, reply,
		                         sizeof(reply), &replyLength, &seqno, kReflectionMs);
	IOFree(message, size);
	// Posted where postLifecycle answered or timed out waiting for the answer. Whether the context
	// closed since the check: its CONTEXT_CLOSE may then have gone before this create, which the
	// device's close then never dropped.
	const bool posted = checked && (r == kIOReturnSuccess || r == kIOReturnTimeout);
	IOLockLock(listLock);
	const bool closedSince = !hasContext || context != ctx;
	IOLockUnlock(listLock);
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
	// A deferred create's slot, taken after its context's close dropped the context's: freed
	// too, as no one takes it now. Its own slot alone, by its seqno: a new context may have the
	// number now.
	if (deferred && closedSince)
		owner->forgetDeferred(seqno);
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

// kRgfxAccelMethodTakeAnswers (RgfxAccelerator.hpp): the answers to the client's deferred
// pipeline creates the device has given, after waiting for the one asked for. Always succeeds with
// what it took, the answers' count out, so none is lost on a wait that timed out: the caller finds
// whether the one it waited for is among them.
IOReturn RgfxAcceleratorUserClient::takeAnswersMethod(IOExternalMethodArguments *a)
{
	IOLockLock(listLock);
	const bool open = hasContext;
	const UInt32 ctx = context;
	IOLockUnlock(listLock);
	if (!open || a->scalarInputCount != 1 || a->scalarOutputCount != 1 || !a->structureOutput)
		return kIOReturnBadArgument;
	UInt32 length = 0, count = 0;
	owner->takeDeferred(ctx, a->scalarInput[0], kReflectionMs,
	                    static_cast<UInt8 *>(a->structureOutput), a->structureOutputSize, &length,
	                    &count);
	a->scalarOutput[0] = count;
	a->structureOutputSize = length;
	return kIOReturnSuccess;
}

// Whether `queue` is one of the client's open queues.
bool RgfxAcceleratorUserClient::ownsQueue(UInt64 queue)
{
	IOLockLock(listLock);
	bool owns = false;
	for (UInt32 i = 0; i < queueCount && !owns; i++)
		owns = queues[i] == queue;
	IOLockUnlock(listLock);
	return owns;
}

bool RgfxAcceleratorUserClient::destroyQueueIfTracked(UInt64 objId)
{
	// The destroy of any other object (most of them, posted through submit) is told at once from
	// the list: an id neither listed nor in the making is none of the client's queues, and
	// waits for no create under the queue lock.
	IOLockLock(listLock);
	bool tracked = objId != 0 && objId == creatingQueue;
	for (UInt32 i = 0; i < queueCount && !tracked; i++)
		tracked = queues[i] == objId;
	IOLockUnlock(listLock);
	if (!tracked)
		return false;
	bool found = false;
	// The whole find, remove, unwatch and close under the queue lock, which QUEUE_CREATE watches
	// and lists a queue under, so a create of this id cannot slip between the check and the post.
	// Off the list first, so no surface write or retire named for it is taken from here.
	IOLockLock(queueLock);
	IOLockLock(listLock);
	for (UInt32 i = 0; i < queueCount; i++) {
		if (queues[i] == objId) {
			queues[i] = queues[--queueCount];
			found = true;
			break;
		}
	}
	IOLockUnlock(listLock);
	if (found) {
		owner->unwatchQueue(this, objId);
		owner->closeQueue(objId);
	}
	IOLockUnlock(queueLock);
	return found;
}

// kRgfxAccelMethodSurfaceWrites (RgfxUserClientABI.hpp): the surfaces a command buffer of one of
// the client's queues writes, whose CPU locks wait for it from here.
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

// kRgfxAccelMethodWaitCompletions (RgfxUserClientABI.hpp): the bundle's completion thread
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

bool RgfxAcceleratorUserClient::beginAsk(UInt32 *ctx)
{
	IOLockLock(listLock);
	const bool begun = rgfxAskBegin(asks, hasContext);
	*ctx = context;
	IOLockUnlock(listLock);
	return begun;
}

void RgfxAcceleratorUserClient::endAsk()
{
	IOLockLock(listLock);
	if (rgfxAskEnd(asks))
		IOLockWakeup(listLock, &asks, false);
	IOLockUnlock(listLock);
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
	// The asks in flight end first, and none begins from here (RgfxAskGate.hpp): what each made
	// is the context's, which the close lets go of below. Out of the queue lock, which a queue's
	// create, one of them, takes.
	IOLockLock(listLock);
	const uint64_t drainFrom = mach_absolute_time();
	const UInt32 inFlight = rgfxAskClose(asks);
	while (!rgfxAskDrained(asks))
		IOLockSleep(listLock, &asks, THREAD_UNINT);
	const UInt32 most = asks.most, asked = context;
	const bool opened = hasContext;
	IOLockUnlock(listLock);
	if (opened && (inFlight || most > 1)) {
		uint64_t ns = 0;
		absolutetime_to_nanoseconds(mach_absolute_time() - drainFrom, &ns);
		IOLog("RgfxAccelerator: context %u: %u asks at once at most; its close waited %llu us "
		      "for %u in flight\n",
		      asked, most, ns / 1000, inFlight);
	}
	IOLockLock(queueLock);
	// Closed to the readers first, before its CONTEXT_CLOSE is posted: a submit that finds it
	// open posts behind the close only to find it closed after (rgfxSubmitDestroysBehind), and
	// no surface write or retire is taken for its queues, whose writes the unwatch forgets. The
	// list's entries stay for the closes below: no create or destroy runs under the queue lock.
	IOLockLock(listLock);
	const bool had = hasContext;
	const UInt32 listed = queueCount;
	hasContext = false;
	queueCount = 0;
	IOLockUnlock(listLock);
	// The process reaches nothing of the region from here on: its window goes before anything
	// it held is freed, and given to another process.
	closeWindow();
	RgfxAccelerator *accel = owner;
	// Under the queue lock, which a QUEUE_CREATE watches its ring under, so none is left.
	if (accel)
		accel->unwatchClient(this);
	if (!had || !accel) {
		IOLockUnlock(queueLock);
		return;
	}
	for (UInt32 i = 0; i < listed; i++)
		accel->closeQueue(queues[i]);
	// What the context held goes only once the device has answered the close: it then holds
	// nothing of it, maps none of its resource ids or GPU addresses to an object, and the
	// context's command buffers have finished. A close answered late lets go of it then.
	int32_t late = -1;
	accel->holdBlocks(this);
	const RgfxAccelerator::CloseAnswer closed =
	    accel->closeContext(context, firstUnit, units, &late);
	// Its deferred creates' answers, taken by no one now; a waiter for one returns.
	accel->dropDeferred(context);
	switch (closed) {
	case RgfxAccelerator::kCloseAnswered:
		accel->freeIdChunks(this);
		accel->freeBlocks(this);
		accel->dropSurfaceBlocks(this, true);
		break;
	case RgfxAccelerator::kCloseLate:
		accel->parkClose(this, late);
		break;
	case RgfxAccelerator::kCloseRefused:
		accel->keepIdChunks(this);
		accel->keepBlocks(this);
		accel->dropSurfaceBlocks(this, false);
		break;
	}
	accel->revokeGrants(this);
	IOLockUnlock(queueLock);
	// The device dropped this context's objects at its close: a block whose release it refused
	// because an object lay in it may now be free to go.
	if (closed == RgfxAccelerator::kCloseAnswered)
		accel->retryRefusedBlocks();
}

// kRgfxAccelMethodDeviceInfo (RgfxUserClientABI.hpp): the device info the host wrote, read from
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
	const void *info = nullptr;
	IOMemoryMap *map = owner->mapDeviceInfo(outSize, &info);
	if (!map)
		return kIOReturnNoResources;
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
