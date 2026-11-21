/*
 * RgfxMetalDevice: see RgfxMetalDevice.h for the class's role and the category
 * split. This file is slice 0: init, identity, and the limit/feature getters,
 * answered from RgfxCaps. The resource/pipeline/command-buffer/sync selectors live
 * in the category files the header names.
 *
 * Our own ObjC++; the only Apple-private names are the forward-declarations in
 * RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice.h"
#include "RgfxLog.hpp"
#include "RgfxLogOnce.hpp"

#import "RgfxBinaryArchive.h"
#import "RgfxBuffer.h"
#import "RgfxInterop.h"
#import "RgfxIOSurface.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include <IOKit/IOKitLib.h>
#include <IOSurface/IOSurfaceRef.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <memory>
#include <mutex>
#include <time.h>
#include <vector>

#include "RgfxAllocationCount.hpp"
#include "RgfxCommitOrder.hpp"
#include "RgfxCompletionThread.hpp"
#include "RgfxFeatureProfile.hpp"
#include "RgfxHeapSizes.hpp"
#include "RgfxKnownPipelines.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"

// The accelerator's user-client protocol, the kext's own header: its type, its selectors and
// the kext's own feature bits.
#include "../../kext/src/RgfxDeferCarry.hpp"
#include "../../kext/src/RgfxQueuePublish.hpp"
#include "../../kext/src/RgfxUserClientABI.hpp"
#include "RgfxDoorbellPage.hpp"

namespace {
// The largest object-lifecycle message the kext takes (ReimsGraphics::kLifecycleMessageMax):
// half the control ring.
constexpr uint32_t kLifecycleMessageMax = RGFX_CONTROL_RING_DATA_SIZE / 2;

// What the lifecycle's forward reaches: the user client, and the reader its replies go to. A
// pipeline create posted without waiting (rgfxKextSubmitDeferred) has its answer kept by the
// kext until the process takes it (rgfxKextAwaitAnswer, rgfxKextDrainAnswers), one take at a
// time (`takeLock`), so a thread that waits for an answer another is handing on waits for that.
struct RgfxKextLink {
	io_connect_t conn = IO_OBJECT_NULL;
	RgfxReplyReader *reader = nullptr;
	std::mutex takeLock;
	std::atomic<uint32_t> deferredPosts{0};
	// The control ring, whose held destroys (RgfxRingWriter::postLater) go before any other call
	// of the process's into the kext (rgfxKextConn); null until it is made.
	RgfxRingWriter *control = nullptr;
};

// The connection of the link at `context`, for a call into the kext, once the destroys the
// control ring holds are posted: every call of the process's comes after them, as each of its
// posts does (RgfxRingWriter::postLater). The kext's answer to a call may hang on what a held
// destroy lets go of (a block's free, a queue's close), and the host's on the order of the ring.
io_connect_t rgfxKextConn(void *context)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	if (link->control != nullptr)
		link->control->flushLater();
	return link->conn;
}

// The answers to deferred creates one take carries at most: an inline struct out (4 KiB).
constexpr size_t kAnswersMax = 4096;
// Every this many deferred posts, the answers the process has not asked for are taken, so they
// never hold the kext's slots (64 a context) while the process makes more.
constexpr uint32_t kDrainEvery = 32;

// Takes the answers the kext keeps for the process's deferred creates, after waiting for the one
// at `waitFor` (0: none), and hands each to the lifecycle reader: how many came. With `link`'s
// take lock held.
uint32_t rgfxKextTakeAnswers(RgfxKextLink *link, uint64_t waitFor)
{
	uint8_t answers[kAnswersMax];
	size_t size = sizeof(answers);
	uint64_t count = 0;
	uint32_t outCount = 1;
	if (IOConnectCallMethod(rgfxKextConn(link), kRgfxAccelMethodTakeAnswers, &waitFor, 1, nullptr,
	                        0, &count, &outCount, answers, &size) != KERN_SUCCESS)
		return 0;
	for (size_t at = 0; at + sizeof(RgfxMsgHeader) <= size;) {
		RgfxMsgHeader header = {};
		std::memcpy(&header, answers + at, sizeof(header));
		if (header.size < sizeof(header) || header.size > size - at)
			break;
		if (link->reader)
			link->reader->deliver(answers + at, header.size);
		at += header.size;
	}
	return static_cast<uint32_t>(count);
}

// Takes every answer the kext keeps for the process, waiting first for the one at `waitFor`.
void rgfxKextTakeAll(RgfxKextLink *link, uint64_t waitFor)
{
	while (rgfxKextTakeAnswers(link, waitFor) != 0)
		waitFor = 0;
}

// The answers no one asked for, taken where no other take runs (a waiter takes them then).
void rgfxKextDrainAnswers(RgfxKextLink *link)
{
	std::unique_lock<std::mutex> take(link->takeLock, std::try_to_lock);
	if (take.owns_lock())
		rgfxKextTakeAll(link, 0);
}

// The control ring's deferred post (RgfxRingWriter::setDeferred): a pipeline create equal to one
// the device accepted, posted with kRgfxSubmitDeferred, which returns once it is posted. Where the
// kext kept no slot for its answer, it posted and answered it as any create, and the answer goes
// to the reader as rgfxKextSubmit's.
bool rgfxKextSubmitDeferred(void *context, const void *message, uint32_t size,
                            uint64_t *out_seqno, bool *out_deferred)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	const uint64_t flags = kRgfxSubmitDeferred;
	uint64_t seqno = 0;
	uint32_t outCount = 1;
	uint8_t reply[128];
	size_t replySize = sizeof(reply);
	kern_return_t kr = IOConnectCallMethod(link->conn, kRgfxAccelMethodSubmit, &flags, 1, message,
	                                       size, &seqno, &outCount, reply, &replySize);
	if (kr != KERN_SUCCESS)
		return false;
	*out_seqno = seqno;
	*out_deferred = replySize == 0;
	if (!*out_deferred) {
		if (replySize >= sizeof(RgfxMsgHeader) && link->reader)
			link->reader->deliver(reply, static_cast<uint32_t>(replySize));
		// Every slot was held: the answers no one asked for free them.
		rgfxKextDrainAnswers(link);
	} else if (link->deferredPosts.fetch_add(1, std::memory_order_relaxed) % kDrainEvery ==
	           kDrainEvery - 1) {
		rgfxKextDrainAnswers(link);
	}
	return true;
}

// The control ring's shared post (RgfxRingWriter::setShared): a pipeline create posted with
// kRgfxSubmitShared and its pipeline key, which returns at once with the answer the kext keeps for
// an equal create of any process, its own answer then kept as a deferred create's. Where the kext
// keeps none, or no slot for the answer, it posted and answered it as any create, and the answer
// goes to the reader as rgfxKextSubmit's.
bool rgfxKextSubmitShared(void *context, const void *message, uint32_t size, const uint8_t *key,
                          uint64_t *out_seqno, bool *out_deferred,
                          RgfxPipelineReflection *out_reflection)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	uint64_t in[1 + kRgfxSubmitKeyWords] = {kRgfxSubmitShared};
	std::memcpy(&in[1], key, kRgfxSubmitKeyWords * sizeof(uint64_t));
	uint64_t out[2] = {};
	uint32_t outCount = 2;
	uint8_t reply[128];
	size_t replySize = sizeof(reply);
	kern_return_t kr = IOConnectCallMethod(link->conn, kRgfxAccelMethodSubmit, in,
	                                       1 + kRgfxSubmitKeyWords, message, size, out, &outCount,
	                                       reply, &replySize);
	if (kr != KERN_SUCCESS)
		return false;
	*out_seqno = out[0];
	*out_deferred = outCount == 2 && out[1] == 1 && replySize >= sizeof(RgfxPipelineReflection);
	if (*out_deferred) {
		std::memcpy(out_reflection, reply, sizeof(*out_reflection));
		if (link->deferredPosts.fetch_add(1, std::memory_order_relaxed) % kDrainEvery ==
		    kDrainEvery - 1)
			rgfxKextDrainAnswers(link);
	} else if (replySize >= sizeof(RgfxMsgHeader) && link->reader) {
		link->reader->deliver(reply, static_cast<uint32_t>(replySize));
	}
	return true;
}

// The control ring's answer wait (RgfxRingWriter::setDeferred): every answer the kext keeps for
// the process, after the one at `seqno`, handed to the reader. True once taken; whether the one
// at `seqno` came is its state's to see.
bool rgfxKextAwaitAnswer(void *context, uint64_t seqno)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	std::lock_guard<std::mutex> take(link->takeLock);
	rgfxKextTakeAll(link, seqno);
	return true;
}

// Whether the kext keeps a deferred pipeline create's answer for later: its feature word
// (kRgfxAccelMethodFeatures) has kRgfxKextFeatureDeferredAnswers.
bool rgfxKextDefersAnswers(const RgfxCaps &caps)
{
	return (caps.kextFeatures() & kRgfxKextFeatureDeferredAnswers) != 0;
}

// The serializer's lifecycle route: the kext posts the message on the control ring, stamping
// its seqno, and hands back the answer the device owes it: a pipeline create's reflection, which
// goes to the reply reader, or a library create's or lookup's COMPLETE, whose status is
// `*out_status`.
bool rgfxKextSubmit(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
                    uint32_t *out_status)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	uint64_t seqno = 0;
	uint32_t outCount = 1;
	uint8_t reply[128];
	size_t replySize = sizeof(reply);
	kern_return_t kr = IOConnectCallMethod(link->conn, kRgfxAccelMethodSubmit, nullptr, 0, message,
	                                       size, &seqno, &outCount, reply, &replySize);
	if (kr != KERN_SUCCESS)
		return false;
	RgfxMsgHeader header = {};
	if (replySize >= sizeof(header))
		std::memcpy(&header, reply, sizeof(header));
	if (header.type == RGFX_EVT_COMPLETE && replySize >= sizeof(RgfxComplete)) {
		RgfxComplete complete = {};
		std::memcpy(&complete, reply, sizeof(complete));
		if (out_status)
			*out_status = complete.status;
	} else if (replySize >= sizeof(RgfxMsgHeader) && link->reader) {
		link->reader->deliver(reply, static_cast<uint32_t>(replySize));
	}
	*out_seqno = seqno;
	return true;
}

// The control ring's batch (RgfxRingWriter::setBatch): the messages it held (destroys, and the
// creates and writes no one waits for that name no block), posted in one call
// (kRgfxAccelMethodSubmitBatch), which they go before: how many the kext posted, from the first.
uint32_t rgfxKextSubmitBatch(void *context, const void *messages, uint32_t bytes)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	uint64_t posted = 0;
	uint32_t outCount = 1;
	if (IOConnectCallMethod(link->conn, kRgfxAccelMethodSubmitBatch, nullptr, 0, messages, bytes,
	                        &posted, &outCount, nullptr, nullptr) != KERN_SUCCESS ||
	    outCount != 1)
		return 0;
	return static_cast<uint32_t>(posted);
}

// The control ring's batch with a queue's publish after it (RgfxRingWriter::setBatchPublish):
// SubmitBatch with the queue, its producer and the publish's flags in, the kext publishing once it
// posted every message (RgfxQueuePublish.hpp): how many it posted, `*out_published` whether it
// published.
uint32_t rgfxKextSubmitBatchPublish(void *context, const void *messages, uint32_t bytes,
                                    const RgfxQueuePublish &publish, bool *out_published)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	const uint64_t in[3] = {publish.queue, publish.producer,
	                        (publish.answered ? uint64_t(kRgfxPublishAnswered) : 0) |
	                            (publish.asks ? uint64_t(kRgfxPublishAsks) : 0)};
	uint64_t out[2] = {};
	uint32_t outCount = 2;
	*out_published = false;
	if (IOConnectCallMethod(link->conn, kRgfxAccelMethodSubmitBatch, in, 3, messages, bytes, out,
	                        &outCount, nullptr, nullptr) != KERN_SUCCESS ||
	    outCount != 2)
		return 0;
	*out_published = out[1] == 1;
	return static_cast<uint32_t>(out[0]);
}

// The serializer's queries: the kext posts the message on the control ring, stamping its seqno,
// and hands back the device's answer, whole. `context` is the kext link (rgfxKextConn).
bool rgfxKextAsk(void *context, const void *message, uint32_t size, void *reply,
                 uint32_t capacity, uint32_t *reply_size)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t seqno = 0;
	uint32_t outCount = 1;
	size_t replySize = capacity;
	if (IOConnectCallMethod(conn, kRgfxAccelMethodSubmit, nullptr, 0, message, size, &seqno,
	                        &outCount, reply, &replySize) != KERN_SUCCESS)
		return false;
	*reply_size = static_cast<uint32_t>(replySize);
	return true;
}

// The kext's blocks, as RgfxRegion's block source; `context` is the kext link (rgfxKextConn).
bool rgfxKextAlloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in[2] = {size, align};
	uint32_t outCount = 1;
	return IOConnectCallScalarMethod(conn, kRgfxAccelMethodAlloc, in, 2, offset, &outCount) ==
	       KERN_SUCCESS;
}

// The region's write-combined source (RgfxRegion::setWriteCombinedSource): the kext's Alloc with
// its third input, the byte the block's write-combined part starts at
// (kRgfxKextFeatureWriteCombined).
bool rgfxKextAllocWriteCombined(void *context, uint64_t size, uint64_t align,
                                uint64_t writeCombinedFrom, uint64_t *offset)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in[3] = {size, align, writeCombinedFrom};
	uint32_t outCount = 1;
	return IOConnectCallScalarMethod(conn, kRgfxAccelMethodAlloc, in, 3, offset, &outCount) ==
	       KERN_SUCCESS;
}

// What the process keeps for reuse, as RgfxRegion's reclaim; `context` is the RgfxSlabs: the
// slabs' empty and cached blocks, and the IOSurface views the keep holds, a cache that costs no
// refusal (a texture over one of their surfaces then asks the kext for its block, as with no
// keep).
bool rgfxRegionReclaim(void *context)
{
	const bool slabs = static_cast<RgfxSlabs *>(context)->reclaim();
	const bool views = rgfxIOSurfaceKeepReclaim();
	return slabs || views;
}

// The slabs' reuse counts (RgfxSlabs, CHURN), one line a process at most every
// RgfxSlabs::kChurnLogMs once they changed: what the slabs' sources would serve with no empty or
// cached slab kept.
static void rgfxLogSlabChurn(void *pool, const RgfxSlabs::Churn &c)
{
	// The wall clock, so a run places each line in its own steps.
	struct timespec now = {};
	clock_gettime(CLOCK_REALTIME, &now);
	os_log(rgfxLog(),
	       "RGFX slab churn at %lld.%03ld s: small slabs made %llu (%llu KiB), emptied %llu, "
	       "refilled while kept %llu (%llu KiB), empty let go %llu; large slabs made %llu, taken "
	       "from the cache %llu (%llu KiB)",
	       static_cast<long long>(now.tv_sec), now.tv_nsec / 1000000, c.classMade,
	       c.classMadeBytes >> 10, c.emptied, c.refilled, c.refilledBytes >> 10,
	       c.emptyLetGo, c.dedicatedMade, c.dedicatedReused, c.dedicatedReusedBytes >> 10);
	// The Shared pool's takes, where its blocks are the slabs (`pool`, the RgfxPoolSlabSource):
	// how long the threads that made the slabs waited for the host.
	if (pool == nullptr)
		return;
	const RgfxPoolSlabSource::Takes t = static_cast<RgfxPoolSlabSource *>(pool)->takes();
	os_log(rgfxLog(),
	       "RGFX pool takes at %lld.%03ld s: %llu taken (%llu KiB), %llu refused, %llu us waited, "
	       "longest %llu us; %llu given back; %llu heaps at the host's addresses",
	       static_cast<long long>(now.tv_sec), now.tv_nsec / 1000000, t.taken, t.takenBytes >> 10,
	       t.refused, t.waitUs, t.longestUs, t.given, t.addressed);
}

// The kext's chunks of texture resource ids, as RgfxIdSpace's chunk source; `context` is the
// kext link (rgfxKextConn). False once every chunk of the guest's pool is held.
bool rgfxKextResourceIds(void *context, uint64_t *first_id, uint64_t *ids)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t out[2] = {};
	uint32_t outCount = 2;
	if (IOConnectCallScalarMethod(conn, kRgfxAccelMethodResourceIds, nullptr, 0, out,
	                              &outCount) != KERN_SUCCESS ||
	    outCount != 2)
		return false;
	*first_id = out[0];
	*ids = out[1];
	return true;
}

// The kext's Private blocks, as RgfxIdSpace's Private block source
// (RGFX_FEATURE_PRIVATE_BLOCKS); `context` is the kext link (rgfxKextConn). The take answers a
// block of at least `size` bytes, its GPU address and length; false where the host refused it.
static bool rgfxKextPrivateTake(void *context, uint64_t size, uint64_t *gpu_address,
                                uint64_t *length)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t out[2] = {};
	uint32_t outCount = 2;
	if (IOConnectCallScalarMethod(conn, kRgfxAccelMethodPrivateTake, &size, 1, out, &outCount) !=
	        KERN_SUCCESS ||
	    outCount != 2 || out[0] == 0 || out[1] < size)
		return false;
	*gpu_address = out[0];
	*length = out[1];
	return true;
}

// The give of a Private block no object of the process's lies in any more; its held destroys
// go first (rgfxKextConn), so the host sees the block empty.
static void rgfxKextPrivateGive(void *context, uint64_t gpu_address, uint64_t length)
{
	io_connect_t conn = rgfxKextConn(context);
	const uint64_t in[2] = {gpu_address, length};
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodPrivateGive, in, 2, nullptr, nullptr);
}

// The completion thread's wait: the kext returns once a completion ring of the process's holds
// records its last pass did not see (the wake count differs from `last`), or at once where it
// already does. `context` is the kext link (rgfxKextConn).
RgfxCompletionThread::WaitResult rgfxKextWaitCompletions(void *context, uint64_t last,
                                                         uint64_t *now)
{
	io_connect_t conn = rgfxKextConn(context);
	uint32_t outCount = 1;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodWaitCompletions, &last, 1, now, &outCount);
	if (kr == KERN_SUCCESS)
		return RgfxCompletionThread::kWoke;
	if (kr == kIOReturnAborted)
		return RgfxCompletionThread::kRetry; // a signal broke the sleep
	os_log_error(rgfxLog(), "RGFX: the completion wait ended kr=0x%x; waiters take the rings",
	             kr);
	// A resource's wait has no ring to take: it waits no more.
	RgfxResourceWaitsStop();
	return RgfxCompletionThread::kStop;
}

void rgfxKextRelease(void *context, uint64_t offset)
{
	io_connect_t conn = rgfxKextConn(context);
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodFree, &offset, 1, nullptr, nullptr);
}

// The kext's shared blocks of surface planes, as RgfxRegion's surface source: the block every
// process that holds the plane's surface gets, found by the pages it maps at `plane`.
bool rgfxKextSurfaceBlock(void *context, const void *plane, uint64_t length, uint64_t size,
                          uint64_t *offset, uint64_t *serial, bool *fresh)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in[3] = {reinterpret_cast<uintptr_t>(plane), length, size};
	uint64_t out[4] = {};
	uint32_t outCount = 4;
	if (IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlock, in, 3, out, &outCount) !=
	        KERN_SUCCESS ||
	    outCount != 4) {
		return false;
	}
	if (out[1] != size) {
		IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlockRelease, &out[0], 1, nullptr,
		                          nullptr);
		return false;
	}
	*offset = out[0];
	*serial = out[2];
	*fresh = out[3] != 0;
	return true;
}

// The resident pages of the surface the process maps at `base`, `length` bytes, in the task's
// pages, which are the kernel's: true where 1 to kRgfxCarryPagesMost are, listed in `resident`
// (RgfxDeferCarry.hpp); false where none or more are, or the query fails. It stops at the first
// page past those.
bool rgfxResidentPagesAt(const void *base, uint64_t length, RgfxSurfaceResident *resident)
{
	const uint64_t page = vm_page_size;
	const uint64_t pages = (length + page - 1) / page;
	constexpr uint64_t kBatch = 1024;
	int dispositions[kBatch];
	*resident = {};
	for (uint64_t at = 0; at < pages; at += kBatch) {
		mach_vm_size_t count = std::min(kBatch, pages - at);
		if (mach_vm_page_range_query(mach_task_self(),
		                             reinterpret_cast<uintptr_t>(base) + at * page, count * page,
		                             reinterpret_cast<mach_vm_address_t>(dispositions),
		                             &count) != KERN_SUCCESS)
			return false;
		for (uint64_t i = 0; i < count; i++)
			if ((dispositions[i] & VM_PAGE_QUERY_PAGE_PRESENT) != 0 &&
			    !rgfxResidentAdd(*resident, (at + i) * page, page))
				return false;
	}
	return resident->count != 0;
}

// The kext's shared blocks of surfaces in place, as RgfxRegion's surface pages source: the
// block every process that holds the surface gets, found by the whole allocation it maps at
// `base`, the device's GPU using those pages, and cached where the kext made it so (six inputs,
// the surface's IOSurfaceID and whether it may be cached; a fifth output), the surface's layout
// its struct input where given, followed by the surface's resident pages where `resident` lists
// them. Unsupported where the device lacks the feature or refused the pages; no room where the
// kext had none, in the region (kIOReturnNoSpace) or its tables (kIOReturnNoResources); refused
// for any other answer, among them a kext that takes fewer inputs.
RgfxRegion::PagesAnswer rgfxKextSurfacePagesListing(void *context, const void *base,
                                                    uint64_t length, uint64_t size,
                                                    uint32_t surfaceId, bool cacheable,
                                                    const RgfxSurfacePagesLayout *layout,
                                                    const RgfxSurfaceResident *resident,
                                                    uint64_t *offset, uint64_t *serial,
                                                    bool *fresh, bool *cached)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in[6] = {reinterpret_cast<uintptr_t>(base), length, size, 1, surfaceId,
	                  cacheable ? 1u : 0u};
	uint64_t out[5] = {};
	uint32_t outCount = 5;
	uint8_t given[sizeof(RgfxSurfacePagesLayout) + sizeof(RgfxSurfaceResident)] = {};
	size_t givenSize = 0;
	if (layout != nullptr) {
		memcpy(given, layout, sizeof(*layout));
		givenSize = sizeof(*layout);
		if (resident != nullptr) {
			memcpy(given + sizeof(*layout), resident, sizeof(*resident));
			givenSize += sizeof(*resident);
		}
	}
	kern_return_t kr =
	    IOConnectCallMethod(conn, kRgfxAccelMethodSurfaceBlock, in, 6,
	                        givenSize != 0 ? given : nullptr, givenSize, out, &outCount,
	                        nullptr, nullptr);
	if (kr == kIOReturnUnsupported)
		return RgfxRegion::kPagesUnsupported;
	if (kr == kIOReturnNoSpace || kr == kIOReturnNoResources)
		return RgfxRegion::kPagesNoRoom;
	if (kr != KERN_SUCCESS || outCount != 5)
		return RgfxRegion::kPagesRefused;
	if (out[1] != size) {
		IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlockRelease, &out[0], 1, nullptr,
		                          nullptr);
		return RgfxRegion::kPagesRefused;
	}
	*offset = out[0];
	*serial = out[2];
	*fresh = out[3] != 0;
	*cached = out[4] != 0;
	return RgfxRegion::kPagesGiven;
}

RgfxRegion::PagesAnswer rgfxKextSurfacePages(void *context, const void *base, uint64_t length,
                                             uint64_t size, uint32_t surfaceId, bool cacheable,
                                             const RgfxSurfacePagesLayout *layout,
                                             uint64_t *offset, uint64_t *serial, bool *fresh,
                                             bool *cached)
{
	return rgfxKextSurfacePagesListing(context, base, length, size, surfaceId, cacheable, layout,
	                                   nullptr, offset, serial, fresh, cached);
}

// As rgfxKextSurfacePages, where a deferral may carry its surface's resident pages
// (RGFX_FEATURE_SURFACE_DEFERRED_CARRY): a cacheable ask with its layout lists them, where 1 to
// kRgfxCarryPagesMost are, and the kext defers the surface fresh but for them where it proves
// them (RgfxDeferCarry.hpp). Measured, VTDecoderXPCService writes page 0 of each new decoder
// output before its first decode, the rest not resident.
RgfxRegion::PagesAnswer rgfxKextSurfacePagesCarrying(void *context, const void *base,
                                                     uint64_t length, uint64_t size,
                                                     uint32_t surfaceId, bool cacheable,
                                                     const RgfxSurfacePagesLayout *layout,
                                                     uint64_t *offset, uint64_t *serial,
                                                     bool *fresh, bool *cached)
{
	RgfxSurfaceResident resident = {};
	const bool listed =
	    cacheable && layout != nullptr && rgfxResidentPagesAt(base, length, &resident);
	return rgfxKextSurfacePagesListing(context, base, length, size, surfaceId, cacheable, layout,
	                                   listed ? &resident : nullptr, offset, serial, fresh,
	                                   cached);
}

// The region's ways of a pages block's memory are the kext's (kRgfxAccelMethodPagesBlock's third
// input).
static_assert(RgfxRegion::kPagesWritable == kRgfxPagesWritable &&
                  RgfxRegion::kPagesUnchanging == kRgfxPagesUnchanging &&
                  RgfxRegion::kPagesCopyOnWrite == kRgfxPagesCopyOnWrite &&
                  RgfxRegion::kPagesSoleHolder == kRgfxPagesSoleHolder &&
                  RgfxRegion::kPagesWhyShift == kRgfxPagesWhyShift,
              "the region's pages block ways are the kext's");

// The kext's pages block of the process's own memory (kRgfxAccelMethodPagesBlock), asked for
// `outputs` outputs: 3 for the GPU address the device gave the block's view too, else 2.
static bool rgfxKextPagesBlockOf(void *context, const void *address, uint64_t length,
                                 uint32_t how, uint64_t *offset, uint64_t *size,
                                 uint64_t *gpuAddress, uint32_t outputs)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in[3] = {reinterpret_cast<uintptr_t>(address), length, how};
	uint64_t out[3] = {};
	uint32_t outCount = outputs;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodPagesBlock, in, 3, out, &outCount);
	if (kr != KERN_SUCCESS || outCount != outputs)
		return false;
	*offset = out[0];
	*size = out[1];
	*gpuAddress = outputs == 3 ? out[2] : 0;
	return true;
}

// The region's pages block source (RgfxRegion::setPagesBlockSource): the kext's pages block of
// the process's own memory (kRgfxAccelMethodPagesBlock), false where it makes none, for any
// reason, and the memory then takes a shadow. No GPU address.
bool rgfxKextPagesBlock(void *context, const void *address, uint64_t length, uint32_t how,
                        uint64_t *offset, uint64_t *size, uint64_t *gpuAddress)
{
	return rgfxKextPagesBlockOf(context, address, length, how, offset, size, gpuAddress, 2);
}

// rgfxKextPagesBlock, with the GPU address the device gave the block's view, where the kext and
// the device took IMPORT_ADDRESSES (kRgfxKextFeatureImportAddresses).
bool rgfxKextPagesBlockAt(void *context, const void *address, uint64_t length, uint32_t how,
                          uint64_t *offset, uint64_t *size, uint64_t *gpuAddress)
{
	return rgfxKextPagesBlockOf(context, address, length, how, offset, size, gpuAddress, 3);
}

// The region's chunked pages source (RgfxRegion::setChunkedPagesSource): the kext's chunks and
// their pages block (kRgfxAccelMethodPagesChunked), false where it makes none, for any reason,
// and the slab then takes plain pages.
bool rgfxKextPagesChunked(void *context, uint64_t length, uint8_t **bytes, uint64_t *offset,
                          uint64_t *size)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t out[3] = {};
	uint32_t outCount = 3;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodPagesChunked, &length, 1, out, &outCount);
	if (kr != KERN_SUCCESS || outCount != 3)
		return false;
	*bytes = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(out[0]));
	*offset = out[1];
	*size = out[2];
	return true;
}

// The pages source's memory (RgfxPagesMemory::pages): plain pages, zero-filled, tagged
// VM_MEMORY_IOACCELERATOR, which the kext's pages block wires.
int rgfxPlainPages(void *, uint64_t size, uint8_t **bytes)
{
	mach_vm_address_t at = 0;
	const kern_return_t kr = mach_vm_allocate(mach_task_self(), &at, size,
	                                          VM_FLAGS_ANYWHERE |
	                                              VM_MAKE_TAG(VM_MEMORY_IOACCELERATOR));
	if (kr == KERN_SUCCESS)
		*bytes = reinterpret_cast<uint8_t *>(at);
	return kr;
}

void rgfxUnmapPages(void *, uint8_t *bytes, uint64_t size)
{
	mach_vm_deallocate(mach_task_self(), reinterpret_cast<mach_vm_address_t>(bytes), size);
}

// The pool source's memory (RgfxPoolMemory): a block of the Shared pool the kext maps into the
// process once the host answers its take, and its give; `context` is the kext link
// (rgfxKextConn, whose held destroys, a slab's heap's among them, go before the give). The take
// asks the kext for `outputs` answers: 3 (the block's address here, its pool offset and its
// length), or 4, its GPU address too, where the kext answers it (kRgfxKextFeaturePoolAddresses).
static int rgfxKextPoolTakeOf(void *context, uint64_t size, uint8_t **bytes, uint64_t *offset,
                              uint64_t *gpu_address, uint32_t outputs)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t out[4] = {};
	uint32_t outCount = outputs;
	const kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodPoolTake, &size, 1, out, &outCount);
	if (kr != KERN_SUCCESS)
		return kr;
	if (outCount != outputs || out[0] == 0)
		return KERN_FAILURE;
	if (out[2] != size) {
		// Not the block asked for (the kext answers the take's length): back at once.
		IOConnectCallScalarMethod(conn, kRgfxAccelMethodPoolGive, &out[1], 1, nullptr, nullptr);
		return KERN_FAILURE;
	}
	*bytes = reinterpret_cast<uint8_t *>(out[0]);
	*offset = out[1];
	*gpu_address = out[3]; // 0 where the device gave none, or the kext was not asked
	return KERN_SUCCESS;
}

int rgfxKextPoolTake(void *context, uint64_t size, uint8_t **bytes, uint64_t *offset,
                     uint64_t *gpu_address)
{
	return rgfxKextPoolTakeOf(context, size, bytes, offset, gpu_address, 3);
}

int rgfxKextPoolTakeAddressed(void *context, uint64_t size, uint8_t **bytes, uint64_t *offset,
                              uint64_t *gpu_address)
{
	return rgfxKextPoolTakeOf(context, size, bytes, offset, gpu_address, 4);
}

void rgfxKextPoolGive(void *context, uint64_t offset)
{
	io_connect_t conn = rgfxKextConn(context);
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodPoolGive, &offset, 1, nullptr, nullptr);
}

void rgfxKextSurfaceRelease(void *context, uint64_t offset)
{
	io_connect_t conn = rgfxKextConn(context);
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlockRelease, &offset, 1, nullptr,
	                          nullptr);
}

// The region's surface write source (RgfxRegion::setSurfaceWriteSource): a command buffer's
// writes of surfaces' shared blocks to the kext, which keeps a CPU lock of one of them waiting
// until they retire. A write the kext could not keep is its to log; the buffer runs alike.
void rgfxKextSurfaceWrites(void *context, uint64_t queue, uint64_t seqno, const uint64_t *writes,
                           uint32_t count)
{
	io_connect_t conn = rgfxKextConn(context);
	std::vector<uint64_t> words;
	words.reserve(3 + 2 * static_cast<size_t>(count));
	words.push_back(queue);
	words.push_back(seqno);
	words.push_back(count);
	words.insert(words.end(), writes, writes + 2 * static_cast<size_t>(count));
	kern_return_t r = IOConnectCallStructMethod(conn, kRgfxAccelMethodSurfaceWrites, words.data(),
	                                            words.size() * sizeof(uint64_t), nullptr, nullptr);
	// A refusal leaves the surfaces' CPU locks waiting for nothing; the kext logs it too.
	static std::atomic<int> refusals{0};
	if (r != KERN_SUCCESS && refusals.fetch_add(1) < 4)
		rgfxLogLine(@"RGFX: the kext refused the naming of %u surface writes of queue %llu's buffer %llu "
		            @"(0x%x)",
		            count, (unsigned long long)queue, (unsigned long long)seqno, r);
}

void rgfxKextSurfaceRetire(void *context, uint64_t queue, uint64_t seqno)
{
	io_connect_t conn = rgfxKextConn(context);
	const uint64_t in[2] = {queue, seqno};
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceRetire, in, 2, nullptr, nullptr);
}

// The region's surface seeds source (RgfxRegion::setSurfaceSeedsSource): a surface's lock and
// unlock seeds from the kext (kRgfxAccelMethodSurfaceSeeds), false where it answers none.
bool rgfxKextSurfaceSeeds(void *context, uint32_t surfaceId, uint32_t *lockSeed,
                          uint32_t *unlockSeed)
{
	io_connect_t conn = rgfxKextConn(context);
	const uint64_t in = surfaceId;
	uint64_t out[2] = {};
	uint32_t outCount = 2;
	if (IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceSeeds, &in, 1, out, &outCount) !=
	        KERN_SUCCESS ||
	    outCount != 2)
		return false;
	*lockSeed = static_cast<uint32_t>(out[0]);
	*unlockSeed = static_cast<uint32_t>(out[1]);
	return true;
}

// The serializer's queue closer: the kext stops watching the queue's completion ring and posts
// its destroy on the control ring, behind the process's messages before it.
void rgfxKextQueueCloser(void *context, RgfxObjId objId)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in = objId;
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodQueueDestroy, &in, 1, nullptr, nullptr);
}

// The serializer's queue opener: the kext posts the queue's create on the control ring and
// returns once the device has its rings.
bool rgfxKextQueueOpener(void *context, const RgfxQueueCreate &q)
{
	io_connect_t conn = rgfxKextConn(context);
	uint64_t in[6] = {q.obj_id, q.submit_offset, q.complete_offset, q.submit_data_size,
	                  q.complete_data_size, q.priority};
	uint64_t status = ~0ull;
	uint32_t outCount = 1;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodQueueCreate, in, 6, &status, &outCount);
	return kr == KERN_SUCCESS && status == RGFX_STATUS_OK;
}
} // namespace

// The device's doorbell trampoline the queue submission rings ring: it forwards the
// ring id to the device, which (once BAR 0's registers are mapped) writes it to the
// DOORBELL register. `context` is the device, valid for each ring writer's life (the
// device owns them).
@interface RgfxMetalDevice ()
- (void)rgfxRingDoorbell:(uint32_t)ringId;
- (uint64_t)rgfxListenQueue;
@end

#if defined(__arm64__)
@interface RgfxMetalDevice () <RgfxAppleRulesDevice>
@end
#endif

namespace {
void rgfxDeviceDoorbell(void *context, uint32_t ring_id)
{
	[(__bridge RgfxMetalDevice *)context rgfxRingDoorbell:ring_id];
}

// What the serializer's sharer reaches (RgfxSerializer::setEventSharer): the accelerator, which
// the process opens each event's port on, the process's context, and the device, whose queue
// takes the values other processes give its events.
struct RgfxEventShareLink {
	mach_port_t accelerator = MACH_PORT_NULL;
	// The device's own connection, which answers whether a port it is handed is a shared event's
	// port of the kext's (rgfxKextTakeEvent).
	io_connect_t connection = IO_OBJECT_NULL;
	// The device's link, whose held destroys go before the port's calls (rgfxKextConn).
	RgfxKextLink *kext = nullptr;
	uint64_t context = 0;
	__weak RgfxMetalDevice *device = nil;
};

// A shared event's port (kRgfxSharedEventUserClientType): a connection of the kext's opened for
// the event, which names it for other processes (kRgfxSharedEventMethodExport), its values from
// them going to the device's listen queue. The connection's send right, which the event holds;
// MACH_PORT_NULL, nothing left open, where the kext or the device refused it.
uint32_t rgfxKextShareEvent(void *context, RgfxObjId event)
{
	RgfxEventShareLink *link = static_cast<RgfxEventShareLink *>(context);
	RgfxMetalDevice *device = link->device;
	const uint64_t listen = [device rgfxListenQueue];
	if (listen == 0)
		return MACH_PORT_NULL;
	io_connect_t port = IO_OBJECT_NULL;
	kern_return_t kr =
	    IOServiceOpen(link->accelerator, mach_task_self(), kRgfxSharedEventUserClientType, &port);
	if (kr != KERN_SUCCESS)
		return MACH_PORT_NULL;
	const uint64_t in[3] = {link->context, event, listen};
	uint64_t status = ~0ull;
	uint32_t outCount = 1;
	rgfxKextConn(link->kext);
	kr = IOConnectCallScalarMethod(port, kRgfxSharedEventMethodExport, in, 3, &status, &outCount);
	if (kr != KERN_SUCCESS || outCount != 1 || status != RGFX_STATUS_OK) {
		static char site;
		if (rgfxLogFirst(&site, (static_cast<uint64_t>(kr) << 32) ^ status))
			os_log_error(rgfxLog(), "RGFX: shared event 0x%llx: no port (kr=0x%x status %llu)",
			             (unsigned long long)event, kr, (unsigned long long)status);
		IOServiceClose(port);
		return MACH_PORT_NULL;
	}
	return port;
}

// The process's new shared event `event` made of the one the port another process's event (or
// this one's) gave names (kRgfxSharedEventMethodImport), its values from the other holders going
// to the device's listen queue: the event's value in `*value`. False where the port is no shared
// event's port, or the kext or the device made nothing.
bool rgfxKextTakeEvent(void *context, uint32_t port, RgfxObjId event, uint64_t *value)
{
	RgfxEventShareLink *link = static_cast<RgfxEventShareLink *>(context);
	RgfxMetalDevice *device = link->device;
	// Whether the port is a shared event's port of the kext's, asked of the device's connection
	// (RgfxAcceleratorUserClient::connectClient): the kernel looks the port up and sends it
	// nothing, so a send right of another process's own port answers nil at once, as Metal's,
	// where the import's call made on it would kill the process (oracle_events connect).
	if (IOConnectAddClient(link->connection, port) != KERN_SUCCESS)
		return false;
	const uint64_t listen = [device rgfxListenQueue];
	if (listen == 0)
		return false;
	const uint64_t in[3] = {link->context, event, listen};
	uint64_t out[2] = {~0ull, 0};
	uint32_t outCount = 2;
	rgfxKextConn(link->kext);
	kern_return_t kr =
	    IOConnectCallScalarMethod(port, kRgfxSharedEventMethodImport, in, 3, out, &outCount);
	if (kr != KERN_SUCCESS || outCount != 2 || out[0] != RGFX_STATUS_OK)
		return false;
	*value = out[1];
	return true;
}
} // namespace

// The device's `architecture`: Metal's class, whose name is the host's answer in the caps blob.
@interface RgfxArchitecture : MTLArchitecture
- (instancetype)initWithArchitectureName:(NSString *)name;
@end

@implementation RgfxArchitecture {
	NSString *_rgfxName;
}

RGFX_LOG_UNRECOGNIZED

- (instancetype)initWithArchitectureName:(NSString *)name
{
	self = [super init];
	if (self)
		_rgfxName = [name copy];
	return self;
}

- (NSString *)name
{
	return _rgfxName;
}

- (id)copyWithZone:(NSZone *)zone
{
	(void)zone;
	return self; // immutable
}
@end

namespace {

// The device's GPU compiler target, which `targetDeviceArchitecture` answers (it says why).
constexpr uint32_t kRgfxTargetCpuType = 0x0100'5247;
constexpr uint32_t kRgfxTargetSubType = 0;
constexpr uint64_t kRgfxTargetDriverVersion = 0x1'0000;

// Metal's MTLTargetDeviceArchVersion, `{MTLTargetDeviceArchVersion=SCC}` (measured: its setter's
// type on macOS 26.7.1 and 27.0.1): a 16-bit and two 8-bit fields, which its versionCombined
// answers as field_00 << 16 | field_02 << 8 | field_03 (tests/inherited_tests.mm reads it back).
struct RgfxTargetArchVersion {
	uint16_t field_00;
	uint8_t field_02;
	uint8_t field_03;
};
static_assert(sizeof(RgfxTargetArchVersion) == 4, "MTLTargetDeviceArchVersion is 4 bytes");

constexpr RgfxTargetArchVersion kRgfxTargetVersion = {1, 0, 0};

// Whether `cls` has the instance method `sel` of the type encoding `types`.
bool rgfxHasMethod(Class cls, SEL sel, const char *types)
{
	Method m = class_getInstanceMethod(cls, sel);
	return m != nullptr && std::strcmp(method_getTypeEncoding(m), types) == 0;
}

// An MTLTargetDeviceArchitecture naming the device's target; nil where the guest's Metal has no
// such class, or its setters are not the ones this sets.
id rgfxTargetDeviceArchitecture()
{
	Class cls = objc_getClass("MTLTargetDeviceArchitecture");
	if (cls == Nil || !rgfxHasMethod(cls, @selector(setCpuType:), "v20@0:8I16") ||
	    !rgfxHasMethod(cls, @selector(setSubType:), "v20@0:8I16") ||
	    !rgfxHasMethod(cls, @selector(setDriverVersion:), "v24@0:8Q16") ||
	    !rgfxHasMethod(cls, @selector(setVersion:), "v20@0:8{MTLTargetDeviceArchVersion=SCC}16")) {
		os_log_error(rgfxLog(), "RGFX: the guest's Metal has no MTLTargetDeviceArchitecture of the "
		                        "setters the device sets; targetDeviceArchitecture answers nil");
		return nil;
	}
	id a = [[cls alloc] init];
	((void (*)(id, SEL, uint32_t))objc_msgSend)(a, @selector(setCpuType:), kRgfxTargetCpuType);
	((void (*)(id, SEL, uint32_t))objc_msgSend)(a, @selector(setSubType:), kRgfxTargetSubType);
	((void (*)(id, SEL, uint64_t))objc_msgSend)(a, @selector(setDriverVersion:),
	                                            kRgfxTargetDriverVersion);
	((void (*)(id, SEL, RgfxTargetArchVersion))objc_msgSend)(a, @selector(setVersion:),
	                                                         kRgfxTargetVersion);
	return a;
}

} // namespace

@implementation RgfxMetalDevice {
	RgfxCaps _caps;              // host device caps; invalid until the region is read
	RgfxVideoCaps _videoCaps;    // the host's video codecs; none until the region is read
	RgfxIdSpace _ids;            // this device's id/address/resource-id lines
	RgfxSerializer *_serializer; // the create route; null until the ring transport lands
	mach_port_t _acceleratorPort;
	RgfxRegion *_region;         // BAR-region allocator; null until the handshake maps it
	RgfxReplyReader *_replyReader; // the lifecycle's replies, from the kext; null until the handshake
	std::unique_ptr<RgfxCompletionThread> _completions; // the queues' completions; null until then
	RgfxKextLink _link;          // the lifecycle's route to the kext
	std::shared_ptr<RgfxKnownPipelines> _knownPipelines; // null where the kext defers no answers
	// The order of its command buffers across its queues (RgfxCommitOrder.hpp), made with it.
	std::shared_ptr<RgfxCommitOrder> _commitOrder;
	std::vector<std::unique_ptr<RgfxRingWriter>> _queueRings; // owned queue submit-ring writers
	std::mutex _queueRingsLock;  // queues are made and end on any thread
	io_connect_t _conn;          // our accelerator user client (BAR-2 region map + doorbell)
	std::unique_ptr<RgfxRingWriter> _controlRing; // the lifecycle, forwarded to the kext
	mach_vm_address_t _mapBase;  // the window's base in this process (0 until the context opens)
	mach_vm_size_t _mapSize;     // the window's size, the region's
	// The context's doorbell page in this process (RgfxDoorbellPage.hpp): null where the kext
	// mapped none, and the kext's doorbell call rings.
	volatile uint32_t *_doorbellPage;
	std::atomic<bool> _metalAssertions; // setMetalAssertionsEnabled:'s, read back
	std::atomic<bool> _maximizeCompilation; // setShouldMaximizeConcurrentCompilation:'s, read back
	RgfxStateCache *_stateCache; // samplers and depth/stencil states; nil until the handshake
	// The slabs Shared and Managed buffers lie in, and their sources: the region's, and before
	// it the Shared pool's where the device has one, else the process's own pages where the host
	// maps them; null until then.
	std::unique_ptr<RgfxRegionSlabSource> _slabSource;
	std::unique_ptr<RgfxPoolSlabSource> _poolSource;
	std::unique_ptr<RgfxPagesSlabSource> _pagesSource;
	std::unique_ptr<RgfxSlabs> _slabs;
	// Shared events across processes: the sharer's link, and the queue whose completion ring
	// takes the values other processes give the device's shared events, made at the first port
	// asked or taken (rgfxListenQueue), under _listenLock.
	RgfxEventShareLink _shareLink;
	std::mutex _listenLock;
	id<MTLCommandQueue> _listenQueue;
	// The bytes its resources take in this process (currentAllocatedSize), made with it.
	std::shared_ptr<RgfxAllocationCount> _allocationCount;
}

RGFX_LOG_UNRECOGNIZED

- (instancetype)initWithAcceleratorPort:(mach_port_t)port
{
	self = [super initWithAcceleratorPort:port];
	if (self == nil) {
		// Apple's IOGPU base declined the accelerator: its own device calls (RgfxIOGPUReplies.hpp)
		// failed in this process, or its sandbox refused them.
		os_log_error(rgfxLog(),
		             "RGFX: the IOGPU base declined the accelerator; no Metal device in this "
		             "process");
		return nil;
	}
	_acceleratorPort = port;
	_allocationCount = std::make_shared<RgfxAllocationCount>();
	_conn = IO_OBJECT_NULL;
	_serializer = nullptr;
	_region = nullptr;
	_replyReader = nullptr;
	_mapBase = 0;
	_mapSize = 0;
	_doorbellPage = nullptr;

	// Wire transport handshake: open our own user client on the accelerator, take the kext's
	// features and selector map, read the caps blob the host wrote, open the process's context,
	// whose window is where its part of the device region (BAR 2) lies in this task, and build the
	// region allocator over the window and the device submission ring (RgfxSerializer) + device
	// completion ring (RgfxReplyReader); the resource / command-queue / command-buffer categories
	// route over our wire. A step that fails refuses the device, logged: the process has no Metal
	// device, the answer every Metal client handles, where a device that works by guesses fails
	// later and elsewhere (a texture's id the kext never granted, a call another kext numbers
	// otherwise).
	kern_return_t kr = IOServiceOpen(port, mach_task_self(), kRgfxAccelUserClientType, &_conn);
	if (kr != KERN_SUCCESS) {
		os_log_error(rgfxLog(),
		             "RGFX: IOServiceOpen(type=%u) failed kr=0x%x; no Metal device in this "
		             "process",
		             kRgfxAccelUserClientType, kr);
		_conn = IO_OBJECT_NULL;
		return nil;
	}

	// The device's feature word, the kext's own and the kext's selector map, first: a kext of
	// another map, or a sandbox that refuses the call, leaves no number of this bundle's sure to
	// mean its call there (RgfxUserClientABI.hpp).
	uint64_t features[3] = {};
	uint32_t featuresCount = 3;
	kr = IOConnectCallScalarMethod(_conn, kRgfxAccelMethodFeatures, nullptr, 0, features,
	                               &featuresCount);
	if (!_caps.takeFeatures(kr, featuresCount, features)) {
		os_log_error(rgfxLog(),
		             "RGFX: the features call (selector %u) answered kr=0x%x, %u words, selector "
		             "map %llu, where this bundle's map is %u: a kext of another selector map, or "
		             "a sandbox that refuses the call; no Metal device in this process",
		             (unsigned)kRgfxAccelMethodFeatures, kr, featuresCount,
		             (unsigned long long)features[2], (unsigned)kRgfxSelectorMap);
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return nil;
	}

	// The caps blob the host wrote (read once at init), which the kext reads for us: its page
	// is no process's to map. The video caps follow it; where the host wrote them in another
	// version, the device has no video codec.
	struct {
		RgfxDeviceInfo info;
		RgfxVideoCaps video;
	} blob = {};
	static_assert(offsetof(decltype(blob), video) == sizeof(RgfxDeviceInfo), "no padding");
	size_t infoSize = sizeof(blob);
	kr = IOConnectCallStructMethod(_conn, kRgfxAccelMethodDeviceInfo, nullptr, 0, &blob,
	                               &infoSize);
	if (kr != KERN_SUCCESS || infoSize != sizeof(blob) || !_caps.load(blob.info)) {
		os_log_error(rgfxLog(),
		             "RGFX: the device info was not read (kr=0x%x, %zu of %zu bytes) or is of "
		             "version %u, where this bundle reads %u; no Metal device in this process",
		             kr, infoSize, sizeof(blob), (unsigned)blob.info.version,
		             (unsigned)RGFX_DEVICE_INFO_VERSION);
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return nil;
	}
	if (blob.video.version == RGFX_VIDEO_CAPS_VERSION) {
		_videoCaps = blob.video;
		if (_videoCaps.codec_count > RGFX_VIDEO_CAPS_CODECS)
			_videoCaps.codec_count = RGFX_VIDEO_CAPS_CODECS;
	} else if (blob.video.version == 0) {
		// A host with no hardware video writes no caps: a macOS host, whose library has no
		// video arm, or a Linux host where neither NVDEC nor VA-API opens on the GPU.
		os_log(rgfxLog(),
		       "RGFX: the host has no hardware video; this process's device has no video codec");
	} else {
		os_log_error(rgfxLog(),
		             "RGFX: the host's video caps are of version %u, where this bundle reads %u; "
		             "this process's device has no video codec",
		             (unsigned)blob.video.version, (unsigned)RGFX_VIDEO_CAPS_VERSION);
	}
	// Where the host orders command buffers of different queues over the surfaces they share,
	// this process names each one's order (rgfxIOSurfaceNameOrder).
	rgfxIOSurfaceSetOrder((_caps.features() & RGFX_FEATURE_SURFACE_ORDER) != 0);
	// Where the host commits in the order the process's queues put their buffers on their rings,
	// each queue takes part in it (newCommandQueue; RgfxCommitOrder.hpp).
	if ((_caps.features() & RGFX_FEATURE_COMMIT_ORDER) != 0)
		_commitOrder = std::make_shared<RgfxCommitOrder>();

	// The process's context: its id and address ranges, its window, its blocks and its chunks
	// of texture resource ids, from the kext, so no two processes' rings, blocks or numbers
	// meet, and this one maps nothing of the region but its own. The span of its GPU addresses
	// comes sixth where the kext answers it; a kext that does not gives RGFX_CONTEXT_VA_SPAN. Its
	// doorbell page comes seventh where the kext answers that (RgfxDoorbellPage.hpp).
	const uint32_t ctxWanted = rgfxOpenContextOutputs(_caps.kextFeatures());
	uint64_t ctx[7] = {0, 0, 0, 0, 0, RGFX_CONTEXT_VA_SPAN, 0};
	uint32_t ctxCount = ctxWanted;
	kr = IOConnectCallScalarMethod(_conn, kRgfxAccelMethodOpenContext, nullptr, 0, ctx, &ctxCount);
	if (kr == kIOReturnNoResources) {
		// The kext has no range of GPU addresses left for another process: no device here, the
		// answer every Metal client handles, where a device whose every create answers nil
		// stops the frameworks that make their first objects at its open.
		os_log_error(rgfxLog(),
		             "RGFX: OPEN_CONTEXT refused: the device holds as many processes as it "
		             "can; no Metal device in this one");
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return nil;
	}
	if (kr != KERN_SUCCESS || ctxCount != ctxWanted || ctx[3] == 0 ||
	    ctx[4] < RGFX_ALLOCATIONS_OFFSET || ctx[5] == 0) {
		os_log_error(rgfxLog(),
		             "RGFX: OPEN_CONTEXT failed kr=0x%x (window 0x%llx, %llu bytes); no Metal "
		             "device in this process",
		             kr, (unsigned long long)ctx[3], (unsigned long long)ctx[4]);
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return nil;
	}
	const mach_vm_address_t addr = ctx[3];
	const mach_vm_size_t mapSize = ctx[4];
	_mapBase = addr;
	_mapSize = mapSize;
	_doorbellPage = rgfxDoorbellPageOf(ctx, ctxCount);
	volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(addr);
	_ids.setBases(ctx[1], ctx[2], ctx[5]);
	// The kext's calls go through the link (rgfxKextConn), after the destroys the control ring
	// holds once it is made.
	_link.conn = _conn;
	void *connContext = &_link;
	// The kext's chunks are of RGFX_RESOURCE_ID_SMALL_CHUNK ids where it took
	// RGFX_FEATURE_SMALL_ID_CHUNKS (RgfxUserClientABI.hpp, kRgfxAccelMethodResourceIds).
	_ids.setResourceIdSource(&rgfxKextResourceIds, connContext,
	                         (_caps.features() & RGFX_FEATURE_SMALL_ID_CHUNKS) != 0
	                             ? RGFX_RESOURCE_ID_SMALL_CHUNK
	                             : RGFX_RESOURCE_ID_CHUNK);
	// Private objects' addresses lie in blocks the host gives where the kext and the device both
	// took them (RgfxObjectModel.hpp).
	if ((_caps.kextFeatures() & kRgfxKextFeaturePrivateBlocks) != 0 &&
	    (_caps.features() & RGFX_FEATURE_PRIVATE_BLOCKS) != 0)
		_ids.setPrivateBlockSource(&rgfxKextPrivateTake, &rgfxKextPrivateGive, connContext);
	_region = new RgfxRegion(base, mapSize);
	_region->setBlockSource(&rgfxKextAlloc, &rgfxKextRelease, connContext);
	// The bytes the process only writes in write-combined parts of their blocks, where the kext
	// maps them so (RgfxRegion.hpp, WRITE-COMBINED PARTS).
	if ((_caps.kextFeatures() & kRgfxKextFeatureWriteCombined) != 0)
		_region->setWriteCombinedSource(&rgfxKextAllocWriteCombined);
	_region->setSurfaceSource(&rgfxKextSurfaceBlock, &rgfxKextSurfaceRelease, connContext);
	// Its asks list their surfaces' few resident pages where a deferral may carry them.
	_region->setSurfacePagesSource((_caps.features() & RGFX_FEATURE_SURFACE_DEFERRED_CARRY) != 0
	                                   ? &rgfxKextSurfacePagesCarrying
	                                   : &rgfxKextSurfacePages,
	                               _caps.linearTextureAlignment(), connContext);
	_region->setSurfaceWriteSource(&rgfxKextSurfaceWrites, &rgfxKextSurfaceRetire, connContext);
	// A copy synced while a CPU held its surface locked for writing syncs again at the unlock,
	// where the kext reads IOSurface's unlock seed (RgfxIOSurface.h).
	const bool seeds = (_caps.kextFeatures() & kRgfxKextFeatureSurfaceSeeds) != 0;
	if (seeds)
		_region->setSurfaceSeedsSource(&rgfxKextSurfaceSeeds, connContext);
	rgfxIOSurfaceNoteSurfaceSeeds(seeds);
	// So does a cached surface's device cache filled while such a lock was held, where the host
	// fills one again when a command buffer's use asks it (RGFX_SURFACE_USE_REFILL).
	rgfxIOSurfaceSetRefill(seeds && (_caps.features() & RGFX_FEATURE_SURFACE_REFILL) != 0);
	// Where the device caches every surface the kext gives in place, a view that is a copy is a
	// fallback, logged by its reason.
	rgfxIOSurfaceSetCacheEvery((_caps.features() & RGFX_FEATURE_SURFACE_CACHE_EVERY) != 0);
	// No-copy buffers over the process's own pages, where the kext makes pages blocks and the
	// host took the feature that maps them.
	if ((_caps.kextFeatures() & kRgfxKextFeaturePagesBlock) != 0 &&
	    (_caps.features() & RGFX_FEATURE_PAGES_BUFFERS) != 0) {
#if defined(__arm64__)
		// Whole 16 KiB pages where the kext took PAGES_16K, as it rounds the blocks' ranges then.
		_region->setPagesPage((_caps.features() & RGFX_FEATURE_PAGES_16K) != 0 ? 0x4000 : 0x1000);
#endif
		// With the GPU address the host gave each block's view, where both took the feature
		// (RgfxBuffer's bufferOverPagesOf places its buffers there).
		const bool addressed =
		    (_caps.kextFeatures() & kRgfxKextFeatureImportAddresses) != 0 &&
		    (_caps.features() & RGFX_FEATURE_IMPORT_ADDRESSES) != 0;
		_region->setPagesBlockSource(addressed ? &rgfxKextPagesBlockAt : &rgfxKextPagesBlock,
		                             connContext);
		// Read-only memory that cannot change, which the host reads once, where both took the
		// feature (RgfxFilePages.hpp); the host offers it with READ_ONLY_PAGES.
		_region->setUnchangingPages(
		    (_caps.kextFeatures() & kRgfxKextFeatureUnchangingPages) != 0 &&
		    (_caps.features() & (RGFX_FEATURE_UNCHANGING_PAGES | RGFX_FEATURE_READ_ONLY_PAGES)) ==
		        (RGFX_FEATURE_UNCHANGING_PAGES | RGFX_FEATURE_READ_ONLY_PAGES));
		// Anonymous memory the app's mapping alone holds among it, where the kext checks no
		// writable block of its own covers it.
		_region->setSoleHolderPages(_region->unchangingPages() &&
		                            (_caps.kextFeatures() & kRgfxKextFeatureSoleHolderPages) != 0);
		// Slabs of the kext's chunks first, where it makes them.
		if ((_caps.kextFeatures() & kRgfxKextFeaturePagesChunked) != 0)
			_region->setChunkedPagesSource(&rgfxKextPagesChunked, connContext);
	}

	// The object lifecycle goes to the kext, which posts it on the control ring and hands back
	// the replies (a pipeline's reflection).
	_replyReader = new RgfxReplyReader();
	_link.reader = _replyReader;
	_controlRing = std::make_unique<RgfxRingWriter>(&rgfxKextSubmit, &_link, kLifecycleMessageMax);
	// The process's destroys, and its creates and writes no one waits for that name no block, go
	// together, where the kext takes them in one call (RgfxRingWriter::postLater).
	if ((_caps.kextFeatures() & kRgfxKextFeatureSubmitBatch) != 0) {
		_controlRing->setBatch(&rgfxKextSubmitBatch, kRgfxSubmitBatchBytes);
		_link.control = _controlRing.get();
		// A commit behind held messages hands its buffer's publish to them, where the kext makes
		// it after them (RgfxQueuePublish.hpp): one call, not the held and then the doorbell.
		if ((_caps.kextFeatures() & kRgfxKextFeatureQueuePublish) != 0)
			_controlRing->setBatchPublish(&rgfxKextSubmitBatchPublish);
	}
	// A pipeline create equal to one the device accepted is posted without waiting, where the
	// kext keeps its answer for later (RgfxKnownPipelines.hpp).
	if (rgfxKextDefersAnswers(_caps)) {
		_controlRing->setDeferred(&rgfxKextSubmitDeferred, &rgfxKextAwaitAnswer, &_link);
		_knownPipelines = std::make_shared<RgfxKnownPipelines>();
		// An equal create of another process is posted without waiting too, where the kext keeps
		// the answers by pipeline key and the host gives them (RgfxSharedPipelines.hpp).
		if ((_caps.kextFeatures() & kRgfxKextFeatureSharedPipelines) != 0 &&
		    (_caps.features() & RGFX_FEATURE_PIPELINE_KEYS) != 0)
			_controlRing->setShared(&rgfxKextSubmitShared);
	}
	_serializer = new RgfxSerializer(*_controlRing, _ids);
	_serializer->setQueueOpener(&rgfxKextQueueOpener, connContext);
	_serializer->setQueueCloser(&rgfxKextQueueCloser, connContext);
	_serializer->setAsker(&rgfxKextAsk, connContext);
	_serializer->setCarriesPrivateRenderState((_caps.features() & RGFX_FEATURE_PRIVATE_RENDER_STATE) !=
	                                          0);
	_serializer->setCarriesTextureDidModifyData(
	    (_caps.features() & RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA) != 0);
	_serializer->setCarriesBufferEscaped((_caps.features() & RGFX_FEATURE_BUFFER_ESCAPED) != 0);
	_serializer->setCarriesRateMaps((_caps.features() & RGFX_FEATURE_RATE_MAPS) != 0);
	_serializer->setSharesPipelines(_controlRing->shares());
	_serializer->setCarriesFunctionPointers((_caps.features() & RGFX_FEATURE_FUNCTION_POINTERS) !=
	                                        0);
	_serializer->setCarriesDynamicLibraries((_caps.features() & RGFX_FEATURE_DYNAMIC_LIBRARIES) !=
	                                        0);
	_serializer->setManagedAtCopy((_caps.features() & RGFX_FEATURE_MANAGED_AT_COPY) != 0);
	// Buffers over pages blocks at the host's address, asked where it is read, where the host
	// answers a block before its view has one (RgfxBuffer's bufferOverPagesOf): both took
	// IMPORT_ADDRESSES and IMPORT_ADDRESSES_LATER.
	const uint64_t later = RGFX_FEATURE_IMPORT_ADDRESSES | RGFX_FEATURE_IMPORT_ADDRESSES_LATER;
	_serializer->setPagesLater((_caps.kextFeatures() & kRgfxKextFeatureImportAddresses) != 0 &&
	                           (_caps.features() & later) == later);
	// Shared events cross processes through the kext's ports where the kext makes them and the
	// host took the feature that names their events (RgfxUserClientABI.hpp,
	// kRgfxSharedEventUserClientType).
	if ((_caps.kextFeatures() & kRgfxKextFeatureSharedEvents) != 0 &&
	    (_caps.features() & RGFX_FEATURE_SHARED_EVENT_NAMES) != 0) {
		_shareLink.accelerator = port;
		_shareLink.connection = _conn;
		_shareLink.kext = &_link;
		_shareLink.context = ctx[0];
		_shareLink.device = self;
		_serializer->setEventSharer(&rgfxKextShareEvent, &rgfxKextTakeEvent, &_shareLink);
	}
	_stateCache = [[RgfxStateCache alloc] initWithDevice:self serializer:_serializer];
	// Managed buffers lie over Shared copies of the Shared buffers' slabs where the host makes
	// them over any Shared buffer (MANAGED_BUFFER_SLABS); else their CPU copies lie in heaps of
	// their own slabs where the host places them there.
	const bool managedAtOffset = (_caps.features() & RGFX_FEATURE_MANAGED_AT_OFFSET) != 0;
	const bool managedOverCopy = (_caps.features() & RGFX_FEATURE_MANAGED_BUFFER_SLABS) != 0;
	_slabSource = std::make_unique<RgfxRegionSlabSource>(*_region, *_serializer, managedAtOffset,
	                                                     managedOverCopy);
	_slabSource->setZeroedBlocks((_caps.kextFeatures() & kRgfxKextFeatureZeroedBlocks) != 0);
	// Every slab is a block of the Shared pool, where the device has one, the kext maps its
	// blocks and Managed buffers lie over Shared copies (MANAGED_BUFFER_SLABS, which every host
	// that offers the pool offers): the pool is the slabs' one source, with no region behind it,
	// so a buffer the pool has no block for is nil (RgfxSlabs.hpp, THE SOURCES). Else the slabs
	// whose buffers lie in heaps, and large Shared buffers, are the kext's chunks, or the process's
	// own plain pages, wired into pages blocks the host makes heaps or buffers over, where the
	// region has a pages block source (above) and the device took HEAP_OVER_PAGES, ahead of the
	// region's; else the region's.
	if ((_caps.kextFeatures() & kRgfxKextFeaturePool) != 0 &&
	    (_caps.features() & RGFX_FEATURE_POOL) != 0 && managedOverCopy) {
		// Its blocks' heaps lie at the GPU addresses the host gives them, where the device and
		// the kext both took POOL_ADDRESSES; else at the process's picks.
		const bool addressed = (_caps.kextFeatures() & kRgfxKextFeaturePoolAddresses) != 0 &&
		                       (_caps.features() & RGFX_FEATURE_POOL_ADDRESSES) != 0;
		RgfxPoolMemory memory;
		memory.context = connContext;
		memory.take = addressed ? &rgfxKextPoolTakeAddressed : &rgfxKextPoolTake;
		memory.give = &rgfxKextPoolGive;
		_poolSource = std::make_unique<RgfxPoolSlabSource>(*_region, *_serializer, memory,
		                                                   managedAtOffset, managedOverCopy);
		_slabs = std::make_unique<RgfxSlabs>(*_poolSource);
	} else if (_region->hasPagesBlocks() &&
	           (_caps.features() & RGFX_FEATURE_HEAP_OVER_PAGES) != 0) {
		RgfxPagesMemory memory;
		memory.pages = &rgfxPlainPages;
		memory.unmap = &rgfxUnmapPages;
		_pagesSource = std::make_unique<RgfxPagesSlabSource>(
		    *_region, *_serializer, memory, [NSProcessInfo processInfo].physicalMemory,
		    managedAtOffset, managedOverCopy);
		_slabs = std::make_unique<RgfxSlabs>(*_slabSource);
		_slabs->setFirst(_pagesSource.get());
	} else {
		_slabs = std::make_unique<RgfxSlabs>(*_slabSource);
	}
	_slabs->setChurnLog(&rgfxLogSlabChurn, _poolSource.get());
	if (!_slabs->start())
		os_log_error(rgfxLog(),
		             "RGFX: no slab thread; creates zero and frees let slabs go themselves");
	// A block the kext has no room for is asked again once the slabs the process keeps went.
	_region->setReclaim(&rgfxRegionReclaim, _slabs.get());
	// The queues' completions, taken as the kext wakes the thread for them.
	_completions = std::make_unique<RgfxCompletionThread>(&rgfxKextWaitCompletions, connContext);
	if (!_completions->start()) {
		os_log_error(rgfxLog(), "RGFX: no completion thread; waiters take the rings");
		RgfxResourceWaitsStop();
		_completions.reset();
	}
	return self;
}

// The queue whose completion ring takes the values other processes give this process's shared
// events (SHARED_EVENT_NAME's listen queue): a queue of the device's own, which no command buffer
// uses, made at the first ask and kept. Its id; 0 where none could be made.
- (uint64_t)rgfxListenQueue
{
	std::lock_guard<std::mutex> hold(_listenLock);
	if (_listenQueue == nil)
		_listenQueue = [self newCommandQueue];
	return _listenQueue != nil ? [(RgfxResource *)_listenQueue rgfxObjId] : 0;
}

// --- accessors the category slices route through -----------------------------

- (RgfxSerializer *)serializer
{
	return _serializer;
}

- (std::shared_ptr<RgfxKnownPipelines>)knownPipelines
{
	return _knownPipelines;
}

- (RgfxIdSpace *)ids
{
	return &_ids;
}

- (std::shared_ptr<RgfxCommitOrder>)commitOrder
{
	return _commitOrder;
}

- (RgfxStateCache *)stateCache
{
	return _stateCache;
}

- (const RgfxVideoCaps *)videoCaps
{
	return &_videoCaps;
}

- (const RgfxCaps *)caps
{
	return &_caps;
}

- (RgfxRegion *)region
{
	return _region;
}

- (RgfxSlabs *)slabs
{
	return _slabs.get();
}

- (RgfxPoolSlabSource *)rgfxPool
{
	return _poolSource.get();
}

// A texture over a legacy IOAccel surface, which is how WindowServer's CoreDisplay reaches the
// surface a display is drawn into. The kext answers where the surface's memory lies, the
// framebuffer's scanout in the region, and lets this process name it; the texture lies over a
// Shared buffer over that span, so what the GPU draws there is what the display shows. Nil for
// a surface that is not the scanout, as the base answers for every surface.
- (id<MTLTexture>)newTextureWithSurface:(NSUInteger)surfaceId buffer:(NSUInteger)buffer
{
	if (_conn == IO_OBJECT_NULL || _region == nullptr || [self serializer] == nullptr) {
		return nil;
	}
	uint64_t in[2] = {surfaceId, buffer};
	uint64_t out[5] = {};
	uint32_t outCount = 5;
	kern_return_t kr = IOConnectCallScalarMethod(
	    rgfxKextConn(&_link), kRgfxAccelMethodSurfaceScanout, in, 2, out, &outCount);
	static std::atomic<int> logged{0};
	if (logged.fetch_add(1) < 8) {
		rgfxLogLine(@"RGFX: newTextureWithSurface:0x%lx buffer:%lu -> 0x%x: offset 0x%llx, %llu bytes, "
		            @"%llux%llu, %llu bytes a row",
		            static_cast<unsigned long>(surfaceId), static_cast<unsigned long>(buffer), kr,
		            out[0], out[1], out[2], out[3], out[4]);
	}
	if (kr != KERN_SUCCESS || outCount != 5) {
		return nil;
	}
	uint64_t offset = out[0], length = out[1], width = out[2], height = out[3];
	uint64_t rowBytes = out[4];
	if (offset > _region->size() || length > _region->size() - offset ||
	    rowBytes * height > length) {
		return nil;
	}
	RgfxBuffer *span =
	    [RgfxBuffer bufferOverHostMemoryWithContents:const_cast<uint8_t *>(_region->base() + offset)
	                                    regionOffset:offset
	                                          length:static_cast<NSUInteger>(length)
	                                         options:MTLResourceStorageModeShared
	                                     deallocator:nil
	                                          device:self
	                                      serializer:[self serializer]];
	if (span == nil) {
		return nil;
	}
	// The scanout's pixels: BGRA8.
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:static_cast<NSUInteger>(width)
	                                                      height:static_cast<NSUInteger>(height)
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModeShared;
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
	return [RgfxTexture textureWithDescriptor:d
	                                   buffer:span
	                                   offset:0
	                              bytesPerRow:rowBytes
	                                   device:self
	                               serializer:[self serializer]];
}

- (RgfxReplyReader *)replyReader
{
	return _replyReader;
}

- (RgfxCompletionThread *)completionThread
{
	return _completions.get();
}

- (RgfxRingWriter *)addQueueSubmitRingAtOffset:(uint64_t)regionOffset dataSize:(uint32_t)dataSize
{
	if (_region == nullptr) {
		return nullptr;
	}
	// The host keys a per-queue ring's doorbell by its submission ring's region offset
	// (device.rs `ring_doorbell` -> `by_ring`, keyed on `submit_offset`), the control
	// ring's id being its offset 0. So the doorbell id is the region offset itself, not a
	// serial counter, or the host finds no ring for the wake and the queue never runs.
	uint32_t ringId = static_cast<uint32_t>(regionOffset);
	auto writer = std::make_unique<RgfxRingWriter>(_region->base() + regionOffset, dataSize, ringId,
	                                               &rgfxDeviceDoorbell, (__bridge void *)self);
	RgfxRingWriter *raw = writer.get();
	std::lock_guard<std::mutex> hold(_queueRingsLock);
	_queueRings.push_back(std::move(writer));
	return raw;
}

- (void)dropQueueSubmitRing:(RgfxRingWriter *)ring
{
	std::unique_ptr<RgfxRingWriter> dropped;
	{
		std::lock_guard<std::mutex> hold(_queueRingsLock);
		auto it = std::find_if(_queueRings.begin(), _queueRings.end(),
		                       [ring](const std::unique_ptr<RgfxRingWriter> &w) { return w.get() == ring; });
		if (it == _queueRings.end())
			return;
		dropped = std::move(*it);
		_queueRings.erase(it);
	}
}

- (void)rgfxRingDoorbell:(uint32_t)ringId
{
	if (_conn == IO_OBJECT_NULL)
		return;
	// After the destroys the control ring holds, as every call of the process's into the kext.
	const io_connect_t conn = rgfxKextConn(&_link);
	// At the context's doorbell page, where the kext mapped one: a store the host takes, waking
	// its consumer of the ring, with no call into the kext (RgfxDoorbellPage.hpp).
	if (_doorbellPage != nullptr) {
		rgfxDoorbellPageRing(_doorbellPage, ringId);
		return;
	}
	// Else through our accelerator user client: the kext writes `ringId` to RGFX_REG_DOORBELL in
	// BAR 0.
	uint64_t scalar = ringId;
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodDoorbell, &scalar, 1, nullptr, nullptr);
}

// --- identity ----------------------------------------------------------------

// The host's name of its GPU, as the device info carries it, or "reims-graphics" where it
// carries none: the name an arm64 guest's device answers, and the one the kext states as the GPU's
// `model`, which System Information shows (the kext's RgfxDeviceName.hpp), on either guest.
static NSString *rgfxHostNameOf(const RgfxCaps &caps)
{
	if (caps.valid() && caps.name()[0] != '\0') {
		return [NSString stringWithUTF8String:caps.name()];
	}
	return @"reims-graphics";
}

// On an x86-64 guest the host's name after the prefix of the host GPU's type (rgfxMetalNameOf):
// "AMD - NVIDIA GeForce RTX 3090" on a discrete GPU, "Intel - AMD Radeon 780M Graphics (RADV
// PHOENIX)" on an integrated one. On an arm64 guest the host's name.
- (NSString *)name
{
#if defined(__x86_64__)
	const std::string name = rgfxMetalNameOf(_caps.name(), _caps.hostIntegrated());
	return [NSString stringWithUTF8String:name.c_str()];
#else
	return rgfxHostNameOf(_caps);
#endif
}

// The MTLDeviceSPI family name, from the host's name (rgfxFamilyNameOf), on either guest:
// "GeForce RTX" for "NVIDIA GeForce RTX 5080 Laptop GPU". Metal's devices answer the name without
// the vendor and the model number, each its own, so the answer is the lane's choice inside what
// they give (dgpu-oracle.md class 2); the IOGPU base answered "", which no Mac does.
- (NSString *)familyName
{
	const std::string name = rgfxFamilyNameOf(rgfxHostNameOf(_caps).UTF8String);
	return [NSString stringWithUTF8String:name.c_str()];
}

// The MTLDeviceSPI vendor name, the host GPU's vendor from the host's name (rgfxVendorNameOf), on
// either guest: "NVIDIA" for "NVIDIA GeForce RTX 5080 Laptop GPU". The IOGPU base answered
// "Apple", on which the system's GL renderer ends every process that makes a GL context.
- (NSString *)vendorName
{
	const std::string name = rgfxVendorNameOf(rgfxHostNameOf(_caps).UTF8String);
	return [NSString stringWithUTF8String:name.c_str()];
}

- (uint64_t)registryID
{
	// The guest's own accelerator's registry entry ID: WindowServer finds the Metal device
	// that drives a display by matching this against the accelerator the display's
	// framebuffer names, so the host GPU's ID would match no display. The host's ID is
	// the answer only where the accelerator has none.
	uint64_t entryId = 0;
	if (_acceleratorPort != MACH_PORT_NULL &&
	    IORegistryEntryGetRegistryEntryID(_acceleratorPort, &entryId) == KERN_SUCCESS && entryId != 0)
		return entryId;
	return _caps.valid() ? _caps.registryId() : 0;
}

- (MTLSize)maxThreadsPerThreadgroup
{
	if (_caps.valid()) {
		return MTLSizeMake(_caps.maxThreadsPerThreadgroup(0), _caps.maxThreadsPerThreadgroup(1),
		                   _caps.maxThreadsPerThreadgroup(2));
	}
	return MTLSizeMake(1024, 1024, 1024);
}

// Every identity and capability getter below answers the caps blob's value, which the host
// writes from its platform's answers (and the lane's, where its device can do less); 0 or NO
// where the blob is invalid or the host has no answer.

- (MTLArchitecture *)architecture
{
	if (_caps.valid() && _caps.architectureName()[0] != '\0') {
		NSString *name = [NSString stringWithUTF8String:_caps.architectureName()];
		return [[RgfxArchitecture alloc] initWithArchitectureName:name];
	}
	// No answer from the host: the base's own, where it has one, else an empty name.
	Class base = class_getSuperclass([RgfxMetalDevice class]);
	if (class_getInstanceMethod(base, @selector(architecture)) != nullptr) {
		struct objc_super sup = {self, base};
		return ((MTLArchitecture * (*)(struct objc_super *, SEL)) objc_msgSendSuper)(
		    &sup, @selector(architecture));
	}
	return [[RgfxArchitecture alloc] initWithArchitectureName:@""];
}

// The device's GPU compiler target, MTLDeviceSPI's `targetDeviceArchitecture`: an object of
// Metal's own class, MTLTargetDeviceArchitecture, as every Mac's device answers (measured:
// tools/oracle/oracle_archtype.m on the AMD 5300M, the Intel 630 and the M3 Max,
// docs/graphics-kext/dgpu-oracle.md Q42). nil, which the IOGPU base answered, is outside that:
// Metal's loader ends the process at it (+[MTLLoader sliceIDForDevice:legacyDriverVersion:
// airntDriverVersion:], read from macOS 27's Metal), and MTLAssetUpgraderD, which asks Metal's
// archive reader about every .metallib of an app it is told of for every device, ended with
// SIGABRT at each app installed in the guest. Each Mac's answer names its own compiler's
// target, so the device's names its own:
// - cpuType 0x01005247: CPU_ARCH_ABI64 with the device's PCI device id, 0x5247. No Metal slice
//   carries it (Apple's GPUs are 0x01000013, AMD's 0x01000014, Intel's 0x01000015, AIR
//   0x01000017), so no archive or dynamic library holds a slice Metal reads as this device's:
//   the device makes its pipelines from AIR on the host, and its archives are of its own format
//   (RgfxBinaryArchiveFile.hpp).
// - subType 0: the device has one target, whatever the host's GPU.
// - version 1.0.0, versionCombined 0x10000, and driverVersion 0x10000, the same value, as the M3
//   Max (0x30000) and the Intel 630 (0x1a0701) answer theirs.
// Where the guest's Metal has no such class, nil.
- (id)targetDeviceArchitecture
{
	static id answer = rgfxTargetDeviceArchitecture();
	return answer;
}

- (BOOL)hasUnifiedMemory
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityHasUnifiedMemory);
}

- (BOOL)isLowPower
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityIsLowPower);
}

- (BOOL)isHeadless
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityIsHeadless);
}

- (BOOL)isRemovable
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityIsRemovable);
}

- (MTLDeviceLocation)location
{
	return static_cast<MTLDeviceLocation>(_caps.size(kRgfxSizeLocation));
}

- (NSUInteger)locationNumber
{
	return _caps.size(kRgfxSizeLocationNumber);
}

- (uint64_t)maxTransferRate
{
	return _caps.size(kRgfxSizeMaxTransferRate);
}

- (uint64_t)peerGroupID
{
	return _caps.size(kRgfxSizePeerGroupId);
}

- (uint32_t)peerIndex
{
	return static_cast<uint32_t>(_caps.size(kRgfxSizePeerIndex));
}

- (uint32_t)peerCount
{
	return static_cast<uint32_t>(_caps.size(kRgfxSizePeerCount));
}

// The bytes the device's resources take in this process: each one it makes counts its
// allocatedSize while it lives, a heap its size (RgfxAllocationCount.hpp, RgfxCounted).
- (NSUInteger)currentAllocatedSize
{
	return _allocationCount ? static_cast<NSUInteger>(_allocationCount->bytes()) : 0;
}

- (std::shared_ptr<RgfxAllocationCount>)rgfxAllocationCount
{
	return _allocationCount;
}

- (uint64_t)recommendedMaxWorkingSetSize
{
	return _caps.size(kRgfxSizeRecommendedMaxWorkingSetSize);
}

// The guest's own logical processor count, at either state of
// shouldMaximizeConcurrentCompilation (rgfxConcurrentCompilationTaskCount, RgfxCaps.hpp, which
// says why), never the host's.
- (NSUInteger)maximumConcurrentCompilationTaskCount
{
	return static_cast<NSUInteger>(rgfxConcurrentCompilationTaskCount());
}

// NO in each new device, then what setShouldMaximizeConcurrentCompilation: set; the count above
// does not follow it.
- (BOOL)shouldMaximizeConcurrentCompilation
{
	return _maximizeCompilation.load(std::memory_order_relaxed);
}

// Kept per device, and passed on to _MTLDevice's setter where the base has it (MTLDevice's from
// macOS 13.3; RgfxMetalPrivate.h), which took every call before the device kept its own: whatever
// the base does with it, which is not measured, it still does. The device's source libraries
// compile through the base, in Metal's compiler service (RgfxCompileOverMetal).
- (void)setShouldMaximizeConcurrentCompilation:(BOOL)maximize
{
	_maximizeCompilation.store(maximize, std::memory_order_relaxed);
	if ([IOGPUMetalDevice instancesRespondToSelector:_cmd])
		[super setShouldMaximizeConcurrentCompilation:maximize];
}

// --- limits (RgfxCaps) -------------------------------------------------------

- (NSUInteger)maxBufferLength
{
	return _caps.size(kRgfxSizeMaxBufferLength);
}

- (NSUInteger)maxThreadgroupMemoryLength
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxThreadgroupMemoryLength) : 32768;
}

- (NSUInteger)maxArgumentBufferSamplerCount
{
	return _caps.size(kRgfxSizeMaxArgumentBufferSamplerCount);
}

- (MTLArgumentBuffersTier)argumentBuffersSupport
{
	return static_cast<MTLArgumentBuffersTier>(_caps.size(kRgfxSizeArgumentBuffersSupport));
}

- (MTLReadWriteTextureTier)readWriteTextureSupport
{
	return static_cast<MTLReadWriteTextureTier>(_caps.size(kRgfxSizeReadWriteTextureSupport));
}

- (NSUInteger)sparseTileSizeInBytes
{
	return _caps.size(kRgfxSizeSparseTileSizeInBytes);
}

- (NSUInteger)minimumTextureBufferAlignmentForPixelFormat:(MTLPixelFormat)format
{
	return _caps.textureBufferAlignment(static_cast<uint64_t>(format));
}

// The format's own linear alignment, the host GPU's (RgfxCaps::minimumLinearTextureAlignment),
// for the formats the x86-64 Macs answer (both answer 256 for each), or in an arm64 guest those
// the M3 Max answers (16 for each); Metal ends the process at every other
// (rgfxEndAtUnalignableLinearFormat). The bundle's own callers ask
// rgfxLinearTextureAlignmentForPixelFormat:, which ends nothing, as Metal's own texture creators
// over a buffer do not end it there.
- (NSUInteger)minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format
{
	rgfxEndAtUnalignableLinearFormat(format, self);
	return [self rgfxLinearTextureAlignmentForPixelFormat:format];
}

- (NSUInteger)rgfxLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format
{
	return _caps.minimumLinearTextureAlignment(static_cast<uint64_t>(format));
}

- (NSUInteger)linearTextureAlignmentBytes
{
	return _caps.limit(kRgfxLimitLinearTextureAlignmentBytes);
}

- (NSUInteger)deviceLinearTextureAlignmentBytes
{
	return _caps.limit(kRgfxLimitDeviceLinearTextureAlignmentBytes);
}

// Metal's 16 for both, never the host's answer (kRgfxIOSurfaceTextureAlignmentBytes says why):
// CoreAnimation makes no texture of a surface whose rows are not at a multiple of it, and other
// code's surfaces carry rows at IOSurface's own alignment, 64 bytes. CoreAnimation also rounds the
// rows of its own backing stores' IOSurfaces to iosurfaceTextureAlignmentBytes, to no less than 64
// bytes, so on a GPU that takes a linear image in place only at rows a multiple of 256 (RADV) many
// of them are the runtime's shadows: measured in a macOS 11.7.11 guest on a Radeon 780M, 1399
// textures over a cold boot and a 15 s Safari scroll, and 11 answering 512.
- (NSUInteger)iosurfaceTextureAlignmentBytes
{
	return kRgfxIOSurfaceTextureAlignmentBytes;
}

- (NSUInteger)iosurfaceReadOnlyTextureAlignmentBytes
{
	return kRgfxIOSurfaceTextureAlignmentBytes;
}

- (MTLSize)sparseTileSizeWithTextureType:(MTLTextureType)textureType
                             pixelFormat:(MTLPixelFormat)pixelFormat
                             sampleCount:(NSUInteger)sampleCount
                          sparsePageSize:(MTLSparsePageSize)sparsePageSize
{
	// The host answers one tile for every texture type, as Metal does.
	(void)textureType;
	uint64_t tile[3];
	_caps.sparseTile(static_cast<uint64_t>(pixelFormat), sampleCount,
	                 static_cast<int64_t>(sparsePageSize), tile);
	return MTLSizeMake(tile[0], tile[1], tile[2]);
}

- (MTLSize)sparseTileSizeWithTextureType:(MTLTextureType)textureType
                             pixelFormat:(MTLPixelFormat)pixelFormat
                             sampleCount:(NSUInteger)sampleCount
{
	// Pages of 16 KiB, as Metal's query without a page size.
	return [self sparseTileSizeWithTextureType:textureType
	                               pixelFormat:pixelFormat
	                               sampleCount:sampleCount
	                            sparsePageSize:MTLSparsePageSize16];
}

// --- feature flags (RgfxCaps) ------------------------------------------------

- (BOOL)areProgrammableSamplePositionsSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagSupportsProgrammableSamplePositions);
}

- (BOOL)areRasterOrderGroupsSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagAreRasterOrderGroupsSupported);
}

- (BOOL)isDepth24Stencil8PixelFormatSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagIsDepth24Stencil8PixelFormatSupported);
}

- (BOOL)isFramebufferReadSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagIsFramebufferReadSupported);
}

// Programmable blending, a fragment function reading its own colour attachments, is answered as
// framebuffer reads are: Metal's devices give both the same answer, YES on the M3 Max and the
// Intel UHD 630 and NO on the AMD Radeon Pro 5300M (measured with respondsToSelector: and the
// method). So it follows the host's isFramebufferReadSupported flag, which the host gives an
// arm64 guest from its GPU and an x86-64 guest as NO.
- (BOOL)supportsProgrammableBlending
{
	return [self isFramebufferReadSupported];
}

- (BOOL)supportsTileShaders
{
	return _caps.valid() && _caps.flag(kRgfxFlagSupportsTileShaders);
}

- (BOOL)supportsNonSquareTileShaders
{
	return _caps.valid() && _caps.flag(kRgfxFlagSupportsNonSquareTileShaders);
}

- (BOOL)supportsImageBlocks
{
	return _caps.valid() && _caps.flag(kRgfxFlagSupportsImageBlocks);
}

- (BOOL)supports32BitFloatFiltering
{
	return _caps.identityFlag(kRgfxIdentitySupports32BitFloatFiltering);
}

- (BOOL)supports32BitMSAA
{
	return _caps.identityFlag(kRgfxIdentitySupports32BitMSAA);
}

- (BOOL)supportsQueryTextureLOD
{
	return _caps.identityFlag(kRgfxIdentitySupportsQueryTextureLOD);
}

- (BOOL)supportsBCTextureCompression
{
	return _caps.identityFlag(kRgfxIdentitySupportsBCTextureCompression);
}

- (BOOL)supportsPullModelInterpolation
{
	return _caps.identityFlag(kRgfxIdentitySupportsPullModelInterpolation);
}

- (BOOL)areBarycentricCoordsSupported
{
	return _caps.identityFlag(kRgfxIdentityAreBarycentricCoordsSupported);
}

- (BOOL)supportsShaderBarycentricCoordinates
{
	return _caps.identityFlag(kRgfxIdentitySupportsShaderBarycentricCoordinates);
}

- (BOOL)supportsDynamicLibraries
{
	return _caps.identityFlag(kRgfxIdentitySupportsDynamicLibraries);
}

- (BOOL)supportsRenderDynamicLibraries
{
	return _caps.identityFlag(kRgfxIdentitySupportsRenderDynamicLibraries);
}

- (BOOL)supportsRaytracing
{
	return _caps.identityFlag(kRgfxIdentitySupportsRaytracing);
}

// The host's answer, where it makes a compute pipeline's handles and tables (its
// FUNCTION_POINTERS feature taken).
- (BOOL)supportsFunctionPointers
{
	return (_caps.features() & RGFX_FEATURE_FUNCTION_POINTERS) != 0 &&
	       _caps.identityFlag(kRgfxIdentitySupportsFunctionPointers);
}

// NO: the bundle carries no render stage's handles or tables (a render pipeline's are
// RGFX_UNSUPPORTED), as the AMD Radeon Pro 5300M answers (measured: macOS 26.7.1, oracle_fnptr).
- (BOOL)supportsFunctionPointersFromRender
{
	return NO;
}

// The device's private compile of a visible function (`v40@0:8@16@24^@32`,
// `v48@0:8@16@24@32^@40`, `v40@0:8@16@24@?32`), as the AMD 5300M and the Intel 630 answer it
// (measured: macOS 26.7.1, tools/oracle/oracle_binfn.m; RgfxBinaryArchive's
// rgfxCompileVisibleFunction:descriptor:archive:): no error, whatever the function and the
// descriptor; the archive form keeps the function in the archive; the handler form calls its
// handler, of one argument, with nil on the caller's thread before it returns. Metal's library
// sends it for MTLFunctionOptionCompileToBinary, and Metal's own link of Core Image's kernels on
// macOS 12 and 13. Where the host makes no function pointers (supportsFunctionPointers NO), each
// answers the unsupported error as before.
- (void)compileVisibleFunction:(id<MTLFunction>)function
                withDescriptor:(id)descriptor
                         error:(NSError **)error
{
	if (![self supportsFunctionPointers]) {
		RGFX_UNSUPPORTED_ERROR(error, "function pointers, where the host makes none");
		return;
	}
	[RgfxBinaryArchive rgfxCompileVisibleFunction:function descriptor:descriptor archive:nil];
}

- (void)compileVisibleFunction:(id<MTLFunction>)function
                withDescriptor:(id)descriptor
      destinationBinaryArchive:(id<MTLBinaryArchive>)archive
                         error:(NSError **)error
{
	if (![self supportsFunctionPointers]) {
		RGFX_UNSUPPORTED_ERROR(error, "function pointers, where the host makes none");
		return;
	}
	[RgfxBinaryArchive rgfxCompileVisibleFunction:function descriptor:descriptor archive:archive];
}

- (void)compileVisibleFunction:(id<MTLFunction>)function
                withDescriptor:(id)descriptor
             completionHandler:(id)completionHandler
{
	NSError *error = nil;
	if (![self supportsFunctionPointers])
		RGFX_UNSUPPORTED_ERROR(&error, "function pointers, where the host makes none");
	else
		[RgfxBinaryArchive rgfxCompileVisibleFunction:function descriptor:descriptor archive:nil];
	if (completionHandler != nil)
		((void (^)(NSError *))completionHandler)(error);
}

// A function's handle of the device (macOS 26's functionHandleWithFunction:, and the SPI's at a
// resource index, `@32@0:8@16Q24`): the AMD 5300M's and the Intel 630's devices raise an
// unrecognized selector at each, answering respondsToSelector: YES (measured: macOS 26.7.1,
// tools/oracle/oracle_binfn.m), and so does this.
- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
{
	(void)function;
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
                                      resourceIndex:(NSUInteger)resourceIndex
{
	(void)function;
	(void)resourceIndex;
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (BOOL)supportsRaytracingFromRender
{
	return _caps.identityFlag(kRgfxIdentitySupportsRaytracingFromRender);
}

- (BOOL)supportsPrimitiveMotionBlur
{
	return _caps.identityFlag(kRgfxIdentitySupportsPrimitiveMotionBlur);
}

- (BOOL)supportsPlacementSparse
{
	return _caps.identityFlag(kRgfxIdentitySupportsPlacementSparse);
}

- (BOOL)supportsFamily:(MTLGPUFamily)gpuFamily
{
	// By raw value: the blob lists every family of the host's SDK and -1009, the device-info
	// bit the host reports among them; any other value names no family, and is no.
	return _caps.supportsFamily(static_cast<int64_t>(gpuFamily));
}

#if defined(__arm64__)
// In an arm64 guest, Metal's rules are an Apple GPU's (RgfxAppleRulesDevice): textures, over a
// surface too, views, Shared heaps and linear alignments (rgfxAppleGPU; dgpu-oracle.md, Q34).
- (BOOL)rgfxAppleRules
{
	return YES;
}
#endif

- (BOOL)supportsFeatureSet:(MTLFeatureSet)featureSet
{
	return _caps.supportsFeatureSet(static_cast<uint64_t>(featureSet));
}

- (BOOL)supportsCounterSampling:(MTLCounterSamplingPoint)samplingPoint
{
	return _caps.supportsCounterSampling(static_cast<uint64_t>(samplingPoint));
}

- (BOOL)supportsVertexAmplificationCount:(NSUInteger)count
{
	// Metal takes every count from 0 up to its largest.
	uint64_t largest = _caps.valid() ? _caps.size(kRgfxSizeLargestVertexAmplificationCount) : 1;
	return count <= largest;
}

// The layer counts the host makes maps of, where it makes them (its RATE_MAPS feature taken).
- (BOOL)supportsRasterizationRateMapWithLayerCount:(NSUInteger)layerCount
{
	if ((_caps.features() & RGFX_FEATURE_RATE_MAPS) == 0)
		return NO;
	return layerCount >= 1 && layerCount <= _caps.size(kRgfxSizeLargestRasterizationRateMapLayerCount);
}

- (BOOL)supportsTextureSampleCount:(NSUInteger)sampleCount
{
	return _caps.valid() && _caps.supportsSampleCount(static_cast<uint32_t>(sampleCount));
}

// --- MTLDeviceSPI (overridden; forward-declared in RgfxMetalPrivate.h) --------

- (NSUInteger)maxVertexSamplers
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxVertexSamplers) : 16;
}

- (NSUInteger)maxFragmentTextures
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxFragmentTextures) : 128;
}

- (NSUInteger)maxComputeTextures
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxComputeTextures) : 128;
}

- (NSUInteger)maxVertexAmplificationCount
{
	return _caps.valid() ? _caps.maxVertexAmplificationCount() : 1;
}

- (BOOL)supportsPrimitiveType:(NSUInteger)primitiveType
{
	return _caps.valid() && _caps.supportsPrimitiveType(static_cast<uint32_t>(primitiveType));
}

// --- MTLDeviceSPI feature queries ----------------------------------------------------------
// The IOGPU base answers each `supports*` query of MTLDeviceSPI from the IOGPU device info, which
// the kext gives all zero, so it answered NO to every one where the M3 Max, the AMD 5300M and the
// Intel UHD 630 all answer YES (measured, tools/base_answers_probe.m; aud-tests-04). The device
// answers those the lane carries out, from the host's answers (RgfxCaps::spiFeature says which
// and why), and leaves the others to the base's NO (tests/protocol_members.txt says why for each).

- (BOOL)supportsArgumentBuffers
{
	return _caps.spiFeature(kRgfxSpiArgumentBuffers);
}

- (BOOL)supportsBaseVertexInstanceDrawing
{
	return _caps.spiFeature(kRgfxSpiBaseVertexInstanceDrawing);
}

- (BOOL)supportsBinaryArchives
{
	return _caps.spiFeature(kRgfxSpiBinaryArchives);
}

- (BOOL)supportsBlackOrWhiteSamplerBorderColors
{
	return _caps.spiFeature(kRgfxSpiBlackOrWhiteSamplerBorderColors);
}

- (BOOL)supportsCombinedMSAAStoreAndResolveAction
{
	return _caps.spiFeature(kRgfxSpiCombinedMSAAStoreAndResolveAction);
}

- (BOOL)supportsComputeMemoryBarrier
{
	return _caps.spiFeature(kRgfxSpiComputeMemoryBarrier);
}

- (BOOL)supportsConcurrentComputeDispatch
{
	return _caps.spiFeature(kRgfxSpiConcurrentComputeDispatch);
}

- (BOOL)supportsCountingOcclusionQuery
{
	return _caps.spiFeature(kRgfxSpiCountingOcclusionQuery);
}

- (BOOL)supportsDepthClipMode
{
	return _caps.spiFeature(kRgfxSpiDepthClipMode);
}

// Render and compute indirect command buffers, where the host makes them, as Metal's devices
// answer YES (measured: M3 Max, AMD 5300M, Intel 630).
- (BOOL)supportsGFXIndirectCommandBuffers
{
	return (_caps.features() & RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS) != 0;
}

- (BOOL)supportsCMPIndirectCommandBuffers
{
	return (_caps.features() & RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS) != 0;
}

// Whether the host makes mesh render pipelines and draws them (RgfxMeshPipelineDevice): its
// MESH_PIPELINES feature taken.
- (BOOL)rgfxMakesMeshPipelines
{
	return (_caps.features() & RGFX_FEATURE_MESH_PIPELINES) != 0;
}

- (BOOL)rgfxMakesOpenCLModePipelines
{
	return (_caps.features() & RGFX_FEATURE_OPENCL_MODE) != 0;
}

// Mesh shaders where the host makes mesh pipelines, as the M3 Max, the AMD 5300M and the Intel 630
// answer YES (measured, tools/base_answers_probe.m, 11-05). Their SPI's supportsMeshShadersInICB
// the x86 Macs answer NO and the M3 Max YES; the device leaves it to the base's NO, as no mesh
// pipeline of it supports indirect command buffers, and requiresMeshShaderEmulation, which the
// x86 Macs answer YES and the M3 Max NO, to the base's NO: the host's GPU draws mesh shaders.
- (BOOL)supportsMeshShaders
{
	return [self rgfxMakesMeshPipelines];
}

- (BOOL)supportsIndirectDrawAndDispatch
{
	return _caps.spiFeature(kRgfxSpiIndirectDrawAndDispatch);
}

- (BOOL)supportsIndirectStageInRegion
{
	return _caps.spiFeature(kRgfxSpiIndirectStageInRegion);
}

- (BOOL)supportsIndirectTessellation
{
	return _caps.spiFeature(kRgfxSpiIndirectTessellation);
}

- (BOOL)supportsLayeredRendering
{
	return _caps.spiFeature(kRgfxSpiLayeredRendering);
}

- (BOOL)supportsLinearTextureFromSharedBuffer
{
	return _caps.spiFeature(kRgfxSpiLinearTextureFromSharedBuffer);
}

- (BOOL)supportsMSAADepthResolve
{
	return _caps.spiFeature(kRgfxSpiMSAADepthResolve);
}

- (BOOL)supportsMSAADepthResolveFilter
{
	return _caps.spiFeature(kRgfxSpiMSAADepthResolveFilter);
}

- (BOOL)supportsMSAAStencilResolve
{
	return _caps.spiFeature(kRgfxSpiMSAAStencilResolve);
}

- (BOOL)supportsMSAAStencilResolveFilter
{
	return _caps.spiFeature(kRgfxSpiMSAAStencilResolveFilter);
}

- (BOOL)supportsMirrorClampToEdgeSamplerMode
{
	return _caps.spiFeature(kRgfxSpiMirrorClampToEdgeSamplerMode);
}

- (BOOL)supportsNonUniformThreadgroupSize
{
	return _caps.spiFeature(kRgfxSpiNonUniformThreadgroupSize);
}

- (BOOL)supportsRenderPassWithoutRenderTarget
{
	return _caps.spiFeature(kRgfxSpiRenderPassWithoutRenderTarget);
}

- (BOOL)supportsResourceHeaps
{
	return _caps.spiFeature(kRgfxSpiResourceHeaps);
}

// NO, as the AMD 5300M and the Intel 630 answer (measured: macOS 26.7.1, tools/base_answers_probe.m
// and oracle_caps), where the IOGPU base answered YES here (measured: the x86-64 guest's device,
// tools/base_answers_probe.m) and the M3 Max answers YES: the device's heaps neither detach nor
// replace their backing (RgfxHeap's detachBacking and replaceBackingWithRanges:readOnly: raise an
// unrecognized selector, as Metal's base heap), so a caller told YES that sends them would end.
- (BOOL)supportsResourceDetachBacking
{
	return NO;
}

- (BOOL)supportsSamplerCompareFunction
{
	return _caps.spiFeature(kRgfxSpiSamplerCompareFunction);
}

- (BOOL)supportsTessellation
{
	return _caps.spiFeature(kRgfxSpiTessellation);
}

- (BOOL)supportsTextureSwizzle
{
	return _caps.spiFeature(kRgfxSpiTextureSwizzle);
}

- (BOOL)supportsViewportAndScissorArray
{
	return _caps.spiFeature(kRgfxSpiViewportAndScissorArray);
}

- (BOOL)supportsArrayOfSamplers
{
	return _caps.spiFeature(kRgfxSpiArrayOfSamplers);
}

- (BOOL)supportsArrayOfTextures
{
	return _caps.spiFeature(kRgfxSpiArrayOfTextures);
}

- (BOOL)supportsFragmentBufferWrites
{
	return _caps.spiFeature(kRgfxSpiFragmentBufferWrites);
}

- (BOOL)supportsInt64
{
	return _caps.spiFeature(kRgfxSpiInt64);
}

- (BOOL)supportsInvariantVertexPosition
{
	return _caps.spiFeature(kRgfxSpiInvariantVertexPosition);
}

- (BOOL)supportsQuadGroup
{
	return _caps.spiFeature(kRgfxSpiQuadGroup);
}

- (BOOL)supportsQuadShufflesAndBroadcast
{
	return _caps.spiFeature(kRgfxSpiQuadShufflesAndBroadcast);
}

- (BOOL)supportsRenderTextureWrites
{
	return _caps.spiFeature(kRgfxSpiRenderTextureWrites);
}

- (BOOL)supportsSIMDGroup
{
	return _caps.spiFeature(kRgfxSpiSIMDGroup);
}

- (BOOL)supportsSIMDShufflesAndBroadcast
{
	return _caps.spiFeature(kRgfxSpiSIMDShufflesAndBroadcast);
}

- (BOOL)supportsShaderMinLODClamp
{
	return _caps.spiFeature(kRgfxSpiShaderMinLODClamp);
}

- (BOOL)supportsStencilFeedback
{
	return _caps.spiFeature(kRgfxSpiStencilFeedback);
}

- (BOOL)supportsTextureCubeArray
{
	return _caps.spiFeature(kRgfxSpiTextureCubeArray);
}

- (BOOL)supportsDynamicAttributeStride
{
	return _caps.spiFeature(kRgfxSpiDynamicAttributeStride);
}

- (BOOL)supportsRasterOrderGroups
{
	return _caps.spiFeature(kRgfxSpiRasterOrderGroups);
}

- (BOOL)supportsSIMDReduction
{
	return _caps.spiFeature(kRgfxSpiSIMDReduction);
}

- (BOOL)supportsTexture2DMultisampleArray
{
	return _caps.spiFeature(kRgfxSpiTexture2DMultisampleArray);
}

- (BOOL)supportsReadWriteTextureArguments
{
	return _caps.spiFeature(kRgfxSpiReadWriteTextureArguments);
}

// --- what Apple's frameworks send the device beyond MTLDevice (RgfxInterop.h) --------
// SkyLight, Core Image and MPS send these; each answers what Metal's device answers, measured
// on the M3 Max (macOS 27), unless it says otherwise.

// The feature profile is the x86 Macs' that the guest's Metal knows (RgfxFeatureProfile.hpp), on
// every host. The others are the host's answers, from the device info: its Metal device's own,
// or, where no Metal device stands for its GPU, the M3 Max's (the Vulkan platform's), and 0 for a
// Metal device that does not respond to one. Measured on the M3 Max (macOS 27): FP configs 0xe
// (half), 0x54 (single) and 0 (double), 40 cores.
- (NSUInteger)featureProfile
{
	static const bool metalHasArchitecture = objc_getClass("MTLArchitecture") != Nil;
	const uint64_t profile = rgfx_profile::forGuest(metalHasArchitecture);
	static std::atomic<bool> logged{false};
	if (!logged.exchange(true)) {
		rgfxLogLine(@"RGFX: feature profile %llu: the guest's Metal %s the MTLArchitecture class "
		            @"(logged once a process)",
		            (unsigned long long)profile, metalHasArchitecture ? "has" : "lacks");
	}
	return profile;
}

- (NSUInteger)halfFPConfig
{
	return _caps.size(kRgfxSizeHalfFpConfig);
}

- (NSUInteger)singleFPConfig
{
	return _caps.size(kRgfxSizeSingleFpConfig);
}

- (NSUInteger)doubleFPConfig
{
	return _caps.size(kRgfxSizeDoubleFpConfig);
}

// `unsigned int`, as Metal's (`I16@0:8`): the host's count, 0 where it has none, on an Apple GPU,
// an arm64 guest's device. Elsewhere, an x86-64 guest's, the device does not respond to it
// (rgfxAnswersGpuCoreCount), as the AMD 5300M's and the Intel 630's do not, and a call raises an
// unrecognized selector, as it does there.
- (unsigned int)gpuCoreCount
{
	if (!rgfxAnswersGpuCoreCount(self)) {
		[self doesNotRecognizeSelector:_cmd];
		return 0;
	}
	return static_cast<unsigned int>(_caps.gpuCoreCount());
}

- (BOOL)respondsToSelector:(SEL)selector
{
	if (selector == @selector(gpuCoreCount))
		return rgfxAnswersGpuCoreCount(self);
	return [super respondsToSelector:selector];
}

// As supportsTextureSampleCount: (the same for each count from 1 to 16).
- (BOOL)supportsSampleCount:(NSUInteger)sampleCount
{
	return [self supportsTextureSampleCount:sampleCount];
}

// NO until set, then what was set.
- (BOOL)metalAssertionsEnabled
{
	return _metalAssertions.load(std::memory_order_relaxed);
}

- (void)setMetalAssertionsEnabled:(BOOL)enabled
{
	_metalAssertions.store(enabled, std::memory_order_relaxed);
}

// A resource type's patching kind: Metal's answers for types 0 to 11, and 0 from 14 on. Types
// 12 and 13 answer values that look read from past the table; 0 here.
- (NSUInteger)resourcePatchingTypeForResourceType:(NSUInteger)resourceType
{
	static const uint8_t kPatching[] = {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0};
	return resourceType < sizeof(kPatching) ? kPatching[resourceType] : 0;
}

// Metal's writes nothing to `indices` (types 0, 1 and 2 tried), and neither does this.
- (void)reserveResourceIndicesForResourceType:(NSUInteger)resourceType
                                      indices:(uint64_t *)indices
                                   indexCount:(NSUInteger)indexCount
{
	(void)resourceType;
	(void)indices;
	(void)indexCount;
}

// As the public query, whose page size this takes first.
- (MTLSize)tileSizeWithSparsePageSize:(MTLSparsePageSize)sparsePageSize
                          textureType:(MTLTextureType)textureType
                          pixelFormat:(MTLPixelFormat)pixelFormat
                          sampleCount:(NSUInteger)sampleCount
{
	return [self sparseTileSizeWithTextureType:textureType
	                               pixelFormat:pixelFormat
	                               sampleCount:sampleCount
	                            sparsePageSize:sparsePageSize];
}

// The IOSurface properties a surface backing a texture of `descriptor` needs, which SkyLight
// adds its own to: a mutable dictionary of its IOSurfaceAllocSize, the bytes the texture takes
// in the device's layout, as Metal answers in its own. The AMD 5300M and the Intel 630 under
// macOS 26.7.1 answer each a mutable dictionary of IOSurfaceAllocSize and keys of their drivers'
// own (kAMDMTL_*, kIntel*), which only their textures read: AMD's size its GPU's layout's, at a
// page; Intel's the texture's linear rows, with no padding past them (oracle_bundle iosurf.props
// and props2: 18 shapes, formats, usages, mipmaps and arrays). The device's textures read a
// surface's rows in place where they are multiples of the host's linear alignment
// (RgfxIOSurface.h), so its size is the texture's linear rows at that alignment
// (rgfxSurfaceTextureAllocSize). nil for no descriptor, or one no texture has.
- (NSDictionary *)copyIOSurfaceSharedTextureProperties:(MTLTextureDescriptor *)descriptor
{
	if (descriptor == nil)
		return nil;
	const uint64_t align = _caps.linearTextureAlignment();
	uint64_t size = 0;
	if (!rgfxSurfaceTextureAllocSize(static_cast<uint32_t>(descriptor.pixelFormat),
	                                 static_cast<uint32_t>(descriptor.textureType), descriptor.width,
	                                 descriptor.height, descriptor.depth,
	                                 static_cast<uint32_t>(descriptor.mipmapLevelCount),
	                                 descriptor.arrayLength, descriptor.sampleCount,
	                                 align != 0 ? align : 256, &size))
		return nil;
	return [NSMutableDictionary dictionaryWithObject:@(size)
	                                          forKey:(__bridge NSString *)kIOSurfaceAllocSize];
}

// No buffer over an IOSurface, as both x86 Macs answer: the AMD 5300M and the Intel 630 under
// macOS 26.7.1 answer nil for 64 x 64 and 100 x 7 BGRA surfaces and a two-plane 420v one
// (oracle_bundle iosurf.buffer). So no buffer of the device lies over a surface, and each
// answers iosurface NULL (RgfxBuffer).
- (id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)surface
{
	(void)surface;
	return nil;
}

// A buffer of the descriptor's length and options, its bytes copied from `contents` where it
// names some. What the device's buffers cannot be (a pinned or parent GPU address, address
// ranges, a pointer tag, sparse placement, an alignment past the GPU addresses' 32 KiB, an
// IOSurface, or no-copy contents, whose deallocator the descriptor does not answer) is refused.
- (id<MTLBuffer>)newBufferWithDescriptor:(id)descriptor
{
	id<RgfxBufferDescriptorFields> d = descriptor;
	if (d == nil)
		return nil;
	if (d.pinnedGPUAddress != 0 || d.parentGPUAddress != 0 || d.parentGPUSize != 0 ||
	    d.pointerTag != 0 || d.addressRanges != nil || d.placementSparsePageSize != 0) {
		RGFX_UNSUPPORTED("buffer descriptors placing a buffer at a GPU address");
		return nil;
	}
	if (d.alignment > 0x8000 || d.iosurface != nullptr || (d.contents != nullptr && d.noCopy)) {
		RGFX_UNSUPPORTED("buffer descriptors with an alignment past 32 KiB, an IOSurface or "
		                 "no-copy contents");
		return nil;
	}
	if (d.contents != nullptr)
		return [self newBufferWithBytes:d.contents length:d.length options:d.resourceOptions];
	return [self newBufferWithLength:d.length options:d.resourceOptions];
}

// As the public create, with the index range Metal's takes nothing from.
- (id<MTLIndirectCommandBuffer>)newIndirectCommandBufferWithDescriptor:
                                    (MTLIndirectCommandBufferDescriptor *)descriptor
                                                              maxCount:(NSUInteger)maxCount
                                                               options:(MTLResourceOptions)options
{
	return [self newIndirectCommandBufferWithDescriptor:descriptor
	                                    maxCommandCount:maxCount
	                                            options:options];
}

// Nil, as Metal's answers.
- (id)newIndirectComputeCommandEncoderWithBuffer:(id<MTLBuffer>)buffer
{
	(void)buffer;
	RGFX_UNSUPPORTED("indirect command encoders (none on the wire)");
	return nil;
}

- (id)newIndirectRenderCommandEncoderWithBuffer:(id<MTLBuffer>)buffer
{
	(void)buffer;
	RGFX_UNSUPPORTED("indirect command encoders (none on the wire)");
	return nil;
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	(void)descriptor;
	RGFX_UNSUPPORTED("visible function tables (function pointers)");
	return nil;
}

// As the public creates, with no options.
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithTileDescriptor:
                                  (MTLTileRenderPipelineDescriptor *)descriptor
                                                                 error:(NSError **)error
{
	return [self newRenderPipelineStateWithTileDescriptor:descriptor
	                                              options:MTLPipelineOptionNone
	                                           reflection:nil
	                                                error:error];
}

- (void)newRenderPipelineStateWithTileDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                               completionHandler:
                                   (MTLNewRenderPipelineStateCompletionHandler)completionHandler
{
	[self newRenderPipelineStateWithTileDescriptor:descriptor
	                                       options:MTLPipelineOptionNone
	                             completionHandler:^(id<MTLRenderPipelineState> state,
	                                                 MTLRenderPipelineReflection *reflection,
	                                                 NSError *error) {
		  (void)reflection;
		  if (completionHandler)
			  completionHandler(state, error);
	  }];
}

// Metal answers objects of its GPU driver's own classes, which the device has no counterpart of.
- (id)newResourceGroupFromResources:(const id<MTLResource> _Nonnull *)resources
                              count:(NSUInteger)count
{
	(void)resources;
	(void)count;
	RGFX_UNSUPPORTED("resource groups");
	return nil;
}

- (id)newTextureLayoutWithDescriptor:(MTLTextureDescriptor *)descriptor
                isHeapOrBufferBacked:(BOOL)isHeapOrBufferBacked
{
	(void)descriptor;
	(void)isHeapOrBufferBacked;
	RGFX_UNSUPPORTED("texture layouts");
	return nil;
}

// Shader profiling's sample buffers: none.
- (BOOL)mapShaderSampleBufferWithBuffer:(void *)buffer
                               capacity:(NSUInteger)capacity
                                   size:(NSUInteger)size
{
	(void)buffer;
	(void)capacity;
	(void)size;
	RGFX_UNSUPPORTED("shader sample buffers (shader profiling)");
	return NO;
}

- (void)unmapShaderSampleBuffer
{
}

@end
