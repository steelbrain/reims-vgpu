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

#import "RgfxBuffer.h"
#import "RgfxInterop.h"
#import "RgfxIOSurface.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include <IOKit/IOKitLib.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <memory>
#include <mutex>
#include <os/log.h>
#include <vector>

#include "RgfxCommitOrder.hpp"
#include "RgfxCompletionThread.hpp"
#include "RgfxKnownPipelines.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"
#include "RgfxSuperpages.hpp"

// The accelerator's user-client protocol, the kext's own header: its type, its selectors and
// the kext's own feature bits.
#include "../../kext/src/RgfxUserClientABI.hpp"

namespace {
// The largest object-lifecycle message the kext takes (ReimsGraphics::kLifecycleMessageMax):
// half the control ring.
constexpr uint32_t kLifecycleMessageMax = RGFX_CONTROL_RING_DATA_SIZE / 2;

// What the lifecycle's forward reaches: the user client, and the reader its replies go to. A
// pipeline create posted without waiting (rgfxKextSubmitDeferred) has its answer kept by the
// kext until the process takes it (rgfxKextAwaitAnswer, rgfxKextDrainAnswers), one take at a
// time (`takeLock`), so a thread that waits for an answer another is handing on waits for that.
struct RgfxKextLink {
	io_connect_t conn;
	RgfxReplyReader *reader;
	std::mutex takeLock;
	std::atomic<uint32_t> deferredPosts{0};
};

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
	if (IOConnectCallMethod(link->conn, kRgfxAccelMethodTakeAnswers, &waitFor, 1, nullptr, 0,
	                        &count, &outCount, answers, &size) != KERN_SUCCESS)
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
// (Features, 0x10f) has kRgfxKextFeatureDeferredAnswers. An older kext's word is 0.
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

// The serializer's queries: the kext posts the message on the control ring, stamping its seqno,
// and hands back the device's answer, whole. `context` is the user client's connect.
bool rgfxKextAsk(void *context, const void *message, uint32_t size, void *reply,
                 uint32_t capacity, uint32_t *reply_size)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t seqno = 0;
	uint32_t outCount = 1;
	size_t replySize = capacity;
	if (IOConnectCallMethod(conn, kRgfxAccelMethodSubmit, nullptr, 0, message, size, &seqno,
	                        &outCount, reply, &replySize) != KERN_SUCCESS)
		return false;
	*reply_size = static_cast<uint32_t>(replySize);
	return true;
}

// The kext's blocks, as RgfxRegion's block source; `context` is the user client's connect.
bool rgfxKextAlloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in[2] = {size, align};
	uint32_t outCount = 1;
	return IOConnectCallScalarMethod(conn, kRgfxAccelMethodAlloc, in, 2, offset, &outCount) ==
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

// The kext's chunks of texture resource ids, as RgfxIdSpace's chunk source; `context` is the
// user client's connect. False once every chunk of the guest's pool is held.
bool rgfxKextResourceIds(void *context, uint64_t *first_id)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint32_t outCount = 1;
	return IOConnectCallScalarMethod(conn, kRgfxAccelMethodResourceIds, nullptr, 0, first_id,
	                                 &outCount) == KERN_SUCCESS;
}

// The completion thread's wait: the kext returns once a completion ring of the process's holds
// records its last pass did not see (the wake count differs from `last`), or at once where it
// already does. `context` is the user client's connect.
RgfxCompletionThread::WaitResult rgfxKextWaitCompletions(void *context, uint64_t last,
                                                         uint64_t *now)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint32_t outCount = 1;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodWaitCompletions, &last, 1, now, &outCount);
	if (kr == KERN_SUCCESS)
		return RgfxCompletionThread::kWoke;
	if (kr == kIOReturnAborted)
		return RgfxCompletionThread::kRetry; // a signal broke the sleep
	os_log_error(OS_LOG_DEFAULT, "RGFX: the completion wait ended kr=0x%x; waiters take the rings",
	             kr);
	return RgfxCompletionThread::kStop;
}

void rgfxKextRelease(void *context, uint64_t offset)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodFree, &offset, 1, nullptr, nullptr);
}

// The kext's shared blocks of surface planes, as RgfxRegion's surface source: the block every
// process that holds the plane's surface gets, found by the pages it maps at `plane`.
bool rgfxKextSurfaceBlock(void *context, const void *plane, uint64_t length, uint64_t size,
                          uint64_t *offset, uint64_t *serial, bool *fresh)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
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

// The kext's shared blocks of surfaces in place, as RgfxRegion's surface pages source: the
// block every process that holds the surface gets, found by the whole allocation it maps at
// `base`, the device's GPU using those pages, and cached where the kext made it so (six inputs,
// the surface's IOSurfaceID and whether it may be cached; a fifth output). Unsupported where the
// device lacks the feature or refused the pages; no room where the kext had none, in the region
// (kIOReturnNoSpace) or its tables (kIOReturnNoResources); refused for any other answer, among
// them a kext that takes fewer inputs.
RgfxRegion::PagesAnswer rgfxKextSurfacePages(void *context, const void *base, uint64_t length,
                                             uint64_t size, uint32_t surfaceId, bool cacheable,
                                             uint64_t *offset, uint64_t *serial, bool *fresh,
                                             bool *cached)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in[6] = {reinterpret_cast<uintptr_t>(base), length, size, 1, surfaceId,
	                  cacheable ? 1u : 0u};
	uint64_t out[5] = {};
	uint32_t outCount = 5;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlock, in, 6, out, &outCount);
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

// The region's pages block source (RgfxRegion::setPagesBlockSource): the kext's pages block of
// the process's own memory (kRgfxAccelMethodPagesBlock), false where it makes none, for any
// reason, and the memory then takes a shadow.
bool rgfxKextPagesBlock(void *context, const void *address, uint64_t length, bool writable,
                        uint64_t *offset, uint64_t *size)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in[3] = {reinterpret_cast<uintptr_t>(address), length, writable ? 1u : 0u};
	uint64_t out[2] = {};
	uint32_t outCount = 2;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodPagesBlock, in, 3, out, &outCount);
	if (kr != KERN_SUCCESS || outCount != 2)
		return false;
	*offset = out[0];
	*size = out[1];
	return true;
}

// The pages source's memory (RgfxPagesMemory): 2 MiB superpages, zeroed and wired, each one run of
// RAM, one kernel call each, so a refusal is safe in a process whose VM guard is on
// (RgfxSuperpages.hpp).
int rgfxSuperpages(void *, uint64_t size, uint8_t **bytes)
{
	return rgfxTakeSuperpages(rgfxKernelSuperpageVm(), size, bytes);
}

void rgfxUnmapPages(void *, uint8_t *bytes, uint64_t size)
{
	rgfxGiveSuperpages(rgfxKernelSuperpageVm(), bytes, size);
}

void rgfxKextSurfaceRelease(void *context, uint64_t offset)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlockRelease, &offset, 1, nullptr,
	                          nullptr);
}

// The region's surface write source (RgfxRegion::setSurfaceWriteSource): a command buffer's
// writes of surfaces' shared blocks to the kext, which keeps a CPU lock of one of them waiting
// until they retire. A write the kext could not keep is its to log; the buffer runs alike.
void rgfxKextSurfaceWrites(void *context, uint64_t queue, uint64_t seqno, const uint64_t *writes,
                           uint32_t count)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
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
		NSLog(@"RGFX: the kext refused the naming of %u surface writes of queue %llu's buffer %llu "
		      @"(0x%x)",
		      count, (unsigned long long)queue, (unsigned long long)seqno, r);
}

void rgfxKextSurfaceRetire(void *context, uint64_t queue, uint64_t seqno)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	const uint64_t in[2] = {queue, seqno};
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceRetire, in, 2, nullptr, nullptr);
}

// The serializer's queue closer: the kext stops watching the queue's completion ring and posts
// its destroy on the control ring, behind the process's messages before it.
void rgfxKextQueueCloser(void *context, RgfxObjId objId)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in = objId;
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodQueueDestroy, &in, 1, nullptr, nullptr);
}

// The serializer's queue opener: the kext posts the queue's create on the control ring and
// returns once the device has its rings.
bool rgfxKextQueueOpener(void *context, const RgfxQueueCreate &q)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
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
@end

namespace {
void rgfxDeviceDoorbell(void *context, uint32_t ring_id)
{
	[(__bridge RgfxMetalDevice *)context rgfxRingDoorbell:ring_id];
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
	std::atomic<bool> _metalAssertions; // setMetalAssertionsEnabled:'s, read back
	RgfxStateCache *_stateCache; // samplers and depth/stencil states; nil until the handshake
	// The slabs Shared and Managed buffers lie in, and their sources: the region's, and the
	// process's own pages before it where the host maps them; null until then.
	std::unique_ptr<RgfxRegionSlabSource> _slabSource;
	std::unique_ptr<RgfxPagesSlabSource> _pagesSource;
	std::unique_ptr<RgfxSlabs> _slabs;
}

RGFX_LOG_UNRECOGNIZED

- (instancetype)initWithAcceleratorPort:(mach_port_t)port
{
	self = [super initWithAcceleratorPort:port];
	if (self == nil) {
		return nil; // Apple's base declined (or the port is not ours)
	}
	_acceleratorPort = port;
	_conn = IO_OBJECT_NULL;
	_serializer = nullptr;
	_region = nullptr;
	_replyReader = nullptr;
	_mapBase = 0;
	_mapSize = 0;

	// Wire transport handshake: open our own user client on the accelerator, read the caps blob
	// the host wrote, open the process's context, whose window is where its part of the device
	// region (BAR 2) lies in this task, and build the region allocator over the window and the
	// device submission ring (RgfxSerializer) + device completion ring (RgfxReplyReader). With
	// these non-null the resource / command-queue / command-buffer categories route over our
	// wire instead of falling back to the IOGPUMetalDevice base (Apple's IOGPU kernel ABI). Any
	// failure leaves the transport stubbed (the getters answer from defaults) rather than
	// crashing.
	kern_return_t kr = IOServiceOpen(port, mach_task_self(), kRgfxAccelUserClientType, &_conn);
	if (kr != KERN_SUCCESS) {
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: IOServiceOpen(type=%u) failed kr=0x%x; transport stubbed",
		             kRgfxAccelUserClientType, kr);
		return self;
	}

	// The caps blob the host wrote (read once at init), which the kext reads for us: its page
	// is no process's to map.
	// The video caps follow it, where the kext reads them too; a kext older than them reads the
	// device info alone, and the device then has no video codec.
	struct {
		RgfxDeviceInfo info;
		RgfxVideoCaps video;
	} blob = {};
	static_assert(offsetof(decltype(blob), video) == sizeof(RgfxDeviceInfo), "no padding");
	RgfxDeviceInfo &info = blob.info;
	size_t infoSize = sizeof(blob);
	kr = IOConnectCallStructMethod(_conn, kRgfxAccelMethodDeviceInfo, nullptr, 0, &blob,
	                               &infoSize);
	if (kr != KERN_SUCCESS) {
		blob = {};
		infoSize = sizeof(info);
		kr = IOConnectCallStructMethod(_conn, kRgfxAccelMethodDeviceInfo, nullptr, 0, &info,
		                               &infoSize);
	} else if (infoSize == sizeof(blob) && blob.video.version == RGFX_VIDEO_CAPS_VERSION) {
		_videoCaps = blob.video;
		if (_videoCaps.codec_count > RGFX_VIDEO_CAPS_CODECS)
			_videoCaps.codec_count = RGFX_VIDEO_CAPS_CODECS;
		infoSize = sizeof(info);
	} else {
		infoSize = sizeof(info);
	}
	if (kr != KERN_SUCCESS || infoSize != sizeof(info) || !_caps.load(info))
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: caps blob not read (kr=0x%x) or invalid; using defaults", kr);
	// The device's feature word and the kext's own, from the kext; a kext older than the call
	// leaves the blob's bits 0 to 2 and no kext feature (RgfxCaps::takeFeatures).
	uint64_t features[2] = {};
	uint32_t featuresCount = 2;
	kr = IOConnectCallScalarMethod(_conn, kRgfxAccelMethodFeatures, nullptr, 0, features,
	                               &featuresCount);
	if (kr == KERN_SUCCESS && featuresCount != 2)
		kr = kIOReturnBadArgument;
	_caps.takeFeatures(kr, features[0], features[1]);
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
	// comes sixth where the kext answers it; a kext that does not gives RGFX_CONTEXT_VA_SPAN.
	const uint32_t ctxWanted = (_caps.kextFeatures() & kRgfxKextFeatureContextSpan) ? 6 : 5;
	uint64_t ctx[6] = {0, 0, 0, 0, 0, RGFX_CONTEXT_VA_SPAN};
	uint32_t ctxCount = ctxWanted;
	kr = IOConnectCallScalarMethod(_conn, kRgfxAccelMethodOpenContext, nullptr, 0, ctx, &ctxCount);
	if (kr == kIOReturnNoResources) {
		// The kext has no range of GPU addresses left for another process: no device here, the
		// answer every Metal client handles, where a device whose every create answers nil
		// stops the frameworks that make their first objects at its open.
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: OPEN_CONTEXT refused: the device holds as many processes as it "
		             "can; no Metal device in this one");
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return nil;
	}
	if (kr != KERN_SUCCESS || ctxCount != ctxWanted || ctx[3] == 0 ||
	    ctx[4] < RGFX_ALLOCATIONS_OFFSET || ctx[5] == 0) {
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: OPEN_CONTEXT failed kr=0x%x (window 0x%llx, %llu bytes); transport "
		             "stubbed",
		             kr, (unsigned long long)ctx[3], (unsigned long long)ctx[4]);
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return self;
	}
	const mach_vm_address_t addr = ctx[3];
	const mach_vm_size_t mapSize = ctx[4];
	_mapBase = addr;
	_mapSize = mapSize;
	volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(addr);
	_ids.setBases(ctx[1], ctx[2], ctx[5]);
	void *connContext = reinterpret_cast<void *>(static_cast<uintptr_t>(_conn));
	// The kext's chunks are of RGFX_RESOURCE_ID_SMALL_CHUNK ids where it took
	// RGFX_FEATURE_SMALL_ID_CHUNKS (RgfxUserClientABI.hpp, kRgfxAccelMethodResourceIds).
	_ids.setResourceIdSource(&rgfxKextResourceIds, connContext,
	                         (_caps.features() & RGFX_FEATURE_SMALL_ID_CHUNKS) != 0
	                             ? RGFX_RESOURCE_ID_SMALL_CHUNK
	                             : RGFX_RESOURCE_ID_CHUNK);
	_region = new RgfxRegion(base, mapSize);
	_region->setBlockSource(&rgfxKextAlloc, &rgfxKextRelease, connContext);
	_region->setSurfaceSource(&rgfxKextSurfaceBlock, &rgfxKextSurfaceRelease, connContext);
	_region->setSurfacePagesSource(&rgfxKextSurfacePages, _caps.linearTextureAlignment(),
	                               connContext);
	_region->setSurfaceWriteSource(&rgfxKextSurfaceWrites, &rgfxKextSurfaceRetire, connContext);
	// No-copy buffers over the process's own pages, where the kext makes pages blocks and the
	// host took the feature that maps them.
	if ((_caps.kextFeatures() & kRgfxKextFeaturePagesBlock) != 0 &&
	    (_caps.features() & RGFX_FEATURE_PAGES_BUFFERS) != 0)
		_region->setPagesBlockSource(&rgfxKextPagesBlock, connContext);

	// The object lifecycle goes to the kext, which posts it on the control ring and hands back
	// the replies (a pipeline's reflection).
	_replyReader = new RgfxReplyReader();
	_link.conn = _conn;
	_link.reader = _replyReader;
	_controlRing = std::make_unique<RgfxRingWriter>(&rgfxKextSubmit, &_link, kLifecycleMessageMax);
	// A pipeline create equal to one the device accepted is posted without waiting, where the
	// kext keeps its answer for later (RgfxKnownPipelines.hpp).
	if (rgfxKextDefersAnswers(_caps)) {
		_controlRing->setDeferred(&rgfxKextSubmitDeferred, &rgfxKextAwaitAnswer, &_link);
		_knownPipelines = std::make_shared<RgfxKnownPipelines>();
	}
	_serializer = new RgfxSerializer(*_controlRing, _ids);
	_serializer->setQueueOpener(&rgfxKextQueueOpener, connContext);
	_serializer->setQueueCloser(&rgfxKextQueueCloser, connContext);
	_serializer->setAsker(&rgfxKextAsk, connContext);
	_stateCache = [[RgfxStateCache alloc] initWithDevice:self serializer:_serializer];
	// Managed buffers lie over Shared copies of the Shared buffers' slabs where the host makes
	// them over any Shared buffer (MANAGED_BUFFER_SLABS); else their CPU copies lie in heaps of
	// their own slabs where the host places them there.
	const bool managedAtOffset = (_caps.features() & RGFX_FEATURE_MANAGED_AT_OFFSET) != 0;
	const bool managedOverCopy = (_caps.features() & RGFX_FEATURE_MANAGED_BUFFER_SLABS) != 0;
	_slabSource = std::make_unique<RgfxRegionSlabSource>(*_region, *_serializer, managedAtOffset,
	                                                     managedOverCopy);
	_slabs = std::make_unique<RgfxSlabs>(*_slabSource);
	// The slabs whose buffers lie in heaps, and large Shared buffers, are the process's own
	// superpages, wired into pages blocks the host makes heaps or buffers over, where the region
	// has a pages block source (above) and the device took HEAP_OVER_PAGES; else, and where the
	// kernel has no superpage, the region's.
	if (_region->hasPagesBlocks() && (_caps.features() & RGFX_FEATURE_HEAP_OVER_PAGES) != 0) {
		RgfxPagesMemory memory;
		memory.superpages = &rgfxSuperpages;
		memory.unmap = &rgfxUnmapPages;
		_pagesSource = std::make_unique<RgfxPagesSlabSource>(
		    *_region, *_serializer, memory, [NSProcessInfo processInfo].physicalMemory,
		    managedAtOffset, managedOverCopy);
		_slabs->setFirst(_pagesSource.get());
	}
	if (!_slabs->start())
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: no slab thread; creates zero and frees let slabs go themselves");
	// A block the kext has no room for is asked again once the slabs the process keeps went.
	_region->setReclaim(&rgfxRegionReclaim, _slabs.get());
	// The queues' completions, taken as the kext wakes the thread for them.
	_completions = std::make_unique<RgfxCompletionThread>(&rgfxKextWaitCompletions, connContext);
	if (!_completions->start()) {
		os_log_error(OS_LOG_DEFAULT, "RGFX: no completion thread; waiters take the rings");
		_completions.reset();
	}
	return self;
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
	kern_return_t kr =
	    IOConnectCallScalarMethod(_conn, kRgfxAccelMethodSurfaceScanout, in, 2, out, &outCount);
	static std::atomic<int> logged{0};
	if (logged.fetch_add(1) < 8) {
		NSLog(@"RGFX: newTextureWithSurface:0x%lx buffer:%lu -> 0x%x: offset 0x%llx, %llu bytes, "
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
	// Ring the ring's doorbell through our accelerator user client: the kext writes `ringId`
	// to RGFX_REG_DOORBELL in BAR 0, waking the host's consumer of that ring.
	if (_conn == IO_OBJECT_NULL)
		return;
	uint64_t scalar = ringId;
	IOConnectCallScalarMethod(_conn, kRgfxAccelMethodDoorbell, &scalar, 1, nullptr, nullptr);
}

// --- identity ----------------------------------------------------------------

- (NSString *)name
{
	if (_caps.valid() && _caps.name()[0] != '\0') {
		return [NSString stringWithUTF8String:_caps.name()];
	}
	return @"reims-graphics";
}

// The MTLDeviceSPI family name, from the name the device answers (rgfxFamilyNameOf): "GeForce
// RTX" for "NVIDIA GeForce RTX 5080 Laptop GPU". Metal's devices answer the name without the
// vendor and the model number, each its own, so the answer is the lane's choice inside what they
// give (dgpu-oracle.md class 2); the IOGPU base answered "", which no Mac does.
- (NSString *)familyName
{
	return [NSString stringWithUTF8String:rgfxFamilyNameOf(self.name.UTF8String).c_str()];
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

// The bytes the device has allocated for this process. The bundle keeps no count of its
// allocations, so it answers 0, as a device that has made nothing does, and logs the ask.
- (NSUInteger)currentAllocatedSize
{
	RGFX_UNSUPPORTED("the device's allocation count (answers 0)");
	return 0;
}

- (uint64_t)recommendedMaxWorkingSetSize
{
	return _caps.size(kRgfxSizeRecommendedMaxWorkingSetSize);
}

- (NSUInteger)maximumConcurrentCompilationTaskCount
{
	return _caps.size(kRgfxSizeMaximumConcurrentCompilationTaskCount);
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

// 256 for every format the x86-64 Macs answer, as both answer it; Metal ends the process at the
// others (rgfxEndAtUnalignableLinearFormat). The bundle's own callers ask
// rgfxLinearTextureAlignmentForPixelFormat:, which ends nothing, as Metal's own texture creators
// over a buffer do not end it there.
- (NSUInteger)minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format
{
	rgfxEndAtUnalignableLinearFormat(format);
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

- (BOOL)supportsFunctionPointers
{
	return _caps.identityFlag(kRgfxIdentitySupportsFunctionPointers);
}

- (BOOL)supportsFunctionPointersFromRender
{
	return _caps.identityFlag(kRgfxIdentitySupportsFunctionPointersFromRender);
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

- (BOOL)supportsRasterizationRateMapWithLayerCount:(NSUInteger)layerCount
{
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

// The host's answers, from the device info: its Metal device's own, or, where no Metal device
// stands for its GPU, the M3 Max's (the Vulkan platform's), and 0 for a Metal device that does
// not respond to one. Measured on the M3 Max (macOS 27): feature profile 10, FP configs 0xe
// (half), 0x54 (single) and 0 (double), 40 cores.
- (NSUInteger)featureProfile
{
	return _caps.size(kRgfxSizeFeatureProfile);
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

// `unsigned int`, as Metal's (`I16@0:8`).
- (unsigned int)gpuCoreCount
{
	return static_cast<unsigned int>(_caps.gpuCoreCount());
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

// Metal answers, for a texture descriptor, the IOSurface properties a surface backing it needs
// in its own layout (its allocation size and compression mode); this device has no such
// layout size to give.
- (NSDictionary *)copyIOSurfaceSharedTextureProperties:(MTLTextureDescriptor *)descriptor
{
	(void)descriptor;
	RGFX_UNSUPPORTED("IOSurface properties for a texture (no layout size on the wire)");
	return nil;
}

// A buffer over an IOSurface would need the surface's GPU copy (RgfxIOSurface.h) in the
// surface's own layout, so that it aliases the surface's textures; none is made.
- (id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)surface
{
	(void)surface;
	RGFX_UNSUPPORTED("buffers over IOSurfaces");
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
