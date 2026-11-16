/*
 * RgfxVtService: see RgfxVtService.hpp. The arm64e kext's VideoToolbox service, its user client,
 * and display 0's framebuffer's side of it: the service's start and stop, and the VT_MESSAGE
 * events it routes.
 */

#include "RgfxVtService.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxIOSurfaceKPI.hpp"
#include "RgfxSurfaceBlocks.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOLib.h>
#include <libkern/version.h>

namespace {
// How long a send waits for its VT_STATUS: the host answers once its side took the message.
constexpr UInt32 kRgfxVtAnswerMs = 10000;
// The status the framework's guest side answers a message it cannot send, kVTParameterErr.
constexpr int32_t kRgfxVtParameterErr = -12902;
// The selectors (RgfxVtService.hpp).
enum : uint32_t {
	kVtSizeBuffer = 0,
	kVtRegisterUuid = 1,
	kVtDeregisterUuid = 2,
	kVtSend = 3,
	kVtReadyForMessages = 4,
};
// The most scalars a send takes: its length and five IOSurface IDs.
constexpr uint32_t kVtSendScalarsMax = 1 + RGFX_VT_MESSAGE_SURFACES;
// The most runs one VT_MESSAGE carries, its own and its surfaces': what its bytes hold.
constexpr uint32_t kVtRunsMax = ReimsGraphics::kLifecycleMessageMax / sizeof(RgfxPageRun);

// Whether `surface` is an IOSurface, as IOSurfaceRoot hands out.
bool isSurface(const OSObject *surface)
{
	const OSMetaClass *meta = surface->getMetaClass();
	return meta && strcmp(meta->getClassName(), "IOSurface") == 0;
}
} // namespace

// MARK: - The service

#define super IOService
OSDefineMetaClassAndStructors(AppleVideoToolboxParavirtualizationDriver, IOService);

AppleVideoToolboxParavirtualizationDriver *
AppleVideoToolboxParavirtualizationDriver::withFramebuffer(ReimsGraphics *fb)
{
	AppleVideoToolboxParavirtualizationDriver *service =
	    OSTypeAlloc(AppleVideoToolboxParavirtualizationDriver);
	if (!service)
		return nullptr;
	service->fb = nullptr;
	service->lock = nullptr;
	service->routes = nullptr;
	service->forgetBuffer = nullptr;
	service->forgetRun = {};
	service->forgetLock = nullptr;
	service->delivered = nullptr;
	service->surfaceRoot = nullptr;
	service->lends = nullptr;
	service->lendCount = 0;
	service->lastLend = 0;
	if (!service->init(nullptr) || !service->attach(fb)) {
		service->release();
		return nullptr;
	}
	if (!service->start(fb)) {
		service->detach(fb);
		service->release();
		return nullptr;
	}
	service->registerService();
	return service;
}

bool AppleVideoToolboxParavirtualizationDriver::start(IOService *provider)
{
	ReimsGraphics *owner = OSDynamicCast(ReimsGraphics, provider);
	if (!owner || !super::start(provider))
		return false;
	owner->retain();
	fb = owner;
	lock = IOLockAlloc();
	forgetLock = IOLockAlloc();
	routes = static_cast<RgfxVtRoutes *>(IOMalloc(sizeof(RgfxVtRoutes)));
	if (routes)
		bzero(routes, sizeof(RgfxVtRoutes));
	delivered = static_cast<uint8_t *>(IOMalloc(RGFX_VT_HOST_MESSAGE_MAX));
	lends = static_cast<Lend *>(IOMalloc(kRgfxVtLendsMax * sizeof(Lend)));
	// IOSurfaceRoot starts with IOSurface.kext, before any display's framebuffer.
	if (OSDictionary *matching = IOService::serviceMatching("IOSurfaceRoot")) {
		surfaceRoot = IOService::copyMatchingService(matching);
		matching->release();
	}
	// The page of the kext's own `-uui`s, in one run of guest RAM.
	forgetBuffer = IOBufferMemoryDescriptor::withOptions(
	    kIODirectionInOut | kIOMemoryPhysicallyContiguous, PAGE_SIZE, PAGE_SIZE);
	if (forgetBuffer && forgetBuffer->prepare() == kIOReturnSuccess) {
		IOByteCount segment = 0;
		forgetRun = {forgetBuffer->getPhysicalSegment(0, &segment, kIOMemoryMapperNone),
		             kRgfxVtHeaderBytes};
	} else {
		OSSafeReleaseNULL(forgetBuffer);
	}
	if (!surfaceRoot)
		IOLog("RgfxVtService: no IOSurfaceRoot: the service lends the host no surface\n");
	if (!lock || !forgetLock || !routes || !delivered || !lends || !forgetBuffer ||
	    !forgetRun.address) {
		IOLog("RgfxVtService: no memory for the VideoToolbox service\n");
		super::stop(provider);
		return false;
	}
	IOLog("RgfxVtService: the guest's VideoToolbox finds the host's decoders here\n");
	return true;
}

void AppleVideoToolboxParavirtualizationDriver::stop(IOService *provider)
{
	// The device stops with the framebuffer: what the host held of the guest's goes with it.
	letGoLends();
	super::stop(provider);
}

void AppleVideoToolboxParavirtualizationDriver::free()
{
	if (forgetBuffer) {
		forgetBuffer->complete();
		forgetBuffer->release();
		forgetBuffer = nullptr;
	}
	if (lends && lock)
		letGoLends();
	if (lends)
		IOFree(lends, kRgfxVtLendsMax * sizeof(Lend));
	lends = nullptr;
	OSSafeReleaseNULL(surfaceRoot);
	if (routes)
		IOFree(routes, sizeof(RgfxVtRoutes));
	routes = nullptr;
	if (delivered)
		IOFree(delivered, RGFX_VT_HOST_MESSAGE_MAX);
	delivered = nullptr;
	if (lock)
		IOLockFree(lock);
	lock = nullptr;
	if (forgetLock)
		IOLockFree(forgetLock);
	forgetLock = nullptr;
	OSSafeReleaseNULL(fb);
	super::free();
}

IOReturn AppleVideoToolboxParavirtualizationDriver::newUserClient(task_t owningTask,
                                                                  void *securityID, UInt32 type,
                                                                  OSDictionary *properties,
                                                                  IOUserClient **handler)
{
	if (type != 0 || !handler)
		return kIOReturnBadArgument;
	AppleVideoToolboxParavirtualizationUserClient *client =
	    OSTypeAlloc(AppleVideoToolboxParavirtualizationUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnNoMemory;
	}
	if (!client->attach(this)) {
		client->release();
		return kIOReturnInternalError;
	}
	if (!client->start(this)) {
		client->detach(this);
		client->release();
		return kIOReturnInternalError;
	}
	*handler = client;
	return kIOReturnSuccess;
}

bool AppleVideoToolboxParavirtualizationDriver::registerUuid(
    const AppleVideoToolboxParavirtualizationUserClient *client, const uint8_t *uuid)
{
	IOLockLock(lock);
	const bool added = routes->add(uuid, client);
	IOLockUnlock(lock);
	return added;
}

void AppleVideoToolboxParavirtualizationDriver::deregisterUuid(
    const AppleVideoToolboxParavirtualizationUserClient *client, const uint8_t *uuid)
{
	IOLockLock(lock);
	const bool removed = routes->remove(uuid, client);
	IOLockUnlock(lock);
	if (removed)
		sendForget(uuid);
}

void AppleVideoToolboxParavirtualizationDriver::deregisterAll(
    const AppleVideoToolboxParavirtualizationUserClient *client)
{
	uint8_t(*held)[kRgfxVtUuidBytes] = static_cast<uint8_t(*)[kRgfxVtUuidBytes]>(
	    IOMalloc(kRgfxVtRoutesMax * kRgfxVtUuidBytes));
	if (!held)
		return;
	IOLockLock(lock);
	const uint32_t n = routes->removeClient(client, held, kRgfxVtRoutesMax);
	IOLockUnlock(lock);
	for (uint32_t i = 0; i < n; i++)
		sendForget(held[i]);
	IOFree(held, kRgfxVtRoutesMax * kRgfxVtUuidBytes);
}

void AppleVideoToolboxParavirtualizationDriver::sendForget(const uint8_t *uuid)
{
	IOLockLock(forgetLock);
	rgfxVtForgetMessage(static_cast<uint8_t *>(forgetBuffer->getBytesNoCopy()), uuid);
	int32_t status = 0;
	const IOReturn r = send(&forgetRun, 1, kRgfxVtHeaderBytes, &status);
	IOLockUnlock(forgetLock);
	static char unsent;
	if ((r != kIOReturnSuccess || status != 0) && rgfxFirstCalls(&unsent, 0, 1))
		IOLog("RgfxVtService: a session's -uui: 0x%x, status %d\n", r, status);
}

IOReturn AppleVideoToolboxParavirtualizationDriver::send(const RgfxPageRun *runs, uint32_t count,
                                                         uint64_t length, int32_t *status)
{
	bool posted = false;
	return send(runs, count, length, nullptr, 0, status, &posted);
}

IOReturn AppleVideoToolboxParavirtualizationDriver::send(const RgfxPageRun *runs, uint32_t count,
                                                         uint64_t length, const RgfxVtLent *lent,
                                                         uint32_t lentCount, int32_t *status,
                                                         bool *posted)
{
	*status = kRgfxVtParameterErr;
	*posted = false;
	if (count == 0)
		return kIOReturnNoSpace;
	constexpr uint32_t kMax = ReimsGraphics::kLifecycleMessageMax;
	uint8_t *message = static_cast<uint8_t *>(IOMalloc(kMax));
	if (!message)
		return kIOReturnNoMemory;
	const uint32_t size = rgfxVtPackMessage(length, runs, count, lent, lentCount, message, kMax);
	if (!size) {
		IOFree(message, kMax);
		return kIOReturnNoSpace;
	}
	UInt8 reply[ReimsGraphics::kLifecycleReplyMax];
	UInt32 replied = 0;
	IOReturn r = fb->postLifecycle(RGFX_CMD_VT_MESSAGE, message, size, RGFX_EVT_VT_STATUS, reply,
	                               sizeof(reply), &replied, nullptr, kRgfxVtAnswerMs);
	IOFree(message, kMax);
	// On the ring, the message is the host's, answered or not.
	*posted = r == kIOReturnSuccess || r == kIOReturnTimeout;
	if (r == kIOReturnSuccess && replied >= sizeof(RgfxVtStatus)) {
		RgfxVtStatus answer;
		memcpy(&answer, reply, sizeof(answer));
		*status = answer.status;
	} else if (r == kIOReturnSuccess) {
		r = kIOReturnUnderrun;
	}
	return r;
}

void AppleVideoToolboxParavirtualizationDriver::deliver(const volatile uint8_t *message,
                                                        uint32_t length)
{
	if (length > RGFX_VT_HOST_MESSAGE_MAX)
		return;
	for (uint32_t i = 0; i < length; i++)
		delivered[i] = message[i];
	uint8_t uuid[kRgfxVtUuidBytes];
	const RgfxVtHostKind kind = rgfxVtHostMessage(delivered, length, uuid);
	if (kind == kRgfxVtReleases) {
		uint64_t released[kRgfxVtLendsMax];
		const uint32_t n = rgfxVtReleasedLends(delivered, length, released, kRgfxVtLendsMax);
		static char unread;
		if (!n && rgfxFirstCalls(&unread, 0, 1))
			IOLog("RgfxVtService: a host rels of %u bytes names no lend\n", length);
		for (uint32_t i = 0; i < n; i++)
			letGoLend(released[i]);
		return;
	}
	if (kind == kRgfxVtUnreadable) {
		static char dropped;
		if (rgfxFirstCalls(&dropped, 0, 1))
			IOLog("RgfxVtService: a host message of %u bytes goes to no client (unreadable)\n",
			      length);
		return;
	}
	if (kind == kRgfxVtForUuid) {
		const void *clients[kRgfxVtRoutesMax];
		IOLockLock(lock);
		const uint32_t n = routes->clientsOf(uuid, clients, kRgfxVtRoutesMax);
		for (uint32_t i = 0; i < n; i++)
			static_cast<const AppleVideoToolboxParavirtualizationUserClient *>(clients[i])
			    ->retain();
		IOLockUnlock(lock);
		for (uint32_t i = 0; i < n; i++) {
			AppleVideoToolboxParavirtualizationUserClient *client =
			    const_cast<AppleVideoToolboxParavirtualizationUserClient *>(
			        static_cast<const AppleVideoToolboxParavirtualizationUserClient *>(
			            clients[i]));
			client->deliver(delivered, length);
			client->release();
		}
		return;
	}
	// A `Bye!`: to each client once for each UUID it registered, that UUID written in.
	RgfxVtRoutes::Route *all = static_cast<RgfxVtRoutes::Route *>(
	    IOMalloc(sizeof(RgfxVtRoutes::Route) * kRgfxVtRoutesMax));
	if (!all)
		return;
	IOLockLock(lock);
	const uint32_t n = routes->count;
	memcpy(all, routes->routes, sizeof(RgfxVtRoutes::Route) * n);
	for (uint32_t i = 0; i < n; i++)
		static_cast<const AppleVideoToolboxParavirtualizationUserClient *>(all[i].client)
		    ->retain();
	IOLockUnlock(lock);
	for (uint32_t i = 0; i < n; i++) {
		AppleVideoToolboxParavirtualizationUserClient *client =
		    const_cast<AppleVideoToolboxParavirtualizationUserClient *>(
		        static_cast<const AppleVideoToolboxParavirtualizationUserClient *>(
		            all[i].client));
		memcpy(delivered + kRgfxVtUuidAt, all[i].uuid, kRgfxVtUuidBytes);
		client->deliver(delivered, length);
		client->release();
	}
	IOFree(all, sizeof(RgfxVtRoutes::Route) * kRgfxVtRoutesMax);
}


IOReturn AppleVideoToolboxParavirtualizationDriver::lend(task_t task, uint32_t surfaceId,
                                                         RgfxVtSurface *record, RgfxPageRun *runs,
                                                         uint32_t max)
{
	if (!surfaceRoot)
		return kIOReturnNotReady;
	OSObject *surface = rgfxIOSurfaceRootLookupSurface(surfaceRoot, surfaceId, task);
	if (!surface)
		return kIOReturnNotFound;
	IOMemoryDescriptor *memory =
	    isSurface(surface) ? OSDynamicCast(IOMemoryDescriptor, rgfxIOSurfaceMemory(surface))
	                       : nullptr;
	// The host writes the surface's pages (a decoder's output) or reads them: a read-write CPU
	// lock of it first, as a process's, puts what the GPU wrote of it in its pages, moves its
	// seed, and has a device cache of it filled from its pages again at its next use
	// (RgfxAcceleratorLocks.cpp). Nothing uses the surface on the GPU while it is lent.
	uint32_t seed = 0;
	if (memory && rgfxIOSurfaceLock(surface, 0, &seed) == kIOReturnSuccess)
		rgfxIOSurfaceUnlock(surface, 0, &seed);
	const uint64_t base = memory ? rgfxIOSurfaceBaseOffset(surface) : 0;
	const uint64_t length = memory ? rgfxIOSurfaceAllocSize(surface) : 0;
	if (!memory || !length || base + length > memory->getLength()) {
		surface->release();
		return kIOReturnBadArgument;
	}
	memory->retain();
	if (memory->prepare(kIODirectionInOut) != kIOReturnSuccess) {
		memory->release();
		surface->release();
		return kIOReturnNoMemory;
	}
	uint32_t n = 0;
	bool walked = true;
	for (uint64_t at = base; walked && at < base + length;) {
		IOByteCount segment = 0;
		const addr64_t phys = memory->getPhysicalSegment(at, &segment, kIOMemoryMapperNone);
		const uint64_t take = segment < base + length - at ? segment : base + length - at;
		walked = phys && segment && rgfxSurfaceAppendRun(runs, &n, max, phys, take);
		at += take;
	}
	*record = {};
	record->length = length;
	record->surface_id = rgfxIOSurfaceID(surface);
	record->pixel_format = rgfxIOSurfacePixelFormat(surface);
	record->width = rgfxIOSurfaceWidth(surface);
	record->height = rgfxIOSurfaceHeight(surface);
	record->plane_count = rgfxIOSurfacePlaneCount(surface);
	record->run_count = n;
	if (record->plane_count == 0) {
		record->planes[0].bytes_per_row = rgfxIOSurfaceBytesPerRow(surface);
		record->planes[0].width = record->width;
		record->planes[0].height = record->height;
	}
	for (uint32_t p = 0; p < record->plane_count && p < RGFX_VT_SURFACE_PLANES; p++) {
		record->planes[p].offset = rgfxIOSurfacePlaneOffset(surface, p);
		record->planes[p].bytes_per_row = rgfxIOSurfacePlaneBytesPerRow(surface, p);
		record->planes[p].width = rgfxIOSurfacePlaneWidth(surface, p);
		record->planes[p].height = rgfxIOSurfacePlaneHeight(surface, p);
	}
	IOLockLock(lock);
	const bool held = walked && lendCount < kRgfxVtLendsMax;
	if (held) {
		lastLend = lastLend + 1 ? lastLend + 1 : 1;
		record->lend = lastLend;
		lends[lendCount++] = {lastLend, surface, memory};
	}
	IOLockUnlock(lock);
	if (!held) {
		static char refused;
		if (rgfxFirstCalls(&refused, walked, 1))
			IOLog("RgfxVtService: IOSurface %u is not lent: %s\n", surfaceId,
			      walked ? "the service holds the most lends it holds"
			             : "its runs are more than a message carries");
		memory->complete(kIODirectionInOut);
		memory->release();
		surface->release();
		return walked ? kIOReturnNoResources : kIOReturnNoSpace;
	}
	return kIOReturnSuccess;
}

void AppleVideoToolboxParavirtualizationDriver::letGoLend(uint64_t name)
{
	Lend gone = {};
	IOLockLock(lock);
	for (uint32_t i = 0; i < lendCount; i++) {
		if (lends[i].name != name)
			continue;
		gone = lends[i];
		lends[i] = lends[--lendCount];
		break;
	}
	IOLockUnlock(lock);
	if (!gone.surface)
		return;
	gone.memory->complete(kIODirectionInOut);
	gone.memory->release();
	gone.surface->release();
}

void AppleVideoToolboxParavirtualizationDriver::letGoLends()
{
	for (;;) {
		IOLockLock(lock);
		const uint64_t name = lendCount ? lends[lendCount - 1].name : 0;
		IOLockUnlock(lock);
		if (!name)
			return;
		letGoLend(name);
	}
}

#undef super

// MARK: - The user client

#define super IOUserClient
OSDefineMetaClassAndStructors(AppleVideoToolboxParavirtualizationUserClient, IOUserClient);

bool AppleVideoToolboxParavirtualizationUserClient::initWithTask(task_t owningTask,
                                                                 void *securityID, UInt32 type,
                                                                 OSDictionary *properties)
{
	task = owningTask;
	service = nullptr;
	buffer = nullptr;
	bufferMap = nullptr;
	capacity = 0;
	runs = nullptr;
	runCount = 0;
	runCapacity = 0;
	queue = nullptr;
	queueMap = nullptr;
	portSet = false;
	kept = nullptr;
	gone = false;
	lock = nullptr;
	sendLock = nullptr;
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	lock = IOLockAlloc();
	sendLock = IOLockAlloc();
	kept = OSArray::withCapacity(4);
	queue = IOSharedDataQueue::withCapacity(kRgfxVtQueueBytes);
	if (!lock || !sendLock || !kept || !queue)
		return false;
	// The queue's memory, mapped into the process, which dequeues from it.
	IOMemoryDescriptor *memory = queue->getMemoryDescriptor();
	if (memory) {
		queueMap = memory->createMappingInTask(owningTask, 0, kIOMapAnywhere);
		memory->release();
	}
	return queueMap != nullptr;
}

bool AppleVideoToolboxParavirtualizationUserClient::start(IOService *provider)
{
	AppleVideoToolboxParavirtualizationDriver *owner =
	    OSDynamicCast(AppleVideoToolboxParavirtualizationDriver, provider);
	if (!owner || !super::start(provider))
		return false;
	owner->retain();
	service = owner;
	return true;
}

void AppleVideoToolboxParavirtualizationUserClient::letGo()
{
	IOLockLock(lock);
	const bool first = !gone;
	gone = true;
	IOLockUnlock(lock);
	if (first && service)
		service->deregisterAll(this);
}

IOReturn AppleVideoToolboxParavirtualizationUserClient::clientClose()
{
	letGo();
	if (!isInactive())
		terminate();
	return kIOReturnSuccess;
}

IOReturn AppleVideoToolboxParavirtualizationUserClient::clientDied()
{
	return clientClose();
}

void AppleVideoToolboxParavirtualizationUserClient::free()
{
	if (lock)
		letGo();
	OSSafeReleaseNULL(queueMap);
	OSSafeReleaseNULL(queue);
	OSSafeReleaseNULL(bufferMap);
	if (buffer) {
		buffer->complete();
		buffer->release();
		buffer = nullptr;
	}
	if (runs)
		IOFree(runs, runCapacity * sizeof(RgfxPageRun));
	runs = nullptr;
	OSSafeReleaseNULL(kept);
	OSSafeReleaseNULL(service);
	if (lock)
		IOLockFree(lock);
	lock = nullptr;
	if (sendLock)
		IOLockFree(sendLock);
	sendLock = nullptr;
	super::free();
}

IOReturn AppleVideoToolboxParavirtualizationUserClient::registerNotificationPort(
    mach_port_t port, UInt32 type, io_user_reference_t refCon)
{
	if (type != 0)
		return kIOReturnUnsupported;
	IOLockLock(lock);
	const bool set = portSet;
	if (!set) {
		portSet = true;
		queue->setNotificationPort(port);
	}
	IOLockUnlock(lock);
	return set ? kIOReturnBadArgument : kIOReturnSuccess;
}

IOReturn AppleVideoToolboxParavirtualizationUserClient::externalMethod(
    uint32_t selector, IOExternalMethodArguments *a, IOExternalMethodDispatch *dispatch,
    OSObject *target, void *reference)
{
	if (!a || !service)
		return kIOReturnBadArgument;
	switch (selector) {
	case kVtSizeBuffer:
		if (a->scalarInputCount != 1 || a->scalarOutputCount < 3)
			return kIOReturnBadArgument;
		return sizeBuffer(a);
	case kVtRegisterUuid:
	case kVtDeregisterUuid: {
		if (a->structureInputSize != kRgfxVtUuidBytes || !a->structureInput)
			return kIOReturnBadArgument;
		const uint8_t *uuid = static_cast<const uint8_t *>(a->structureInput);
		if (selector == kVtDeregisterUuid) {
			service->deregisterUuid(this, uuid);
			return kIOReturnSuccess;
		}
		return service->registerUuid(this, uuid) ? kIOReturnSuccess : kIOReturnNoResources;
	}
	case kVtSend:
		if (a->scalarInputCount < 1 || a->scalarInputCount > kVtSendScalarsMax ||
		    a->scalarOutputCount < 1)
			return kIOReturnBadArgument;
		return sendMessage(a);
	case kVtReadyForMessages:
		IOLockLock(lock);
		deliverKept();
		IOLockUnlock(lock);
		return kIOReturnSuccess;
	default:
		return kIOReturnBadArgument;
	}
}

// Selector 0: the message buffer, of at least the size asked, wired and shared with the process,
// and its runs of guest RAM, which a send's VT_MESSAGE names.
IOReturn AppleVideoToolboxParavirtualizationUserClient::sizeBuffer(IOExternalMethodArguments *a)
{
	const uint64_t asked = a->scalarInput[0];
	if (asked > RGFX_VT_MESSAGE_MAX) {
		IOLog("RgfxVtService: a message buffer of %llu bytes is more than the %u it holds\n",
		      asked, static_cast<unsigned>(RGFX_VT_MESSAGE_MAX));
		return kIOReturnBadArgument;
	}
	IOLockLock(sendLock);
	IOReturn r = kIOReturnSuccess;
	if (asked > capacity || !buffer) {
		const uint64_t bytes = (asked + PAGE_SIZE - 1) & ~static_cast<uint64_t>(PAGE_SIZE - 1);
		IOBufferMemoryDescriptor *made = IOBufferMemoryDescriptor::inTaskWithOptions(
		    kernel_task, kIODirectionInOut | kIOMemoryKernelUserShared, bytes ? bytes : PAGE_SIZE,
		    PAGE_SIZE);
		IOMemoryMap *mapped = nullptr;
		const uint32_t most = static_cast<uint32_t>((bytes ? bytes : PAGE_SIZE) / PAGE_SIZE);
		RgfxPageRun *walked = static_cast<RgfxPageRun *>(IOMalloc(most * sizeof(RgfxPageRun)));
		uint32_t n = 0;
		bool ok = made && walked && made->prepare() == kIOReturnSuccess;
		const bool prepared = ok;
		if (ok)
			mapped = made->createMappingInTask(task, 0, kIOMapAnywhere);
		ok = ok && mapped;
		for (uint64_t at = 0; ok && at < made->getLength();) {
			IOByteCount segment = 0;
			const addr64_t phys = made->getPhysicalSegment(at, &segment, kIOMemoryMapperNone);
			ok = phys && segment && rgfxSurfaceAppendRun(walked, &n, most, phys, segment);
			at += segment;
		}
		if (!ok) {
			OSSafeReleaseNULL(mapped);
			if (made && prepared)
				made->complete();
			OSSafeReleaseNULL(made);
			if (walked)
				IOFree(walked, most * sizeof(RgfxPageRun));
			r = kIOReturnNoMemory;
		} else {
			OSSafeReleaseNULL(bufferMap);
			if (buffer) {
				buffer->complete();
				buffer->release();
			}
			if (runs)
				IOFree(runs, runCapacity * sizeof(RgfxPageRun));
			buffer = made;
			bufferMap = mapped;
			runs = walked;
			runCount = n;
			runCapacity = most;
			capacity = asked;
		}
	}
	if (r == kIOReturnSuccess) {
		a->scalarOutput[0] = capacity;
		a->scalarOutput[1] = bufferMap->getAddress();
		a->scalarOutput[2] = queueMap->getAddress();
		a->scalarOutputCount = 3;
	}
	IOLockUnlock(sendLock);
	return r;
}

// Selector 3: the buffer's first `length` bytes to the host, lending it the surfaces of the IDs
// that follow, and its status back.
IOReturn AppleVideoToolboxParavirtualizationUserClient::sendMessage(IOExternalMethodArguments *a)
{
	const uint64_t length = a->scalarInput[0];
	int32_t status = kRgfxVtParameterErr;
	IOReturn r = kIOReturnSuccess;
	IOLockLock(sendLock);
	RgfxPageRun *message = static_cast<RgfxPageRun *>(IOMalloc(kVtRunsMax * sizeof(RgfxPageRun)));
	RgfxPageRun *surfaceRuns =
	    static_cast<RgfxPageRun *>(IOMalloc(kVtRunsMax * sizeof(RgfxPageRun)));
	RgfxVtLent lent[RGFX_VT_MESSAGE_SURFACES] = {};
	uint32_t lentCount = 0;
	if (!buffer || length > capacity) {
		r = kIOReturnBadArgument;
	} else if (!message || !surfaceRuns) {
		r = kIOReturnNoMemory;
	} else {
		const uint32_t n = rgfxVtMessageRuns(runs, runCount, length, message,
		                                     runCapacity < kVtRunsMax ? runCapacity : kVtRunsMax);
		uint32_t used = 0;
		bool lending = n != 0;
		bool posted = false;
		for (uint32_t i = 1; lending && i < a->scalarInputCount; i++) {
			const uint32_t id = static_cast<uint32_t>(a->scalarInput[i]);
			if (!id)
				continue;
			RgfxVtLent &l = lent[lentCount];
			lending = service->lend(task, id, &l.record, surfaceRuns + used, kVtRunsMax - used) ==
			          kIOReturnSuccess;
			if (lending) {
				l.runs = surfaceRuns + used;
				used += l.record.run_count;
				lentCount++;
			}
		}
		if (!n) {
			r = kIOReturnNoMemory;
		} else if (lending) {
			r = service->send(message, n, length, lent, lentCount, &status, &posted);
		}
		// What was lent of a message never posted goes back at once; once posted, the host lets
		// go of each with a `rels`.
		if (!posted) {
			for (uint32_t i = 0; i < lentCount; i++)
				service->letGoLend(lent[i].record.lend);
		}
	}
	if (message)
		IOFree(message, kVtRunsMax * sizeof(RgfxPageRun));
	if (surfaceRuns)
		IOFree(surfaceRuns, kVtRunsMax * sizeof(RgfxPageRun));
	IOLockUnlock(sendLock);
	a->scalarOutput[0] = static_cast<uint64_t>(static_cast<int64_t>(status));
	a->scalarOutputCount = 1;
	return r != kIOReturnSuccess ? r : static_cast<IOReturn>(status);
}

void AppleVideoToolboxParavirtualizationUserClient::deliver(const uint8_t *message, uint32_t length)
{
	IOLockLock(lock);
	if (!gone) {
		const bool queued = kept->getCount() == 0 &&
		                    queue->enqueue(const_cast<uint8_t *>(message), length);
		if (!queued && kept->getCount() < kRgfxVtPendingMax) {
			if (OSData *copy = OSData::withBytes(message, length)) {
				kept->setObject(copy);
				copy->release();
			}
		} else if (!queued) {
			static char full;
			if (rgfxFirstCalls(&full, 0, 1))
				IOLog("RgfxVtService: a client keeps %u host messages already: one dropped\n",
				      kRgfxVtPendingMax);
		}
	}
	IOLockUnlock(lock);
}

void AppleVideoToolboxParavirtualizationUserClient::deliverKept()
{
	while (kept->getCount()) {
		OSData *first = OSDynamicCast(OSData, kept->getObject(0));
		if (first && !queue->enqueue(const_cast<void *>(first->getBytesNoCopy()),
		                             first->getLength()))
			break;
		kept->removeObject(0);
	}
}

#undef super

// MARK: - Display 0's framebuffer's side

void ReimsGraphics::startVtService()
{
	if (!vtMessagesTaken || __atomic_load_n(&vtService, __ATOMIC_ACQUIRE))
		return;
	RgfxVideoCaps caps;
	volatile const UInt8 *at = ring(RGFX_VIDEO_CAPS_OFFSET);
	UInt8 *to = reinterpret_cast<UInt8 *>(&caps);
	for (UInt32 i = 0; i < sizeof(caps); i++)
		to[i] = at[i];
	if (!rgfxVtServesDecoders(caps)) {
		IOLog("ReimsGraphics: the host decodes no video: no VideoToolbox service\n");
		return;
	}
	if (!rgfxVtServesRelease(version_major)) {
		IOLog("ReimsGraphics: no VideoToolbox service on kernel %d: macOS 26 and later only\n",
		      version_major);
		return;
	}
	AppleVideoToolboxParavirtualizationDriver *made =
	    AppleVideoToolboxParavirtualizationDriver::withFramebuffer(this);
	if (!made) {
		IOLog("ReimsGraphics: the VideoToolbox service did not start\n");
		return;
	}
	__atomic_store_n(&vtService, made, __ATOMIC_RELEASE);
}

void ReimsGraphics::stopVtService()
{
	AppleVideoToolboxParavirtualizationDriver *service =
	    __atomic_exchange_n(&vtService, nullptr, __ATOMIC_ACQ_REL);
	if (!service)
		return;
	service->terminate();
	service->release();
}

void ReimsGraphics::vtMessage(const volatile UInt8 *payload, UInt32 size)
{
	// The event's length and its padding word, then its bytes.
	constexpr UInt32 kBytesAt = RGFX_VT_HOST_MESSAGE_BYTES_OFFSET - sizeof(RgfxMsgHeader);
	AppleVideoToolboxParavirtualizationDriver *service =
	    __atomic_load_n(&vtService, __ATOMIC_ACQUIRE);
	const UInt32 length = size >= kBytesAt ? *reinterpret_cast<const volatile UInt32 *>(payload)
	                                       : 0;
	if (!service || size < kBytesAt || length > size - kBytesAt) {
		static char unread;
		if (rgfxFirstCalls(&unread, size, 1))
			IOLog("ReimsGraphics: a VT_MESSAGE of %u bytes goes nowhere%s\n", size,
			      service ? "" : " (no VideoToolbox service is up)");
		return;
	}
	service->deliver(payload + kBytesAt, length);
}
