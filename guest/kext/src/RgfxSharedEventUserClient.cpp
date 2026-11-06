/*
 * RgfxSharedEventUserClient: see RgfxSharedEventUserClient.hpp. A shared event's port: its
 * export and its imports, each posted through the caller's own context's client.
 */

#include "RgfxSharedEventUserClient.hpp"

#include "RgfxAccelerator.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxUserClientABI.hpp"

#include <IOKit/IOLib.h>

#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxSharedEventUserClient, IOUserClient);

bool RgfxSharedEventUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                             OSDictionary *properties)
{
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	accelerator = nullptr;
	port = RgfxPortName();
	detached = false;
	lock = IOLockAlloc();
	return lock != nullptr;
}

bool RgfxSharedEventUserClient::start(IOService *provider)
{
	RgfxAccelerator *owner = OSDynamicCast(RgfxAccelerator, provider);
	if (!owner || !super::start(provider))
		return false;
	owner->retain();
	accelerator = owner;
	return true;
}

void RgfxSharedEventUserClient::letGoOfAccelerator()
{
	IOLockLock(lock);
	const bool first = !detached;
	detached = true;
	IOLockUnlock(lock);
	if (first && accelerator)
		detach(accelerator);
}

// The process closed its connection: the port lives on for every task that holds it.
IOReturn RgfxSharedEventUserClient::clientClose()
{
	letGoOfAccelerator();
	return kIOReturnSuccess;
}

// Its creator died, or no task holds the port: in either case the port lives as long as a send
// right of it does, and the free that follows its last lets go of the name.
IOReturn RgfxSharedEventUserClient::clientDied()
{
	letGoOfAccelerator();
	return kIOReturnSuccess;
}

void RgfxSharedEventUserClient::free()
{
	if (accelerator) {
		// The export's name, where it gave one: no process holds the port now, so no import can
		// name it.
		if (port.name) {
			IOLog("RGFX: shared event port gone: name %llu let go of\n", port.name);
			accelerator->unnameSharedEvent(port.name);
		}
		accelerator->release();
		accelerator = nullptr;
	}
	if (lock)
		IOLockFree(lock);
	lock = nullptr;
	super::free();
}

IOReturn RgfxSharedEventUserClient::externalMethod(uint32_t selector,
                                                   IOExternalMethodArguments *a,
                                                   IOExternalMethodDispatch *dispatch,
                                                   OSObject *target, void *reference)
{
	if (!a || !accelerator)
		return kIOReturnBadArgument;
	if (selector == kRgfxSharedEventMethodExport && a->scalarInputCount == 3 &&
	    a->scalarOutputCount == 1)
		return exportMethod(a);
	if (selector == kRgfxSharedEventMethodImport && a->scalarInputCount == 3 &&
	    a->scalarOutputCount == 2)
		return importMethod(a);
	return kIOReturnUnsupported;
}

// kRgfxSharedEventMethodExport: the caller's shared event named for other processes, once.
IOReturn RgfxSharedEventUserClient::exportMethod(IOExternalMethodArguments *a)
{
	const uint64_t ctx = a->scalarInput[0], eventId = a->scalarInput[1],
	               listen = a->scalarInput[2];
	if (!rgfxPortIdsInContext(ctx, eventId, listen))
		return kIOReturnBadArgument;
	IOLockLock(lock);
	const bool begun = port.beginExport();
	IOLockUnlock(lock);
	if (!begun)
		return kIOReturnExclusiveAccess;
	RgfxAcceleratorUserClient *client = accelerator->copyContextClient(current_task(), ctx);
	if (!client) {
		IOLockLock(lock);
		port.endExport(0);
		IOLockUnlock(lock);
		return kIOReturnBadArgument;
	}
	const UInt64 name = accelerator->takeSharedEventName();
	UInt32 status = RGFX_STATUS_REFUSED;
	bool posted = false;
	IOReturn r = client->exportSharedEvent(ctx, eventId, listen, name, &status, &posted);
	client->release();
	const bool named = r == kIOReturnSuccess && status == RGFX_STATUS_OK;
	IOLockLock(lock);
	port.endExport(named ? name : 0);
	IOLockUnlock(lock);
	// A name posted and not taken in time may still be taken: let go of it behind the post, so
	// the device holds nothing of it.
	if (posted && !named)
		accelerator->unnameSharedEvent(name);
	a->scalarOutput[0] = status;
	return r;
}

// kRgfxSharedEventMethodImport: the caller's new shared event made of the one the port names.
IOReturn RgfxSharedEventUserClient::importMethod(IOExternalMethodArguments *a)
{
	const uint64_t ctx = a->scalarInput[0], eventId = a->scalarInput[1],
	               listen = a->scalarInput[2];
	if (!rgfxPortIdsInContext(ctx, eventId, listen))
		return kIOReturnBadArgument;
	IOLockLock(lock);
	const UInt64 name = port.forImport();
	IOLockUnlock(lock);
	if (!name)
		return kIOReturnNotReady;
	RgfxAcceleratorUserClient *client = accelerator->copyContextClient(current_task(), ctx);
	if (!client)
		return kIOReturnBadArgument;
	UInt32 status = RGFX_STATUS_REFUSED;
	UInt64 value = 0;
	IOReturn r = client->importSharedEvent(ctx, eventId, listen, name, &status, &value);
	client->release();
	a->scalarOutput[0] = status;
	a->scalarOutput[1] = value;
	return r;
}

// MARK: - The accelerator's side

namespace {
// The last name RgfxAccelerator::takeSharedEventName gave, 0 at the kext's start: one device,
// so one count.
UInt64 gSharedEventNames = 0;
} // namespace

IOReturn RgfxAccelerator::newSharedEventPort(task_t owningTask, void *securityID, UInt32 type,
                                             OSDictionary *properties, IOUserClient **handler)
{
	if (!(featureWord() & RGFX_FEATURE_SHARED_EVENT_NAMES))
		return kIOReturnUnsupported;
	RgfxSharedEventUserClient *client = OSTypeAlloc(RgfxSharedEventUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnBadArgument;
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

RgfxAcceleratorUserClient *RgfxAccelerator::copyContextClient(task_t caller, UInt64 ctx)
{
	OSIterator *clients = getClientIterator();
	if (!clients)
		return nullptr;
	RgfxAcceleratorUserClient *found = nullptr;
	while (OSObject *o = clients->getNextObject()) {
		RgfxAcceleratorUserClient *c = OSDynamicCast(RgfxAcceleratorUserClient, o);
		if (c && c->holdsContext(caller, ctx)) {
			c->retain();
			found = c;
			break;
		}
	}
	clients->release();
	return found;
}

UInt64 RgfxAccelerator::takeSharedEventName()
{
	return __atomic_add_fetch(&gSharedEventNames, 1, __ATOMIC_RELAXED);
}

void RgfxAccelerator::unnameSharedEvent(UInt64 name)
{
	RgfxSharedEventUnname m = rgfxSharedEventUnnameMessage(name);
	IOReturn r = postLifecycle(RGFX_CMD_SHARED_EVENT_UNNAME, &m, sizeof(m), 0, nullptr, 0, nullptr,
	                           nullptr, 0);
	if (r != kIOReturnSuccess)
		IOLog("RGFX: shared event name %llu: its unname was not posted (0x%x)\n", name, r);
}
