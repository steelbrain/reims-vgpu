/*
 * RgfxVtService: the arm64e kext's VideoToolbox service (design.md, "Decided", 17), of the class
 * an arm64 guest's VideoToolboxParavirtualizationSupport looks for,
 * AppleVideoToolboxParavirtualizationDriver: where it finds one and opens it, the guest's
 * VideoToolbox counts itself as running in a guest and sends its host side `vtpv` messages, which
 * the service carries to the device as VT_MESSAGEs. It is a child of display 0's framebuffer,
 * published by it where the guest took VT_MESSAGES and the host's caps list a decoder, and
 * stopped with it.
 *
 * Its user client, AppleVideoToolboxParavirtualizationUserClient, of the class name the guest's
 * sandboxes let a process open (VideoToolbox's encoder and decoder XPC services' profiles and
 * the app sandbox's name it alone, read from macOS 27's; a sandboxed process's VideoToolbox finds
 * no paravirtual codec through a user client of another name), answers the framework's five
 * selectors as Apple's kext of the class does (AppleVideoToolboxParavirtualization.kext 15.0.0,
 * read from the macOS 27.2.1 arm64e VM kernel collection; RgfxVtRoutes.hpp):
 *  - 0, one scalar in, three out: sizes the client's message buffer to at least the size asked,
 *    at most RGFX_VT_MESSAGE_MAX, and answers its capacity and the process's addresses of the
 *    buffer and of its host-message queue, an IOSharedDataQueue of kRgfxVtQueueBytes;
 *  - 1 and 2, a 16-byte structure in: register and deregister a session UUID for the client;
 *  - 3, the message's length and up to five IOSurface IDs in, one scalar out: sends the buffer's
 *    first `length` bytes, lending the host the IOSurfaces of the IDs (looked up in the client's
 *    task), and answers the host's status for it, as its output and its return. A surface the
 *    task cannot see, or more lends than the service holds, answers kVTParameterErr, posting
 *    nothing;
 *  - 4: moves the host messages kept while the queue was full into it.
 * Its notification port, type 0, is the queue's.
 *
 * A lent surface is held, retained and its memory wired, from the send until the host's `rels`
 * names its lend, whatever becomes of the client, and is let go of at the service's stop. The
 * host writes a decoded picture into its pages meanwhile. Apple's kext of the class binds only to
 * Apple's VM device (106b:1a03), which no VM of ours has, so the two never load together.
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_VT_SERVICE_HPP_
#define RGFX_VT_SERVICE_HPP_

#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOLocks.h>
#include <IOKit/IOService.h>
#include <IOKit/IOSharedDataQueue.h>
#include <IOKit/IOUserClient.h>

#include "RgfxVtRoutes.hpp"

class ReimsGraphics;
class AppleVideoToolboxParavirtualizationUserClient;

class AppleVideoToolboxParavirtualizationDriver : public IOService {
	OSDeclareDefaultStructors(AppleVideoToolboxParavirtualizationDriver);

public:
	// The service of display 0's framebuffer `fb`, started and registered; nullptr where it did
	// not start.
	static AppleVideoToolboxParavirtualizationDriver *withFramebuffer(ReimsGraphics *fb);

	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	void free() override;
	IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type,
	                       OSDictionary *properties, IOUserClient **handler) override;

	// Selector 1: `uuid` registered for `client`; false where the routes are full.
	bool registerUuid(const AppleVideoToolboxParavirtualizationUserClient *client,
	                  const uint8_t *uuid);
	// Selector 2: every registration of `uuid` for `client` let go of, and the host sent `-uui`
	// of it where there was one.
	void deregisterUuid(const AppleVideoToolboxParavirtualizationUserClient *client,
	                    const uint8_t *uuid);
	// A client's close: each UUID it registered let go of, and the host sent `-uui` of each.
	void deregisterAll(const AppleVideoToolboxParavirtualizationUserClient *client);
	// Sends the message of `length` bytes in the guest RAM runs `runs` (`count` of them) as a
	// VT_MESSAGE, and waits for its VT_STATUS, its status into *status. Thread context only.
	IOReturn send(const RgfxPageRun *runs, uint32_t count, uint64_t length, int32_t *status);
	// The same, lending the `lentCount` surfaces of `lent`, each lent already (lend); *posted
	// says whether the message reached the ring, after which the host lets go of the lends.
	IOReturn send(const RgfxPageRun *runs, uint32_t count, uint64_t length, const RgfxVtLent *lent,
	              uint32_t lentCount, int32_t *status, bool *posted);
	// A host message, on the device's work loop, from handleEvent's VT_MESSAGE or a VT_RING's
	// take of the VideoToolbox ring (ReimsGraphics::vtMessage): its `length` bytes, to the
	// clients RgfxVtRoutes.hpp routes it to, or the lends a `rels` lets go of. It copies them
	// before it returns.
	void deliver(const volatile uint8_t *message, uint32_t length);
	// Lends the host the IOSurface `surfaceId` as `task` sees it: held, its memory wired, its
	// record, named by its new lend, into *record and its runs into `runs` (at most `max`); where
	// they are more, a run table of them, which the lend holds, and the table's runs instead
	// (RGFX_VT_SURFACE_RUN_TABLE). kIOReturnSuccess, or why not, with nothing held.
	IOReturn lend(task_t task, uint32_t surfaceId, RgfxVtSurface *record, RgfxPageRun *runs,
	              uint32_t max);
	// Lets go of the lend `name`, where the service holds it.
	void letGoLend(uint64_t name);

private:
	// One lend the service holds.
	struct Lend {
		uint64_t name;
		OSObject *surface;
		IOMemoryDescriptor *memory;
		// The surface's run table, prepared, where its runs were more than its message holds.
		IOBufferMemoryDescriptor *table;
	};
	// The run table of `memory`'s `length` bytes from `base`: wired memory of the service's,
	// prepared, of whole pages, that holds their runs (RgfxPageRun, in order), its own runs into
	// `runs` (at most `max`) and *count. nullptr where it is not made or its runs are more.
	static IOBufferMemoryDescriptor *runTable(IOMemoryDescriptor *memory, uint64_t base,
	                                          uint64_t length, RgfxPageRun *runs, uint32_t max,
	                                          uint32_t *count);
	// Lets go of every lend.
	void letGoLends();
	// Sends the host `-uui` of `uuid`, from forgetBuffer.
	void sendForget(const uint8_t *uuid);

	// The framebuffer, retained from start until free.
	ReimsGraphics *fb;
	// Under lock: the routes, IOMalloc'd.
	IOLock *lock;
	RgfxVtRoutes *routes;
	// The page the kext writes its own `-uui`s in, wired, and its run; under forgetLock.
	IOBufferMemoryDescriptor *forgetBuffer;
	RgfxPageRun forgetRun;
	IOLock *forgetLock;
	// The work loop's copy of the host message it routes, RGFX_VT_HOST_MESSAGE_MAX bytes.
	uint8_t *delivered;
	// IOSurfaceRoot, retained, which surfaces are looked up in.
	IOService *surfaceRoot;
	// Under lock: the lends held, kRgfxVtLendsMax of them IOMalloc'd, and the last lend's name.
	Lend *lends;
	uint32_t lendCount;
	uint64_t lastLend;
};

class AppleVideoToolboxParavirtualizationUserClient : public IOUserClient {
	OSDeclareDefaultStructors(AppleVideoToolboxParavirtualizationUserClient);

public:
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	bool start(IOService *provider) override;
	IOReturn clientClose() override;
	IOReturn clientDied() override;
	void free() override;
	// The queue's port: the 64-bit reference's, which a 64-bit process's
	// IOConnectSetNotificationPort reaches, and the 32-bit one's IOUserClient's own.
	using IOUserClient::registerNotificationPort;
	IOReturn registerNotificationPort(mach_port_t port, UInt32 type, io_user_reference_t refCon)
	    override;
	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;
	// A host message for this client, on the work loop: into its queue, or kept while the queue
	// is full or holds kept ones.
	void deliver(const uint8_t *message, uint32_t length);

private:
	IOReturn sizeBuffer(IOExternalMethodArguments *a);
	IOReturn sendMessage(IOExternalMethodArguments *a);
	// Moves the kept messages into the queue while it has room. Under lock.
	void deliverKept();
	// Lets go of the service's routes, once.
	void letGo();

	task_t task;
	// The service, retained from start until free.
	AppleVideoToolboxParavirtualizationDriver *service;
	// Under lock: the message buffer, its capacity, its process mapping and its runs; the queue,
	// its process mapping and its port; the kept messages; and whether the client let go.
	IOLock *lock;
	IOBufferMemoryDescriptor *buffer;
	IOMemoryMap *bufferMap;
	uint64_t capacity;
	RgfxPageRun *runs;
	uint32_t runCount;
	uint32_t runCapacity;
	IOSharedDataQueue *queue;
	IOMemoryMap *queueMap;
	bool portSet;
	OSArray *kept;
	bool gone;
	// Serializes the client's sends, which share its buffer.
	IOLock *sendLock;
};

#endif // RGFX_VT_SERVICE_HPP_
