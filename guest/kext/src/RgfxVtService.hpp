/*
 * RgfxVtService: the arm64e kext's VideoToolbox service (design.md, "Decided", 17), of the class
 * an arm64 guest's VideoToolboxParavirtualizationSupport looks for,
 * AppleVideoToolboxParavirtualizationDriver: where it finds one and opens it, the guest's
 * VideoToolbox counts itself as running in a guest and sends its host side `vtpv` messages, which
 * the service carries to the device as VT_MESSAGEs. It is a child of display 0's framebuffer,
 * published by it where the guest took VT_MESSAGES and the host's caps list a decoder, and
 * stopped with it.
 *
 * Its user client, RgfxVtUserClient, answers the framework's five selectors as Apple's kext of
 * the class does (AppleVideoToolboxParavirtualization.kext 15.0.0, read from the macOS 27.2.1
 * arm64e VM kernel collection; RgfxVtRoutes.hpp):
 *  - 0, one scalar in, three out: sizes the client's message buffer to at least the size asked,
 *    at most RGFX_VT_MESSAGE_MAX, and answers its capacity and the process's addresses of the
 *    buffer and of its host-message queue, an IOSharedDataQueue of kRgfxVtQueueBytes;
 *  - 1 and 2, a 16-byte structure in: register and deregister a session UUID for the client;
 *  - 3, the message's length and up to five IOSurface IDs in, one scalar out: sends the buffer's
 *    first `length` bytes, and answers the host's status for it, as its output and its return. A
 *    message that names a surface is answered kVTParameterErr, posting nothing: nothing the host
 *    answers yet takes one;
 *  - 4: moves the host messages kept while the queue was full into it.
 * Its notification port, type 0, is the queue's. Apple's kext of the class binds only to Apple's
 * VM device (106b:1a03), which no VM of ours has, so the two never load together.
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
class RgfxVtUserClient;

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
	bool registerUuid(const RgfxVtUserClient *client, const uint8_t *uuid);
	// Selector 2: every registration of `uuid` for `client` let go of, and the host sent `-uui`
	// of it where there was one.
	void deregisterUuid(const RgfxVtUserClient *client, const uint8_t *uuid);
	// A client's close: each UUID it registered let go of, and the host sent `-uui` of each.
	void deregisterAll(const RgfxVtUserClient *client);
	// Sends the message of `length` bytes in the guest RAM runs `runs` (`count` of them) as a
	// VT_MESSAGE, and waits for its VT_STATUS, its status into *status. Thread context only.
	IOReturn send(const RgfxPageRun *runs, uint32_t count, uint64_t length, int32_t *status);
	// handleEvent's VT_MESSAGE, on the device's work loop: the `length` bytes of the host's
	// message, to the clients RgfxVtRoutes.hpp routes it to.
	void deliver(const volatile uint8_t *message, uint32_t length);

private:
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
};

class RgfxVtUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxVtUserClient);

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
