/*
 * RgfxAcceleratorUserClient: the connection the guest Metal bundle opens to the
 * RgfxAccelerator over OUR protocol (RgfxUserClientABI.hpp names the ids and selectors): the
 * doorbell, the process's context and its window.
 *   - The window (RgfxAcceleratorUserClientWindow.cpp): the range of the process where its part
 *     of the region lies, each byte at the window's base plus its region offset. The kext maps
 *     in only what is the client's: its blocks, the shared blocks of the surface planes it
 *     holds and the scanout it is granted, and nothing of the rings, the device page or another
 *     process's. The bundle's RgfxRegion and RgfxRingWriter write rings and resources there,
 *     in place in the device's memory. The device info comes from a selector.
 *   - externalMethod(kRgfxAccelMethodDoorbell): writes a ring id to RGFX_REG_DOORBELL
 *     so the host's consumer of that queue's submission ring wakes.
 *
 * This is deliberately NOT Apple's IOGPU submission / allocation user-client ABI. That
 * ABI (its types and selectors) is Apple-private and boot-gated — see the base-class
 * note in RgfxAccelerator.hpp — so it is neither modelled nor invented here.
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_ACCELERATOR_USER_CLIENT_HPP_
#define RGFX_ACCELERATOR_USER_CLIENT_HPP_

#include <IOKit/IOUserClient.h>

#include "RgfxAccelerator.hpp"
#include "RgfxAskGate.hpp"
#include "RgfxQueueLimits.hpp"

class IOBufferMemoryDescriptor;

class RgfxAcceleratorUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxAcceleratorUserClient);

public:
	// IOUserClient lifecycle.
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	IOReturn clientClose() override;
	IOReturn clientDied() override;
	void free() override;

	// Wakes the client's completion waiters (kRgfxAccelMethodWaitCompletions): a completion
	// ring of one of its queues holds records it has not taken. Any thread, no lock of the
	// client's held but its own.
	void wakeCompletions();

	// Maps nothing: the region reaches the process only through the window.
	IOReturn clientMemoryForType(UInt32 type, IOOptionBits *options,
	                             IOMemoryDescriptor **memory) override;

	// The window (RgfxAcceleratorUserClientWindow.cpp): maps the region's `length` bytes at
	// `offset` in at the window's base plus `offset`, a span of the client's own
	// (RgfxWindowSpans.hpp), before the process learns of it; unmapSpan maps the window's own
	// bytes back over the span, read-only, before the span can be anyone else's. Any thread; the
	// window's lock alone, which a caller may take under the accelerator's context or surface
	// lock. mapSpan fails, the op it serves with it, where the window is closed or the span is not
	// one; an unmapSpan that cannot map the window back closes the window.
	IOReturn mapSpan(UInt64 offset, UInt64 length);
	void unmapSpan(UInt64 offset, UInt64 length);

	// Whether the client's task is WindowServer (initWithTask): its blocks may take the reserve
	// no other process is given (RgfxBlockTable).
	bool isWindowServer() const { return windowServer; }

	// A shared event's port's calls (RgfxSharedEventUserClient), for the port's caller `caller`
	// in its context `ctx`: whether this client holds that context open for that task
	// (rgfxPortCallerHolds); and the export and the import, each an ask of this client's, so its
	// close waits for them, posted on the control ring and answered by the device: the export's
	// SHARED_EVENT_NAME of the caller's shared event `eventId` as `name`, its status in
	// `*status` and whether it was posted in `*posted`; the import's SHARED_EVENT_IMPORT of the
	// caller's new shared event `eventId` from `name`, its status and the event's value in
	// `*status` and `*value`, destroyed behind where the device did not answer in time. Each
	// names the caller's queue `listen`, which must be one of this client's.
	// kIOReturnBadArgument where the context is not this client's or is closing, or `listen` is no
	// queue of it; else postLifecycle's return.
	bool holdsContext(task_t caller, UInt64 ctx);
	IOReturn exportSharedEvent(UInt64 ctx, UInt64 eventId, UInt64 listen, UInt64 name,
	                           UInt32 *status, bool *posted);
	IOReturn importSharedEvent(UInt64 ctx, UInt64 eventId, UInt64 listen, UInt64 name,
	                           UInt32 *status, UInt64 *value);
	// IOConnectAddClient(this client's connection, port), which the bundle calls before it calls
	// on a port it was handed: kIOReturnSuccess where the port is a shared event's port of this
	// client's accelerator (rgfxPortIsShareOf), kIOReturnBadArgument where it is another user
	// client. The kernel looks the port up in the caller's space and sends it nothing, and
	// answers kIOReturnBadArgument itself for a port that is no IOKit connection, so a send right
	// of another process's own port is refused at once, where an IOKit call made on it kills the
	// caller (measured on the 5300M and the 630, oracle_events connect). Connects nothing.
	IOReturn connectClient(IOUserClient *client) override;

	// Dispatch our selectors (the doorbell).
	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;

	// RGFX DIAG (Wall-2 trace; remove before merge): log the legacy dispatch paths.
	IOExternalMethod *getTargetAndMethodForIndex(IOService **targetP, UInt32 index) override;
	IOExternalTrap *getTargetAndTrapForIndex(IOService **targetP, UInt32 index) override;

private:
	static IOReturn doorbellMethod(OSObject *target, void *reference,
	                               IOExternalMethodArguments *arguments);

	// The process's context (RgfxAcceleratorUserClientContexts.cpp): its selectors, and its
	// release at the client's close or death (its queues, then the context, then its blocks).
	IOReturn contextMethod(uint32_t selector, IOExternalMethodArguments *arguments);
	// The IOGPU base's device calls on its type-5 client: its configuration and device info.
	IOReturn deviceMethod(uint32_t selector, IOExternalMethodArguments *arguments);
	IOReturn submitMethod(IOExternalMethodArguments *arguments);
	IOReturn takeAnswersMethod(IOExternalMethodArguments *arguments);
	IOReturn waitCompletionsMethod(IOExternalMethodArguments *arguments);
	void releaseContext();
	// An ask the close must not overtake begins (RgfxAskGate.hpp): false where the context is not
	// open or its close has begun, else `*ctx` is the context it asks for. endAsk ends one begun,
	// waking the close where it was the last. Under no lock; each takes listLock.
	bool beginAsk(UInt32 *ctx);
	void endAsk();
	// An ask begun, ended as it goes out of scope.
	struct Ask {
		RgfxAcceleratorUserClient *client;
		bool begun;
		~Ask()
		{
			if (begun)
				client->endAsk();
		}
	};
	// Makes the window at the context's open; closeWindow lets it go, every span in it with it,
	// at the release, before the context's blocks go.
	IOReturn openWindow();
	void closeWindow();
	IOReturn deviceInfoMethod(IOExternalMethodArguments *arguments);
	// The surfaces a command buffer writes (kRgfxAccelMethodSurfaceWrites), and whether a queue is
	// one of the client's open ones.
	IOReturn surfaceWritesMethod(IOExternalMethodArguments *arguments);
	bool ownsQueue(UInt64 queue);
	// Ends one of the client's command queues `objId` the way kRgfxAccelMethodQueueDestroy does
	// (stop watching its completion ring, drop it from queues[], post its destroy), under the
	// queue lock; false where `objId` is not one of the client's queues. QUEUE_DESTROY and a
	// queue's OBJECT_DESTROY posted through submit both go through it, so the kext never keeps
	// watching a queue the host has torn down.
	bool destroyQueueIfTracked(UInt64 objId);

	// As many as the device watches for one process (RgfxQueueLimits.hpp).
	static const UInt32 kMaxQueues = kRgfxMaxQueuesPerClient;
	// The most bytes one kRgfxAccelMethodSurfaceWrites names: 1024 surfaces.
	static const UInt64 kMaxSurfaceWritesBytes = (3 + 2 * 1024) * sizeof(UInt64);
	// The context and the queues, changed under both locks: queueLock is held across what
	// changes them (a context's open and release, a queue's create and destroy), each of which
	// may wait for the device. listLock is held only to read or write these fields, so the
	// process's submits, its commits' surface writes and its completions' retires, which only read
	// them, never wait behind another thread's wait for the device.
	//
	// The asks a close must not overtake (a surface's block, a pages block, a block, a chunk of
	// resource ids, a queue, the scanout's grant) are counted under listLock (`asks`,
	// RgfxAskGate.hpp) and run beside each other, under no lock of the client's but a queue's
	// create, which takes queueLock among the queue creates and destroys for the queue list. The
	// close waits for those in flight before it takes queueLock, and none begins after.
	RgfxAskGate asks;
	bool hasContext;
	UInt32 context;
	// The context's run of units of GPU addresses (RgfxContextSlots.hpp): its first unit and
	// how many.
	UInt32 firstUnit;
	UInt32 units;
	UInt32 queueCount;
	UInt64 queues[kMaxQueues];
	// The queue a QUEUE_CREATE is making, its device's answer not yet in; 0 for none.
	UInt64 creatingQueue;
	IOLock *queueLock;
	IOLock *listLock;

	// Under completionLock: the count wakeCompletions raises, which a waiter sleeps on, and
	// whether the context is released (every waiter then returns).
	IOLock *completionLock;
	UInt64 completionCount;
	bool completionsClosed;

	// Under windowLock: the window's memory and its mapping in the task, its base there and its
	// size, null and 0 while closed; how many spans were mapped in and taken back, and the time
	// it took, logged at the close.
	IOLock *windowLock;
	IOBufferMemoryDescriptor *windowMemory;
	IOMemoryMap *windowMap;
	UInt64 windowBase;
	UInt64 windowSize;
	UInt64 grants;
	UInt64 grantNs;
	UInt64 revokes;
	UInt64 revokeNs;

	task_t task;
	// Whether the task is WindowServer (initWithTask), whose blocks may take the reserve the
	// others are never given (RgfxBlockTable).
	bool windowServer;
	RgfxAccelerator *owner;
	// The type it was opened as: ours (kRgfxAccelUserClientType) or the IOGPU base's device
	// client (kRgfxIOGPUDeviceUserClientType); it decides which selectors it answers.
	UInt32 ucType;
};

#endif // RGFX_ACCELERATOR_USER_CLIENT_HPP_
