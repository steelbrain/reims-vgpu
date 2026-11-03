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
	// changes them (a context's open and release, a queue's create and destroy) and across a
	// surface block's ask and a resource id chunk's, which a close must not overtake, each of
	// which may wait for the device. listLock is held only to read or write these fields, so the
	// process's submits, its commits' surface writes and its completions' retires, which only read
	// them, never wait behind another thread's wait for the device.
	bool hasContext;
	UInt32 context;
	UInt32 slot;
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
