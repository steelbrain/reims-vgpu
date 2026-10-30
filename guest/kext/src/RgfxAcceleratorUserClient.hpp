/*
 * RgfxAcceleratorUserClient: the connection the guest Metal bundle opens to the
 * RgfxAccelerator over OUR protocol (RgfxAccelerator.hpp names the ids and selectors).
 * It does exactly two things, and no more:
 *   - clientMemoryForType(kRgfxAccelMemoryRegion): hands back the device's BAR-2
 *     region so the client's IOConnectMapMemory maps it into the Metal process; the
 *     bundle's RgfxRegion / RgfxRingWriter then write rings and resources into real
 *     device memory, and read the caps page the host wrote in the region's reserved
 *     head.
 *   - externalMethod(kRgfxAccelMethodDoorbell): writes a ring id to RGFX_REG_DOORBELL
 *     so the host's consumer of that queue's submission ring wakes.
 *
 * This is deliberately NOT Apple's IOGPU submission / allocation user-client ABI. That
 * ABI (its types and selectors) is Apple-private and boot-gated — see the base-class
 * note in RgfxAccelerator.hpp — so it is neither modelled nor invented here. Our
 * transport is region-in-place plus a doorbell, which is all the bundle's committed
 * RgfxRingWriter / RgfxReplyReader need.
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_ACCELERATOR_USER_CLIENT_HPP_
#define RGFX_ACCELERATOR_USER_CLIENT_HPP_

#include <IOKit/IOUserClient.h>

#include "RgfxAccelerator.hpp"

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

	// Map the device region (kRgfxAccelMemoryRegion) into the client task.
	IOReturn clientMemoryForType(UInt32 type, IOOptionBits *options,
	                             IOMemoryDescriptor **memory) override;

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
	IOReturn submitMethod(IOExternalMethodArguments *arguments);
	void releaseContext();

	static const UInt32 kMaxQueues = 256;
	bool hasContext;
	UInt32 context;
	UInt32 slot;
	UInt32 queueCount;
	UInt64 queues[kMaxQueues];
	IOLock *queueLock;

	task_t task;
	RgfxAccelerator *owner;
	UInt32 ucType; // RGFX DIAG (remove before merge): the opened user-client type
};

#endif // RGFX_ACCELERATOR_USER_CLIENT_HPP_
