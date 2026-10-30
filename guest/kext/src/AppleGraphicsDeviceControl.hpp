/*
 * The interface of AppleGraphicsDeviceControl (AGDC), the System-KC library
 * class (com.apple.AppleGraphicsDeviceControl, AppleGraphicsControl.kext's
 * plug-in) that RgfxDeviceControl derives from. The SDK has no header for it:
 * this is our own declaration of the parts a subclass binds, from the names the
 * macOS 26 KDK's x86_64 binary exports and the order of its vtable's slots.
 *
 * - Size 0x110, after IOService's 0x88. Its fields are the library's own; a
 *   subclass only needs their extent, so its own members land after them.
 * - Vtable, 276 slots: IOService's 0-265, then the two vendor hooks (266, 267)
 *   and eight reserved slots (268-275). The class overrides IOService's start,
 *   stop, terminate, message and the five-argument newUserClient, so they are
 *   declared here: without them, a subclass's vtable would name IOService's
 *   methods in those slots, and AGDC's own would never run.
 *
 * What a subclass answers (vendor_doDeviceAttribute with the method
 * arguments) is the device's attributes: the library's get, set and do entry
 * points funnel into the AGDCClientState_t hook, whose own body forwards to
 * the pure one. A get has no input and an output of *outputSize bytes; a set
 * has inputSize bytes of input and no output.
 */

#ifndef RGFX_APPLE_GRAPHICS_DEVICE_CONTROL_HPP_
#define RGFX_APPLE_GRAPHICS_DEVICE_CONTROL_HPP_

#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>

struct AGDCClientState_t;

class AppleGraphicsDeviceControl : public IOService {
	OSDeclareAbstractStructors(AppleGraphicsDeviceControl);

public:
	bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
	void stop(IOService *provider) APPLE_KEXT_OVERRIDE;
	bool terminate(IOOptionBits options) APPLE_KEXT_OVERRIDE;
	IOReturn message(UInt32 type, IOService *provider, void *argument) APPLE_KEXT_OVERRIDE;
	IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type,
	                       OSDictionary *properties, IOUserClient **handler) APPLE_KEXT_OVERRIDE;

	// Slot 266: the subclass's answer to one attribute.
	virtual IOReturn vendor_doDeviceAttribute(UInt32 attribute, uintptr_t *input, size_t inputSize,
	                                          uintptr_t *output, size_t *outputSize,
	                                          IOExternalMethodArguments *arguments) = 0;
	// Slot 267: what the library calls; forwards to slot 266.
	virtual IOReturn vendor_doDeviceAttribute(UInt32 attribute, uintptr_t *input, size_t inputSize,
	                                          uintptr_t *output, size_t *outputSize,
	                                          AGDCClientState_t *client);

	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 0);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 1);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 2);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 3);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 4);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 5);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 6);
	OSMetaClassDeclareReservedUnused(AppleGraphicsDeviceControl, 7);

private:
	UInt8 field_88[0x110 - 0x88];
};

static_assert(sizeof(IOService) == 0x88, "IOService's size is not the KDK's");
static_assert(sizeof(AppleGraphicsDeviceControl) == 0x110,
              "AppleGraphicsDeviceControl's size is not the KDK's");

#endif /* RGFX_APPLE_GRAPHICS_DEVICE_CONTROL_HPP_ */
