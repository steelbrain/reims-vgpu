// RgfxPointerDevice: the guest's own relative pointing device, a HID mouse with no bus under it,
// which the framebuffer publishes where the guest took POINTER_MOTION and hands each
// POINTER_MOTION as reports (RgfxPointerMotion.hpp, design.md "A pointer for games"). macOS's
// HID stack drives it as any USB mouse: IOHIDEventDriver turns its reports into pointer events,
// which WindowServer accelerates and moves the cursor by, and GCMouse and IOHIDManager list it.
//
// Its name and numbers are ours: "Reims vGPU Mouse" of "Reims vGPU", vendor 0x1234 and product
// 0x5247, the PCI IDs of the device it belongs to. It is a child of display 0's framebuffer,
// started by it once the guest took the feature and stopped with it.
//
// Our own code.
#ifndef RGFX_POINTER_DEVICE_HPP_
#define RGFX_POINTER_DEVICE_HPP_

#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/hid/IOHIDDevice.h>

#include "RgfxPointerMotion.hpp"

class RgfxPointerDevice final : public IOHIDDevice
{
	OSDeclareFinalStructors(RgfxPointerDevice);

public:
	// The device, attached to `provider` and started: nullptr where it does not start.
	static RgfxPointerDevice *withProvider(IOService *provider);

	// Hands the HID stack the reports of one POINTER_MOTION, in order. On the framebuffer's work
	// loop, one call at a time.
	void post(const RgfxMotion &motion);

	IOReturn newReportDescriptor(IOMemoryDescriptor **descriptor) const override;
	OSString *newManufacturerString() const override;
	OSString *newProductString() const override;
	OSNumber *newVendorIDNumber() const override;
	OSNumber *newProductIDNumber() const override;
	OSNumber *newPrimaryUsageNumber() const override;
	OSNumber *newPrimaryUsagePageNumber() const override;

private:
	void free() override;

	// The buffer each report is written into before the HID stack reads it.
	IOBufferMemoryDescriptor *reportBuffer;
};

#endif // RGFX_POINTER_DEVICE_HPP_
