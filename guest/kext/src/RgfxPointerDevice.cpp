/*
 * RgfxPointerDevice: the guest's own mouse, which POINTER_MOTION drives (RgfxPointerDevice.hpp).
 */

#include "RgfxPointerDevice.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/hid/IOHIDUsageTables.h>

OSDefineMetaClassAndFinalStructors(RgfxPointerDevice, IOHIDDevice);

#if !RGFX_GUEST_ARM64
RgfxPointerDevice *RgfxPointerDevice::withProvider(IOService *provider, bool scroller)
#else
RgfxPointerDevice *RgfxPointerDevice::withProvider(IOService *provider)
#endif
{
	RgfxPointerDevice *device = OSTypeAlloc(RgfxPointerDevice);
	if (!device)
		return nullptr;
	device->reportBuffer = nullptr;
#if !RGFX_GUEST_ARM64
	device->buttons = 0;
	device->scroller = scroller;
	device->rest[0] = device->rest[1] = 0;
#endif
	if (!device->init(nullptr)) {
		device->release();
		return nullptr;
	}
#if !RGFX_GUEST_ARM64
	// The scroller's acceleration off, so each count scrolls 10 points at any speed
	// (RgfxPointerMotion.hpp): IOHIDInterface copies HIDSupportsScrollAcceleration from its device
	// (IOHIDInterface.cpp:961), the event service from its interface (IOHIDEventService.cpp:2876),
	// and the scroll filter accelerates no event of a service where it reads false
	// (IOHIDPointerScrollFilter.cpp:917-922, IOHIDFamily-2238.100.59). Copied on 26.7.2 and
	// 11.7.11 alike (measured, potluck jobs m3-max-75987 and m3-max-75994).
	if (scroller)
		device->setProperty("HIDSupportsScrollAcceleration", false);
#endif
	device->reportBuffer =
	    IOBufferMemoryDescriptor::withCapacity(kRgfxMouseReportBytes, kIODirectionOutIn);
	if (!device->reportBuffer || !device->attach(provider)) {
		device->release();
		return nullptr;
	}
	if (!device->start(provider)) {
		device->detach(provider);
		device->release();
		return nullptr;
	}
	return device;
}

void RgfxPointerDevice::post(const RgfxMotion &motion)
{
	uint8_t reports[kRgfxMouseReportsMax][kRgfxMouseReportBytes];
	const uint32_t n = rgfxMouseReports(motion, reports);
	for (uint32_t i = 0; i < n; i++) {
		reportBuffer->writeBytes(0, reports[i], kRgfxMouseReportBytes);
		reportBuffer->setLength(kRgfxMouseReportBytes);
		handleReport(reportBuffer, kIOHIDReportTypeInput, 0);
	}
#if !RGFX_GUEST_ARM64
	buttons = motion.buttons;
#endif
}

#if !RGFX_GUEST_ARM64
void RgfxPointerDevice::scroll(const RgfxScrolled &scrolled)
{
	RgfxMotion motion = {};
	motion.wheel = scrolled.notchesUp;
	motion.pan = scrolled.notchesRight;
	motion.buttons = buttons;
	post(motion);
}

void RgfxPointerDevice::scrollFine(const RgfxScrolled &scrolled)
{
	RgfxMotion motion = {};
	motion.wheel = rgfxScrollerCounts(&rest[0], scrolled.fineUp);
	motion.pan = rgfxScrollerCounts(&rest[1], scrolled.fineRight);
	if (motion.wheel || motion.pan)
		post(motion);
}
#endif

IOReturn RgfxPointerDevice::newReportDescriptor(IOMemoryDescriptor **descriptor) const
{
#if !RGFX_GUEST_ARM64
	IOBufferMemoryDescriptor *buffer =
	    scroller ? IOBufferMemoryDescriptor::withBytes(kRgfxScrollerReportDescriptor,
	                                                   sizeof(kRgfxScrollerReportDescriptor),
	                                                   kIODirectionOutIn)
	             : IOBufferMemoryDescriptor::withBytes(kRgfxMouseReportDescriptor,
	                                                   sizeof(kRgfxMouseReportDescriptor),
	                                                   kIODirectionOutIn);
#else
	IOBufferMemoryDescriptor *buffer = IOBufferMemoryDescriptor::withBytes(
	    kRgfxMouseReportDescriptor, sizeof(kRgfxMouseReportDescriptor), kIODirectionOutIn);
#endif
	if (!buffer)
		return kIOReturnNoResources;
	*descriptor = buffer;
	return kIOReturnSuccess;
}

OSString *RgfxPointerDevice::newManufacturerString() const
{
	return OSString::withCString("Reims vGPU");
}

OSString *RgfxPointerDevice::newProductString() const
{
#if !RGFX_GUEST_ARM64
	if (scroller)
		return OSString::withCString("Reims vGPU Scroller");
#endif
	return OSString::withCString("Reims vGPU Mouse");
}

OSNumber *RgfxPointerDevice::newVendorIDNumber() const
{
	return OSNumber::withNumber(0x1234ULL, 32);
}

OSNumber *RgfxPointerDevice::newProductIDNumber() const
{
#if !RGFX_GUEST_ARM64
	if (scroller)
		return OSNumber::withNumber(0x5248ULL, 32);
#endif
	return OSNumber::withNumber(0x5247ULL, 32);
}

OSNumber *RgfxPointerDevice::newPrimaryUsageNumber() const
{
	return OSNumber::withNumber(static_cast<unsigned long long>(kHIDUsage_GD_Mouse), 32);
}

OSNumber *RgfxPointerDevice::newPrimaryUsagePageNumber() const
{
	return OSNumber::withNumber(static_cast<unsigned long long>(kHIDPage_GenericDesktop), 32);
}

#if defined(__arm64e__)
OSString *RgfxPointerDevice::newTransportString() const
{
	return OSString::withCString(kRgfxMouseTransport);
}
#endif

bool RgfxPointerDevice::willTerminate(IOService *provider, IOOptionBits options)
{
	return IOHIDDevice::willTerminate(provider, options);
}

bool RgfxPointerDevice::didTerminate(IOService *provider, IOOptionBits options, bool *defer)
{
	return IOHIDDevice::didTerminate(provider, options, defer);
}

IOReturn RgfxPointerDevice::updateElementValues(IOHIDElementCookie *cookies, UInt32 cookieCount,
                                                IOOptionBits options, UInt32 completionTimeout,
                                                IOHIDCompletion *completion,
                                                IOBufferMemoryDescriptor *elementData)
{
	return IOHIDDevice::updateElementValues(cookies, cookieCount, options, completionTimeout,
	                                        completion, elementData);
}

IOReturn RgfxPointerDevice::postElementValues(IOHIDElementCookie *cookies, UInt32 cookieCount,
                                              IOOptionBits options, UInt32 completionTimeout,
                                              IOHIDCompletion *completion)
{
	return IOHIDDevice::postElementValues(cookies, cookieCount, options, completionTimeout,
	                                      completion);
}

void RgfxPointerDevice::free()
{
	if (reportBuffer) {
		reportBuffer->release();
		reportBuffer = nullptr;
	}
	IOHIDDevice::free();
}
