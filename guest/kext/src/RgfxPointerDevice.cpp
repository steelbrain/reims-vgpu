/*
 * RgfxPointerDevice: the guest's own mouse, which POINTER_MOTION drives (RgfxPointerDevice.hpp).
 */

#include "RgfxPointerDevice.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/hid/IOHIDUsageTables.h>

OSDefineMetaClassAndFinalStructors(RgfxPointerDevice, IOHIDDevice);

RgfxPointerDevice *RgfxPointerDevice::withProvider(IOService *provider)
{
	RgfxPointerDevice *device = OSTypeAlloc(RgfxPointerDevice);
	if (!device)
		return nullptr;
	device->reportBuffer = nullptr;
	if (!device->init(nullptr)) {
		device->release();
		return nullptr;
	}
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
}

IOReturn RgfxPointerDevice::newReportDescriptor(IOMemoryDescriptor **descriptor) const
{
	IOBufferMemoryDescriptor *buffer = IOBufferMemoryDescriptor::withBytes(
	    kRgfxMouseReportDescriptor, sizeof(kRgfxMouseReportDescriptor), kIODirectionOutIn);
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
	return OSString::withCString("Reims vGPU Mouse");
}

OSNumber *RgfxPointerDevice::newVendorIDNumber() const
{
	return OSNumber::withNumber(0x1234ULL, 32);
}

OSNumber *RgfxPointerDevice::newProductIDNumber() const
{
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
