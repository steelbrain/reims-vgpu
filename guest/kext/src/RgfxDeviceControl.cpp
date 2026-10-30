/*
 * RgfxDeviceControl: see RgfxDeviceControl.hpp for the attributes it answers
 * and why the device needs one.
 */

#include "RgfxDeviceControl.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/graphics/IOFramebuffer.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <libkern/OSAtomic.h>

#define super AppleGraphicsDeviceControl
OSDefineMetaClassAndStructors(RgfxDeviceControl, AppleGraphicsDeviceControl);

namespace {

constexpr UInt32 kAttributeDescription = 0x1;
constexpr UInt32 kAttributeInit = 0x3;
constexpr UInt32 kAttributeGpuIndex = 0x301;
constexpr UInt32 kAttributeDisplays = 0x980;
constexpr UInt32 kAttributeLimits = 0x921;

constexpr UInt32 kInitFramebuffers = 1u << 0;
constexpr UInt32 kInitAccelerator = 1u << 1;

constexpr UInt32 kDescriptionVersion = 0x10000;
constexpr char kDescriptionName[] = "ReimsGraphics";
// The +0x24 half-word IOPresentment's platform check reads, and the device's
// PCI device ID after it.
constexpr UInt16 kDescriptionVendor = 0x106b;
constexpr UInt16 kDescriptionDevice = 0x5247;
// The +0x28 word that makes the control a display device to the wrangler and
// to IOPresentment.
constexpr UInt32 kDescriptionKind = 2;

// The one display the device drives.
constexpr UInt32 kDisplayCount = 1;

constexpr SInt32 kLoggedCalls = 256;

struct Description {
	UInt32 field_00;     // kDescriptionVersion
	char field_04[0x20]; // kDescriptionName, NUL-padded
	UInt16 field_24;     // kDescriptionVendor
	UInt16 field_26;     // kDescriptionDevice
	UInt32 field_28;     // kDescriptionKind
};
static_assert(sizeof(Description) == 0x2c, "attribute 1 is 0x2c bytes");

// Attribute 0x980, IOPresentment's "AGDC GPU capabilities": without it, it
// builds no pipe. Four masks of the display ports (bit 1 + i for display i),
// the display count five times, the control's provider, and each display's
// framebuffer.
struct __attribute__((packed)) Displays {
	UInt64 field_00[4];
	UInt32 field_20[5];
	UInt64 field_34;
	UInt64 field_3c[20];
};
static_assert(sizeof(Displays) == 0xdc, "attribute 0x980 is 0xdc bytes");

// Attribute 0x921, 0xb0 bytes, which IOPresentment reads once it has a pipe.
// What its words mean is not known; these are the values a display device
// answers.
struct __attribute__((packed)) Limits {
	UInt64 field_00;
	UInt64 field_08; // 1
	UInt64 field_10;
	UInt32 field_18;
	UInt32 field_1c; // 0x500
	UInt32 field_20; // 0x400
	UInt8 field_24[0x20];
	UInt32 field_44; // 0x500
	UInt64 field_48;
	UInt32 field_50;
	UInt32 field_54; // 0x400
	UInt8 field_58[0x2c];
	UInt32 field_84; // 1
	UInt8 field_88[0x28];
};
static_assert(sizeof(Limits) == 0xb0, "attribute 0x921 is 0xb0 bytes");

} // namespace

bool RgfxDeviceControl::start(IOService *provider)
{
	if (!OSDynamicCast(IOPCIDevice, provider))
		return false;
	// The library's start reads attribute 1, waits for the wrangler and
	// registers the control.
	bool started = super::start(provider);
	IOLog("RgfxDeviceControl: start %s\n", started ? "done" : "failed");
	return started;
}

IOReturn RgfxDeviceControl::vendor_doDeviceAttribute(UInt32 attribute, uintptr_t *input,
                                                     size_t inputSize, uintptr_t *output,
                                                     size_t *outputSize,
                                                     IOExternalMethodArguments *)
{
	size_t capacity = outputSize ? *outputSize : 0;
	IOReturn r = answer(attribute, input, inputSize, output, outputSize);
	if (OSIncrementAtomic(&loggedCalls) < kLoggedCalls)
		IOLog("RgfxDeviceControl: attribute 0x%x in %lu out %lu: 0x%x\n", attribute,
		      input ? static_cast<unsigned long>(inputSize) : 0UL,
		      output ? static_cast<unsigned long>(capacity) : 0UL, r);
	return r;
}

IOFramebuffer *RgfxDeviceControl::framebuffer(IOService *provider)
{
	if (!provider)
		return nullptr;
	IOFramebuffer *found = nullptr;
	OSIterator *children = provider->getChildIterator(gIOServicePlane);
	if (!children)
		return nullptr;
	while (OSObject *child = children->getNextObject()) {
		found = OSDynamicCast(IOFramebuffer, child);
		if (found)
			break;
	}
	children->release();
	return found;
}

IOReturn RgfxDeviceControl::answer(UInt32 attribute, uintptr_t *input, size_t inputSize,
                                   uintptr_t *output, size_t *outputSize)
{
	bool get = output && outputSize;
	switch (attribute) {
	case kAttributeDescription: {
		if (!get || *outputSize < sizeof(Description))
			return kIOReturnBadArgument;
		Description d = {};
		d.field_00 = kDescriptionVersion;
		strlcpy(d.field_04, kDescriptionName, sizeof d.field_04);
		d.field_24 = kDescriptionVendor;
		d.field_26 = kDescriptionDevice;
		d.field_28 = kDescriptionKind;
		memcpy(output, &d, sizeof d);
		*outputSize = sizeof d;
		return kIOReturnSuccess;
	}
	case kAttributeLimits: {
		if (!get || *outputSize < sizeof(Limits))
			return kIOReturnBadArgument;
		Limits l = {};
		l.field_08 = 1;
		l.field_1c = 0x500;
		l.field_20 = 0x400;
		l.field_44 = 0x500;
		l.field_54 = 0x400;
		l.field_84 = 1;
		memcpy(output, &l, sizeof l);
		*outputSize = sizeof l;
		return kIOReturnSuccess;
	}
	case kAttributeDisplays: {
		if (!get || *outputSize < sizeof(Displays))
			return kIOReturnBadArgument;
		Displays d = {};
		UInt64 ports = ((1ULL << kDisplayCount) - 1) << 1;
		for (auto &mask : d.field_00)
			mask = ports;
		for (auto &count : d.field_20)
			count = kDisplayCount;
		IOService *provider = getProvider();
		d.field_34 = reinterpret_cast<uintptr_t>(provider);
		d.field_3c[0] = reinterpret_cast<uintptr_t>(framebuffer(provider));
		memcpy(output, &d, sizeof d);
		*outputSize = sizeof d;
		return kIOReturnSuccess;
	}
	case kAttributeInit:
		if (get) {
			if (*outputSize < sizeof(UInt32))
				return kIOReturnBadArgument;
			UInt32 mask = kInitFramebuffers | kInitAccelerator;
			memcpy(output, &mask, sizeof mask);
			*outputSize = sizeof mask;
			return kIOReturnSuccess;
		}
		if (!input || inputSize < sizeof(UInt32))
			return kIOReturnBadArgument;
		memcpy(&initFlags, input, sizeof initFlags);
		setProperty("RgfxInitFlags", initFlags, 32);
		return kIOReturnSuccess;
	case kAttributeGpuIndex:
		if (!input || inputSize < sizeof(UInt16))
			return kIOReturnBadArgument;
		memcpy(&gpuIndex, input, sizeof gpuIndex);
		setProperty("RgfxGPUIndex", gpuIndex, 16);
		return kIOReturnSuccess;
	default:
		return kIOReturnUnsupported;
	}
}
