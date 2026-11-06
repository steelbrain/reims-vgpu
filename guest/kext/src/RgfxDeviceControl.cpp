/*
 * RgfxDeviceControl: see RgfxDeviceControl.hpp for the attributes it answers
 * and why the device needs one.
 */

#include "RgfxDeviceControl.hpp"
#include "RgfxGammaCapability.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/graphics/IOFramebuffer.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <libkern/OSAtomic.h>

#define super AppleGraphicsDeviceControl
OSDefineMetaClassAndStructors(RgfxDeviceControl, AppleGraphicsDeviceControl);

namespace {

constexpr UInt32 kAttributeDescription = 0x1;
constexpr UInt32 kAttributeInit = 0x3;
constexpr UInt32 kAttribute921 = 0x921;
constexpr UInt32 kAttributeDisplays = 0x980;
constexpr UInt32 kAttributeConnection = 0x925;
constexpr UInt32 kAttributeQuery = 0x711;
constexpr UInt32 kAttributeEdid = 0x986;

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

// The one display the device drives, and its framebuffer's index, by which
// the display's attributes name it.
constexpr UInt32 kDisplayCount = 1;
constexpr UInt32 kDisplayIndex = 0;

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
struct __attribute__((packed)) Attribute921 {
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
static_assert(sizeof(Attribute921) == 0xb0, "attribute 0x921 is 0xb0 bytes");

// Attribute 0x925's request and answer: the display (its index plus 1), and
// in the answer, whether it is connected.
struct Connection {
	UInt32 field_00;
	UInt32 field_04;
	UInt32 field_08;
};
static_assert(sizeof(Connection) == 12, "attribute 0x925 is 12 bytes");

// Attribute 0x711's request and answer, the same bytes both ways: the
// display (its index), the request (one bit), and in the answer, its entry
// count and its entries in the body, 0x32c bytes apart (IOPresentment reads
// at most 8; RgfxGammaCapability.hpp).
struct Query {
	UInt32 field_00;
	UInt32 field_04;
	UInt32 field_08;
	UInt8 field_0c[0x1960];
};
static_assert(sizeof(Query) == 0x196c, "attribute 0x711 is 0x196c bytes");
static_assert(__builtin_offsetof(Query, field_08) == kRgfxCapabilityCount &&
                  __builtin_offsetof(Query, field_0c) == kRgfxCapabilityEntries,
              "the count and the entries are where IOPresentment reads them");

// Each answer holds one entry.
constexpr UInt32 kQueryEntries = 1;

// The requests the display answers: each bit of the low 16 but 0x40. Every
// answer's entry is zeros, which names no plane, but those of requests 1, 2
// and 0x10.
constexpr UInt32 kQueryRequests = 0xffbf;
constexpr UInt32 kQuery1 = 0x1;
constexpr UInt32 kQueryGamma = 0x2;
constexpr UInt32 kQueryMode = 0x10;

// Two of the answers carry floats: their IEEE 754 single-precision bits, so
// the kext does no floating point.
constexpr UInt32 kFloatOne = 0x3f800000;
constexpr UInt32 kFloatMinusOne = 0xbf800000;

// Request 1's answer. What its words mean is not known.
struct __attribute__((packed)) Query1 {
	UInt32 field_00; // 5
	UInt64 field_04;
	UInt32 field_0c;
	UInt32 field_10; // 8
	UInt32 field_14;
	UInt64 field_18;
	UInt32 field_20; // 0x30300
	UInt32 field_24; // -1.0f
	UInt32 field_28; // 1.0f
};
static_assert(sizeof(Query1) == 0x2c, "request 1's answer is 0x2c bytes");

// Request 0x10's answer: two of these after 0x1c bytes of zeros, each the
// current mode's size.
struct __attribute__((packed)) QueryModeSize {
	UInt64 field_00; // 1
	UInt32 field_08; // the mode's height
	UInt32 field_0c; // its width
	UInt32 field_10; // its height
	UInt32 field_14; // its width
	UInt32 field_18; // 0x10000
	UInt8 field_1c[0x16c];
};
static_assert(sizeof(QueryModeSize) == 0x188, "a mode size is 0x188 bytes");

struct __attribute__((packed)) QueryMode {
	UInt8 field_00[0x1c];
	QueryModeSize field_1c[2];
};
static_assert(sizeof(QueryMode) <= sizeof(Query::field_0c), "request 0x10's answer fits");

// Attribute 0x986's request and answer: the display (its index plus 1), the
// block's index in the request, and the block in the answer.
struct Edid {
	UInt32 field_00;
	UInt32 field_04;
	UInt32 field_08;
	UInt8 field_0c[128];
	UInt32 field_8c;
};
static_assert(sizeof(Edid) == 0x90, "attribute 0x986 is 0x90 bytes");

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
	size_t capacity = output && outputSize ? *outputSize : 0;
	if (!input)
		inputSize = 0;
	IOReturn r = answer(attribute, input, inputSize, capacity ? output : nullptr, capacity);
	if (OSIncrementAtomic(&loggedCalls) < kLoggedCalls) {
		// A display's attributes name it, and 0x711 its request, in the
		// input's first two words.
		UInt32 words[2] = {};
		if (inputSize >= sizeof words)
			memcpy(words, input, sizeof words);
		IOLog("RgfxDeviceControl: attribute 0x%x in %lu [0x%x 0x%x] out %lu: 0x%x\n", attribute,
		      static_cast<unsigned long>(inputSize), words[0], words[1],
		      static_cast<unsigned long>(capacity), r);
	}
	return r;
}

IOFramebuffer *RgfxDeviceControl::copyFramebuffer() const
{
	IOService *provider = getProvider();
	if (!provider)
		return nullptr;
	OSIterator *children = provider->getChildIterator(gIOServicePlane);
	if (!children)
		return nullptr;
	IOFramebuffer *found = nullptr;
	while (OSObject *child = children->getNextObject()) {
		found = OSDynamicCast(IOFramebuffer, child);
		if (found) {
			found->retain();
			break;
		}
	}
	children->release();
	return found;
}

// `output` is null for a call without one, else `outputSize` bytes.
IOReturn RgfxDeviceControl::answer(UInt32 attribute, const void *input, size_t inputSize,
                                   void *output, size_t outputSize)
{
	switch (attribute) {
	case kAttributeDescription: {
		if (!output || outputSize != sizeof(Description))
			return kIOReturnUnsupported;
		Description d = {};
		d.field_00 = kDescriptionVersion;
		strlcpy(d.field_04, kDescriptionName, sizeof d.field_04);
		d.field_24 = kDescriptionVendor;
		d.field_26 = kDescriptionDevice;
		d.field_28 = kDescriptionKind;
		memcpy(output, &d, sizeof d);
		return kIOReturnSuccess;
	}
	case kAttributeInit:
		if (output && outputSize == sizeof(UInt32)) {
			UInt32 mask = kInitFramebuffers | kInitAccelerator;
			memcpy(output, &mask, sizeof mask);
			return kIOReturnSuccess;
		}
		if (inputSize != sizeof(UInt32))
			return kIOReturnUnsupported;
		memcpy(&initFlags, input, sizeof initFlags);
		setProperty("RgfxInitFlags", initFlags, 32);
		return kIOReturnSuccess;
	case kAttribute921: {
		if (!output || outputSize != sizeof(Attribute921))
			return kIOReturnUnsupported;
		Attribute921 a = {};
		a.field_08 = 1;
		a.field_1c = 0x500;
		a.field_20 = 0x400;
		a.field_44 = 0x500;
		a.field_54 = 0x400;
		a.field_84 = 1;
		memcpy(output, &a, sizeof a);
		return kIOReturnSuccess;
	}
	case kAttributeDisplays: {
		if (!output || outputSize != sizeof(Displays))
			return kIOReturnUnsupported;
		Displays d = {};
		UInt64 ports = ((1ULL << kDisplayCount) - 1) << 1;
		for (auto &mask : d.field_00)
			mask = ports;
		for (auto &count : d.field_20)
			count = kDisplayCount;
		d.field_34 = reinterpret_cast<uintptr_t>(getProvider());
		// The pointer names the framebuffer; the reference is not kept.
		if (IOFramebuffer *fb = copyFramebuffer()) {
			d.field_3c[kDisplayIndex] = reinterpret_cast<uintptr_t>(fb);
			fb->release();
		}
		memcpy(output, &d, sizeof d);
		return kIOReturnSuccess;
	}
	case kAttributeConnection:
		return answerConnection(input, inputSize, output, outputSize);
	case kAttributeQuery:
		return answerQuery(input, inputSize, output, outputSize);
	case kAttributeEdid:
		return answerEdid(input, inputSize, output, outputSize);
	default:
		return kIOReturnUnsupported;
	}
}

// Success for a request naming the display, whatever the output: an output of
// at least 12 bytes gets the connection in its third word, and nothing else.
IOReturn RgfxDeviceControl::answerConnection(const void *input, size_t inputSize, void *output,
                                             size_t outputSize)
{
	Connection request;
	if (inputSize != sizeof request)
		return kIOReturnError;
	memcpy(&request, input, sizeof request);
	if (request.field_00 != kDisplayIndex + 1)
		return kIOReturnError;
	if (!output || outputSize < sizeof(Connection))
		return kIOReturnSuccess;
	uintptr_t connected = 0;
	if (IOFramebuffer *fb = copyFramebuffer()) {
		if (fb->getAttributeForConnection(0, kConnectionCheckEnable, &connected) != kIOReturnSuccess)
			connected = 0;
		fb->release();
	}
	UInt32 word = connected ? 1 : 0;
	memcpy(static_cast<UInt8 *>(output) + __builtin_offsetof(Connection, field_08), &word, sizeof word);
	return kIOReturnSuccess;
}

// An unknown request is unsupported; a known one for another display, or
// without the whole output, is an error. The answer's first two words stay as
// the caller left them; the rest is the answer.
IOReturn RgfxDeviceControl::answerQuery(const void *input, size_t inputSize, void *output,
                                        size_t outputSize)
{
	Query *reply = static_cast<Query *>(output);
	UInt32 head[2];
	if (inputSize != sizeof(Query))
		return kIOReturnError;
	memcpy(head, input, sizeof head);
	UInt32 display = head[0], request = head[1];
	if (!request || (request & (request - 1)) || !(request & kQueryRequests))
		return kIOReturnUnsupported;
	if (display != kDisplayIndex || !reply || outputSize != sizeof(Query))
		return kIOReturnError;

	// The mode's size first: the answer's bytes may be the request's.
	QueryModeSize size = {};
	if (request == kQueryMode) {
		IOFramebuffer *fb = copyFramebuffer();
		if (!fb)
			return kIOReturnError;
		IODisplayModeID mode = 0;
		IOIndex depth = 0;
		IODisplayModeInformation info = {};
		IOReturn r = fb->getCurrentDisplayMode(&mode, &depth);
		if (r == kIOReturnSuccess)
			r = fb->getInformationForDisplayMode(mode, &info);
		fb->release();
		if (r != kIOReturnSuccess || !info.nominalWidth || !info.nominalHeight)
			return kIOReturnError;
		size.field_00 = 1;
		size.field_08 = info.nominalHeight;
		size.field_0c = info.nominalWidth;
		size.field_10 = info.nominalHeight;
		size.field_14 = info.nominalWidth;
		size.field_18 = 0x10000;
	}

	reply->field_08 = kQueryEntries;
	bzero(reply->field_0c, sizeof reply->field_0c);
	switch (request) {
	case kQuery1: {
		Query1 f = {};
		f.field_00 = 5;
		f.field_10 = 8;
		f.field_20 = 0x30300;
		f.field_24 = kFloatMinusOne;
		f.field_28 = kFloatOne;
		memcpy(reply->field_0c, &f, sizeof f);
		break;
	}
	case kQueryGamma: {
		// The display's gamma, as the AMD's display answers it: plane 0 takes
		// tables of 256 entries.
		const RgfxGammaCapability g = rgfxGammaCapability();
		memcpy(reply->field_0c, &g, sizeof g);
		break;
	}
	case kQueryMode: {
		QueryMode m = {};
		m.field_1c[0] = size;
		m.field_1c[1] = size;
		memcpy(reply->field_0c, &m, sizeof m);
		break;
	}
	default:
		break;
	}
	return kIOReturnSuccess;
}

// The block, from the framebuffer's getDDCBlock (whose blocks count from 1),
// for a request naming the display and an output of at least 0x90 bytes.
IOReturn RgfxDeviceControl::answerEdid(const void *input, size_t inputSize, void *output,
                                       size_t outputSize)
{
	UInt32 head[3];
	if (inputSize != sizeof(Edid))
		return kIOReturnError;
	memcpy(head, input, sizeof head);
	if (head[0] != kDisplayIndex + 1 || !output || outputSize < sizeof(Edid) || head[2] >= 0xff)
		return kIOReturnError;
	IOFramebuffer *fb = copyFramebuffer();
	if (!fb)
		return kIOReturnError;
	UInt8 block[sizeof(Edid::field_0c)];
	IOByteCount length = sizeof block;
	IOReturn r = fb->getDDCBlock(0, head[2] + 1, kIODDCBlockTypeEDID, 0, block, &length);
	fb->release();
	if (r != kIOReturnSuccess || length != sizeof block)
		return kIOReturnError;
	memcpy(static_cast<UInt8 *>(output) + __builtin_offsetof(Edid, field_0c), block, sizeof block);
	return kIOReturnSuccess;
}
