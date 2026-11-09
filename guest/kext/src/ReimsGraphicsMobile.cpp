/*
 * ReimsGraphics' arm64e front end: the IOMobileFramebufferLegacy subclass an arm64 WindowServer
 * drives (ReimsGraphics.hpp). The device, its rings, its display and its pipe clients are
 * ReimsGraphics.cpp's, as on x86-64; this file is what the class asks of a subclass.
 *
 * Each override logs its calls (the first ones and then one in kLogEvery). The class queues the
 * swaps a process submits, and swap_begin_gated takes them off its queue as Apple's paravirtual
 * display does: the device presents a swap's layer-0 surface, and the swap completes once the host
 * answers the present. The device's refreshes reach the class as its vblank note. The class's own
 * pure methods answer what a display with no frame info, no factory configuration and no CRC
 * answers.
 *
 * None of the class's methods a subclass overrides is forwarded to the class: what the class
 * implements there is for no display at all, on every arm64 release from 13 to 27.0.1 (read from
 * each release's IOMobileGraphicsFamily). Its start_hardware answers false, which fails its start
 * ("start_hardware failed"); its stop_hardware_gated and swap_begin_gated do nothing; its
 * verify_swap answers false, which fails every swap ("verify_swap failed for swap id"); and its
 * swap_layer_map, surface_layer_map and surface_layer_unmap map a layer's surface for the display
 * hardware's DMA, through the mapper an AppleARMIODevice provider names. Apple's paravirtual
 * display, the class's subclass in a VM (AppleParavirtDisplay), overrides each of the seven: its
 * start_hardware starts its display and joins power management, its verify_swap answers true, and
 * its three maps map nothing. ReimsGraphics does the same: the device reads a surface by its
 * IOSurface ID, so no layer's surface is mapped. Its swap_begin_gated begins each swap the class
 * queued, in the class's own fields at each release's offsets (RgfxLegacySwapLayout.hpp), and
 * the class's own methods, which the paravirtual display calls, clean the queue, drop a swap and
 * complete one.
 *
 * Of what the class implements for every subclass, ReimsGraphics overrides what reads the
 * class's provider as an AppleARMIODevice unchecked where a process reaches it: get_dot_pitch.
 *
 * A process finds the display as IOMobileFramebuffer.framework counts displays: by a node of the
 * device tree (attachMobileNode), then the framebuffer's service and the properties it publishes
 * (publishMobileDisplay).
 */

#include "ReimsGraphics.hpp"

#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOLib.h>
#include <libkern/libkern.h>

#include "RgfxDisplayIdentity.hpp"

#if defined(__arm64e__)

#define super IOMobileFramebufferLegacy

// An IOReturn as a word: its 32 bits, which a caller of a method returning IOReturn reads.
static constexpr RgfxIOMFBWord word(IOReturn r)
{
	return static_cast<uint32_t>(r);
}

// IOSurface's global ID (IOSurface.kext's exported member, as RgfxIOSurfaceKPI.hpp declares the
// others): the ID PRESENT_SURFACE names a surface by.
extern "C" uint32_t rgfxIOSurfaceSurfaceID(const OSObject *surface)
    __asm__("__ZNK9IOSurface12getSurfaceIDEv");

// IOMobileFramebufferLegacy's own methods for its swap queue, which the paravirtual display's
// swap_begin_gated and clean_queue_gated call and ReimsGraphics overrides none of: called directly,
// as the SDK stand-in's note_vbl_ISR is, under the names every release from 14 to 27.0.1 exports.
// Each runs in the class's gate. dropAbortedSwaps_gated takes a bool from macOS 15 on and none
// on 14; Graft binds this import to either (graft-arm64e.json), and 14's ignores the bool.
extern "C" void rgfxLegacySwapComplete(IOMobileFramebufferLegacy *self)
    __asm__("__ZN25IOMobileFramebufferLegacy19swap_complete_gatedEv");
extern "C" void rgfxLegacyDropSwap(IOMobileFramebufferLegacy *self, void *swap, bool field_1)
    __asm__("__ZN25IOMobileFramebufferLegacy14dropSwap_gatedEP18IOMFBSwapIORequestb");
extern "C" void rgfxLegacyUpdateSwapInfoState(IOMobileFramebufferLegacy *self, void *swap,
                                              uint32_t state)
    __asm__("__ZN25IOMobileFramebufferLegacy28update_swap_info_state_gatedEP18IOMFBSwapIORequest24IOMFBSwapCompletionState");
extern "C" uint32_t rgfxLegacyDropObsoleteSwaps(IOMobileFramebufferLegacy *self)
    __asm__("__ZN25IOMobileFramebufferLegacy23dropObsoleteSwaps_gatedEv");
extern "C" void rgfxLegacyDropNoOpSwaps(IOMobileFramebufferLegacy *self)
    __asm__("__ZN25IOMobileFramebufferLegacy19dropNoOpSwaps_gatedEv");
extern "C" void rgfxLegacyDropAbortedSwaps(IOMobileFramebufferLegacy *self, bool field_1)
    __asm__("__ZN25IOMobileFramebufferLegacy22dropAbortedSwaps_gatedEb");
extern "C" void rgfxLegacyDropDummySwaps(IOMobileFramebufferLegacy *self)
    __asm__("__ZN25IOMobileFramebufferLegacy20dropDummySwaps_gatedEv");

// The IOMFBSwapCompletionState the paravirtual display gives a swap as it begins it.
constexpr uint32_t kSwapStateAtBegin = 2;

namespace {

// The methods whose calls are logged, each counted apart.
enum MobileCall : unsigned {
	kSetNotifyTagMask,
	kSupportedFrameInfo,
	kFactoryPortal,
	kExecuteFactoryConfigurations,
	kSetFrameInfoNotificationState,
	kDoSwapInfoNotify,
	kNotifySwapDone,
	kNotifyCrcReady,
	kAreTimingsEnabled,
	kStartHardware,
	kStopHardware,
	kSwapBegin,
	kVerifySwap,
	kSwapLayerMap,
	kSurfaceLayerMap,
	kSurfaceLayerUnmap,
	kGetDotPitch,
	kVblank,
	kPresented,
	kSwapShown,
	kSwapDropped,
	kSwapCursor,
	kMobileCalls,
};

constexpr UInt32 kLogFirst = 16;
constexpr UInt32 kLogEvery = 1024;
UInt32 callCounts[kMobileCalls];

// Whether this call of `call` is one to log: the first kLogFirst, then one in kLogEvery.
bool logged(MobileCall call)
{
	const UInt32 n = __atomic_fetch_add(&callCounts[call], 1, __ATOMIC_RELAXED);
	return n < kLogFirst || n % kLogEvery == 0;
}

} // namespace

bool ReimsGraphics::startMobile()
{
	mobilePresentSeqno = 0;
	mobileTransaction = 0;
	mobileSwapShowing = nullptr;
	// The release, by the size of the class this one derives from as the kernel has it.
	const uint32_t legacySize = IOMobileFramebufferLegacy::gMetaClass.getClassSize();
	mobileSwapLayout = rgfxLegacySwapLayoutOf(legacySize);
	if (mobileSwapLayout)
		IOLog("ReimsGraphics: the swap queue as macOS %s lays it out\n", mobileSwapLayout->release);
	else
		IOLog("ReimsGraphics: IOMobileFramebufferLegacy is 0x%x bytes, no release's size, so no "
		      "swap is begun\n",
		      legacySize);
	mobilePipe = addPipe(this, mobileVblank, mobilePresented);
	if (!mobilePipe) {
		IOLog("ReimsGraphics: no pipe for the mobile framebuffer\n");
		return false;
	}
	mobilePowerJoined = false;
	IOLog("ReimsGraphics: mobile framebuffer, display %u\n", displayId);
	publishMobileDisplay();
	attachMobileNode();
	return true;
}

void ReimsGraphics::stopMobile()
{
	detachMobileNode();
	if (mobilePipe)
		removePipe(this);
	mobilePipe = false;
	// The answer's event source leaves the class's work loop; free lets it go, once no pipe hook
	// can still signal it.
	if (IOInterruptEventSource *answer = mobileSwapAnswer) {
		answer->disable();
		if (mobileSwapLoop)
			mobileSwapLoop->removeEventSource(answer);
	}
}

// MARK: - The display as a process finds it

namespace {

// A dictionary's entries, each left out where it cannot be made.
void putNumber(OSDictionary *dict, const char *key, uint32_t value)
{
	if (OSNumber *n = OSNumber::withNumber(value, 32)) {
		dict->setObject(key, n);
		n->release();
	}
}

void putString(OSDictionary *dict, const char *key, const char *value)
{
	if (OSString *s = OSString::withCString(value)) {
		dict->setObject(key, s);
		s->release();
	}
}

// Puts `value` (which may be null) and lets it go.
void putDictionary(OSDictionary *dict, const char *key, OSDictionary *value)
{
	if (value) {
		dict->setObject(key, value);
		value->release();
	}
}

// A colour's point as DisplayAttributes gives it: X and Y in 16.16 fixed point, and a Gamma of 1.0
// in 16.16, as Apple's paravirtual display gives its own (read from macOS 27.0.1's
// AppleParavirtGPU, createColorPointDict).
OSDictionary *colorPoint(RgfxDisplayChromaticity c)
{
	OSDictionary *dict = OSDictionary::withCapacity(3);
	if (!dict)
		return nullptr;
	putNumber(dict, "X", (c.x * 65536 + 500) / 1000);
	putNumber(dict, "Y", (c.y * 65536 + 500) / 1000);
	putNumber(dict, "Gamma", 0x10000);
	return dict;
}

// What the device tree node's device_type holds: IOMobileFramebuffer.framework counts a child of
// /arm-io whose device_type begins with these 17 bytes as one display.
constexpr char kDisplayNodeType[] = "display-subsystem";

} // namespace

// The display's node in the device tree. A process waits for as many IOMobileFramebuffer services
// as IOMobileFramebuffer.framework counts displays (iomfb_populate_all_display_infos), and it
// counts the children of IODeviceTree:/arm-io whose device_type is "display-subsystem" or
// "ext-display-subsystem", and the DisplayPortCount of an AppleParavirtGPU where a child's
// compatible names Apple's paravirtual GPU (read from macOS 14.6.1's and 27.0.1's framework). A
// VM's device tree has no such child for this device: the framework counted no display, and
// WindowServer ran headless ("FB Count = 0").
//
// The framework counts once in a process (dispatch_once), and only a count above 0 starts its wait,
// up to 30 seconds, for the IOMobileFramebuffer services. The kexts load from the auxiliary kernel
// collection, after userspace starts: WindowServer counts before start runs, so a node attached
// here is counted only by the processes that start after it. A node the firmware placed at the same
// path, before the kernel started, is counted by every process: start uses it as it stands, and
// attaches a node of its own only where the firmware placed none. Nothing in the kernel reads the
// node: no driver matches its name or its device_type.
void ReimsGraphics::attachMobileNode()
{
	mobileNode = nullptr;
	mobileNodeAttached = false;
	char name[32];
	snprintf(name, sizeof(name), "reims-display%u", displayId);
	char path[48];
	snprintf(path, sizeof(path), "/arm-io/%s", name);
	if (IORegistryEntry *placed = IORegistryEntry::fromPath(path, gIODTPlane)) {
		mobileNode = placed;
		IOLog("ReimsGraphics: display %u has the firmware's IODeviceTree:%s\n", displayId, path);
		return;
	}
	IORegistryEntry *armIO = IORegistryEntry::fromPath("/arm-io", gIODTPlane);
	if (!armIO) {
		IOLog("ReimsGraphics: no /arm-io in the device tree, so no process counts display %u\n",
		      displayId);
		return;
	}
	OSDictionary *props = OSDictionary::withCapacity(2);
	OSData *nameData = OSData::withBytes(name, static_cast<unsigned>(strlen(name) + 1));
	OSData *typeData = OSData::withBytes(kDisplayNodeType, sizeof(kDisplayNodeType));
	IORegistryEntry *node = OSTypeAlloc(IORegistryEntry);
	bool attached = false;
	if (props && nameData && typeData && node) {
		props->setObject("name", nameData);
		props->setObject("device_type", typeData);
		if (node->init(props)) {
			node->setName(name);
			attached = node->attachToParent(armIO, gIODTPlane);
		}
	}
	if (attached) {
		mobileNode = node;
		mobileNodeAttached = true;
		node = nullptr;
	}
	OSSafeReleaseNULL(node);
	OSSafeReleaseNULL(typeData);
	OSSafeReleaseNULL(nameData);
	OSSafeReleaseNULL(props);
	armIO->release();
	IOLog("ReimsGraphics: display %u %s IODeviceTree:/arm-io/%s, a %s\n", displayId,
	      attached ? "attached" : "could not attach", name, kDisplayNodeType);
}

void ReimsGraphics::detachMobileNode()
{
	if (!mobileNode)
		return;
	if (mobileNodeAttached)
		mobileNode->detachAbove(gIODTPlane);
	mobileNodeAttached = false;
	OSSafeReleaseNULL(mobileNode);
}

// The framebuffer's properties a process reads of the display, as Apple's paravirtual display
// publishes its own (read from macOS 27.0.1's AppleParavirtGPU, AppleParavirtDisplay's
// createDisplayAttributes), with the values the x86-64 framebuffer's EDID gives
// (RgfxDisplayIdentity.hpp):
// - external: false for the device's first display, which IOMobileFramebuffer.framework then
//   names primary-0, and true for the others, which it names external-N;
// - IOMFBUUID: the display's UUID, a fixed one of ours: "RVG" in ASCII, the product code, the
//   version and variant bits of RFC 9562's custom UUIDs (8), and the serial;
// - DisplayAttributes: the product (manufacturer, product code, serial, name, year), sRGB's
//   primaries and D65 white, sRGB the default colour space, no PQ transfer function, and the
//   native mode's size in pixels, in whole centimetres and as an aspect ratio in 16.16;
// - Transport: "Other" upstream and downstream, as no link carries the pixels;
// - IOMFBMaxSrcPixels: the largest source buffer and source rectangle a layer takes, the largest
//   mode's size; and its MaxSrcRectTotal and IOMFBMaxCompressedSizeInBytes 0, as the paravirtual
//   display gives them.
// The paravirtual display's TimingElements and ColorElements, IOAVFamily's objects, are not
// published.
void ReimsGraphics::publishMobileDisplay()
{
	IOLockLock(stateLock);
	const Mode native = nativeMode(Mode{});
	const UInt32 id = displayId;
	IOLockUnlock(stateLock);
	const uint32_t serial = rgfxDisplaySerial(id);

	setProperty("external", id != 0);
	char uuid[40];
	snprintf(uuid, sizeof(uuid), "52564700-%04X-8000-8000-%012X", kRgfxDisplayProduct, serial);
	setProperty("IOMFBUUID", uuid);

	OSDictionary *attributes = OSDictionary::withCapacity(12);
	if (attributes) {
		OSDictionary *product = OSDictionary::withCapacity(6);
		if (product) {
			putString(product, "ManufacturerID", kRgfxDisplayManufacturer);
			putNumber(product, "ProductID", kRgfxDisplayProduct);
			putNumber(product, "SerialNumber", serial);
			putString(product, "ProductName", kRgfxDisplayName);
			putNumber(product, "WeekOfManufacture", 0);
			putNumber(product, "YearOfManufacture", kRgfxDisplayYear);
		}
		putDictionary(attributes, "ProductAttributes", product);
		OSDictionary *chromaticity = OSDictionary::withCapacity(3);
		if (chromaticity) {
			putDictionary(chromaticity, "Red", colorPoint(kRgfxDisplayRed));
			putDictionary(chromaticity, "Green", colorPoint(kRgfxDisplayGreen));
			putDictionary(chromaticity, "Blue", colorPoint(kRgfxDisplayBlue));
		}
		putDictionary(attributes, "Chromaticity", chromaticity);
		putDictionary(attributes, "DefaultWhitePoint", colorPoint(kRgfxDisplayWhite));
		attributes->setObject("DefaultColorSpaceIsSRGB", kOSBooleanTrue);
		attributes->setObject("SupportsPQEOTF", kOSBooleanFalse);
		if (native.width && native.height) {
			putNumber(attributes, "MaxHorizontalImageSize",
			          (rgfxDisplayMillimetres(native.width, native.width) + 9) / 10);
			putNumber(attributes, "MaxVerticalImageSize",
			          (rgfxDisplayMillimetres(native.height, native.width) + 9) / 10);
			putNumber(attributes, "NativeFormatHorizontalPixels", native.width);
			putNumber(attributes, "NativeFormatVerticalPixels", native.height);
			putNumber(attributes, "PreciseAspectRatio",
			          static_cast<uint32_t>((static_cast<uint64_t>(native.width) << 16) /
			                                native.height));
		}
		setProperty("DisplayAttributes", attributes);
		attributes->release();
	}

	if (OSDictionary *transport = OSDictionary::withCapacity(2)) {
		putString(transport, "Upstream", "Other");
		putString(transport, "Downstream", "Other");
		setProperty("Transport", transport);
		transport->release();
	}

	if (OSDictionary *source = OSDictionary::withCapacity(5)) {
		putNumber(source, "MaxSrcBufferWidth", kRgfxDisplayMaxDimension);
		putNumber(source, "MaxSrcBufferHeight", kRgfxDisplayMaxDimension);
		putNumber(source, "MaxSrcRectWidth", kRgfxDisplayMaxDimension);
		putNumber(source, "MaxSrcRectTotal", 0);
		putNumber(source, "IOMFBMaxCompressedSizeInBytes", 0);
		setProperty("IOMFBMaxSrcPixels", source);
		source->release();
	}
	IOLog("ReimsGraphics: display %u is %s, %ux%u native, UUID %s\n", id,
	      id != 0 ? "external" : "primary", native.width, native.height, uuid);
}

void ReimsGraphics::mobileVblank(OSObject *client, UInt64 count, UInt64 time, UInt64 next)
{
	ReimsGraphics *self = static_cast<ReimsGraphics *>(client);
	if (logged(kVblank))
		IOLog("ReimsGraphics: vblank %llu at %llu, next %llu\n", count, time, next);
	self->note_vbl_ISR(static_cast<uint32_t>(count));
}

void ReimsGraphics::mobilePresented(OSObject *client, UInt64 seqno, UInt32 status, UInt64 time)
{
	ReimsGraphics *self = static_cast<ReimsGraphics *>(client);
	if (seqno != __atomic_load_n(&self->mobilePresentSeqno, __ATOMIC_ACQUIRE))
		return;
	if (logged(kPresented))
		IOLog("ReimsGraphics: present %llu shown, status %u at %llu\n", seqno, status, time);
	// The swap it shows completes in the class's gate (completeMobileSwap).
	if (IOInterruptEventSource *answer = __atomic_load_n(&self->mobileSwapAnswer, __ATOMIC_ACQUIRE))
		answer->interruptOccurred(nullptr, nullptr, 0);
}

// MARK: - The swaps

void ReimsGraphics::beginMobileSwaps()
{
	if (!mobileSwapLayout)
		return;
	RgfxLegacySwapQueue queue(this, *mobileSwapLayout);
	for (;;) {
		// The queue cleaned as the paravirtual display's clean_queue_gated cleans it, once no swap
		// is shown and one waits: none begins where the class keeps an obsolete swap's turn.
		if (queue.current() || !queue.first())
			return;
		if (rgfxLegacyDropObsoleteSwaps(this) != 0)
			return;
		rgfxLegacyDropNoOpSwaps(this);
		rgfxLegacyDropAbortedSwaps(this, false);
		rgfxLegacyDropDummySwaps(this);
		void *swap = queue.first();
		if (!swap)
			return;
		if (!queue.firstLinked()) {
			// Where the paravirtual display stops the kernel: the queue is left as it is.
			IOLog("ReimsGraphics: swap %p is not linked as the queue's first; no swap is begun\n",
			      swap);
			return;
		}
		if (!queue.begins(swap)) {
			// The class's dropSwap_gated takes the swap off the queue and completes it.
			if (logged(kSwapDropped))
				IOLog("ReimsGraphics: swap %p dropped at its state\n", swap);
			rgfxLegacyDropSwap(this, swap, false);
			continue;
		}
		queue.setCurrent(swap);
		rgfxLegacyUpdateSwapInfoState(this, swap, kSwapStateAtBegin);
		queue.removeFirst();
		if (queue.surface(swap, 1) && logged(kSwapCursor))
			IOLog("ReimsGraphics: swap %p has a layer-1 surface, which is not shown\n", swap);
		// Layer 0's surface, by its IOSurface ID. The swap completes when the host answers its
		// present (mobilePresented), or here where nothing is presented or no answer can reach
		// the gate. The present's seqno is written before the host can answer it.
		OSObject *surface = static_cast<OSObject *>(queue.surface(swap, 0));
		const uint32_t id = surface ? rgfxIOSurfaceSurfaceID(surface) : 0;
		IOReturn presented = kIOReturnNotReady;
		if (id) {
			const UInt32 transaction = ++mobileTransaction;
			mobileSwapShowing = mobileSwapAnswer ? swap : nullptr;
			presented = postPresent(id, transaction, 0, &mobilePresentSeqno);
			if (presented != kIOReturnSuccess)
				mobileSwapShowing = nullptr;
		}
		if (logged(kSwapShown))
			IOLog("ReimsGraphics: swap %p begun: surface %u, present 0x%x seqno %llu\n", swap, id,
			      presented, __atomic_load_n(&mobilePresentSeqno, __ATOMIC_RELAXED));
		if (mobileSwapShowing)
			return;
		rgfxLegacySwapComplete(this);
	}
}

void ReimsGraphics::completeMobileSwap()
{
	void *swap = mobileSwapShowing;
	if (!swap || !mobileSwapLayout)
		return;
	mobileSwapShowing = nullptr;
	// The class may have finished the swap itself meanwhile (an abort or a stop).
	RgfxLegacySwapQueue queue(this, *mobileSwapLayout);
	if (queue.current() != swap)
		return;
	rgfxLegacySwapComplete(this);
	beginMobileSwaps();
}

void ReimsGraphics::mobileSwapAnswered(OSObject *owner, IOInterruptEventSource *, int)
{
	static_cast<ReimsGraphics *>(owner)->completeMobileSwap();
}

// MARK: - The class's own pure methods

RgfxIOMFBWord ReimsGraphics::set_notify_tag_mask_gated(IOMFB_NotificationType type,
                                                       IOMFBNotifyRequest *request, uint32_t mask)
{
	if (logged(kSetNotifyTagMask))
		IOLog("ReimsGraphics: set_notify_tag_mask_gated(%u, %p, 0x%x)\n",
		      static_cast<uint32_t>(type), request, mask);
	return word(kIOReturnSuccess);
}

RgfxIOMFBWord ReimsGraphics::supported_frame_info(uint32_t index, uint32_t *count, char (*names)[64],
                                                  uint32_t *field_3)
{
	if (logged(kSupportedFrameInfo))
		IOLog("ReimsGraphics: supported_frame_info(%u, %p, %p, %p)\n", index, count, names,
		      field_3);
	return word(kIOReturnUnsupported);
}

RgfxIOMFBWord ReimsGraphics::factory_portal(void *field_0, task *owner)
{
	if (logged(kFactoryPortal))
		IOLog("ReimsGraphics: factory_portal(%p, %p)\n", field_0, owner);
	return word(kIOReturnUnsupported);
}

RgfxIOMFBWord ReimsGraphics::executeFactoryConfigurations_gated()
{
	if (logged(kExecuteFactoryConfigurations))
		IOLog("ReimsGraphics: executeFactoryConfigurations_gated()\n");
	return word(kIOReturnSuccess);
}

RgfxIOMFBWord ReimsGraphics::set_frame_info_notification_state_gated(IOMFBNotifyRequest *request,
                                                                     bool on)
{
	if (logged(kSetFrameInfoNotificationState))
		IOLog("ReimsGraphics: set_frame_info_notification_state_gated(%p, %d)\n", request, on);
	return word(kIOReturnSuccess);
}

RgfxIOMFBWord ReimsGraphics::do_swap_info_notify_gated(IOMFBNotifyRequest *request,
                                                       const IOMobileFramebufferLegacy::SwapInfoBlob *blob)
{
	if (logged(kDoSwapInfoNotify))
		IOLog("ReimsGraphics: do_swap_info_notify_gated(%p, %p)\n", request, blob);
	return word(kIOReturnSuccess);
}

RgfxIOMFBWord ReimsGraphics::notify_swap_done_gated(uint32_t field_0, ::SwapInfoBlob *blob,
                                                    uint32_t *field_2, uint32_t field_3)
{
	if (logged(kNotifySwapDone))
		IOLog("ReimsGraphics: notify_swap_done_gated(%u, %p, %p, %u)\n", field_0, blob, field_2,
		      field_3);
	return word(kIOReturnSuccess);
}

RgfxIOMFBWord ReimsGraphics::notify_crc_ready_gated(uint32_t field_0)
{
	if (logged(kNotifyCrcReady))
		IOLog("ReimsGraphics: notify_crc_ready_gated(%u)\n", field_0);
	return word(kIOReturnSuccess);
}

bool ReimsGraphics::are_timings_enabled() const
{
	if (logged(kAreTimingsEnabled))
		IOLog("ReimsGraphics: are_timings_enabled()\n");
	return true;
}

// MARK: - What a subclass implements for its display

namespace {

// The power states the display registers, as Apple's paravirtual display registers its own: off,
// and on and usable.
IOPMPowerState mobilePowerStates[] = {
	{kIOPMPowerStateVersion1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{kIOPMPowerStateVersion1, kIOPMDeviceUsable, kIOPMPowerOn, kIOPMPowerOn, 0, 0, 0, 0, 0, 0, 0, 0},
};

} // namespace

RgfxIOMFBWord ReimsGraphics::start_hardware()
{
	// The class's start calls this last, and fails unless it answers true. The device runs
	// already (start). Power management's state reaches the class through its setPowerState,
	// which records it: the class drops every swap submitted while the display is off. The
	// vblanks start here, once the class has its work loop and swap queue.
	PMinit();
	if (IOService *provider = getProvider())
		provider->joinPMtree(this);
	const IOReturn power = registerPowerDriver(this, mobilePowerStates,
	                                           sizeof(mobilePowerStates) / sizeof(mobilePowerStates[0]));
	mobilePowerJoined = true;
	// The present's answer reaches the class's gate through an event source on its work loop.
	if (!mobileSwapAnswer && mobileSwapLayout) {
		IOWorkLoop *loop = getWorkLoop();
		IOInterruptEventSource *answer =
		    loop ? IOInterruptEventSource::interruptEventSource(this, mobileSwapAnswered) : nullptr;
		if (answer && loop->addEventSource(answer) == kIOReturnSuccess) {
			loop->retain();
			mobileSwapLoop = loop;
			answer->enable();
			__atomic_store_n(&mobileSwapAnswer, answer, __ATOMIC_RELEASE);
		} else {
			IOLog("ReimsGraphics: no event source on the framebuffer's work loop; no swap is "
			      "begun\n");
			OSSafeReleaseNULL(answer);
		}
	}
	if (mobilePipe)
		setPipeVblankEnabled(this, true);
	if (logged(kStartHardware))
		IOLog("ReimsGraphics: start_hardware(): power driver 0x%x, vblanks on\n", power);
	return 1;
}

RgfxIOMFBWord ReimsGraphics::stop_hardware_gated()
{
	if (mobilePipe)
		setPipeVblankEnabled(this, false);
	if (mobilePowerJoined)
		PMstop();
	mobilePowerJoined = false;
	if (logged(kStopHardware))
		IOLog("ReimsGraphics: stop_hardware_gated()\n");
	return 0;
}

RgfxIOMFBWord ReimsGraphics::swap_begin_gated()
{
	if (logged(kSwapBegin))
		IOLog("ReimsGraphics: swap_begin_gated()\n");
	beginMobileSwaps();
	return 0;
}

RgfxIOMFBWord ReimsGraphics::verify_swap(IOMFBSwapIORequest *request)
{
	if (logged(kVerifySwap))
		IOLog("ReimsGraphics: verify_swap(%p) = 1\n", request);
	return 1;
}

RgfxIOMFBWord ReimsGraphics::swap_layer_map(IOMFBSwapIORequest *request, uint32_t layer)
{
	if (logged(kSwapLayerMap))
		IOLog("ReimsGraphics: swap_layer_map(%p, %u)\n", request, layer);
	return word(kIOReturnSuccess);
}

RgfxIOMFBWord ReimsGraphics::surface_layer_map(IOMFB::AP::DisplayedData *data, uint32_t layer,
                                               IOSurface *surface)
{
	// Nothing to map: the swap's begin presents its surface by ID.
	const RgfxIOMFBWord r = word(kIOReturnSuccess);
	if (logged(kSurfaceLayerMap))
		IOLog("ReimsGraphics: surface_layer_map(%p, %u, %p: surface %u) = 0x%llx\n", data, layer,
		      surface, surface ? rgfxIOSurfaceSurfaceID(reinterpret_cast<const OSObject *>(surface)) : 0,
		      r);
	return r;
}

RgfxIOMFBWord ReimsGraphics::surface_layer_unmap(IOMFB::AP::DisplayedData *data, uint32_t layer)
{
	if (logged(kSurfaceLayerUnmap))
		IOLog("ReimsGraphics: surface_layer_unmap(%p, %u)\n", data, layer);
	return 0;
}

// IOMobileFramebufferLegacy's start keeps its provider as an AppleARMIODevice, NULL where the
// provider is not one, as a PCI device is not, and leaves nothing else unset. Of the class's
// functions that read it, get_dot_pitch alone calls through it unchecked: it reads the provider's
// "dot-pitch" property, and IOMobileFramebufferUserClient's s_get_dot_pitch reaches it from any
// process, so the class's own faulted (macOS 26.6.2 and 27.0.1, from photolibraryd). The others
// do not fault: the thread find_swap_function_gated starts hands the provider to
// AppleARMFunction::withProvider, which answers none for a NULL provider, and allocate_carveout
// reads it only under a flag the family never sets (read from the IOMobileGraphicsFamily of
// macOS 14.6.1, 15.6.1, 26.6.2 and 27.0.1; tests/legacy_overrides_test.cpp). The display's dot
// pitch is its pixels per inch, as the paravirtual display answers from its own size.
RgfxIOMFBWord ReimsGraphics::get_dot_pitch(uint32_t *pitch)
{
	if (!pitch)
		return word(kIOReturnBadArgument);
	IOLockLock(stateLock);
	const Mode native = nativeMode(Mode{});
	IOLockUnlock(stateLock);
	*pitch = rgfxDisplayPixelsPerInch(native.width);
	if (logged(kGetDotPitch))
		IOLog("ReimsGraphics: get_dot_pitch() = %u\n", *pitch);
	return word(kIOReturnSuccess);
}

#endif // __arm64e__
