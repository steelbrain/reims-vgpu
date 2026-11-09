/*
 * ReimsGraphics' arm64e front end: the IOMobileFramebufferLegacy subclass an arm64 WindowServer
 * drives (ReimsGraphics.hpp). The device, its rings, its display and its pipe clients are
 * ReimsGraphics.cpp's, as on x86-64; this file is what the class asks of a subclass.
 *
 * Each override logs its calls (the first ones and then one in kLogEvery). The class queues the
 * swaps a process submits, and swap_begin_gated takes them off its queue as Apple's paravirtual
 * display does: the device presents a swap's layer-0 surface, and the swap completes once the host
 * answers the present. The device's refreshes reach the class as its vblank note. Each completed
 * swap's presentation time reaches WindowServer as frame info, which the paravirtual display has
 * none of: supported_frame_info answers the tags the device fills, and a record of the swap goes to
 * each notify request's client, through its shared data queue (RgfxLegacySwapInfo.hpp). The class's other pure methods answer what a display with no factory
 * configuration and no CRC answers.
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
 * class's provider as an AppleARMIODevice unchecked where a process reaches it: get_dot_pitch. And
 * as the paravirtual display does, it overrides set_digital_out_mode, which the class answers
 * kIOReturnUnsupported, to set the mode WindowServer picks of those it publishes. Unlike the
 * paravirtual display, which publishes three colour modes, it publishes one, and the device lists
 * one mode, so WindowServer picks none: the display starts in its preferred mode (start_hardware).
 * And it overrides swap_set_color_matrix, which the class answers kIOReturnUnsupported, so
 * WindowServer, which sets its accessibility matrix at each swap until the call succeeds, sets it
 * once: as the paravirtual display does, it answers 0 and the display shows the matrix
 * (RgfxMobileColor.hpp).
 *
 * A process finds the display as IOMobileFramebuffer.framework counts displays: by a node of the
 * device tree (attachMobileNode), then the framebuffer's service and the properties it publishes
 * (publishMobileDisplay).
 */

#include "ReimsGraphics.hpp"

#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOLib.h>
#include <kern/clock.h>
#include <libkern/libkern.h>

#include "RgfxDisplayIdentity.hpp"
#include "RgfxMobileColor.hpp"
#include "RgfxMobileModes.hpp"

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
// The class's own dispatch of a swap-info record to each notify request it holds, through the
// display's do_swap_info_notify_gated (RgfxLegacySwapInfo.hpp), which the paravirtual display's
// notify_swap_done_gated calls: called directly, under the name every release from 14 to 27.0.1
// exports, in the class's gate.
extern "C" void rgfxLegacySwapInfoNotifyDispatch(IOMobileFramebufferLegacy *self,
                                                 RgfxSwapInfoRecord *record)
    __asm__("__ZN25IOMobileFramebufferLegacy31swap_info_notify_dispatch_gatedEPNS_12SwapInfoBlobE");

// IOMobileFramebufferUserClient's enqueue of `bytes` bytes at `words` in its shared data queue,
// which IOMobileFramebuffer.framework reads a frame info record from (RgfxLegacySwapInfo.hpp):
// called directly, under the name every release from 14 to 27.0.1 exports, on the user client a
// notify request is for. Answers kIOReturnSuccess, or an error where the queue holds no room.
extern "C" IOReturn rgfxUserClientEnqueueFrameInfo(void *client, uint64_t *words, uint32_t bytes)
    __asm__("__ZN29IOMobileFramebufferUserClient26enqueueInIOSharedDataQueueEPyj");

// IOAVFamily's constructors of a timing element and of a colour element, and a timing's colour
// elements, which the paravirtual display calls to make the elements it publishes
// (RgfxMobileModes.hpp): called directly, under the names every release from 14 to 27.0.1 exports.
// IOAVFamily is a library of IOMobileGraphicsFamily's on each of them, so it is loaded before the
// kext, and Graft binds each (graft-arm64e.json). Each constructor answers a retained element, or
// null. setColorElements retains the array it is given.
extern "C" OSObject *rgfxIOAVTimingWithData(const RgfxIOAVTimingData *data, uint32_t standardType,
                                            uint32_t timingStandard)
    __asm__("__ZN22IOAVVideoTimingElement8withDataEPK19IOAVVideoTimingData16IOAVStandardType23IOAVVideoTimingStandard");
extern "C" OSObject *rgfxIOAVColorWithCapabilities(
    uint32_t standardType, uint32_t depth, uint32_t pixelEncoding, uint32_t dynamicRange,
    uint32_t eotf, uint32_t colorimetry, bool supportsDsc, const OSDictionary *chromaticity,
    const OSDictionary *whitePoint, const OSDictionary *displayLuminance,
    const OSSet *hdrStaticMetadataTypes)
    __asm__("__ZN21IOAVVideoColorElement16withCapabilitiesE16IOAVStandardTypej22IOAVVideoPixelEncoding26IOAVVideoColorDynamicRange18IOAVVideoColorEOTF20IOAVVideoColorimetrybPK12OSDictionaryS7_S7_PK5OSSet");
extern "C" void rgfxIOAVTimingSetColorElements(OSObject *timing, OSArray *colors)
    __asm__("__ZN22IOAVVideoTimingElement16setColorElementsEP7OSArray");

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
	kSetDigitalOutMode,
	kSwapSetColorMatrix,
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
	mobileSwapRate.clear();
	mobileSecond = 0;
	mobileShownAt = 0;
	mobilePresentedAt = 0;
	mobileSwapInfoKept.head = 0;
	mobileSwapInfoKept.tail = 0;
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
//   display gives them;
// - IOMFBScalingLimits: no layer scaled, up (RGBLayer_MaxScale 1) or down
//   (RGBLayer_MinScaleFraction 1), where the paravirtual display gives 4 each way: the device shows
//   layer 0's whole surface, whatever rectangles the swap gives it;
// - TimingElements and ColorElements: the device's modes (publishMobileModes).
// Of the rest the paravirtual display publishes, CursorPlane is not, since no swap's cursor layer
// is shown; nor are its own ParavirtDisplayPrefs and ParavirtBrightnessCaps, nor the
// DisplayContainerID its host may give it.
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

	if (OSDictionary *scaling = OSDictionary::withCapacity(2)) {
		putNumber(scaling, "RGBLayer_MinScaleFraction", 1);
		putNumber(scaling, "RGBLayer_MaxScale", 1);
		setProperty("IOMFBScalingLimits", scaling);
		scaling->release();
	}
	publishMobileModes();
	IOLog("ReimsGraphics: display %u is %s, %ux%u native, UUID %s\n", id,
	      id != 0 ? "external" : "primary", native.width, native.height, uuid);
}

namespace {

// Writes an IOAVFamily element's ID and Score (RgfxMobileModes.hpp).
void setElementId(OSObject *element, uint32_t id, uint32_t score)
{
	uint8_t *bytes = reinterpret_cast<uint8_t *>(element);
	*reinterpret_cast<uint64_t *>(bytes + kRgfxIOAVElementId) = id;
	*reinterpret_cast<uint32_t *>(bytes + kRgfxIOAVElementScore) = score;
}

} // namespace

// The display's modes as IOMobileFramebuffer.framework reads them, as the paravirtual display
// publishes its own (RgfxMobileModes.hpp): ColorElements, the one colour mode, and TimingElements,
// a timing for each mode the device lists, with the reduced blanking the x86-64 framebuffer's
// detailed timing gives it (timingFor) at the device's rate, and the colour mode as its ColorModes.
// WindowServer sets one of them with set_digital_out_mode where it has two to pick from.
void ReimsGraphics::publishMobileModes()
{
	Mode modes[kMaxModes];
	IOLockLock(stateLock);
	const UInt32 n = modeSet.modeCount;
	for (UInt32 i = 0; i < n; i++)
		modes[i] = modeSet.modes[i];
	const int32_t preferred = modeSet.preferredId;
	const UInt32 id = displayId;
	IOLockUnlock(stateLock);
	RgfxMobileTiming timings[kMaxModes];
	const uint32_t count = rgfxMobileTimings(modes, n, preferred, timings);

	OSObject *color =
	    rgfxIOAVColorWithCapabilities(kRgfxIOAVStandardType, kRgfxColorDepth, 0, 0, 0,
	                                  kRgfxIOAVColorimetry, false, nullptr, nullptr, nullptr, nullptr);
	OSArray *colors = color ? OSArray::withCapacity(1) : nullptr;
	OSArray *timingElements = colors ? OSArray::withCapacity(count) : nullptr;
	if (!timingElements) {
		IOLog("ReimsGraphics: display %u: no colour element, so no mode is published\n", id);
		OSSafeReleaseNULL(colors);
		OSSafeReleaseNULL(color);
		return;
	}
	setElementId(color, kRgfxColorElementId, kRgfxColorElementId);
	colors->setObject(color);
	uint32_t published = 0;
	for (uint32_t i = 0; i < count; i++) {
		const RgfxMobileTiming &t = timings[i];
		Timing timing;
		timingFor(t.mode, &timing);
		const RgfxIOAVTimingData data = rgfxIOAVTimingData(timing, t.mode.refreshMhz);
		OSObject *element = rgfxIOAVTimingWithData(&data, kRgfxIOAVStandardType,
		                                           kRgfxIOAVTimingStandard);
		// A timing's own array, as the paravirtual display gives each of its timings.
		OSArray *modeColors = element ? OSArray::withCapacity(1) : nullptr;
		if (!modeColors) {
			IOLog("ReimsGraphics: display %u: mode %ux%u has no timing element\n", id, t.mode.width,
			      t.mode.height);
			OSSafeReleaseNULL(element);
			continue;
		}
		setElementId(element, t.id, t.score);
		modeColors->setObject(color);
		rgfxIOAVTimingSetColorElements(element, modeColors);
		timingElements->setObject(element);
		published++;
		modeColors->release();
		element->release();
	}
	setProperty("ColorElements", colors);
	setProperty("TimingElements", timingElements);
	IOLog("ReimsGraphics: display %u publishes %u timings (preferred %u) and colour mode %u\n", id,
	      published, count ? timings[0].id : 0, kRgfxColorElementId);
	timingElements->release();
	colors->release();
	color->release();
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
	// The refresh it showed in, for the swap's frame info (completeMobileSwap).
	__atomic_store_n(&self->mobileShownAt, time, __ATOMIC_RELEASE);
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
			const UInt64 now = mach_absolute_time();
			noteMobileSwapRate(now);
			mobileSwapRate.drop(now);
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
		const UInt64 now = mach_absolute_time();
		noteMobileSwapRate(now);
		mobileSwapRate.begin(now, id, queue.surface(swap, 1) != nullptr,
		                     presented == kIOReturnSuccess);
		if (mobileSwapShowing)
			return;
		// Nothing was shown: its frame info says so (notify_swap_done_gated).
		mobilePresentedAt = 0;
		rgfxLegacySwapComplete(this);
		mobileSwapRate.complete(now);
	}
}

// The line of the second of swaps just over, once a swap comes after it: none while the display
// is idle.
void ReimsGraphics::noteMobileSwapRate(UInt64 now)
{
	if (!mobileSecond)
		nanoseconds_to_absolutetime(1000000000ull, &mobileSecond);
	const RgfxSwapRate &r = mobileSwapRate;
	if (!r.due(now, mobileSecond))
		return;
	uint64_t ns = 0;
	absolutetime_to_nanoseconds(now - r.start, &ns);
	IOLog("ReimsGraphics: swaps over %llu ms: %u begun, %u completed, %u dropped, %u not shown, "
	      "%u with layer 1; layer 0: %u x%u, %u x%u, %u x%u, %u x%u, others x%u\n",
	      ns / 1000000, r.begun, r.completed, r.dropped, r.unshown, r.layer1, r.ids[0], r.counts[0],
	      r.ids[1], r.counts[1], r.ids[2], r.counts[2], r.ids[3], r.counts[3], r.others);
	mobileSwapRate.clear();
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
	// The swap's presentation time, the refresh the host answered in, in mach_continuous_time's
	// clock, which its frame info carries (notify_swap_done_gated, which the class's
	// swap_complete_gated calls).
	const UInt64 shown = __atomic_exchange_n(&mobileShownAt, 0, __ATOMIC_ACQ_REL);
	uint64_t absolute = 0, continuous = 0;
	rgfxClocksTogether([] { return mach_absolute_time(); }, [] { return mach_continuous_time(); },
	                   &absolute, &continuous);
	mobilePresentedAt = rgfxFrameInfoTime(shown, absolute, continuous);
	rgfxLegacySwapComplete(this);
	const UInt64 now = mach_absolute_time();
	noteMobileSwapRate(now);
	mobileSwapRate.complete(now);
	beginMobileSwaps();
}

void ReimsGraphics::mobileSwapAnswered(OSObject *owner, IOInterruptEventSource *, int)
{
	static_cast<ReimsGraphics *>(owner)->completeMobileSwap();
}

// MARK: - The class's own pure methods

// A notify request's tag mask, for the swap-info notification alone, as the paravirtual display
// sets it; any other type is answered kIOReturnBadArgument, as the paravirtual display answers it
// (RgfxLegacySwapInfo.hpp).
RgfxIOMFBWord ReimsGraphics::set_notify_tag_mask_gated(IOMFB_NotificationType type,
                                                       IOMFBNotifyRequest *request, uint32_t mask)
{
	IOReturn r = kIOReturnBadArgument;
	if (static_cast<uint32_t>(type) == kRgfxSwapInfoNotification && request) {
		*reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t *>(request) +
		                              kRgfxNotifyRequestMask) = mask;
		r = kIOReturnSuccess;
	}
	if (logged(kSetNotifyTagMask))
		IOLog("ReimsGraphics: set_notify_tag_mask_gated(%u, %p, 0x%x) = 0x%x\n",
		      static_cast<uint32_t>(type), request, mask, r);
	return word(r);
}

// The frame info tags the device fills (RgfxLegacySwapInfo.hpp), one a call: the count, and tag
// `index`'s name in `names[0]`, which IOMobileFramebuffer.framework asks for each index below the
// count it is given at index 0. The records go through the client's shared data queue
// (`dataQueue` 1, which the user client keeps and makes the queue at its frame info port by), as
// the class adds frame info requests with no async reference (RgfxLegacySwapInfo.hpp).
RgfxIOMFBWord ReimsGraphics::supported_frame_info(uint32_t index, uint32_t *count, char (*names)[64],
                                                  uint32_t *dataQueue)
{
	IOReturn r = kIOReturnBadArgument;
	const char *tag = rgfxFrameInfoTag(index);
	if (count)
		*count = kRgfxFrameInfoTagCount;
	if (dataQueue)
		*dataQueue = 1;
	if (tag && names) {
		strlcpy(names[0], tag, sizeof(names[0]));
		r = kIOReturnSuccess;
	}
	if (logged(kSupportedFrameInfo))
		IOLog("ReimsGraphics: supported_frame_info(%u) = 0x%x: %s\n", index, r, tag ? tag : "none");
	return word(r);
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

// A swap's frame info record sent to one notify request's client: the values the request's tag
// mask asks for (rgfxFrameInfoRecord), enqueued in the shared data queue of the user client the
// request is for, where the client's library reads them (RgfxLegacySwapInfo.hpp), answering the
// enqueue's result: an error where the queue holds no room, or kIOReturnNotAttached where the
// request names no client. WindowServer learns its swaps' presentation times from them.
RgfxIOMFBWord ReimsGraphics::do_swap_info_notify_gated(IOMFBNotifyRequest *request,
                                                       const IOMobileFramebufferLegacy::SwapInfoBlob *blob)
{
	IOReturn r = kIOReturnBadArgument;
	if (request && blob) {
		uint8_t *bytes = reinterpret_cast<uint8_t *>(request);
		void *client = *reinterpret_cast<void *const *>(bytes + kRgfxNotifyRequestClient);
		uint64_t words[kRgfxSwapInfoMaxArgs + 1];
		const uint32_t n = rgfxFrameInfoRecord(
		    bytes, *reinterpret_cast<const RgfxSwapInfoRecord *>(blob), words);
		r = client ? rgfxUserClientEnqueueFrameInfo(client, words,
		                                            static_cast<uint32_t>(n * sizeof(words[0])))
		           : kIOReturnNotAttached;
	}
	if (logged(kDoSwapInfoNotify))
		IOLog("ReimsGraphics: do_swap_info_notify_gated(%p, %p) = 0x%x\n", request, blob, r);
	return word(r);
}

// The class's swap_complete_gated calls this with the completed swap's ID. As the paravirtual
// display does, the swap's record (rgfxSwapInfoRecord) is made, here from its presentation time
// (completeMobileSwap) and the swap's own fields, at the release's offsets
// (RgfxLegacySwapLayout.hpp), and dispatched to each notify request
// (swap_info_notify_dispatch_gated), after the records kept while the class held them. A swap
// whose state word is not 0 gets none, answered kIOReturnUnsupported.
RgfxIOMFBWord ReimsGraphics::notify_swap_done_gated(uint32_t field_0, ::SwapInfoBlob *blob,
                                                    uint32_t *field_2, uint32_t field_3)
{
	IOReturn r = kIOReturnUnsupported;
	const RgfxLegacySwapLayout *l = mobileSwapLayout;
	const uint8_t *swap =
	    l ? static_cast<const uint8_t *>(RgfxLegacySwapQueue(this, *l).current()) : nullptr;
	if (swap && *reinterpret_cast<const uint32_t *>(swap + l->state) == 0) {
		const uint8_t *self = reinterpret_cast<const uint8_t *>(this);
		RgfxSwapInfoRecord record = rgfxSwapInfoRecord(
		    field_0, mobilePresentedAt,
		    *reinterpret_cast<const uint64_t *>(swap + l->swapRecordField00),
		    *reinterpret_cast<const uint32_t *>(swap + l->swapFlags));
		if (*reinterpret_cast<const uint16_t *>(self + l->notifyHeld) != 0) {
			mobileSwapInfoKept.keep(record);
		} else {
			mobileSwapInfoKept.flush([this](RgfxSwapInfoRecord &kept) {
				rgfxLegacySwapInfoNotifyDispatch(this, &kept);
			});
			rgfxLegacySwapInfoNotifyDispatch(this, &record);
		}
		r = kIOReturnSuccess;
	}
	if (logged(kNotifySwapDone))
		IOLog("ReimsGraphics: notify_swap_done_gated(%u, %p, %p, %u) = 0x%x\n", field_0, blob,
		      field_2, field_3, r);
	return word(r);
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
	// The display starts in its preferred mode, as a panel does. WindowServer reads an internal
	// display's timings only where it publishes two colour modes or two timings: with one of each
	// it sets no mode and takes the display's size from get_display_size, which answers the
	// class's fields (read from 14.6.1's and 27.0.1's QuartzCore, IOMFBDisplay's
	// update_framebuffer_locked and set_physical_and_canvas_sizes). The device lists one mode,
	// so the size must be there before WindowServer opens the framebuffer.
	IOLockLock(stateLock);
	const uint32_t startTiming =
	    rgfxMobileStartTimingId(modeSet.modes, modeSet.modeCount, modeSet.preferredId);
	IOLockUnlock(stateLock);
	Mode start{};
	const IOReturn shown = startTiming ? setMobileMode(kRgfxColorElementId, startTiming, &start)
	                                   : kIOReturnNotReady;
	IOLog("ReimsGraphics: display %u starts in timing %u: 0x%x, %ux%u at %u.%03u Hz\n", displayId,
	      startTiming, shown, start.width, start.height, start.refreshMhz / 1000,
	      start.refreshMhz % 1000);
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

// The mode WindowServer picks of the timings and colour modes the display publishes
// (publishMobileModes), by their IDs: the colour mode's, then the timing's, as
// IOMobileFramebufferUserClient passes them from IOMobileFramebufferSetDigitalOutMode (read from
// macOS 14.6.1's QuartzCore and IOMobileFramebuffer.framework, and 27.0.1's AppleParavirtGPU, whose
// set_digital_out_mode looks the second up among its timings and the first among that timing's
// colour modes). The device shows the mode, as the x86-64 framebuffer's setDisplayMode has it
// shown; then, as the paravirtual display does, the class's own fields get the mode's size, in the
// class's gate (RgfxLegacySwapLayout.hpp), which its get_display_size answers and its swap_submit
// and default framebuffer surface read, and the framebuffer publishes the size and the timing ID:
// DisplayWidth, DisplayHeight and DPTimingModeId. A pair the display does not publish is answered
// kIOReturnUnsupported, as the paravirtual display leaves it to the class, which answers that.
// start_hardware sets the display's first mode the same way (setMobileMode).
RgfxIOMFBWord ReimsGraphics::set_digital_out_mode(uint32_t colorId, uint32_t timingId)
{
	Mode mode{};
	const IOReturn r = setMobileMode(colorId, timingId, &mode);
	if (logged(kSetDigitalOutMode))
		IOLog("ReimsGraphics: set_digital_out_mode(%u, %u) = 0x%x: %ux%u at %u.%03u Hz\n", colorId,
		      timingId, r, mode.width, mode.height, mode.refreshMhz / 1000, mode.refreshMhz % 1000);
	return word(r);
}

// The colour matrix WindowServer sets with a swap, by IOMobileFramebufferSwapSetColorMatrix:
// QuartzCore sets its accessibility matrix so (function 2) at each swap until the call succeeds
// (RgfxMobileColor.hpp). As the paravirtual display does, any function and field are answered 0,
// NULL sets the identity, and the display shows the matrix from the next present: the device's
// DISPLAY_COLOR matrix stage, which the host stages for the next PRESENT_SURFACE it takes and
// keeps until it is set again, as the x86-64 display pipe sets it (RgfxPipeColor.hpp); an
// identity turns the stage off. The user client calls it ungated, in WindowServer's thread, so
// the post may wait for the ring's room. A host without RGFX_FEATURE_DISPLAY_COLOR shows no
// matrix, and the call is still answered 0, as the paravirtual display answers whatever its host
// does with it.
RgfxIOMFBWord ReimsGraphics::swap_set_color_matrix(IOMFBColorFixedMatrix *matrix,
                                                   IOMFBColorMatrixFunction function,
                                                   uint32_t field_2)
{
	RgfxColorEntry rows[kRgfxMobileMatrixRows];
	rgfxMobileColorMatrix(reinterpret_cast<const int64_t *>(matrix), rows);
	// A swap's colour has no pipe transaction: its parts carry 0, which the host only logs.
	const IOReturn posted =
	    displayColor() ? postDisplayColor(0, RGFX_COLOR_MATRIX, kRgfxMobileMatrixRows,
	                                      reinterpret_cast<const UInt8 *>(rows))
	                   : kIOReturnUnsupported;
	if (logged(kSwapSetColorMatrix))
		IOLog("ReimsGraphics: swap_set_color_matrix(%s, %u, %u): rows %08x %08x %08x / %08x %08x "
		      "%08x / %08x %08x %08x, posted 0x%x\n",
		      matrix ? "matrix" : "NULL", static_cast<uint32_t>(function), field_2, rows[0].x,
		      rows[0].y, rows[0].z, rows[1].x, rows[1].y, rows[1].z, rows[2].x, rows[2].y, rows[2].z,
		      posted);
	return word(kIOReturnSuccess);
}

IOReturn ReimsGraphics::setMobileMode(uint32_t colorId, uint32_t timingId, Mode *shown)
{
	IOLockLock(stateLock);
	Mode mode{};
	const bool published =
	    rgfxMobileModeFor(colorId, timingId, modeSet.modes, modeSet.modeCount, &mode);
	IOLockUnlock(stateLock);
	*shown = mode;
	IOReturn r = kIOReturnUnsupported;
	if (published) {
		r = setScanout(mode);
		if (r == kIOReturnSuccess && mobileSwapLayout) {
			IOWorkLoop::Action write = [](OSObject *owner, void *layout, void *width, void *height,
			                              void *) -> IOReturn {
				rgfxLegacySetDisplaySize(owner, *static_cast<const RgfxLegacySwapLayout *>(layout),
				                         static_cast<uint32_t>(reinterpret_cast<uintptr_t>(width)),
				                         static_cast<uint32_t>(reinterpret_cast<uintptr_t>(height)));
				return kIOReturnSuccess;
			};
			void *layout = const_cast<RgfxLegacySwapLayout *>(mobileSwapLayout);
			void *width = reinterpret_cast<void *>(static_cast<uintptr_t>(mode.width));
			void *height = reinterpret_cast<void *>(static_cast<uintptr_t>(mode.height));
			if (IOWorkLoop *loop = getWorkLoop())
				loop->runAction(write, this, layout, width, height);
			else
				write(this, layout, width, height, nullptr);
		}
		if (r == kIOReturnSuccess) {
			setProperty("DisplayWidth", mode.width, 32);
			setProperty("DisplayHeight", mode.height, 32);
			setProperty("DPTimingModeId", timingId, 64);
		}
	}
	return r;
}

#endif // __arm64e__
