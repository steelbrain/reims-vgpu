/*
 * ReimsGraphics' arm64e front end: the IOMobileFramebufferLegacy subclass an arm64 WindowServer
 * drives (ReimsGraphics.hpp). The device, its rings, its display and its pipe clients are
 * ReimsGraphics.cpp's, as on x86-64; this file is what the class asks of a subclass.
 *
 * This first cut learns the class's calls: each override logs its calls (the first ones and then
 * one in kLogEvery) and forwards to the class where it implements the method. A layer's surface
 * map presents the layer-0 surface on the device, and the device's refreshes reach the class as
 * its vblank note. The class's own pure methods answer what a display with no frame info, no
 * factory configuration and no CRC answers.
 */

#include "ReimsGraphics.hpp"

#include <IOKit/IOLib.h>

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
	kVblank,
	kPresented,
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
	mobilePipe = addPipe(this, mobileVblank, mobilePresented);
	if (!mobilePipe) {
		IOLog("ReimsGraphics: no pipe for the mobile framebuffer\n");
		return false;
	}
	setPipeVblankEnabled(this, true);
	IOLog("ReimsGraphics: mobile framebuffer, display %u\n", displayId);
	return true;
}

void ReimsGraphics::stopMobile()
{
	if (mobilePipe)
		removePipe(this);
	mobilePipe = false;
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

// MARK: - What the class implements, logged and forwarded

RgfxIOMFBWord ReimsGraphics::start_hardware()
{
	const RgfxIOMFBWord r = super::start_hardware();
	if (logged(kStartHardware))
		IOLog("ReimsGraphics: start_hardware() = 0x%llx\n", r);
	return r;
}

RgfxIOMFBWord ReimsGraphics::stop_hardware_gated()
{
	const RgfxIOMFBWord r = super::stop_hardware_gated();
	if (logged(kStopHardware))
		IOLog("ReimsGraphics: stop_hardware_gated() = 0x%llx\n", r);
	return r;
}

RgfxIOMFBWord ReimsGraphics::swap_begin_gated()
{
	const RgfxIOMFBWord r = super::swap_begin_gated();
	if (logged(kSwapBegin))
		IOLog("ReimsGraphics: swap_begin_gated() = 0x%llx\n", r);
	return r;
}

RgfxIOMFBWord ReimsGraphics::verify_swap(IOMFBSwapIORequest *request)
{
	const RgfxIOMFBWord r = super::verify_swap(request);
	if (logged(kVerifySwap))
		IOLog("ReimsGraphics: verify_swap(%p) = 0x%llx\n", request, r);
	return r;
}

RgfxIOMFBWord ReimsGraphics::swap_layer_map(IOMFBSwapIORequest *request, uint32_t layer)
{
	const RgfxIOMFBWord r = super::swap_layer_map(request, layer);
	if (logged(kSwapLayerMap))
		IOLog("ReimsGraphics: swap_layer_map(%p, %u) = 0x%llx\n", request, layer, r);
	return r;
}

RgfxIOMFBWord ReimsGraphics::surface_layer_map(IOMFB::AP::DisplayedData *data, uint32_t layer,
                                               IOSurface *surface)
{
	const RgfxIOMFBWord r = super::surface_layer_map(data, layer, surface);
	const uint32_t id =
	    surface ? rgfxIOSurfaceSurfaceID(reinterpret_cast<const OSObject *>(surface)) : 0;
	IOReturn presented = kIOReturnNotReady;
	UInt64 seqno = 0;
	if (layer == 0 && id) {
		const UInt32 transaction = ++mobileTransaction;
		presented = postPresent(id, transaction, 0, &seqno);
		if (presented == kIOReturnSuccess)
			__atomic_store_n(&mobilePresentSeqno, seqno, __ATOMIC_RELEASE);
	}
	if (logged(kSurfaceLayerMap))
		IOLog("ReimsGraphics: surface_layer_map(%p, %u, %p: surface %u) = 0x%llx, present 0x%x "
		      "seqno %llu\n",
		      data, layer, surface, id, r, presented, seqno);
	return r;
}

RgfxIOMFBWord ReimsGraphics::surface_layer_unmap(IOMFB::AP::DisplayedData *data, uint32_t layer)
{
	const RgfxIOMFBWord r = super::surface_layer_unmap(data, layer);
	if (logged(kSurfaceLayerUnmap))
		IOLog("ReimsGraphics: surface_layer_unmap(%p, %u) = 0x%llx\n", data, layer, r);
	return r;
}

#endif // __arm64e__
