/*
 * IOMobileFramebufferLegacy: a declaration of IOMobileGraphicsFamily's class of that name for the
 * arm64e ReimsGraphics, which no SDK declares. It holds only what ReimsGraphics overrides or
 * calls: the class's own pure methods, the methods ReimsGraphics overrides and forwards to it, and
 * start, stop and free, each with the signature its symbol in the arm64 kernel collections of
 * macOS 13.6 to 27.0.1 names. The types are named for the manglings and declared no further.
 *
 * No symbol carries a return type. Where nothing says what a method returns, it returns
 * RgfxIOMFBWord, a 64-bit word: a forward hands whatever the class returned (an IOReturn, a bool
 * or a pointer) back unchanged, and ReimsGraphics' own answers are 0, or 1 for a yes.
 *
 * Graft builds the vtable of the class's subclass from the live class at bind time and places each
 * override by its selector (Graft's notes/arm64-display.md), so this declaration's vtable order is
 * not the live one. Its fields are a reserve, field_reserve, inside which the live class's own
 * fields lie; Graft refuses to bind where the live class is bigger than IOService and the reserve.
 */

#ifndef RGFX_IOMOBILEFRAMEBUFFERLEGACY_HPP_
#define RGFX_IOMOBILEFRAMEBUFFERLEGACY_HPP_

#include <IOKit/IOService.h>

class IOSurface;
struct IOMFBGammaTable;
struct IOMFBNotifyRequest;
struct IOMFBSwapIORequest;
struct IOMFBColorFixedMatrix;
struct SwapInfoBlob;
namespace IOMFB {
namespace AP {
struct DisplayedData;
}
}
enum IOMFB_NotificationType : uint32_t;
enum DisplayColorRemapMode : uint32_t;
enum IOMFBColorMatrixFunction : uint32_t;
enum IOMFBParameterName : uint32_t;

typedef uint64_t RgfxIOMFBWord;

// The reserve holds the live IOMobileFramebufferLegacy's own fields, which its inherited methods
// reach at their real offsets, so ReimsGraphics' own fields follow it rather than overlapping
// them. The live class is at most 0x1dd8 bytes across macOS 13 to 27 (Ambrosia's per-release
// measurement), and this stand-in's IOService base is 0x88, so the reserve covers 0x1dd8 - 0x88
// with a little room; Graft refuses the bind where a release's live class is bigger still.
//
// The live chain is IOMobileFramebufferLegacy : IOMobileFramebufferService : IOMobileFramebuffer :
// IOService. The intermediate IOMobileFramebufferService changes neither Legacy's method manglings
// nor, since Graft sizes the subclass to the live Legacy, its layout, so the stand-in leaves it
// out.
constexpr size_t kRgfxIOMFBLegacyReserve = 0x2000;

class IOMobileFramebuffer : public IOService {
	OSDeclareAbstractStructors(IOMobileFramebuffer);
};

class IOMobileFramebufferLegacy : public IOMobileFramebuffer {
	OSDeclareAbstractStructors(IOMobileFramebufferLegacy);

public:
	struct SwapInfoBlob;

	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	void free() override;

	// The class's own pure methods: each subclass implements them.
	virtual RgfxIOMFBWord set_notify_tag_mask_gated(IOMFB_NotificationType type,
	                                                IOMFBNotifyRequest *request, uint32_t mask) = 0;
	// macOS 14 and later; macOS 13's takes no fourth argument (Graft takes either).
	virtual RgfxIOMFBWord supported_frame_info(uint32_t index, uint32_t *count, char (*names)[64],
	                                           uint32_t *field_3) = 0;
	virtual RgfxIOMFBWord factory_portal(void *field_0, task *owner) = 0;
	virtual RgfxIOMFBWord executeFactoryConfigurations_gated() = 0;
	virtual RgfxIOMFBWord set_frame_info_notification_state_gated(IOMFBNotifyRequest *request,
	                                                              bool on) = 0;
	virtual RgfxIOMFBWord do_swap_info_notify_gated(IOMFBNotifyRequest *request,
	                                                const SwapInfoBlob *blob) = 0;
	virtual RgfxIOMFBWord notify_swap_done_gated(uint32_t field_0, ::SwapInfoBlob *blob,
	                                             uint32_t *field_2, uint32_t field_3) = 0;
	virtual RgfxIOMFBWord notify_crc_ready_gated(uint32_t field_0) = 0;
	virtual bool are_timings_enabled() const = 0;

	// What the class implements, which ReimsGraphics overrides and forwards to it.
	virtual RgfxIOMFBWord start_hardware();
	virtual RgfxIOMFBWord stop_hardware_gated();
	virtual RgfxIOMFBWord swap_begin_gated();
	virtual RgfxIOMFBWord verify_swap(IOMFBSwapIORequest *request);
	virtual RgfxIOMFBWord swap_layer_map(IOMFBSwapIORequest *request, uint32_t layer);
	virtual RgfxIOMFBWord surface_layer_map(IOMFB::AP::DisplayedData *data, uint32_t layer,
	                                        IOSurface *surface);
	virtual RgfxIOMFBWord surface_layer_unmap(IOMFB::AP::DisplayedData *data, uint32_t layer);

	// What ReimsGraphics calls: a refresh of the display.
	void note_vbl_ISR(uint32_t field_0);

private:
	UInt8 field_reserve[kRgfxIOMFBLegacyReserve];
};

#endif // RGFX_IOMOBILEFRAMEBUFFERLEGACY_HPP_
