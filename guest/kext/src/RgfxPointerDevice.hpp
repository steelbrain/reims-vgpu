// RgfxPointerDevice: the guest's own relative pointing device, a HID mouse with no bus under it,
// which the framebuffer publishes where the guest took POINTER_MOTION and hands each
// POINTER_MOTION as reports (RgfxPointerMotion.hpp, design.md "A pointer for games"). macOS's
// HID stack drives it as any USB mouse: IOHIDEventDriver turns its reports into pointer events,
// which WindowServer accelerates and moves the cursor by, and GCMouse and IOHIDManager list it.
//
// Its name and numbers are ours: "Reims vGPU Mouse" of "Reims vGPU", vendor 0x1234 and product
// 0x5247, the PCI IDs of the device it belongs to. It is a child of display 0's framebuffer,
// started by it once the guest took the feature and stopped with it. The x86-64 kext's
// framebuffer starts a second beside it, the precise scroller, "Reims vGPU Scroller", product
// 0x5248, which takes a SCROLL's distance (RgfxPointerMotion.hpp) with its acceleration off.
//
// It imports only what every release of both sets defines (x86-64 11 to 26, arm64 14 to 27).
// The SDK's IOHIDDevice.h is the newest release's, and a class inheriting a method imports it by
// name: its vtable names the parent's definition, which the kernel collection must define.
// Ambrosia's linker binds each import by name and patches no vtable, so a name an older release
// lacks refuses the kext there, and the whole display with it (arm64 14.6.1 and 15.6.1 lack
// IOHIDDevice::didTerminate); OpenCore's linker left the kext out of x86-64 11.7.11 (measured).
// Each IOHIDDevice method an older release lacks is overridden below, and the override calls
// IOHIDDevice's: a qualified call to a virtual method, which under -fapple-kext loads the method
// from IOHIDDevice's own vtable at its slot, not by its name. So every release runs what its
// IOHIDDevice keeps in that slot: its own method where it defines one, IOService's where it does
// not. guest/'s kc-check holds the sets to this (guest/Makefile).
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
#if !RGFX_GUEST_ARM64
	// The device, the mouse or (`scroller`) the precise scroller, attached to `provider` and
	// started: nullptr where it does not start.
	static RgfxPointerDevice *withProvider(IOService *provider, bool scroller);
#else
	// The device, attached to `provider` and started: nullptr where it does not start.
	static RgfxPointerDevice *withProvider(IOService *provider);
#endif

	// Hands the HID stack the reports of one POINTER_MOTION, in order. On the framebuffer's work
	// loop, one call at a time.
	void post(const RgfxMotion &motion);
#if !RGFX_GUEST_ARM64
	// Hands it a SCROLL's notches as reports, with the buttons the last POINTER_MOTION left held,
	// as post does.
	void scroll(const RgfxScrolled &scrolled);
	// Hands the scroller a SCROLL's distance as counts of 10 points, keeping the rest for the
	// next (rgfxScrollerCounts); nothing where it makes no count.
	void scrollFine(const RgfxScrolled &scrolled);
#endif

	IOReturn newReportDescriptor(IOMemoryDescriptor **descriptor) const override;
	OSString *newManufacturerString() const override;
	OSString *newProductString() const override;
	OSNumber *newVendorIDNumber() const override;
	OSNumber *newProductIDNumber() const override;
	OSNumber *newPrimaryUsageNumber() const override;
	OSNumber *newPrimaryUsagePageNumber() const override;
#if defined(__arm64e__)
	// Its HID transport, "Virtual" (kIOHIDTransportVirtualValue), on arm64 alone. There display
	// 0's framebuffer, its provider, carries IOMobileFramebuffer's own "Transport", a dictionary
	// (ReimsGraphicsMobile.cpp), which a lookup of the mouse's transport through its parents finds
	// where the mouse names none: macOS 26.6.2's ReportIntervalServiceFilter took it for a string,
	// and WindowServer died of the exception at each start, so no desktop came (lane a64ptr,
	// potluck job m3-max-35949). The x86-64 kext's framebuffer has no "Transport", so its mouse
	// names none, as before.
	OSString *newTransportString() const override;
#endif

	// IOHIDDevice's own methods that an older release's IOHIDDevice does not define, each called
	// through IOHIDDevice's vtable (above). willTerminate and didTerminate are IOService's slots:
	// x86-64 11 and 12 define neither for IOHIDDevice, and x86-64 13 to 15 and arm64 14 and 15
	// not didTerminate, so those releases run IOService's.
	bool willTerminate(IOService *provider, IOOptionBits options) override;
	bool didTerminate(IOService *provider, IOOptionBits options, bool *defer) override;

	// The forms that take a completion fill IOHIDDevice's reserved slots 16 and 17 from macOS 13
	// on. On x86-64 11 and 12 those slots are still placeholders, which a vtable entry naming a
	// method there does not match: a linker that patches vtables (kxld's rule) refuses such a kext
	// as built against a newer class, and leaves alone an entry the kext defines itself. The
	// forwards would load the placeholder there, which nothing calls: those releases' HID stack
	// does not know these methods.
	using IOHIDDevice::postElementValues;
	using IOHIDDevice::updateElementValues;
	IOReturn updateElementValues(IOHIDElementCookie *cookies, UInt32 cookieCount,
	                             IOOptionBits options, UInt32 completionTimeout,
	                             IOHIDCompletion *completion = 0,
	                             IOBufferMemoryDescriptor *elementData = 0) override;
	IOReturn postElementValues(IOHIDElementCookie *cookies, UInt32 cookieCount,
	                           IOOptionBits options, UInt32 completionTimeout,
	                           IOHIDCompletion *completion = 0) override;

private:
	void free() override;

	// IOService's IOReporting slots, which IOHIDDevice overrides as private methods from macOS
	// 13 on and x86-64 11 and 12 do not: forwarded through IOHIDDevice's vtable as the others, in
	// a file of their own (RgfxPointerReporting.cpp).
	IOReturn configureReport(IOReportChannelList *channels, IOReportConfigureAction action,
	                         void *result, void *destination) override;
	IOReturn updateReport(IOReportChannelList *channels, IOReportUpdateAction action,
	                      void *result, void *destination) override;

	// The buffer each report is written into before the HID stack reads it.
	IOBufferMemoryDescriptor *reportBuffer;
#if !RGFX_GUEST_ARM64
	// The buttons the last report held.
	uint32_t buttons;
	// The scroller (withProvider), and the distance its counts have left, up and right, in 1/256
	// point.
	bool scroller;
	int64_t rest[2];
#endif
};

#endif // RGFX_POINTER_DEVICE_HPP_
