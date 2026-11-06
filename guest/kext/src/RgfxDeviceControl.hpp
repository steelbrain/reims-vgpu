/*
 * RgfxDeviceControl: the device's graphics device control, an
 * AppleGraphicsDeviceControl (AGDC) on the PCI function beside ReimsGraphics
 * and RgfxAccelerator.
 *
 * WindowServer gets an accelerated display only through a display pipe, and
 * IOPresentment builds one only for a framebuffer whose PCI device has an
 * AGDC. AppleGPUWrangler, which matches IOResources, pairs each AGDC it sees
 * published with its GPU's record, asks it what it is and what it can make,
 * and tells it what to make. The library class does all of that talking; a
 * subclass only answers the device's attributes (vendor_doDeviceAttribute).
 * A get answers only an output of the attribute's exact size, and every
 * attribute the control does not answer is unsupported.
 *
 * The control's own attributes:
 * - 0x1, the device's description, 0x2c bytes. AGDC's start, the wrangler
 *   and IOPresentment (its user client's selector 1) read it. The word at
 *   +0x28 must be 1, 2 or 7 for the wrangler to pair the control, and 1 or 2
 *   for IOPresentment to build a pipe; we answer 2. IOPresentment's platform
 *   check reads the half-word at +0x24, which 0x106b passes.
 * - 0x3, what to make. A get (a 4-byte output) answers the mask of what the
 *   control can make (bit 0 the framebuffers, bit 1 the accelerator); the
 *   wrangler goes on only for a mask with both. A set (a 4-byte input) is
 *   the wrangler's InitFlags, which we record: ReimsGraphics and
 *   RgfxAccelerator match the device on their own.
 * - 0x980, the displays, 0xdc bytes: AGDC's start and IOPresentment read it,
 *   and IOPresentment builds no pipe without it ("Unable to get AGDC GPU
 *   capabilities").
 * - 0x921, 0xb0 bytes, which IOPresentment reads once it has a pipe.
 *
 * The display's attributes, answered from the framebuffer's state through
 * IOFramebuffer's own interface, so they follow what ReimsGraphics tells
 * IOFramebuffer. IOPresentment asks them once it has the pipe, naming the
 * display by its index (0, the one framebuffer) or that index plus 1:
 * - 0x925, 12 bytes each way: whether the display is connected.
 * - 0x711, 0x196c bytes each way: one query about the display's
 *   capabilities, by a request bit. Request 2 answers its gamma: plane 0
 *   takes tables, as on the AMD's display (RgfxGammaCapability.hpp), so
 *   CoreDisplay fades and calibrates through the display pipe's gamma
 *   selectors. Request 0x10 answers the current mode's size, request 1 has
 *   words of its own, and the rest name no plane.
 * - 0x986, 0x90 bytes each way: one 128-byte block of the display's EDID.
 *
 * The first calls are logged, so a boot shows what was asked.
 *
 * The library (com.apple.AppleGraphicsDeviceControl) is in the System KC; the
 * kext names it in OSBundleLibraries, and its base-class sites are bound as
 * IOGraphicsFamily's are. Its interface is AppleGraphicsDeviceControl.hpp.
 */

#ifndef RGFX_DEVICE_CONTROL_HPP_
#define RGFX_DEVICE_CONTROL_HPP_

#include "AppleGraphicsDeviceControl.hpp"

class IOFramebuffer;

class RgfxDeviceControl : public AppleGraphicsDeviceControl {
	OSDeclareDefaultStructors(RgfxDeviceControl);

public:
	bool start(IOService *provider) override;

	IOReturn vendor_doDeviceAttribute(UInt32 attribute, uintptr_t *input, size_t inputSize,
	                                  uintptr_t *output, size_t *outputSize,
	                                  IOExternalMethodArguments *arguments) override;
	using AppleGraphicsDeviceControl::vendor_doDeviceAttribute;

private:
	// The provider's framebuffer, retained, or null before it attaches.
	IOFramebuffer *copyFramebuffer() const;
	IOReturn answer(UInt32 attribute, const void *input, size_t inputSize, void *output,
	                size_t outputSize);
	IOReturn answerConnection(const void *input, size_t inputSize, void *output,
	                          size_t outputSize);
	IOReturn answerQuery(const void *input, size_t inputSize, void *output, size_t outputSize);
	IOReturn answerEdid(const void *input, size_t inputSize, void *output, size_t outputSize);

	SInt32 loggedCalls;
	UInt32 initFlags;
};

#endif /* RGFX_DEVICE_CONTROL_HPP_ */
