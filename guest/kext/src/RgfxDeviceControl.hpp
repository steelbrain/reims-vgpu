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
 * subclass only answers the device's attributes (vendor_doDeviceAttribute):
 *
 * - 0x1, the device's description, 0x2c bytes. AGDC's start, the wrangler
 *   and IOPresentment (its user client's selector 1) read it. The word at
 *   +0x28 must be 1, 2 or 7 for the wrangler to pair the control, and 1 or 2
 *   for IOPresentment to build a pipe; we answer 2. IOPresentment's platform
 *   check reads the half-word at +0x24, which 0x106b passes.
 * - 0x3, what to make. A get answers the mask of what the control can make
 *   (bit 0 the framebuffers, bit 1 the accelerator); the wrangler goes on
 *   only for a mask with both. A set is the wrangler's InitFlags, which we
 *   record: ReimsGraphics and RgfxAccelerator match the device on their own.
 * - 0x301, set: the GPU's index in the wrangler, recorded.
 * - 0x980, the displays, 0xdc bytes: AGDC's start and IOPresentment read it,
 *   and IOPresentment builds no pipe without it ("Unable to get AGDC GPU
 *   capabilities").
 * - 0x921, 0xb0 bytes, which IOPresentment reads once it has a pipe.
 * Every other attribute is unsupported. The first calls are logged, so a boot
 * shows what was asked.
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
	static IOFramebuffer *framebuffer(IOService *provider);
	IOReturn answer(UInt32 attribute, uintptr_t *input, size_t inputSize, uintptr_t *output,
	                size_t *outputSize);

	SInt32 loggedCalls;
	UInt32 initFlags;
	UInt16 gpuIndex;
};

#endif /* RGFX_DEVICE_CONTROL_HPP_ */
