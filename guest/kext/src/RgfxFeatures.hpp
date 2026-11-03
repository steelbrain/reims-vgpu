// RgfxFeatures: the device's feature bits as the kext negotiates them (design.md, "The wire's
// version").
//
// The host offers its bits in FEATURES_LO and FEATURES_HI. Bits 0 to 2 need no answer: the host
// acts on them as it offers them, and the kext and the bundle use them where offered. From bit 3,
// the host acts on a bit only where the guest's driver took it: the framebuffer writes
// DRIVER_FEATURES_LO and _HI after its reset and before DRIVER_READY with the bits the kext and its
// bundle carry out, masked by the offer, and reads back the ones the host took. A host older than
// the registers reads them as 0, so nothing from bit 3 is taken there. The kext and its bundle
// carry out none of those bits yet, so the framebuffer writes 0.
//
// The accelerator answers the bundle's Features call with the feature word (rgfxFeatureWord), and
// grants a queue's flags (kRgfxAccelMethodQueueCreate's seventh input) only where their feature
// was taken (rgfxQueueFlagsGranted).
//
// Header-only and free of the kernel's headers, so its native test (tests/features_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_FEATURES_HPP_
#define RGFX_FEATURES_HPP_

#include "reims_graphics_wire.h"

// The bits the host acts on as it offers them, with no write: bits 0 to 2.
constexpr uint64_t kRgfxOfferedFeatures =
    RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_SURFACE_CACHE | RGFX_FEATURE_CB_SUBMITTED;

// The bits from bit 3 the kext and its bundle carry out, which the framebuffer writes to
// DRIVER_FEATURES where the host offers them: none yet.
constexpr uint64_t kRgfxKextFeatures = 0;

// What the framebuffer writes to DRIVER_FEATURES, given the host's offer.
inline uint64_t rgfxDriverFeatures(uint64_t offered)
{
	return offered & kRgfxKextFeatures;
}

// The feature word the bundle works by: bits 0 to 2 as offered, and from bit 3 the ones the host
// took (DRIVER_FEATURES read back), of those the kext carries out.
inline uint64_t rgfxFeatureWord(uint64_t offered, uint64_t taken)
{
	return (offered & kRgfxOfferedFeatures) | (taken & kRgfxKextFeatures);
}

// The flags of `wanted` (RGFX_QUEUE_*) a queue gets, given the feature word: each one whose
// feature was taken. A queue granted none is created without the flags tail, as a wire-2 kext
// creates it.
inline uint32_t rgfxQueueFlagsGranted(uint32_t wanted, uint64_t word)
{
	uint32_t allowed = 0;
	if (word & RGFX_FEATURE_QUIET_SCHEDULED)
		allowed |= RGFX_QUEUE_QUIET_SCHEDULED;
	if (word & RGFX_FEATURE_CONSUMED_DOORBELL)
		allowed |= RGFX_QUEUE_CONSUMED_DOORBELL;
	return wanted & allowed;
}

#endif // RGFX_FEATURES_HPP_
