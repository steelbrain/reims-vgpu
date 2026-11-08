// RgfxPciClass: the PCI class the kext states for the GPU, as its IOPCIDevice's `class-code`: a
// VGA-compatible display controller's, 0x030000, the class an x86 Mac's GPUs state, where the
// device's own class is a display controller of another kind, 0x038000 (RGFX_PCI_CLASS).
//
// Software in the guest that lists the Mac's GPUs by that class finds none where the class reads
// 0x038000. On a macOS 11.7.11 guest, Safari 14.1.3 made no WebGL context ("Could not create a
// WebGL context."), and WebKit's ANGLE, on the OpenGL backend that Safari's WebGL runs on there,
// refused to start: eglInitialize answered EGL_NOT_INITIALIZED, "Unable to query ANGLE's
// SystemInfo.", in a plain process as in one with no WindowServer connection (measured:
// EGL_KHR_debug). With the class stated as 0x030000, ANGLE started there and inside the web
// content sandbox, and Safari made its WebGL context (measured on the same guest). Both GPUs of an
// Intel Mac state 0x030000 (measured: macOS 26.7.1, the Intel UHD Graphics 630 and the AMD Radeon
// Pro 5300M, ioreg).
//
// So the accelerator sets `class-code` to those bytes on its PCI device when it starts. Only the
// registry property changes: the device's configuration space, which the PCI family matched
// drivers by at boot, keeps the device's own class.
//
// Header-only and free of the kernel's headers, so its native test (tests/pci_class_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_PCI_CLASS_HPP_
#define RGFX_PCI_CLASS_HPP_

#include "reims_graphics_wire.h"

#include <stdint.h>

// `class-code` as an x86 Mac's GPU states it: four bytes, little-endian, programming interface,
// subclass, base class, then 0.
static constexpr uint8_t kRgfxMacGpuClassCode[4] = {0x00, 0x00, 0x03, 0x00};

// A `class-code` property's value: its four bytes, little-endian.
inline uint32_t rgfxClassCodeValue(const uint8_t (&bytes)[4])
{
	return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
	       static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
}

#endif // RGFX_PCI_CLASS_HPP_
