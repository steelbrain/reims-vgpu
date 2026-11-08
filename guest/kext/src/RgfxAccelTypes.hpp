// RgfxAccelTypes: the framebuffer's IOAccelTypes, the service-plane path of the accelerator that
// draws for it. IOAccelFindAccelerator follows it from a framebuffer to its accelerator, and so
// does OpenGL's list of renderers in a process with no WindowServer connection (libGFXShared's
// plugin data, from each framebuffer: read from macOS 11.7.11's code).
//
// The path names the PCI device by its name, so a rename of the device after the path was set
// leaves the path leading nowhere. On macOS 11.7.11, with the PCI device renamed GFX0 after the
// framebuffer had named its accelerator through the device's earlier name, S20, a process with no
// WindowServer connection was offered the software renderer alone; with the path as it stands,
// the accelerated renderer too (measured: CGLQueryRendererInfo after
// CGSSetDenyWindowServerConnections; IOAccelFindAccelerator answered 0xe00002bc, then 0).
//
// So the accelerator names itself on the framebuffer again whenever it finds the framebuffer
// (RgfxAccelerator::copyFramebuffer), where the framebuffer's path is not its own as it stands.
//
// Header-only and free of the kernel's headers, so its native test (tests/accel_types_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_ACCEL_TYPES_HPP_
#define RGFX_ACCEL_TYPES_HPP_

#include <string.h>

// Whether the framebuffer's IOAccelTypes, `named` (nullptr where it has none), is to be set to
// `path`, the accelerator's service-plane path as it stands: where it has none, or another.
inline bool rgfxAccelTypesStale(const char *named, const char *path)
{
	return named == nullptr || strcmp(named, path) != 0;
}

#endif // RGFX_ACCEL_TYPES_HPP_
