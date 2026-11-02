/*
 * RgfxBundlePaths.hpp — the paths the accelerator looks up for its Metal bundle
 * (RgfxAccelerator.cpp, "the bundle"): the executable Metal loads, and the marker
 * the bundle's package carries for the wire's version it speaks
 * (guest/bundle/Makefile, `package`), both under /Library/GPUBundles/NAME.bundle,
 * NAME the accelerator's MetalPluginName.
 *
 * The bundle maps the region and speaks the messages, so it must be of the
 * kext's wire version (design.md, "The wire's version"). The kext registers the
 * accelerator only where the marker of its own version is there: a bundle of
 * another version leaves it unregistered, and WindowServer composites in
 * software, as without a bundle.
 *
 * Header-only and free of kernel types, so tests/bundle_paths_test.cpp checks it
 * on the Mac.
 */

#ifndef RGFX_BUNDLE_PATHS_HPP_
#define RGFX_BUNDLE_PATHS_HPP_

#ifdef KERNEL
#include <libkern/libkern.h>
#else
#include <stdio.h>
#endif

struct RgfxBundlePaths {
	// /Library/GPUBundles/NAME.bundle/Contents/MacOS/NAME
	char executable[256];
	// /Library/GPUBundles/NAME.bundle/Contents/Resources/wire-version-VERSION
	char marker[256];
};

// Both paths for `name` and the wire's `version`; false where either does not fit.
inline bool rgfxBundlePaths(const char *name, unsigned version, RgfxBundlePaths *paths)
{
	int executable = snprintf(paths->executable, sizeof(paths->executable),
	                          "/Library/GPUBundles/%s.bundle/Contents/MacOS/%s", name, name);
	int marker = snprintf(paths->marker, sizeof(paths->marker),
	                      "/Library/GPUBundles/%s.bundle/Contents/Resources/wire-version-%u", name,
	                      version);
	return executable >= 0 && static_cast<size_t>(executable) < sizeof(paths->executable) &&
	       marker >= 0 && static_cast<size_t>(marker) < sizeof(paths->marker);
}

#endif // RGFX_BUNDLE_PATHS_HPP_
