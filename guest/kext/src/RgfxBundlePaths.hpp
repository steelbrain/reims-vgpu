/*
 * RgfxBundlePaths.hpp — the paths the accelerator looks up for its Metal bundle
 * (RgfxAccelerator.cpp, "the bundle"): the executable Metal loads, and the marker
 * the bundle's package carries for the wire's version it speaks
 * (guest/bundle/Makefile, `package`), both under /Library/GPUBundles/NAME.bundle,
 * NAME the accelerator's MetalPluginName; and when what it finds there counts,
 * given RgfxGate's install pass.
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

// The IOResources property RgfxGate publishes for its install pass (guest/rgfxgate, RgfxGate.hpp's
// kInstallDoneKey, kept in step by hand): false from the pass's start, true once it has ended. A
// guest without RgfxGate has none.
#define kRgfxGateInstallDoneKey "RgfxGateInstallDone"

// RgfxGate's install pass, as its property gives it.
enum RgfxGatePass : unsigned {
	// No property: no RgfxGate, or one older than the property.
	kRgfxGateAbsent,
	// The pass is under way.
	kRgfxGateInstalling,
	// The pass has ended.
	kRgfxGateDone,
};

// Whether the accelerator registers for what a check finds: its own bundle on disk (`ours`, the
// executable with this kext's marker), no install pass of RgfxGate's under way, and the
// framebuffer started (`framebuffer`). A pass under way may be about to rewrite the bundle under
// the marker an older pass left (an update that keeps the wire version), so a bundle found then
// does not count. Every message of a process's context reaches the device through the
// framebuffer's rings, so an accelerator registered before the framebuffer has started would hand
// Metal a GPU that cannot answer: its contexts' messages and their closes would all be refused.
inline bool rgfxBundleRegisters(RgfxGatePass gate, bool ours, bool framebuffer)
{
	return gate != kRgfxGateInstalling && ours && framebuffer;
}

// The IOResources property arm64's RgfxGate publishes while it watches the Data volumes a boot that
// started in its BaseSystem moves to (guest/rgfxgate, RgfxGate.hpp's kPassRootKey and rgfx_pass.h,
// kept in step by hand): the id of the root file system its last pass ended under, then false
// once the watch has ended. A boot that starts in its full system, and x86-64, have none.
#define kRgfxGatePassRootKey "RgfxGatePassRoot"

// Whether a process's read of the accelerator's MetalPluginName waits: RgfxGate's watch runs
// (`watching`, its property a root's id, `passRoot`) and the root the reader is on (`root`, where
// there is one, `haveRoot`) is not the one its last pass ended under. A boot that leaves arm64's
// BaseSystem for its full system keeps the kernel, and the accelerator the BaseSystem's pass
// registered, so the full system's WindowServer finds a registered accelerator before its own
// Data volume has had a pass; that volume may hold another build's bundle, which RgfxGate's
// hooks do not admit. Metal reads the name before it loads the bundle, so holding the read
// until the pass for the reader's root has ended keeps every load after that pass.
inline bool rgfxMetalNameWaits(bool watching, unsigned long long passRoot, bool haveRoot,
                               unsigned long long root)
{
	return watching && haveRoot && passRoot != root;
}

#endif // RGFX_BUNDLE_PATHS_HPP_
