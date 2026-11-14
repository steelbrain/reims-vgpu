// RgfxKeptService: which service a lookup keeps, and when the kept one is handed out again
// (RgfxAccelerator::copyFramebuffer's framebuffer). A lookup walks the PCI device's clients and
// builds the accelerator's path under the registry's lock, which every message the kext posts
// paid for; so the first lookup that finds the framebuffer started keeps it, and the lookups after
// take the kept one, but only while it is live: started (ReimsGraphics::hasStarted, cleared as its
// stop begins) and not terminated (IOService::isInactive, set as its termination begins). One that
// is not is let go, and the lookup is made again, which finds the framebuffer that replaced it, or
// none: a stopped or terminated framebuffer is never handed out from the keep.
//
// Header-only and free of the kernel's headers, over the service's own calls (`Ops`), so the
// native test (tests/kept_service_test.cpp) checks the rule as the kext keeps it.
//
// Our own code.
#ifndef RGFX_KEPT_SERVICE_HPP_
#define RGFX_KEPT_SERVICE_HPP_

// `Ops`, for a service `T`:
//   bool live(T *)     started and not terminated;
//   void retain(T *)   one reference more.
// Both callable under the keep's lock.

// The kept service, retained for the caller, where it is live; nullptr where none is kept or the
// kept one is not live, which is then dropped from the keep and handed back in `*dropped`, its
// reference the keep's, for the caller to release out of the lock. Under the keep's lock.
template <class T, class Ops>
inline T *rgfxKeptTake(T *&kept, Ops &ops, T **dropped)
{
	*dropped = nullptr;
	if (kept == nullptr)
		return nullptr;
	if (!ops.live(kept)) {
		*dropped = kept;
		kept = nullptr;
		return nullptr;
	}
	ops.retain(kept);
	return kept;
}

// Keeps `found`, a lookup's answer, where it is live and none is kept: the keep takes a reference
// of its own. Whether it was kept. Under the keep's lock.
template <class T, class Ops>
inline bool rgfxKeptOffer(T *&kept, Ops &ops, T *found)
{
	if (found == nullptr || kept != nullptr || !ops.live(found))
		return false;
	ops.retain(found);
	kept = found;
	return true;
}

#endif // RGFX_KEPT_SERVICE_HPP_
