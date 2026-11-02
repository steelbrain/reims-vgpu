// RgfxIOSurfaceKPI: what the accelerator calls of IOSurface.kext's kernel interface, which no SDK
// header declares.
//
// Each is one of IOSurface.kext's exported symbols (nm of the kext), declared here under its own
// name as a C function whose first argument is the object the member function is called on. What
// each takes and returns, and when the GPU drivers call it, was measured with fbt (arguments and
// returns only) on a macOS 26.6 guest's IOGPU stack and on a MacBook Pro's AMD Radeon Pro 5300M
// under macOS 26.7 (IOAcceleratorFamily2), both with IOSurface 393.5.8:
//   - a driver gives a surface its device cache with deviceCacheForAccelerator(surface, the
//     driver's accelerator, the driver's accelerator ID, 0), which makes the cache at its first
//     call for that accelerator and returns the same one after, and hands IOSurface its control
//     function with setControlFunc(cache, its context, the function);
//   - at a command buffer's submission it calls setCurrent on the cache of each surface the
//     buffer writes, none for a surface it only reads; the surface's seed moves once for it;
//   - a CPU lock of a surface whose cache is current then calls the control function with the
//     accelerator, the cache and op 0 (sync), then op 2 (wait: its first argument 1 for a
//     read-only lock, 0 for a read-write one), then op 3, and returns once the function does;
//     a read-write lock of a surface with a cache that is not current calls op 2 with 0, then
//     op 3; a read-only one, none. An AvoidSync lock of a surface whose cache is current answers
//     kIOReturnCannotLock with no call, until a plain lock has synced it.
// The accelerator ID came from IOSurfaceRoot::generateUniqueAcceleratorID(the accelerator) on
// both, 1 for IOGPU and 2 for the AMD's (taken before the trace began).
//
// Our own code.
#ifndef RGFX_IOSURFACE_KPI_HPP_
#define RGFX_IOSURFACE_KPI_HPP_

#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOTypes.h>

// A surface's device cache (IOSurfaceDeviceCache), opaque here.
struct RgfxDeviceCache;

// The control function a device cache calls: the accelerator its cache was made for, the cache,
// the op (0 sync, 2 wait, 3 done), and the op's two arguments (op 2's first: 1 for a read-only
// lock, 0 for a read-write one; the second measured 0).
typedef void (*RgfxDeviceCacheControl)(void *accelerator, RgfxDeviceCache *cache, uint32_t op,
                                       uint64_t a, uint64_t b);

extern "C" {
// IOSurfaceRoot::lookupSurface(unsigned int, task*): the surface of ID `surfaceId` as `task` may
// see it, retained, or null.
OSObject *rgfxIOSurfaceRootLookupSurface(void *root, uint32_t surfaceId, task_t task)
    __asm__("__ZN13IOSurfaceRoot13lookupSurfaceEjP4task");
// IOSurfaceRoot::generateUniqueAcceleratorID(void*): an ID for `accelerator`'s device caches.
uint32_t rgfxIOSurfaceRootAcceleratorID(void *root, void *accelerator)
    __asm__("__ZN13IOSurfaceRoot27generateUniqueAcceleratorIDEPv");
// IOSurface::deviceCacheForAccelerator(void*, unsigned int, unsigned int): the surface's device
// cache for `accelerator` of `acceleratorId`, made at the first call; `third` is 0 from both
// drivers measured.
RgfxDeviceCache *rgfxIOSurfaceDeviceCache(OSObject *surface, void *accelerator,
                                          uint32_t acceleratorId, uint32_t third)
    __asm__("__ZN9IOSurface25deviceCacheForAcceleratorEPvjj");
// IOSurface::getMemoryDescriptor() const: the surface's memory.
IOMemoryDescriptor *rgfxIOSurfaceMemory(const OSObject *surface)
    __asm__("__ZNK9IOSurface19getMemoryDescriptorEv");
// IOSurfaceDeviceCache::setControlFunc(void*, void (*)(void*, IOSurfaceDeviceCache*, unsigned
// int, unsigned long long, unsigned long long)): the function a CPU lock calls, and its context.
void rgfxDeviceCacheSetControl(RgfxDeviceCache *cache, void *context, RgfxDeviceCacheControl fn)
    __asm__("__ZN20IOSurfaceDeviceCache14setControlFuncEPvPFvS0_PS_jyyE");
// IOSurfaceDeviceCache::setCurrent(): the GPU holds the surface's newest contents.
void rgfxDeviceCacheSetCurrent(RgfxDeviceCache *cache)
    __asm__("__ZN20IOSurfaceDeviceCache10setCurrentEv");
}

#endif // RGFX_IOSURFACE_KPI_HPP_
