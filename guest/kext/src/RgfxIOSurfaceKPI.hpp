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
//     call for that accelerator and returns the same one after, retained once a call (the
//     caller's reference, the only one: a cache just made has a retain count of 1, measured in
//     the guest), and hands IOSurface its control function with setControlFunc(cache, its
//     context, the function);
//   - the caller's release of a cache to a retain count of 0 frees it at once in the guest,
//     calling the control function with op 4 from the cache's free (measured; the AMD's driver,
//     whose resource holds the reference, has IOSurface keep its cache past the release);
//   - at a command buffer's submission it calls setCurrent on the cache of each surface the
//     buffer writes, none for a surface it only reads; the surface's seed moves once for it;
//   - a CPU lock of a surface whose cache is current then calls the control function with the
//     accelerator, the cache and op 0 (sync), then op 2 (wait: its first argument 1 for a
//     read-only lock, 0 for a read-write one), then op 3, and returns once the function does;
//     a read-write lock of a surface with a cache that is not current calls op 2 with 0, then
//     op 3; a read-only one, none. An AvoidSync lock of a surface whose cache is current answers
//     kIOReturnCannotLock with no call, until a plain lock has synced it;
//   - IOSurfaceDeviceCache::purge calls op 5, at each client of the surface going and when a
//     surface set empty is made nonvolatile again;
//   - setPurgeable calls op 7 with the new state as IOMemoryDescriptor numbers it
//     (kIOMemoryPurgeable*: 1 to read it, 2 nonvolatile, 3 volatile, 4 empty), but a change to
//     nonvolatile after empty, or after volatile while the AMD's resource held the cache, calls
//     op 6 instead, from IOSurfaceDeviceCache::unpurge, its second argument the address of a
//     byte: the AMD's driver sets the byte from 0 to 0xff both after volatile, where IOSurface
//     then answers the old state volatile, and after empty, where it answers empty and calls
//     op 5. In the guest, a control function that leaves the byte 0 has IOSurface answer empty
//     after volatile too, and one that sets it to 0xff has IOSurface answer volatile after
//     volatile and after empty alike: the AMD's answer of empty comes by a way the control
//     function's arguments do not show;
//   - IOSurface takes nothing from the control function's return: answering op 6 with 0 or 1,
//     and op 7 with 0 or 0x8000 (what the AMD's resource returns for volatile), left every
//     answer of IOSurface's as it was (measured in the guest);
//   - IOSurface retains the cache before each call of a lock's ops 0 and 2, of op 6 and of
//     op 7, and ends each with IOSurfaceDeviceCache::accelRelease, which calls op 3 once for
//     each reference it took (a change to nonvolatile by op 6 takes two). The AMD's control
//     function releases the cache at each op 3 (its stacks). In the guest, a control function
//     that releases nothing there leaves the cache one reference more for each op 3, and the
//     cache and its surface are never freed (measured: retain counts at each op).
// The accelerator ID came from IOSurfaceRoot::generateUniqueAcceleratorID(the accelerator) on
// both, 1 for IOGPU and 2 for the AMD's (taken before the trace began).
//
// A surface's seed, and its CPU locks, read from IOSurface 393.5.8's code (the guest's macOS
// 26.6.1, 25G76) and measured in the guest with IOSurfaceGetSeed:
//   - getLockSeed is the seed IOSurfaceGetSeed reads: 0 for a fresh surface, 4K ones included;
//     a read-only lock leaves it, a read-write lock moves it as it locks, and a command buffer
//     that writes the surface moves it at its commit (the kext's setCurrent); a texture made
//     over the surface leaves it (measured);
//   - lockSurface holds the surface's mutex throughout but while synchronize_device_caches calls
//     the device caches' control functions, during which other syncs wait for it to end; a
//     read-write lock moves the seed after those calls, under the mutex again. A lock of a
//     surface with no device cache never lets the mutex go. So a read-only lockSurface returns a
//     seed that counts every read-write lock begun before it, and with no device cache current,
//     calls no control function (read from code);
//   - IOSurfaceDeviceCache::init retains its surface, which the cache holds until it is freed
//     (read from code).
//
// Our own code.
#ifndef RGFX_IOSURFACE_KPI_HPP_
#define RGFX_IOSURFACE_KPI_HPP_

#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOTypes.h>
#include <libkern/c++/OSDictionary.h>

// A surface's device cache (IOSurfaceDeviceCache), opaque here.
struct RgfxDeviceCache;

// The control function a device cache calls: the accelerator its cache was made for, the cache,
// the op (0 sync, 2 wait, 3 done: the function releases the cache, 4 freed, 5 purged, 6 made
// nonvolatile, 7 purgeable), and the op's two arguments (op 2's first: 1 for a read-only lock, 0
// for a read-write one; op 7's the new state; op 6's second the address of the byte it answers
// in; the others measured 0). IOSurface declares it returning nothing (setControlFunc's mangled
// name) and reads no return.
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
// IOSurface::getClientRetainCount() const: the surface's clients, one for each process's
// IOSurfaceClient of it: one more at a process's creation or lookup of it (clientRetain), one
// fewer as each goes (clientRelease), and 0 at the op 5 of the last one's going (measured in the
// guest).
int32_t rgfxIOSurfaceClientCount(const OSObject *surface)
    __asm__("__ZNK9IOSurface20getClientRetainCountEv");
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
// IOSurface::getLockSeed() const: the surface's seed, as IOSurfaceGetSeed reads it.
uint32_t rgfxIOSurfaceLockSeed(const OSObject *surface) __asm__("__ZNK9IOSurface11getLockSeedEv");
// IOSurface::lockSurface(unsigned int, unsigned int*) and unlockSurface(unsigned int, unsigned
// int*): a CPU lock of the surface, as a process's, with IOSurface's lock options
// (kRgfxIOSurfaceLockReadOnly), the seed after it in `*seed` where not null.
IOReturn rgfxIOSurfaceLock(OSObject *surface, uint32_t options, uint32_t *seed)
    __asm__("__ZN9IOSurface11lockSurfaceEjPj");
IOReturn rgfxIOSurfaceUnlock(OSObject *surface, uint32_t options, uint32_t *seed)
    __asm__("__ZN9IOSurface13unlockSurfaceEjPj");
}

// A GL window's buffers (RgfxGLUserClients.cpp), made as IOGPU makes them (read from IOGPUFamily's
// x86_64 code in macOS 26.7.1's KDK, IOGPUSurfaceMTL::shape_buffers and its callers): createSurface
// in the window surface's task, clientRetain, and, when the buffer goes, release_surface_async
// alone, which takes over the reference createSurface gave and the client retain (read from
// IOSurface's code: it holds the surface in its pending set, releases the caller's reference, and
// lets the rest go on a thread call of its own).
extern "C" {
// IOSurfaceRoot::createSurface(task*, OSDictionary*): a new IOSurface of `properties` in `task`,
// retained for the caller, or null.
OSObject *rgfxIOSurfaceRootCreate(void *root, task_t task, OSDictionary *properties)
    __asm__("__ZN13IOSurfaceRoot13createSurfaceEP4taskP12OSDictionary");
// IOSurfaceRoot::release_surface_async(IOSurface*): lets go of a surface createSurface made and
// clientRetain held.
void rgfxIOSurfaceRootReleaseAsync(void *root, OSObject *surface)
    __asm__("__ZN13IOSurfaceRoot21release_surface_asyncEP9IOSurface");
// IOSurfaceRoot::copyPortNameForSurfaceInTask(task*, IOSurface*, unsigned int*): a send right
// to `surface` in `task`, its name in `*name`, which IOSurfaceLookupFromMachPort opens there.
IOReturn rgfxIOSurfaceRootPortName(void *root, task_t task, OSObject *surface, uint32_t *name)
    __asm__("__ZN13IOSurfaceRoot28copyPortNameForSurfaceInTaskEP4taskP9IOSurfacePj");
// IOSurface::clientRetain(): one client more.
void rgfxIOSurfaceClientRetain(OSObject *surface) __asm__("__ZN9IOSurface12clientRetainEv");
// IOSurface::setValue(char const*, OSMetaClassBase const*, unsigned int*): one of the surface's
// values, as IOSurfaceSetValue sets it, `*error` 0 or IOSurface's error.
IOReturn rgfxIOSurfaceSetValue(OSObject *surface, const char *key, const OSMetaClassBase *value,
                               uint32_t *error)
    __asm__("__ZN9IOSurface8setValueEPKcPK15OSMetaClassBasePj");
}

// The rows IOSurface lays out (RgfxSurfaceRows.hpp), as a GPU's accelerator has IOSurface lay
// them out at its start (read from the code of IOSurface and IOAcceleratorFamily2 in macOS 26.7's
// KDK, 25G241): IOSurfaceRoot keeps two masks, one it aligns a surface's plane offsets to and one
// it aligns the bytes per row it works out for a surface to, each the cache line's bytes less one
// until an accelerator ORs more bits in, and answers both to IOSurfaceGetPropertyAlignment.
extern "C" {
// IOSurfaceRoot::updateLimits(unsigned int, unsigned int, unsigned int, unsigned int, unsigned
// int): ORs `planeMask` into the plane offsets' mask and `rowMask` into the bytes per row's, and
// lowers the root's three other limits to `a`, `b` and `c` where those are less
// (kRgfxIOSurfaceNoLimit leaves each as it is).
void rgfxIOSurfaceRootUpdateLimits(void *root, uint32_t planeMask, uint32_t rowMask, uint32_t a,
                                   uint32_t b, uint32_t c)
    __asm__("__ZN13IOSurfaceRoot12updateLimitsEjjjjj");
}

// What IOSurfaceRoot starts its three other limits at, which updateLimits lowers no further.
constexpr uint32_t kRgfxIOSurfaceNoLimit = 0x7fffffff;

// IOSurface's kIOSurfaceLockReadOnly.
constexpr uint32_t kRgfxIOSurfaceLockReadOnly = 1;

#endif // RGFX_IOSURFACE_KPI_HPP_
