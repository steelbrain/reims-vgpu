/*
 * RgfxAccelerator: see RgfxAccelerator.hpp for the service's role, the Metal
 * discovery keys it carries, and the boot-gated base-class decision.
 */

#include "RgfxAccelerator.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxDisplayPipeUserClient.hpp"
#include "RgfxSurfaceUserClient.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/graphics/IOGraphicsInterfaceTypes.h>
#include <kern/clock.h>
#include <kern/thread_call.h>
#include <sys/vnode.h>

#include "RgfxTrace.hpp"

#define super IOAccelerator
OSDefineMetaClassAndStructors(RgfxAccelerator, IOAccelerator);

// MARK: - The bundle
//
// Metal finds a GPU by its registered IOAccelerator, and loads the bundle the accelerator's
// MetalPluginName names. Where the accelerator is registered and its bundle is not on disk (an
// install boot before the bundle is installed, Recovery, the installer), every WindowServer
// aborts in CoreDisplay as it brings the display up, and launchd starts another until it gives
// up: the screen stays black. Where no accelerator is registered, WindowServer composites in
// software on the framebuffer (measured on msi: an accelerator that does not start leaves
// WindowServer up with a console session and the desktop drawn, the framebuffer still naming it
// in IOAccelTypes; a registered accelerator, the framebuffer not naming it, still aborts).
//
// So the accelerator registers only once the bundle is on disk. It starts as before, but a
// check repeats from its start: none before the root file system is mounted, then a lookup of
// the bundle's executable every kBundleCheckMs. /Library/GPUBundles is on the data volume,
// which launchd mounts early in boot, well before it starts WindowServer, so on a guest with the
// bundle the accelerator is registered before WindowServer looks for a GPU. WindowServer's
// first server connection to the framebuffer settles it (settleBundle, from
// ReimsGraphics::newUserClient, before IOFramebuffer makes the connection, which WindowServer
// opens before any of the accelerator's user clients): one last check there, and none after.
// Without the bundle by then, the accelerator stays unregistered for the rest of the boot, so
// WindowServer never sees a GPU appear under it; a bundle installed later is used from the next
// boot. Neither the check nor its wait runs on the kext's start or the framebuffer's console.
namespace {
// How often the check repeats until the bundle is found or WindowServer settles it.
constexpr UInt32 kBundleCheckMs = 100;
// How long the settling connection waits for a registration in flight to finish.
constexpr UInt32 kBundleRegisterWaitMs = 5000;

enum : UInt32 {
	// The check repeats.
	kBundleWatching,
	// The bundle was found; registerService is under way (the service may be published already).
	kBundleRegistering,
	// registerService has returned: the accelerator is registered.
	kBundleRegistered,
	// WindowServer opened the display without the bundle: the accelerator stays unregistered.
	kBundleSettled,
};

// One accelerator per device, one device per guest. The state stays out of the class, as the
// contexts' does (RgfxAcceleratorContexts.cpp).
struct BundleWatch {
	// Under lock: where the accelerator stands (above).
	IOLock *lock;
	UInt32 state;
	// The repeating check.
	thread_call_t call;
	// The bundle's executable: /Library/GPUBundles/NAME.bundle/Contents/MacOS/NAME, NAME the
	// accelerator's MetalPluginName (the bundle's Makefile names its executable so).
	char path[256];
};
BundleWatch bundle = {nullptr, kBundleWatching, nullptr, {}};

UInt32 bundleState()
{
	if (!bundle.lock)
		return kBundleSettled;
	IOLockLock(bundle.lock);
	UInt32 state = bundle.state;
	IOLockUnlock(bundle.lock);
	return state;
}

// Moves the watch from Watching to `to`, and ends the repeating check where `to` is not Watching;
// false where it was no longer Watching.
bool leaveWatching(UInt32 to)
{
	IOLockLock(bundle.lock);
	bool moved = bundle.state == kBundleWatching;
	if (moved)
		bundle.state = to;
	IOLockUnlock(bundle.lock);
	if (moved)
		thread_call_cancel(bundle.call);
	return moved;
}

// Whether the bundle's executable is on disk; false before the root file system is mounted.
bool bundleOnDisk()
{
	vnode_t root = vfs_rootvnode();
	if (!root)
		return false;
	vnode_put(root);
	vfs_context_t context = vfs_context_create(nullptr);
	if (!context)
		return false;
	vnode_t found = nullptr;
	errno_t error = vnode_lookup(bundle.path, 0, &found, context);
	if (!error)
		vnode_put(found);
	vfs_context_rele(context);
	return !error;
}

void armBundleCheck()
{
	uint64_t deadline = 0;
	clock_interval_to_deadline(kBundleCheckMs, kMillisecondScale, &deadline);
	thread_call_enter_delayed(bundle.call, deadline);
}

// Registers the accelerator, which the caller moved to Registering, and marks it Registered once
// registerService returns: synchronously, so the matching (and Metal's notification) is done.
void registerAccelerator(RgfxAccelerator *accelerator, const char *when)
{
	IOLog("RgfxAccelerator: the bundle is at %s%s: registered\n", bundle.path, when);
	accelerator->registerService(kIOServiceSynchronous);
	IOLockLock(bundle.lock);
	bundle.state = kBundleRegistered;
	IOLockUnlock(bundle.lock);
}

// The repeating check, on a thread call: registers the accelerator once the bundle is there.
void checkBundle(thread_call_param_t accelerator, thread_call_param_t)
{
	bool found = bundleOnDisk();
	IOLockLock(bundle.lock);
	bool watching = bundle.state == kBundleWatching;
	if (watching && found)
		bundle.state = kBundleRegistering;
	else if (watching)
		armBundleCheck();
	IOLockUnlock(bundle.lock);
	if (watching && found)
		registerAccelerator(static_cast<RgfxAccelerator *>(accelerator), "");
}
} // namespace

bool RgfxAccelerator::watchBundle()
{
	OSString *name = OSDynamicCast(OSString, getProperty("MetalPluginName"));
	if (!name) {
		IOLog("RgfxAccelerator: no MetalPluginName: not registered\n");
		return false;
	}
	const char *n = name->getCStringNoCopy();
	if (static_cast<size_t>(snprintf(bundle.path, sizeof(bundle.path),
	                                 "/Library/GPUBundles/%s.bundle/Contents/MacOS/%s", n, n)) >=
	    sizeof(bundle.path)) {
		IOLog("RgfxAccelerator: MetalPluginName %s is too long: not registered\n", n);
		return false;
	}
	IOLock *lock = IOLockAlloc();
	thread_call_t call = thread_call_allocate(checkBundle, this);
	if (!lock || !call) {
		IOLog("RgfxAccelerator: no memory for the bundle's check: not registered\n");
		if (lock)
			IOLockFree(lock);
		if (call)
			thread_call_free(call);
		return false;
	}
	bundle.call = call;
	bundle.lock = lock;
	IOLockLock(bundle.lock);
	armBundleCheck();
	IOLockUnlock(bundle.lock);
	return true;
}

void RgfxAccelerator::settleBundle()
{
	if (bundleState() == kBundleWatching) {
		if (bundleOnDisk()) {
			if (leaveWatching(kBundleRegistering))
				registerAccelerator(this, " as WindowServer opens the display");
		} else if (leaveWatching(kBundleSettled)) {
			IOLog("RgfxAccelerator: no bundle at %s as WindowServer opens the display: not "
			      "registered this boot\n",
			      bundle.path);
		}
	}
	// A registration the repeating check began finishes before the connection is made.
	for (UInt32 waited = 0; bundleState() == kBundleRegistering && waited < kBundleRegisterWaitMs;
	     waited++)
		IOSleep(1);
}

// MARK: - Registers

UInt32 RgfxAccelerator::readRegister(UInt32 offset) const
{
	return *reinterpret_cast<const volatile UInt32 *>(registers + offset);
}

void RgfxAccelerator::writeRegister(UInt32 offset, UInt32 value)
{
	*reinterpret_cast<volatile UInt32 *>(registers + offset) = value;
}

// MARK: - IOService

bool RgfxAccelerator::start(IOService *provider)
{
	ReimsGraphics *fb = nullptr;
	pci = OSDynamicCast(IOPCIDevice, provider);
	if (!pci)
		return false;
	doorbellLock = IOSimpleLockAlloc();
	if (!doorbellLock)
		return false;

	// The framebuffer enables these too; doing so again is harmless. The
	// accelerator needs BAR 0 for the doorbell and BAR 2 to hand to the bundle.
	pci->setMemoryEnable(true);
	pci->setBusLeadEnable(true);

	registerMap = pci->mapDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0);
	if (!registerMap) {
		IOLog("RgfxAccelerator: BAR0 not mapped\n");
		goto fail;
	}
	registers = reinterpret_cast<volatile UInt8 *>(registerMap->getVirtualAddress());
	if (readRegister(RGFX_REG_MAGIC) != RGFX_MAGIC || readRegister(RGFX_REG_VERSION) != RGFX_VERSION) {
		IOLog("RgfxAccelerator: not the device: magic 0x%x, version %u\n",
		      readRegister(RGFX_REG_MAGIC), readRegister(RGFX_REG_VERSION));
		goto fail;
	}

	region = pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress2);
	if (!region) {
		IOLog("RgfxAccelerator: no BAR2\n");
		goto fail;
	}
	region->retain();
	regionSize = (static_cast<UInt64>(readRegister(RGFX_REG_REGION_SIZE_HI)) << 32) |
	             readRegister(RGFX_REG_REGION_SIZE_LO);
	if (regionSize > region->getLength())
		regionSize = region->getLength();
	if (regionSize <= RGFX_ALLOCATIONS_OFFSET) {
		IOLog("RgfxAccelerator: the region is too small: 0x%llx bytes\n", regionSize);
		goto fail;
	}

	// The Metal discovery keys (MetalPluginName / MetalPluginClassName) and the PCI
	// match that make Metal load the bundle are in src/Info.plist's accelerator
	// personality, so they are already registry properties on this service. The
	// accelerator does not touch pixels or the control ring (the framebuffer owns
	// those); it only publishes those keys and vends the region-map + doorbell user
	// client. registerService lets Metal's IOServiceMatching("IOAccelerator")
	// notification see it, once the bundle is on disk ("The bundle", above).
	if (!super::start(provider))
		goto fail;
	// The accelerator's index, the one the framebuffer names it by (ReimsGraphics' IOAccelIndex):
	// IOPresentment reads it here, before registerService, to match the display to this GPU.
	setProperty(kIOAccelIndexKey, 0ull, 32);
	// Name this accelerator on the framebuffer, if it has started already (else the display
	// pipe's first association does).
	OSSafeReleaseNULL(fb);
	fb = copyFramebuffer();
	// Without the check (no bundle named, no memory for it) the accelerator stays unregistered,
	// as without the bundle.
	if (watchBundle())
		IOLog("RgfxAccelerator: ready, region 0x%llx bytes, framebuffer %s; registers once %s "
		      "is on disk\n",
		      regionSize, fb ? "named" : "not yet started", bundle.path);
	OSSafeReleaseNULL(fb);
	return true;
fail:
	teardown();
	return false;
}

void RgfxAccelerator::stop(IOService *provider)
{
	// The check ends: none runs after this.
	if (bundle.lock) {
		leaveWatching(kBundleSettled);
		thread_call_cancel_wait(bundle.call);
	}
	// The framebuffer stops calling queueInterrupt (and lets this accelerator go).
	if (ReimsGraphics *fb = copyFramebuffer()) {
		fb->setQueueInterrupt(nullptr, nullptr);
		fb->release();
	}
	super::stop(provider);
	teardown();
}

void RgfxAccelerator::teardown()
{
	OSSafeReleaseNULL(region);
	OSSafeReleaseNULL(registerMap);
	registers = nullptr;
}

void RgfxAccelerator::free()
{
	if (doorbellLock)
		IOSimpleLockFree(doorbellLock);
	doorbellLock = nullptr;
	super::free();
}

// MARK: - The user client

IOReturn RgfxAccelerator::newUserClient(task_t owningTask, void *securityID, UInt32 type,
                                        OSDictionary *properties, IOUserClient **handler)
{
	// The types served: the display pipe (4), the one the IOGPUMetalDevice base builds its
	// device over (5), and ours. Any other is refused, logged once per type. IOAccel surfaces
	// (0) are among them: CoreDisplay opens them at display start for its display surface, and
	// maps the framebuffer's memory for it when they are refused.
	RgfxCaller caller;
	IOLog("RGFX: newUserClient type=%u from %s[%d]\n", type, caller.name, caller.pid);
	// Unregistered, without its bundle, the accelerator is no GPU ("The bundle"): a task that
	// reaches it by path (the framebuffer's IOAccelTypes) is refused, logged once per type.
	UInt32 standing = bundleState();
	if (standing != kBundleRegistering && standing != kBundleRegistered) {
		static char site;
		if (rgfxFirstCalls(&site, type, 1))
			IOLog("RGFX: user client type %u from %s[%d] refused: the accelerator is not "
			      "registered\n",
			      type, caller.name, caller.pid);
		return kIOReturnNotReady;
	}
	if (type == kRgfxDisplayPipeUserClientType)
		return newDisplayPipe(owningTask, securityID, type, properties, handler);
	if (type != kRgfxAccelUserClientType && type != kRgfxIOGPUDeviceUserClientType) {
		static char site;
		if (rgfxFirstCalls(&site, type, 1))
			IOLog("RGFX: user client type %u from %s[%d] is not supported\n", type, caller.name,
			      caller.pid);
		return kIOReturnUnsupported;
	}

	RgfxAcceleratorUserClient *client = OSTypeAlloc(RgfxAcceleratorUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnBadArgument;
	}
	if (!client->attach(this)) {
		client->release();
		return kIOReturnInternalError;
	}
	if (!client->start(this)) {
		client->detach(this);
		client->release();
		return kIOReturnInternalError;
	}
	*handler = client;
	return kIOReturnSuccess;
}

// The display pipe WindowServer presents through (RgfxDisplayPipeUserClient).
IOReturn RgfxAccelerator::newDisplayPipe(task_t owningTask, void *securityID, UInt32 type,
                                         OSDictionary *properties, IOUserClient **handler)
{
	RgfxDisplayPipeUserClient *client = OSTypeAlloc(RgfxDisplayPipeUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnBadArgument;
	}
	if (!client->attach(this)) {
		client->release();
		return kIOReturnInternalError;
	}
	if (!client->start(this)) {
		client->detach(this);
		client->release();
		return kIOReturnInternalError;
	}
	*handler = client;
	return kIOReturnSuccess;
}

// An IOAccel surface (RgfxSurfaceUserClient).
IOReturn RgfxAccelerator::newSurface(task_t owningTask, void *securityID, UInt32 type,
                                     OSDictionary *properties, IOUserClient **handler)
{
	RgfxSurfaceUserClient *client = OSTypeAlloc(RgfxSurfaceUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnBadArgument;
	}
	if (!client->attach(this)) {
		client->release();
		return kIOReturnInternalError;
	}
	if (!client->start(this)) {
		client->detach(this);
		client->release();
		return kIOReturnInternalError;
	}
	*handler = client;
	return kIOReturnSuccess;
}

// MARK: - Reached from the user client

ReimsGraphics *RgfxAccelerator::copyFramebuffer()
{
	if (!pci)
		return nullptr;
	ReimsGraphics *found = nullptr;
	OSIterator *clients = pci->getClientIterator();
	if (clients) {
		while (OSObject *o = clients->getNextObject()) {
			found = OSDynamicCast(ReimsGraphics, o);
			if (found) {
				found->retain();
				break;
			}
		}
		clients->release();
	}
	if (found && !found->getProperty("IOAccelTypes")) {
		char path[512];
		int length = sizeof(path);
		if (getPath(path, &length, gIOServicePlane))
			found->setProperty("IOAccelTypes", path);
	}
	return found;
}

IODeviceMemory *RgfxAccelerator::copyRegion() const
{
	if (region)
		region->retain();
	return region;
}

void RgfxAccelerator::ringDoorbell(UInt32 ringId)
{
	if (!registers)
		return;
	IOInterruptState s = IOSimpleLockLockDisableInterrupt(doorbellLock);
	writeRegister(RGFX_REG_DOORBELL, ringId);
	IOSimpleLockUnlockEnableInterrupt(doorbellLock, s);
}

#if RGFX_FB_TRACE
// MARK: - Trace: the properties processes read

OSObject *RgfxAccelerator::copyProperty(const OSSymbol *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("accel", aKey ? aKey->getCStringNoCopy() : nullptr, o != nullptr);
	return o;
}

OSObject *RgfxAccelerator::copyProperty(const OSString *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("accel", aKey ? aKey->getCStringNoCopy() : nullptr, o != nullptr);
	return o;
}

OSObject *RgfxAccelerator::copyProperty(const char *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("accel", aKey, o != nullptr);
	return o;
}
#endif
