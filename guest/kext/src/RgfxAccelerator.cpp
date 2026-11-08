/*
 * RgfxAccelerator: see RgfxAccelerator.hpp for the service's role, the Metal
 * discovery keys it carries, and the boot-gated base-class decision.
 */

#include "RgfxAccelerator.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxDisplayPipeUserClient.hpp"
#include "RgfxGLUserClients.hpp"
#include "RgfxSurfaceUserClient.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/graphics/IOGraphicsInterfaceTypes.h>
#include <kern/clock.h>
#include <kern/thread_call.h>
#include <sys/vnode.h>

#include "RgfxBundlePaths.hpp"
#include "RgfxFeatures.hpp"
#include "RgfxTrace.hpp"
#include "RgfxVideoCodecs.hpp"

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
//
// The bundle must also be of the kext's wire version (design.md, "The wire's version"): it maps
// the region and speaks the messages. Its package carries an empty marker named for the version
// it was built with (RgfxBundlePaths.hpp), and only the executable with the marker of this
// kext's RGFX_VERSION counts as the bundle on disk. A bundle of another version is left as if
// there were none, so WindowServer composites in software: a bundle that refused to make a
// device under a registered accelerator would leave WindowServer aborting, as with no bundle.
//
// Where RgfxGate installs the bundle, nothing on disk counts while its install pass is under way
// (rgfxBundleRegisters): an update that keeps the wire version finds the old bundle and its
// marker before the pass drops the marker and rewrites the files. The pass publishes itself on
// IOResources (kRgfxGateInstallDoneKey), and the check looks again until it has ended. The
// settling connection waits up to kGatePassWaitMs for a pass under way; one still under way then
// leaves the accelerator unregistered for the boot. Without RgfxGate there is no property, and
// the check registers for what it finds.
namespace {
// How often the check repeats until the bundle is found or WindowServer settles it.
constexpr UInt32 kBundleCheckMs = 100;
// How long the settling connection waits for a registration in flight to finish.
constexpr UInt32 kBundleRegisterWaitMs = 5000;
// How long the settling connection waits for RgfxGate's install pass to end, and how often it
// looks.
constexpr UInt32 kGatePassWaitMs = 5000;
constexpr UInt32 kGatePassLookMs = 10;

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
	// The repeating check, and under lock whether it is running (stop waits for it to end).
	thread_call_t call;
	bool running;
	// The bundle's executable and its marker for this kext's wire version, NAME the
	// accelerator's MetalPluginName (the bundle's Makefile names its executable so).
	RgfxBundlePaths paths;
};
BundleWatch bundle = {nullptr, kBundleWatching, nullptr, false, {}};

// What the lookups find.
enum BundleFound : UInt32 {
	// No executable, or no root file system yet.
	kBundleAbsent,
	// The executable, without the marker of this kext's wire version.
	kBundleOtherVersion,
	// The executable and the marker: the bundle to register for.
	kBundleOurs,
};

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

// Whether `path` is there.
bool onDisk(const char *path, vfs_context_t context)
{
	vnode_t found = nullptr;
	errno_t error = vnode_lookup(path, 0, &found, context);
	if (!error)
		vnode_put(found);
	return !error;
}

// RgfxGate's install pass, from its property on IOResources (above).
RgfxGatePass gatePass()
{
	IOService *resources = IOService::getResourceService();
	OSObject *value = resources ? resources->copyProperty(kRgfxGateInstallDoneKey) : nullptr;
	OSBoolean *done = OSDynamicCast(OSBoolean, value);
	RgfxGatePass pass = !done ? kRgfxGateAbsent
	                          : done->isTrue() ? kRgfxGateDone : kRgfxGateInstalling;
	if (value)
		value->release();
	return pass;
}

// Which bundle is on disk (above); none before the root file system is mounted.
BundleFound bundleOnDisk()
{
	vnode_t root = vfs_rootvnode();
	if (!root)
		return kBundleAbsent;
	vnode_put(root);
	vfs_context_t context = vfs_context_create(nullptr);
	if (!context)
		return kBundleAbsent;
	BundleFound found = kBundleAbsent;
	if (onDisk(bundle.paths.executable, context))
		found = onDisk(bundle.paths.marker, context) ? kBundleOurs : kBundleOtherVersion;
	vfs_context_rele(context);
	return found;
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
	IOLog("RgfxAccelerator: the bundle is at %s%s: registered\n", bundle.paths.executable, when);
	accelerator->registerService(kIOServiceSynchronous);
	IOLockLock(bundle.lock);
	bundle.state = kBundleRegistered;
	IOLockUnlock(bundle.lock);
}

// The repeating check, on a thread call: registers the accelerator once the bundle is there.
void checkBundle(thread_call_param_t accelerator, thread_call_param_t)
{
	IOLockLock(bundle.lock);
	bundle.running = true;
	IOLockUnlock(bundle.lock);
	// The pass first: once it has ended, the disk holds what it left.
	const RgfxGatePass gate = gatePass();
	bool found = rgfxBundleRegisters(gate, gate != kRgfxGateInstalling &&
	                                           bundleOnDisk() == kBundleOurs);
	IOLockLock(bundle.lock);
	bool watching = bundle.state == kBundleWatching;
	if (watching && found)
		bundle.state = kBundleRegistering;
	else if (watching)
		armBundleCheck();
	IOLockUnlock(bundle.lock);
	if (watching && found)
		registerAccelerator(static_cast<RgfxAccelerator *>(accelerator), "");
	IOLockLock(bundle.lock);
	bundle.running = false;
	IOLockWakeup(bundle.lock, &bundle.running, false);
	IOLockUnlock(bundle.lock);
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
	if (!rgfxBundlePaths(n, RGFX_VERSION, &bundle.paths)) {
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
		RgfxGatePass gate = gatePass();
		for (UInt32 waited = 0; gate == kRgfxGateInstalling && waited < kGatePassWaitMs;
		     waited += kGatePassLookMs) {
			IOSleep(kGatePassLookMs);
			gate = gatePass();
		}
		BundleFound found = gate == kRgfxGateInstalling ? kBundleAbsent : bundleOnDisk();
		if (rgfxBundleRegisters(gate, found == kBundleOurs)) {
			if (leaveWatching(kBundleRegistering))
				registerAccelerator(this, " as WindowServer opens the display");
		} else if (leaveWatching(kBundleSettled)) {
			if (gate == kRgfxGateInstalling)
				IOLog("RgfxAccelerator: RgfxGate's install pass has not ended as WindowServer "
				      "opens the display: not registered this boot\n");
			else if (found == kBundleOtherVersion)
				IOLog("RgfxAccelerator: the bundle at %s is not of wire version %u (no %s) as "
				      "WindowServer opens the display: not registered this boot. The bundle, the "
				      "kext and the host must come from one build.\n",
				      bundle.paths.executable, RGFX_VERSION, bundle.paths.marker);
			else
				IOLog("RgfxAccelerator: no bundle at %s as WindowServer opens the display: not "
				      "registered this boot\n",
				      bundle.paths.executable);
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

uint64_t RgfxAccelerator::featureWord() const
{
	const uint64_t offered = (static_cast<uint64_t>(readRegister(RGFX_REG_FEATURES_HI)) << 32) |
	                         readRegister(RGFX_REG_FEATURES_LO);
	const uint64_t taken = (static_cast<uint64_t>(readRegister(RGFX_REG_DRIVER_FEATURES_HI)) << 32) |
	                       readRegister(RGFX_REG_DRIVER_FEATURES_LO);
	return rgfxFeatureWord(offered, taken);
}

// MARK: - IOService

bool RgfxAccelerator::start(IOService *provider)
{
	ReimsGraphics *fb = nullptr;
	pci = OSDynamicCast(IOPCIDevice, provider);
	if (!pci)
		return false;
	// The framebuffer enables these too; doing so again is harmless. The
	// accelerator needs BAR 0 for the doorbell and BAR 2 to hand to the bundle.
	pci->setMemoryEnable(true);
	// setBusMasterEnable rather than setBusLeadEnable, which macOS 11 does not export: a call
	// through IOPCIDevice's vtable, the same slot on macOS 11 to 26 (ReimsGraphics::start).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	pci->setBusMasterEnable(true);
#pragma clang diagnostic pop

	registerMap = pci->mapDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0);
	if (!registerMap) {
		IOLog("RgfxAccelerator: BAR0 not mapped\n");
		goto fail;
	}
	registers = reinterpret_cast<volatile UInt8 *>(registerMap->getVirtualAddress());
	if (readRegister(RGFX_REG_MAGIC) != RGFX_MAGIC) {
		IOLog("RgfxAccelerator: not the device: magic 0x%x\n", readRegister(RGFX_REG_MAGIC));
		goto fail;
	}
	// The wire's version names the region's layout and the messages (design.md, "The wire's
	// version"): a kext of another version misreads them, so it leaves the device.
	if (readRegister(RGFX_REG_VERSION) != RGFX_VERSION) {
		IOLog("RgfxAccelerator: the device speaks wire version %u, this kext version %u: not starting. "
		      "The kext and the host must come from one build.\n",
		      readRegister(RGFX_REG_VERSION), RGFX_VERSION);
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
	// The GPU's memory, as a Mac's GPU states it (RgfxMemorySize.hpp): VRAM,totalMB on the PCI
	// device, where system_profiler reads a discrete GPU's, and on the accelerator, where an
	// Intel GPU states it. A host that gave no working set has none stated.
	{
		RgfxMemorySize memory;
		if (readMemorySize(&memory) && memory.vramMB) {
			pci->setProperty("VRAM,totalMB", memory.vramMB, 32);
			setProperty("VRAM,totalMB", memory.vramMB, 32);
		}
	}
	// The GPU's name, as a Mac's GPU states it (RgfxDeviceName.hpp): `model` on the PCI device,
	// which System Information shows as its Chipset Model. A host that gave no name has none
	// stated.
	{
		uint8_t model[kRgfxDeviceModelMax];
		const size_t length = readDeviceModel(model);
		if (OSData *data = length ? OSData::withBytes(model, static_cast<unsigned>(length)) : nullptr) {
			pci->setProperty("model", data);
			data->release();
		}
	}
	nameProviderForVideo();
	// AppleGVA reads the video codecs off this service once Metal's matching sees it, after
	// registerService.
	publishVideoCodecs();
	// Name this accelerator on the framebuffer, if it has started already (else the display
	// pipe's first association does).
	OSSafeReleaseNULL(fb);
	fb = copyFramebuffer();
	// Without the check (no bundle named, no memory for it) the accelerator stays unregistered,
	// as without the bundle.
	if (watchBundle())
		IOLog("RgfxAccelerator: ready, region 0x%llx bytes, framebuffer %s; registers once %s "
		      "and %s are on disk\n",
		      regionSize, fb ? "named" : "not yet started", bundle.paths.executable,
		      bundle.paths.marker);
	OSSafeReleaseNULL(fb);
	return true;
fail:
	teardown();
	return false;
}

void RgfxAccelerator::stop(IOService *provider)
{
	// The check ends: a pending one is cancelled, a running one (registerService included)
	// finishes first, and one that starts after finds the watch settled and does nothing.
	if (bundle.lock) {
		leaveWatching(kBundleSettled);
		IOLockLock(bundle.lock);
		while (bundle.running)
			IOLockSleep(bundle.lock, &bundle.running, THREAD_UNINT);
		IOLockUnlock(bundle.lock);
		thread_call_free(bundle.call);
	}
	// The framebuffer stops calling queueInterrupt and the release hook (and lets this
	// accelerator go).
	if (ReimsGraphics *fb = copyFramebuffer()) {
		fb->setQueueInterrupt(nullptr, nullptr);
		fb->setReleaseAnswer(nullptr, nullptr);
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


// MARK: - The user client

IOReturn RgfxAccelerator::newUserClient(task_t owningTask, void *securityID, UInt32 type,
                                        OSDictionary *properties, IOUserClient **handler)
{
	// The types served: the display pipe (4), the one the IOGPUMetalDevice base builds its
	// device over (5), ours, a shared event's port (RgfxSharedEventUserClient), and a GL
	// window's drawable (7) and surface (0x20) (RgfxGLUserClients.hpp). Any other is refused,
	// logged once per type. IOAccel surfaces (0) are among them: CoreDisplay opens them at
	// display start for its display surface, and maps the framebuffer's memory for it when
	// they are refused.
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
	if (type == kRgfxSharedEventUserClientType)
		return newSharedEventPort(owningTask, securityID, type, properties, handler);
	if (type == kRgfxGLDrawableUserClientType || type == kRgfxGLSurfaceUserClientType)
		return newGLClient(owningTask, securityID, type, properties, handler);
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

// A GL window's drawable or surface (RgfxGLUserClients.hpp).
IOReturn RgfxAccelerator::newGLClient(task_t owningTask, void *securityID, UInt32 type,
                                      OSDictionary *properties, IOUserClient **handler)
{
	IOUserClient *client = nullptr;
	if (type == kRgfxGLDrawableUserClientType)
		client = OSTypeAlloc(RgfxGLDrawableUserClient);
	else
		client = OSTypeAlloc(RgfxGLSurfaceUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnNoResources;
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

bool RgfxAccelerator::readMemorySize(RgfxMemorySize *out) const
{
	if (!region)
		return false;
	// The device info's pages that hold the two answers, mapped into the kernel write-back as
	// the user client's device info is: the region is BAR memory, which a physical copy does not
	// read.
	const UInt64 first = RGFX_DEVICE_INFO_OFFSET + kRgfxMemorySizeFirst;
	const UInt64 page = first & ~0xfffull;
	const UInt64 length =
	    ((RGFX_DEVICE_INFO_OFFSET + kRgfxMemorySizeEnd + 0xfffull) & ~0xfffull) - page;
	IOMemoryMap *map = region->createMappingInTask(
	    kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache | kIOMapReadOnly, page, length);
	if (!map)
		return false;
	*out = rgfxMemorySizeFrom(reinterpret_cast<const uint8_t *>(map->getVirtualAddress()) +
	                          (first - page));
	map->release();
	return true;
}

size_t RgfxAccelerator::readDeviceModel(uint8_t (&out)[kRgfxDeviceModelMax]) const
{
	const void *info = nullptr;
	IOMemoryMap *map = mapDeviceInfo(kRgfxDeviceNameEnd, &info);
	if (!map)
		return 0;
	const size_t length =
	    rgfxDeviceModelFrom(static_cast<const uint8_t *>(info) + kRgfxDeviceNameFirst, out);
	map->release();
	return length;
}

IOMemoryMap *RgfxAccelerator::mapDeviceInfo(UInt64 size, const void **info) const
{
	if (!region)
		return nullptr;
	// Its pages, mapped into the kernel write-back as the framebuffer maps the rings: the
	// region is BAR memory, which a physical copy does not read.
	const UInt64 page = RGFX_DEVICE_INFO_OFFSET & ~0xfffull;
	const UInt64 length = ((RGFX_DEVICE_INFO_OFFSET + size + 0xfffull) & ~0xfffull) - page;
	IOMemoryMap *map = region->createMappingInTask(
	    kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache | kIOMapReadOnly, page, length);
	if (map)
		*info = reinterpret_cast<const UInt8 *>(map->getVirtualAddress()) +
		        (RGFX_DEVICE_INFO_OFFSET - page);
	return map;
}

void RgfxAccelerator::nameProviderForVideo()
{
	// AppleGVA's hardware video routes to its vendor driver through the GPU it finds by the
	// name real AMD Macs give the GPU's PCI device, GFX0 (on an iMac20,1 with
	// forceOfflineRenderer it looks for GFX0/IGPU). HEVC decode needs it: without it
	// VTDecompressionSessionCreate refuses every HEVC stream with -12913 and our driver is
	// never called, while H.264 decodes all the same and the IOGVACodec name makes no
	// difference (measured on the msi 5080, 2026-11-03: three boots with the provider's
	// default name refused HEVC, a fourth that only named it GFX0 decoded every frame). So
	// the provider is named GFX0, as the hardware does, before registerService lets any
	// AppleGVA lookup see it. Where another device already carries the name (a renamer like
	// WhateverGreen, or a second GPU), the name is left as it was, so two GFX0s are never
	// published.
	if (OSDictionary *match = IOService::nameMatching("GFX0")) {
		IOService *existing = IOService::copyMatchingService(match);
		match->release();
		if (existing) {
			IOLog("RgfxAccelerator: a device is already named GFX0; leaving the provider named %s\n",
			      pci->getName());
			existing->release();
			return;
		}
	}
	pci->setName("GFX0");
}

void RgfxAccelerator::publishVideoCodecs()
{
	static_assert(RGFX_VIDEO_CAPS_OFFSET == RGFX_DEVICE_INFO_OFFSET + sizeof(RgfxDeviceInfo),
	              "the video caps follow the device info");
	const void *info = nullptr;
	IOMemoryMap *map = mapDeviceInfo(sizeof(RgfxDeviceInfo) + sizeof(RgfxVideoCaps), &info);
	if (!map)
		return;
	RgfxVideoCaps caps;
	memcpy(&caps, reinterpret_cast<const UInt8 *>(info) + sizeof(RgfxDeviceInfo), sizeof(caps));
	map->release();
	RgfxHevcDecode hevc;
	if (!rgfxHevcDecode(caps, &hevc))
		return;
	OSArray *profiles = OSArray::withCapacity(hevc.count);
	OSDictionary *details = OSDictionary::withCapacity(hevc.count);
	OSDictionary *capabilities = OSDictionary::withCapacity(2);
	OSNumber *level = OSNumber::withNumber(hevc.maxLevel, 32);
	OSDictionary *profileDetail = OSDictionary::withCapacity(1);
	bool made = profiles && details && capabilities && level && profileDetail &&
	            profileDetail->setObject("VTMaxDecodeLevel", level);
	for (uint32_t i = 0; made && i < hevc.count; i++) {
		char key[4];
		snprintf(key, sizeof key, "%u", hevc.profiles[i]);
		OSNumber *profile = OSNumber::withNumber(hevc.profiles[i], 32);
		made = profile && profiles->setObject(profile) && details->setObject(key, profileDetail);
		OSSafeReleaseNULL(profile);
	}
	if (made && capabilities->setObject("VTSupportedProfileArray", profiles) &&
	    capabilities->setObject("VTPerProfileDetails", details)) {
		setProperty("IOGVAHEVCDecode", "1");
		setProperty("IOGVAHEVCDecodeCapabilities", capabilities);
		IOLog("RgfxAccelerator: HEVC decode, %u profiles, level %u\n", hevc.count, hevc.maxLevel);
	}
	OSSafeReleaseNULL(profileDetail);
	OSSafeReleaseNULL(level);
	OSSafeReleaseNULL(capabilities);
	OSSafeReleaseNULL(details);
	OSSafeReleaseNULL(profiles);
}

// A queue's doorbell exits to the host, which may offer the ring's next command buffer to its
// decoder before the write returns: the write holds no lock and leaves interrupts as they are, so
// the CPU takes its IPIs (a TLB shootdown's among them) however long the host takes, and the
// other CPUs' rings go to the host meanwhile. A ring id is one 32-bit write, which needs no lock.
void RgfxAccelerator::ringDoorbell(UInt32 ringId)
{
	if (!registers)
		return;
	writeRegister(RGFX_REG_DOORBELL, ringId);
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
