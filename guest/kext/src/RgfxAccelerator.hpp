/*
 * RgfxAccelerator: the reims-graphics device's Metal accelerator for x86-64 macOS
 * guests (docs/graphics-kext/design.md, "Stage 3 guest stack"). It is a SECOND
 * IOService on the same PCI device as the Stage-2 framebuffer (ReimsGraphics):
 * the framebuffer matches in category IOFramebuffer, this matches in category
 * IOAccelerator, and both are children of the one IOPCIDevice (the notes require
 * the accelerator and the framebuffer to share a provider).
 *
 * What it is for: Metal does not scan /Library/GPUBundles. It loads our guest GPU
 * bundle from two properties on THIS service's IOKit personality — MetalPluginName
 * (the bundle path) and MetalPluginClassName (the Obj-C class Metal instantiates as
 * the MTLDevice, RgfxMetalDevice) — which it finds because the service matches
 * IOServiceMatching("IOAccelerator") under IOMatchCategory IOAccelerator. Those two
 * keys live in src/Info.plist's accelerator personality (stage3-apple-metal-iface.md
 * section 1). This service then exposes, over OUR protocol (not Apple's IOGPU
 * user-client ABI), the two things the bundle's transport needs:
 *   - the process's part of the device's BAR-2 region, mapped into the Metal client's
 *     task in its window (RgfxAcceleratorUserClient.hpp), so the bundle's RgfxRegion /
 *     RgfxRingWriter write into real device memory, and
 *   - the doorbell: a write of a ring id to RGFX_REG_DOORBELL in BAR 0, to wake the
 *     host's consumer of a queue's submission ring (ringDoorbell, reached from the
 *     user client's external method).
 *
 * ========================= BOOT-GATED DECISION (UNKNOWN #1) =========================
 * BASE CLASS / KERNEL FAMILY. stage3-apple-metal-iface.md section 3 and
 * docs/graphics-kext/design.md Decided 15 record that our guest runs the ported
 * IOGPUFamily stack, that IOAcceleratorFamily2 cannot start in a kernel where
 * IOGPUFamily runs, and that our accelerator therefore aligns with IOGPU. The exact
 * IOGPU kernel base class (an IOGPU, itself an IOAccelerator) and its submission /
 * allocation user-client ABI are Apple-private and are NOT in the macOS SDK or the
 * KDK, so they cannot be subclassed off the guest — only a guest boot + IORegistry +
 * user-client trace settles them (notes UNKNOWN #1; design.md Open).
 *
 * This scaffold subclasses the base the SDK DOES expose, IOAccelerator
 * (IOKit/graphics/IOAccelerator.h), which is what Metal's IOServiceMatching(
 * "IOAccelerator") enumeration matches. That is enough to publish the Metal
 * discovery keys and to stand up OUR own region-map + doorbell user client, which is
 * all that compiles offline. On the guest the base very likely becomes the IOGPU
 * family class; the Metal keys, the PCI match, and our user-client protocol below are
 * expected to carry over unchanged. DO NOT fabricate the IOGPU base or its selectors
 * here.
 * ===================================================================================
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_ACCELERATOR_HPP_
#define RGFX_ACCELERATOR_HPP_

#include <IOKit/IOLib.h>
#include <IOKit/graphics/IOAccelerator.h>
#include <IOKit/pci/IOPCIDevice.h>

#include "RgfxBlockTable.hpp"
#include "reims_graphics_wire.h"

// OUR user-client protocol (not Apple's IOGPU ABI). The bundle's wire handshake opens the
// accelerator with IOServiceOpen(acceleratorPort, task, kRgfxAccelUserClientType, &connect),
// reads the device info (kRgfxAccelMethodDeviceInfo), opens its context and window
// (kRgfxAccelMethodOpenContext), and rings a ring's doorbell with
// IOConnectCallScalarMethod(kRgfxAccelMethodDoorbell, {ringId}).

// Our own user-client type: a private one ('RGFX'), clear of the public IOAccel surface type 0
// (kRgfxSurfaceUserClientType) and of the types Apple's frameworks open.
static const UInt32 kRgfxAccelUserClientType = 0x52474658;
// The IOAccel surface client, the public kIOAccelSurfaceClientType (RgfxSurfaceUserClient).
static const UInt32 kRgfxSurfaceUserClientType = 0;
// The type Apple's IOGPUMetalDevice base opens to build its device
// (RgfxAcceleratorUserClient::deviceMethod).
static const UInt32 kRgfxIOGPUDeviceUserClientType = 5;

// clientMemoryForType ids: the whole BAR-2 device region, which is refused
// (kIOReturnUnsupported): the region holds every process's blocks, the rings and the device
// page, and a process maps only its own part of it, in its window.
enum {
	kRgfxAccelMemoryRegion = 0,
};

// External-method selectors the user client dispatches.
enum {
	// Ring a ring's doorbell: scalarInput[0] is the ring id, written to
	// RGFX_REG_DOORBELL. No output.
	kRgfxAccelMethodDoorbell = 0,
	kRgfxAccelMethodCount = 1,
};

// The process's device context (RgfxAcceleratorContexts.cpp), on our own user client (type
// 0) only, numbered clear of the selectors Apple's IOGPU base drives on its type-0 client.
enum {
	// Opens the client's context and its window. No input; out: the context number c, the
	// object-id base (c << RGFX_CONTEXT_OBJ_ID_SHIFT), the GPU-address base of the context's
	// slot, and the window's base in the process and its size (the region's): region offset o
	// of what the client is given lies at the base plus o, and nothing else of the region is
	// mapped there (RgfxAcceleratorUserClient.hpp).
	kRgfxAccelMethodOpenContext = 0x100,
	// A region block: in size, align (a power of two); out: its region offset.
	kRgfxAccelMethodAlloc = 0x101,
	// Frees the client's block at the region offset scalarInput[0], and returns at once: the
	// block is no longer the client's to name, and is given out again once the device answers
	// its BLOCK_RELEASE OK (RgfxBlockTable.hpp). kIOReturnBusy while a create naming it is in
	// flight.
	kRgfxAccelMethodFree = 0x102,
	// Opens a command queue: in obj_id, submit offset, complete offset, submit data size,
	// complete data size, priority; out: the device's status (RGFX_STATUS_OK, or the
	// refusal). Each ring must lie in one of the client's blocks. The device has the queue's
	// rings when it returns.
	kRgfxAccelMethodQueueCreate = 0x103,
	// Closes the client's queue scalarInput[0].
	kRgfxAccelMethodQueueDestroy = 0x104,
	// Posts one object-lifecycle message on the control ring: struct in, the whole message
	// (its header's type set; the kext stamps its size and seqno), a create or destroy of an
	// object in the client's context (its obj_id, 16 bytes in, in the context's range), at most
	// ReimsGraphics::kLifecycleMessageMax bytes, naming only the client's own blocks of the region
	// (RgfxRegionSpans.hpp); out: the seqno, and for a create the device answers (a pipeline's,
	// a library's), its answer as struct out once the device has given it. A create not answered
	// in time fails, and the kext destroys its object behind it.
	kRgfxAccelMethodSubmit = 0x105,
	// Gives the client's context a chunk of RGFX_RESOURCE_ID_CHUNK texture resource ids, from
	// the guest's one pool of RGFX_RESOURCE_ID_CHUNKS: no input; out: the chunk's first id.
	// kIOReturnNoResources once every chunk is held. The device has the grant
	// (RESOURCE_ID_CHUNK) before it returns; the context holds the chunk until its close.
	kRgfxAccelMethodResourceIds = 0x106,
	// Waits for the client's command buffers to complete: in the last count this returned (0
	// at first); out: the count now. Returns once the count is not the one given, which it
	// becomes when a completion ring of one of the client's queues holds records the client
	// has not taken (the device's interrupt, RgfxAccelerator::queueInterrupt). kIOReturnAborted
	// when the thread is interrupted, kIOReturnNotOpen once the client's context is released.
	kRgfxAccelMethodWaitCompletions = 0x107,
	// The scanout an IOAccel surface of the client's process is drawn into, for a texture over
	// it: in the surface's id and its buffer (0, the one buffer); out its region offset, its
	// bytes, its width, height and bytes per row (BGRA8 pixels). The surface must be the
	// framebuffer's whole scanout (RgfxSurfaceUserClient::isScanout); from then on the client's
	// lifecycle messages may name the span, until its context is released or the surface goes.
	kRgfxAccelMethodSurfaceScanout = 0x108,
	// The shared block of an IOSurface plane the client's process holds (RgfxSurfaceBlocks.hpp):
	// in the plane's CPU address in the process, its bytes there, and the block's size (whole
	// pages, at least those bytes); out the block's region offset, its size, its serial (unique
	// to the block) and 1 where this ask made it (its bytes are not yet the plane's), else 0.
	// Every process that holds the surface gets the same block for the plane, and its lifecycle
	// messages may name it from then on, until it lets go of its last ask. The device has the
	// hold (BLOCK_SHARE) before this returns. kIOReturnBadArgument where the range is not mapped
	// in the process, or the plane's block has another size.
	kRgfxAccelMethodSurfaceBlock = 0x109,
	// Lets go of one ask of the shared block at the region offset scalarInput[0]; the block goes
	// once no process holds it, through its BLOCK_RELEASE.
	kRgfxAccelMethodSurfaceBlockRelease = 0x10a,
	// The device info the host wrote at RGFX_DEVICE_INFO_OFFSET (RgfxDeviceInfo): no input;
	// struct out, exactly sizeof(RgfxDeviceInfo) bytes. Its page holds the control completion
	// ring's tail, so no process maps it; the kext reads it for the process.
	kRgfxAccelMethodDeviceInfo = 0x10b,
};

class RgfxAcceleratorUserClient;
class RgfxSurfaceUserClient;
class ReimsGraphics;

// A command queue's completion ring the accelerator watches for its client
// (RgfxAcceleratorContexts.cpp): the ring's header, mapped into the kernel, and the producer
// count last seen there.
struct RgfxWatchedRing {
	RgfxAcceleratorUserClient *client;
	UInt64 objId;
	IOMemoryMap *map;
	volatile UInt8 *header;
	UInt64 lastProducer;
};

class RgfxAccelerator : public IOAccelerator {
	OSDeclareDefaultStructors(RgfxAccelerator);

public:
	// IOService
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	void free() override;

	// Our own user client: the region-map + doorbell transport the bundle opens.
	IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type,
	                       OSDictionary *properties, IOUserClient **handler) override;

	// Reached from the user client. The BAR-2 device region to map into the Metal
	// client's task (retained for the caller), and the doorbell write that wakes the
	// host's consumer of ring `ringId`.
	IODeviceMemory *copyRegion() const;
	void ringDoorbell(UInt32 ringId);
	// BAR 2's size, as the device's REGION_SIZE registers give it (no more than the BAR's).
	UInt64 getRegionSize() const { return regionSize; }

	// The framebuffer on our PCI device (ReimsGraphics), retained for the caller, or nullptr
	// before it has started. The first lookup that finds it names this accelerator on it
	// (IOAccelTypes), as a display pipe's accelerator does.
	ReimsGraphics *copyFramebuffer();

	// The processes' contexts (RgfxAcceleratorContexts.cpp). A context number is handed out once
	// (from 1); a slot of the GPU-address space is held until the close.
	IOReturn openContext(UInt32 *context, UInt32 *slot);
	// True where the device answered the close OK: it holds nothing of the context, and every
	// command buffer of it has finished.
	bool closeContext(UInt32 context, UInt32 slot);
	// Chunks of the texture resource ids, each owned by `owner` for context `context` and posted
	// to the device (RESOURCE_ID_CHUNK) before takeIdChunk returns its first id; the owner's are
	// freed after its context's close is answered.
	IOReturn takeIdChunk(const void *owner, UInt32 context, UInt64 *firstId);
	void freeIdChunks(const void *owner);
	// Blocks of the region past RGFX_CONTEXT_ALLOCATIONS_OFFSET (RgfxBlockTable.hpp), each owned
	// by `owner` of context `context`. freeBlock returns at once; the block is given out again
	// once the device answers its BLOCK_RELEASE OK, posted from a thread of the kext's own, and
	// is kIOReturnBusy while a create naming it is in flight (pinRange). freeBlocks frees the
	// owner's blocks after its close is answered; keepBlocks keeps them from use for good after
	// a close that was not.
	// allocBlock maps the block into the client's window before it returns; freeBlock takes it
	// back from the window before the block is freed.
	IOReturn allocBlock(RgfxAcceleratorUserClient *owner, UInt32 context, UInt64 size,
	                    UInt64 align, UInt64 *offset);
	IOReturn freeBlock(RgfxAcceleratorUserClient *owner, UInt64 offset);
	void freeBlocks(const void *owner);
	void keepBlocks(const void *owner);
	// Pins the owner's live block that holds the `length` bytes at `offset`, for a create that
	// names them, until unpinBlock: false where no block of the owner's does.
	bool pinRange(const void *owner, UInt64 offset, UInt64 length, RgfxBlockPin *pin);
	void unpinBlock(const RgfxBlockPin &pin);
	// The shared blocks of IOSurface planes (RgfxSurfaceBlocks.hpp), for our user client `client`
	// of context `context` in `task`: the block of the plane of `length` bytes at `address` in
	// the task, of `size` bytes (kRgfxAccelMethodSurfaceBlock), and letting go of one ask of
	// it. dropSurfaceBlocks lets go of every one of the client's after its close, the device
	// answered (`closed`) or not; pinRange pins a span of one the client holds. Each holder has
	// the block mapped in its window from its first ask until its hold ends, which takes it
	// back before the block can go to anyone else.
	struct SurfaceBlock {
		UInt64 offset;
		UInt64 size;
		UInt64 serial;
		bool fresh;
	};
	IOReturn acquireSurfaceBlock(RgfxAcceleratorUserClient *client, UInt32 context, task_t task,
	                             UInt64 address, UInt64 length, UInt64 size,
	                             SurfaceBlock *block);
	IOReturn releaseSurfaceBlock(RgfxAcceleratorUserClient *client, UInt64 offset);
	void dropSurfaceBlocks(const void *client, bool closed);
	// The processes' object lifecycle, on the framebuffer's control ring
	// (ReimsGraphics::postLifecycle). openQueue's QUEUE_CREATE is answered once the device has
	// the queue's rings; closeContext's CONTEXT_CLOSE once the device has dropped the context's
	// objects.
	IOReturn postLifecycle(UInt32 type, const void *message, UInt32 size, UInt32 replyType,
	                       void *reply, UInt32 replyCapacity, UInt32 *replyLength, UInt64 *seqno,
	                       UInt32 timeoutMs);
	UInt32 openQueue(const RgfxQueueCreate &create);
	void closeQueue(UInt64 objId);
	// The completion rings of the clients' queues: watchQueue maps queue `objId`'s ring header
	// at region offset `completeOffset`, and from then on each interrupt of the device's vector
	// that finds records the client has not taken there wakes the client's completion waiters
	// (queueInterrupt, on the framebuffer's work loop). unwatchQueue and unwatchClient stop it.
	IOReturn watchQueue(RgfxAcceleratorUserClient *client, UInt64 objId, UInt64 completeOffset);
	void unwatchQueue(RgfxAcceleratorUserClient *client, UInt64 objId);
	void unwatchClient(RgfxAcceleratorUserClient *client);
	void queueInterrupt();

	// The IOAccel surfaces (RgfxAcceleratorSurfaces.cpp). registerSurface names `surface`'s id
	// (kIOReturnExclusiveAccess where another surface holds it); unregisterSurface drops it and
	// the grants made for it.
	IOReturn registerSurface(RgfxSurfaceUserClient *surface, UInt32 surfaceId);
	void unregisterSurface(RgfxSurfaceUserClient *surface);
	// The framebuffer's scanout as it is now: its place and bytes in the region, its shape.
	struct Scanout {
		UInt64 offset;
		UInt64 length;
		UInt32 width;
		UInt32 height;
		UInt32 rowBytes;
	};
	bool currentScanout(Scanout *scanout);
	// For our user client `client` in `task`, of context `context`: the scanout surface
	// `surfaceId` of that task is drawn into, whose span `client` may name from then on
	// (grantedRange) until revokeGrants or the surface goes. The device is told the span is
	// `context`'s (BLOCK_GRANT) before this returns, and the span is mapped in the client's
	// window while the grant lasts.
	IOReturn grantSurfaceScanout(RgfxAcceleratorUserClient *client, UInt32 context, task_t task,
	                             UInt32 surfaceId, Scanout *scanout);
	void revokeGrants(const void *client);
	bool grantedRange(const void *client, UInt64 offset, UInt64 length);

#if RGFX_FB_TRACE
	// A trace build logs the properties processes read of the accelerator.
	using IOAccelerator::copyProperty;
	OSObject *copyProperty(const OSSymbol *aKey) const override;
	OSObject *copyProperty(const OSString *aKey) const override;
	OSObject *copyProperty(const char *aKey) const override;
#endif

private:
	IOReturn newDisplayPipe(task_t owningTask, void *securityID, UInt32 type,
	                        OSDictionary *properties, IOUserClient **handler);
	IOReturn newSurface(task_t owningTask, void *securityID, UInt32 type,
	                    OSDictionary *properties, IOUserClient **handler);
	UInt32 readRegister(UInt32 offset) const;
	void writeRegister(UInt32 offset, UInt32 value);
	void teardown();

	IOPCIDevice *pci;
	IOMemoryMap *registerMap;
	volatile UInt8 *registers;
	IODeviceMemory *region;
	UInt64 regionSize;

	// Serialises doorbell writes (Metal rings from app threads).
	IOSimpleLock *doorbellLock;

	// The contexts' state lives in RgfxAcceleratorContexts.cpp, not in this class, so the
	// class's layout and its own code stay as the loader's sites were taken from; it is made
	// at the first context's open.
	bool startContexts();
};

#endif // RGFX_ACCELERATOR_HPP_
