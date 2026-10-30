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
 *   - the device BAR-2 region, mapped into the Metal client's task, so the bundle's
 *     RgfxRegion / RgfxRingWriter write into real device memory (clientMemoryForType
 *     on RgfxAcceleratorUserClient), and
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

// OUR user-client protocol (not Apple's IOGPU ABI). The bundle's wire handshake
// (a boot-gated stub in RgfxMetalDevice today) opens the accelerator with
// IOServiceOpen(acceleratorPort, task, kRgfxAccelUserClientType, &connect), maps the
// region with IOConnectMapMemory(kRgfxAccelMemoryRegion), and rings a ring's doorbell
// with IOConnectCallScalarMethod(kRgfxAccelMethodDoorbell, {ringId}).

// The only user-client type this accelerator vends.
static const UInt32 kRgfxAccelUserClientType = 0;

// clientMemoryForType ids: the whole BAR-2 device region (control rings, caps page
// and the allocations past RGFX_ALLOCATIONS_OFFSET), mapped into the client task.
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
	// Opens the client's context. No input; out: the context number c, the object-id base
	// (c << RGFX_CONTEXT_OBJ_ID_SHIFT) and the GPU-address base of the context's slot.
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
};

class RgfxAcceleratorUserClient;
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
	IOReturn allocBlock(const void *owner, UInt32 context, UInt64 size, UInt64 align,
	                    UInt64 *offset);
	IOReturn freeBlock(const void *owner, UInt64 offset);
	void freeBlocks(const void *owner);
	void keepBlocks(const void *owner);
	// Pins the owner's live block that holds the `length` bytes at `offset`, for a create that
	// names them, until unpinBlock: false where no block of the owner's does.
	bool pinRange(const void *owner, UInt64 offset, UInt64 length, RgfxBlockPin *pin);
	void unpinBlock(const RgfxBlockPin &pin);
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
