/*
 * RgfxSurfaceUserClient: an IOAccel surface, user-client type 0 on the accelerator
 * (kIOAccelSurfaceClientType, IOAccelClientConnect.h). IOKit's IOAccelCreateSurface opens one
 * connection per surface and drives the surface methods of IOAccelSurfaceConnect.h on it.
 * WindowServer makes its displays' surfaces this way, and its compositors draw the display
 * into the one on the framebuffer: the software compositor into the memory a lock gives, the
 * Metal compositor into the texture the bundle makes over the surface
 * (-newTextureWithSurface:buffer:, through kRgfxAccelMethodSurfaceScanout).
 *
 * The methods answered, with the shapes WindowServer was measured to use:
 *   7   SetIDMode(id, mode bits)                               scalar 2
 *   0x11 SetShapeBackingAndLength(options, framebuffer index,
 *        backing, row bytes, backing length) + the region      scalar 5, struct in
 *   9   SetShape(options, framebuffer index) + the region      scalar 2, struct in
 *   0, 3 ReadLockOptions, WriteLockOptions(options)
 *        -> IOAccelSurfaceInformation                         scalar 1, struct out
 *   1, 4 ReadUnlockOptions, WriteUnlockOptions(options)       scalar 1
 * Every other method, and any other shape, is refused as unsupported, logged once per method.
 *
 * A surface's memory is:
 *   - the backing the client gave (SetShapeBackingAndLength with a backing), its own memory;
 *   - else, for a surface on framebuffer 0 whose shape is the display's whole current mode,
 *     the framebuffer's scanout in the device region, mapped into the client's task at its
 *     first lock: what is drawn there is what the display shows;
 *   - else none, and a lock fails with kIOReturnNoMemory.
 *
 * Our own code; it names only SDK-public IOKit classes and the SDK's IOAccel headers.
 */

#ifndef RGFX_SURFACE_USER_CLIENT_HPP_
#define RGFX_SURFACE_USER_CLIENT_HPP_

#include <IOKit/IOLocks.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/graphics/IOAccelSurfaceConnect.h>

class RgfxAccelerator;

class RgfxSurfaceUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxSurfaceUserClient);

public:
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	IOReturn clientClose() override;
	IOReturn clientDied() override;
	void free() override;

	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;

	// The task that opened the surface.
	task_t owningTask() const { return task; }
	// Whether the surface's memory is the framebuffer's scanout of `width` x `height`: on
	// framebuffer 0, no backing of the client's, its shape the whole `width` x `height`.
	bool isScanout(UInt32 width, UInt32 height);

private:
	IOReturn setIDMode(IOExternalMethodArguments *a);
	IOReturn setShape(IOExternalMethodArguments *a, bool withBacking);
	IOReturn lock(IOExternalMethodArguments *a);
	void dropSurface();

	task_t task;
	RgfxAccelerator *owner;
	// Under stateLock: the surface's id and mode (SetIDMode), its shape (the region's bounds and
	// framebuffer index), the backing the client gave, and the scanout's mapping in the task.
	IOLock *stateLock;
	bool haveId;
	UInt32 surfaceId;
	UInt32 modeBits;
	bool haveShape;
	UInt32 framebufferIndex;
	IOAccelBounds bounds;
	mach_vm_address_t backing;
	UInt32 backingRowBytes;
	UInt64 backingLength;
	IOMemoryMap *scanoutMap;
	UInt32 scanoutRowBytes;
};

#endif // RGFX_SURFACE_USER_CLIENT_HPP_
