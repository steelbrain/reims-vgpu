/*
 * RgfxDisplayPipeUserClient: the display pipe WindowServer presents through, user-client type 4
 * on the accelerator. IOAccelerator.framework's IOAccelDisplayPipeCreate, which IOPresentment
 * calls for each framebuffer naming an accelerator (IOAccelIndex), opens this type and drives
 * the display-pipe interface macOS's IOAccelerator.framework speaks: associate a framebuffer
 * index, read the pipe's capabilities, register for VBL and transaction notifications, and
 * begin, submit and wait for transactions, each of which names the IOSurfaces the display
 * shows (docs/graphics-kext/design.md, "Compositor/present").
 *
 * The selectors, their argument counts and sizes are the interface's:
 *   0  associate(framebuffer index) -> accelerator id     scalar 1 / scalar 1
 *   1  mode scaler                                        struct out 24: unsupported
 *   2  capabilities, a serialized dictionary              struct out variable
 *   3  notification registration (VBL or transaction)     struct in 24, asynchronous
 *   4  begin a transaction -> its id                      scalar out 1
 *   7  (no arguments)                                     accepted, logged
 *   8  submit a transaction                               struct in 280
 *   9  wait for a transaction                             struct in 12
 * Every other selector is refused as unsupported and logged.
 *
 * A submitted transaction's plane 0, where it is an IOSurface, goes to the host on the control
 * ring as PRESENT_SURFACE through the framebuffer (ReimsGraphics owns that ring), and the
 * transaction completes when the host has answered: its completion notification goes out and
 * a waiter returns. VBL notifications follow the device's VBLANK events while one is
 * registered.
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_
#define RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_

#include <IOKit/IOLocks.h>
#include <IOKit/IOUserClient.h>

class RgfxAccelerator;
class ReimsGraphics;

// The user-client type IOAccelDisplayPipeCreate opens.
static const UInt32 kRgfxDisplayPipeUserClientType = 4;

class RgfxDisplayPipeUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxDisplayPipeUserClient);

public:
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	IOReturn clientClose() override;
	void free() override;

	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;

private:
	// The registrations a client keeps (selector 3's kinds).
	enum { kKindVbl = 0, kKindTransaction = 1, kKinds = 2 };

	static IOReturn sAssociate(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sModeScaler(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sCapabilities(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sRequestNotify(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sBegin(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSeven(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSubmit(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sWait(OSObject *target, void *reference, IOExternalMethodArguments *a);

	// The framebuffer's VBLANK hook (ReimsGraphics::setPipeVblank), on its work loop.
	static void vblankHook(OSObject *client, UInt64 count, UInt64 time);
	void sendVbl(UInt64 count, UInt64 time);
	void sendCompletion(UInt32 transactionId, UInt64 cookie, UInt64 time);
	void releaseRegistrations();

	task_t task;
	RgfxAccelerator *owner;
	ReimsGraphics *framebuffer; // retained once associated
	bool associated;

	// Under lock: the registrations, the transaction begun and the last completed.
	IOLock *lock;
	bool registered[kKinds];
	OSAsyncReference64 references[kKinds];
	UInt64 cookies[kKinds];
	UInt32 nextTransaction;
	UInt32 begun;
	UInt32 completed;
	UInt64 lastVblTime;
	UInt32 logged;
};

#endif // RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_
