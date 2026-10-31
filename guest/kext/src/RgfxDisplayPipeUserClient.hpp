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
 * ring as PRESENT_SURFACE through the framebuffer (ReimsGraphics owns that ring), and submit
 * returns at once. The transaction completes when the host answers that PRESENT_SURFACE, which
 * it does once the display shows the surface: its completion notification goes out and a waiter
 * returns. An answer completes every transaction submitted before it, oldest first. At most two
 * are outstanding; a third submit waits up to five seconds for room, then fails unsent. A
 * transaction with no IOSurface on plane 0 keeps the surface shown and completes at once. VBL
 * notifications follow the device's VBLANK events while one is registered.
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
	// The transactions submitted to the host and not yet answered, oldest first.
	enum { kMaxOutstanding = 2 };
	struct Outstanding {
		UInt64 seqno;
		UInt32 id;
		UInt64 cookie;
	};

	static IOReturn sAssociate(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sModeScaler(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sCapabilities(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sRequestNotify(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sBegin(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSeven(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSubmit(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sWait(OSObject *target, void *reference, IOExternalMethodArguments *a);

	// The framebuffer's hooks (ReimsGraphics::addPipe), on its work loop.
	static void vblankHook(OSObject *client, UInt64 count, UInt64 time);
	static void presentHook(OSObject *client, UInt64 seqno, UInt32 status);
	void presented(UInt64 seqno, UInt32 status);
	void completeLocked(UInt32 transactionId, UInt64 cookie, UInt64 time);
	void sendVbl(UInt64 count, UInt64 time);

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
	Outstanding outstanding[kMaxOutstanding];
	UInt32 outstandingCount;
};

#endif // RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_
