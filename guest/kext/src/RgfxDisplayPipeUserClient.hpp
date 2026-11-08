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
 *   5  plane gamma table                                  struct in variable
 *   6  gamma table before the colour matrix               struct in variable
 *   7  gamma table after the colour matrix                struct in variable
 *   8  submit a transaction                               struct in 280
 *   9  wait for a transaction                             struct in 12
 * Every other selector is refused as unsupported and logged: 10 and 11 (the pipe's linearization
 * before its matrix and gamma after it, in a configuration format unread here), 12 (a surface
 * copy) and 13 (triage), which no WindowServer seen calls.
 *
 * The display's colour (RgfxPipeColor.hpp): where the host took RGFX_FEATURE_DISPLAY_COLOR,
 * selectors 5 to 7 decode their gamma table as macOS's kernel does and stage it for the begun
 * transaction, and its submit sends each staged table, and the colour matrix where its dirty mask
 * has the matrix's bit, as DISPLAY_COLOR parts before its PRESENT_SURFACE, with no other pipe
 * client's post between them (IOPresentment opens several, and any may submit; the host applies
 * the parts to the next present it takes, and drops a stage whose parts a present cuts short).
 * A begin drops what the
 * transaction before it staged and never submitted. A transaction with no IOSurface that sets a
 * colour goes as a PRESENT_SURFACE that keeps the surface shown (RGFX_PRESENT_KEEP), and
 * completes when the host answers it, at the refresh that shows the colour. Where the
 * host did not take the feature, 5 and 6 are refused as unsupported, and 7 takes no structure.
 *
 * A submitted transaction's plane 0, where it is an IOSurface, goes to the host on the control
 * ring as PRESENT_SURFACE through the framebuffer (ReimsGraphics owns that ring), and submit
 * returns at once. The transaction completes when the host answers that PRESENT_SURFACE, which
 * it does once the display shows the surface: a waiter returns, and its completion notification
 * goes out where it carries a cookie (RgfxPipeTransactions.hpp). An answer completes every
 * transaction submitted before it, oldest first, each at the time of the refresh the host
 * answered in (the device's VBLANK stamp in the guest's clock). At most two are outstanding; a
 * third submit waits up to five seconds for room, then fails unsent. A transaction with no
 * IOSurface on plane 0 keeps the surface shown and completes with the one before it, or at once.
 *
 * A VBL registration is a request for one notification, at the next refresh, as the interface's
 * kernel side answers it: the request's callback and refcon (its first two words), then the
 * refresh's time and the next one's, each 64-bit value split in two words, low half first. Up
 * to kMaxVblRequests wait at once; the device's VBLANK events are on while one waits.
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_
#define RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_

#include <IOKit/IOLocks.h>
#include <IOKit/IOUserClient.h>

#include "RgfxPipeTransactions.hpp"

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
	// The transactions submitted to the host and not yet answered.
	enum { kMaxOutstanding = 2 };
	// The VBL requests waiting for the next refresh.
	enum { kMaxVblRequests = 4 };
	typedef RgfxPipeTransactions<kMaxOutstanding> Transactions;

	static IOReturn sAssociate(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sModeScaler(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sCapabilities(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sRequestNotify(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sBegin(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSeven(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sGamma(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSubmit(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sWait(OSObject *target, void *reference, IOExternalMethodArguments *a);

	// The framebuffer's hooks (ReimsGraphics::addPipe), on its work loop.
	static void vblankHook(OSObject *client, UInt64 count, UInt64 time, UInt64 next);
	static void presentHook(OSObject *client, UInt64 seqno, UInt32 status, UInt64 time);
	void presented(UInt64 seqno, UInt32 status, UInt64 time);
	void completeLocked(const Transactions::Completion &c, UInt64 time);
	// sSubmit's transaction, submitLock held: sent to the host, or done with the one before it.
	IOReturn submit(const UInt8 *t, UInt32 id, UInt32 planeType, UInt32 surface, UInt64 cookie);
	// submit's posts once the transaction has room, submitLock and the framebuffer's presents
	// lock held and `lock` not: its colour, then its present, or its completion with the one
	// before it where it sends nothing.
	IOReturn post(const UInt8 *t, UInt32 id, UInt32 planeType, UInt32 surface, UInt64 cookie,
	              bool notified);
	// submit's colour, submitLock held and `lock` not: the stages staged for transaction `id`,
	// and the matrix where transaction `t` carries one, as DISPLAY_COLOR parts; whether any went.
	IOReturn postColor(const UInt8 *t, UInt32 id, bool *sent);
	// Lets go of the gamma tables staged (dropStaged: those of a transaction not submitted).
	void dropStaged();
	void sendVbl(UInt64 time, UInt64 next);

	void releaseRegistrations();

	task_t task;
	RgfxAccelerator *owner;
	ReimsGraphics *framebuffer; // retained once associated
	bool associated;

	// A VBL request: its wake port's reference, and the callback and refcon it names.
	struct VblRequest {
		OSAsyncReference64 reference;
		UInt64 callback;
		UInt64 refcon;
	};

	// Held across a submit, which may wait for the control ring's room with `lock` let go: the
	// submits keep their order. The work loop never takes it.
	IOLock *submitLock;
	// Under lock: the transaction registration, the VBL requests waiting, the transaction begun,
	// the last completed, and the words of a completion, set at the first submit (0 before it;
	// RgfxPipeNotifications.hpp).
	IOLock *lock;
	bool registered;
	OSAsyncReference64 transactionRef;
	VblRequest vblRequests[kMaxVblRequests];
	UInt32 vblWaiting;
	UInt32 nextTransaction;
	UInt32 begun;
	UInt32 completed;
	UInt32 completionWords;
	UInt32 logged;
	UInt32 unnotifiedLogged;
	Transactions transactions;

	// A gamma table staged (selectors 5 to 7): its entries, 16 bytes each, in a buffer of
	// `count` entries (nullptr for none, the stage off), and whether the transaction set it.
	struct GammaStage {
		UInt8 *entries;
		UInt32 count;
		bool set;
	};
	// Under lock: the host took RGFX_FEATURE_DISPLAY_COLOR (read at the association), the
	// transaction the stages are staged for, the stages (plane 0's, before and after the
	// matrix), and the colour calls logged.
	bool displayColor;
	UInt32 stagedFor;
	GammaStage staged[3];
	UInt32 colorLogged;
};

#endif // RGFX_DISPLAY_PIPE_USER_CLIENT_HPP_
