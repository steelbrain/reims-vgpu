// RgfxScanoutOwner: which task may reach the framebuffer's scanout, the display's pixels, from
// user space. Header-only and free of the kernel's headers, so its native test
// (tests/scanout_owner_test.cpp) builds it as the kext does.
//
// The scanout's owner is the task that holds the framebuffer's server connection
// (kIOFBServerConnectType), the WindowServer process that drives the display: IOFramebuffer
// opens that connection for one task at a time (a second open while one is held fails with
// kIOReturnExclusiveAccess), and the task that holds it maps the scanout through it already
// (the system aperture, getApertureRange). So the rule gives no task what IOFramebuffer does not
// give it, and every other task, whatever surfaces it makes, reaches none of the display's
// pixels. Read from IOGraphics' open source: IOFramebuffer::newUserClient makes the server
// connection for the task that opens it, and the connection's clientClose terminates it, which
// the kext reads as its end (IOService::isInactive).
//
// What the rule guards (ReimsGraphics.cpp, RgfxSurfaceUserClient.cpp, RgfxAcceleratorSurfaces.cpp):
//   - the lock of an IOAccel surface with no backing of its own, whose memory is the scanout
//     (mapped into the task at the lock);
//   - the scanout's grant to a context (kRgfxAccelMethodSurfaceScanout), which maps the scanout
//     into the context's window;
//   - the framebuffer's VRAM range (getVRAMRange), which IOFramebuffer maps into the task that
//     asks for it on its server connection, and on its shared connection (kIOFBSharedConnectType)
//     into any task of the console's user: only the task that holds the server connection gets
//     it, once the connection is made (rgfxMayMapVram).
// A task the rule refuses gets an error at that op (kIOReturnNotPermitted, or no memory for the
// VRAM range); nothing is mapped and nothing panics.
//
// While a server connection's open is in flight, its task owns the scanout too: IOFramebuffer
// sets the display up inside the open (its open()), before the connection exists. The VRAM range
// it asks for there is refused (rgfxMayMapVram). An open while a connection is held fails, so it
// marks nothing. IOFramebuffer sets the display up again for each new mode (setupForCurrentConfig,
// its only caller of the full setup, read from macOS 26.7.1's IOGraphicsFamily), on the thread of
// the task that set the mode, the owner's: the record marks that thread while the setup runs, and
// the VRAM range it asks for there is refused too. So the framebuffer never publishes a memory
// size (kIOFBMemorySizeKey), as the Intel 630's and the stock paravirtual framebuffer do not.
//
// One task owns the scanout at a time: a new server connection's task replaces the one before
// it, and every other task's hold of the scanout is taken back then (rgfxKeepsScanout): each
// grant of it, its span mapped back out of the context's window, and each lock's mapping of it,
// unmapped from the task. What IOFramebuffer itself mapped for a task, the VRAM range on its
// connections, is IOFramebuffer's, and goes only with its connection.
//
// Our own code.
#ifndef RGFX_SCANOUT_OWNER_HPP_
#define RGFX_SCANOUT_OWNER_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The scanout's room in the region: RGFX_SCANOUT_SPAN bytes at RGFX_ALLOCATIONS_OFFSET, where a
// region of `regionSize` bytes, its last `cursorBytes` the cursor's, holds that much past its
// head; else what it holds. The accelerator's blocks start past the span
// (RGFX_CONTEXT_ALLOCATIONS_OFFSET), so a mode, the VRAM range or a grant inside the room touches
// no process's memory. 0 where the region holds no room.
inline uint64_t rgfxScanoutRoom(uint64_t regionSize, uint64_t cursorBytes)
{
	if (regionSize <= RGFX_ALLOCATIONS_OFFSET + cursorBytes)
		return 0;
	const uint64_t room = regionSize - RGFX_ALLOCATIONS_OFFSET - cursorBytes;
	return room < RGFX_SCANOUT_SPAN ? room : RGFX_SCANOUT_SPAN;
}

// The owner's record, its tasks and connection as opaque pointers (a task_t, an IOUserClient).
// Not locked: the framebuffer holds its state lock around each call.
struct RgfxScanoutOwner {
	// The task that holds the server connection, and the connection; null for none.
	const void *task = nullptr;
	const void *client = nullptr;
	// The task whose server connection's open is in flight; null for none.
	const void *opening = nullptr;
	// The thread running IOFramebuffer's setup for the current mode; null for none.
	const void *settingUp = nullptr;

	// Whether `who` owns the scanout, `active` whether the recorded connection is still open.
	bool owns(const void *who, bool active) const
	{
		return who && (who == opening || holdsConnection(who, active));
	}

	// Whether `who` holds the server connection, made and still open (`active`): an owner whose
	// open is no longer in flight.
	bool holdsConnection(const void *who, bool active) const
	{
		return who && who == task && client && active;
	}

	// A server connection's open by `who` begins, `active` whether the recorded one is still
	// open. Returns whether `who` was marked as opening, which endOpen then clears.
	bool beginOpen(const void *who, bool active)
	{
		if (!who || (client && active) || opening)
			return false;
		opening = who;
		return true;
	}

	// The open `beginOpen` began for `who` ended, `made` the connection it made (null where it
	// failed), `marked` beginOpen's answer. Where it made one, `who` and `made` are recorded, and
	// the task and the connection they replace go to `*oldTask` and `*oldClient` for the caller
	// to let go (the task's reference, the connection's retain); else both are null.
	void endOpen(const void *who, const void *made, bool marked, const void **oldTask,
	             const void **oldClient)
	{
		*oldTask = nullptr;
		*oldClient = nullptr;
		if (marked && opening == who)
			opening = nullptr;
		if (!who || !made)
			return;
		*oldTask = task;
		*oldClient = client;
		task = who;
		client = made;
	}

	// The record goes (the framebuffer's free): what it held, to let go.
	void clear(const void **oldTask, const void **oldClient)
	{
		*oldTask = task;
		*oldClient = client;
		task = client = opening = nullptr;
	}

	// IOFramebuffer's setup for the current mode begins on `thread`: returns the thread of the
	// setup it runs inside (null for none), which endSetup puts back.
	const void *beginSetup(const void *thread)
	{
		const void *outer = settingUp;
		settingUp = thread;
		return outer;
	}

	// The setup beginSetup began ended: `outer` is its answer.
	void endSetup(const void *outer) { settingUp = outer; }

	// Whether `thread` runs IOFramebuffer's setup for the current mode.
	bool inSetup(const void *thread) const { return thread && thread == settingUp; }
};

// Whether a grant or a lock's mapping of the scanout that `holder` (its task) holds stays when
// the scanout's owner moves to `owner`: only the owner's own.
inline bool rgfxKeepsScanout(const void *holder, const void *owner)
{
	return holder && holder == owner;
}

// Whether `who` may be given the framebuffer's VRAM range: the scanout's owner on the server
// connection it holds (`connected`, RgfxScanoutOwner::holdsConnection), which maps it through that
// connection. IOFramebuffer's own setup asks too, in the kernel, inside the connection's open and,
// for a new mode, on the thread of the task that set it (`inSetup`, RgfxScanoutOwner::inSetup), and
// gets none: it publishes the range's size (kIOFBMemorySizeKey) from that answer alone.
inline bool rgfxMayMapVram(const void *who, const void *kernel, bool connected, bool inSetup)
{
	return who && who != kernel && connected && !inSetup;
}

#endif // RGFX_SCANOUT_OWNER_HPP_
