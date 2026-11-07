/*
 * RgfxGLUserClients: a GL window's surface (RgfxGLSurfaceUserClient, user client type 0x20) and
 * the GL drawables over it (RgfxGLDrawableUserClient, type 7), IOGPU's protocol as an app's
 * OpenGL contexts drive it (RgfxGLSurfaces.hpp, which holds the rules and what was measured).
 *
 * A surface's buffers are IOSurfaces the kext makes in the surface's task, as IOGPU does, which
 * the drawables' lookups hand to their own tasks as port names: the app draws into them through
 * the bundle, and WindowServer composites the window from them. The kext touches no pixel.
 *
 * Every surface and drawable of the accelerator is under one lock, IOGPU's surface lock: the
 * surfaces by id, each surface's drawables and buffers, and each drawable's surface. A drawable
 * holds its surface retained; a surface that goes lets go of its buffers and detaches its
 * drawables, whose lookups then fail.
 *
 * Our own code; it names SDK-public IOKit classes and IOSurface.kext's exports
 * (RgfxIOSurfaceKPI.hpp).
 */

#ifndef RGFX_GL_USER_CLIENTS_HPP_
#define RGFX_GL_USER_CLIENTS_HPP_

#include <IOKit/IOUserClient.h>

#include "RgfxGLSurfaces.hpp"

class RgfxGLDrawableUserClient;

class RgfxGLSurfaceUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxGLSurfaceUserClient);

public:
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	void stop(IOService *provider) override;
	IOReturn clientClose() override;
	void free() override;

	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;

private:
	friend class RgfxGLDrawableUserClient;

	// The most drawables one surface holds at once.
	static constexpr uint32_t kMaxDrawables = 8;

	IOReturn setIDMode(IOExternalMethodArguments *a);
	IOReturn setShape(IOExternalMethodArguments *a);
	// Under the GL lock, IOGPU's steps: the request bits start again from the drawables'
	// (reset_req_bits), the slots no longer asked for go (prune_buffers), and the slots the shape
	// and mode need are made (update_shape).
	void resetRequestsLocked();
	void pruneLocked();
	void reshapeLocked();
	// Under the GL lock: the slots of `mask` go.
	void releaseSlotsLocked(uint32_t mask);
	// Under the GL lock: slot `slot` made anew for the shape and mode.
	void makeSlotLocked(uint32_t slot);
	// Under the GL lock: the slots that hold a buffer.
	uint32_t existingLocked() const;
	// The surface goes: out of the accelerator's list, its drawables detached, its buffers gone.
	void dropSurface();
	// Under the GL lock: the surface of id `id`, the first listed that holds it, or null.
	static RgfxGLSurfaceUserClient *findLocked(uint32_t id);

	task_t task;
	// Under the GL lock.
	RgfxGLSurface state;
	OSObject *slots[kRgfxGLSlots];
	RgfxGLDrawableUserClient *drawables[kMaxDrawables];
	uint32_t drawableCount;
	bool listed;
	bool dropped;
};

class RgfxGLDrawableUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxGLDrawableUserClient);

public:
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	IOReturn clientClose() override;
	void free() override;

	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;

	// The methods, by selector (IOGPU's): 0 set surface, 1 set surface and answer its config,
	// 2 swap rect, 3 swap interval, 4 shared event, 5 lookup.
	static IOReturn sSetSurface(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSetSurfaceGetConfig(OSObject *target, void *reference,
	                                     IOExternalMethodArguments *a);
	static IOReturn sSetSwapRect(OSObject *target, void *reference, IOExternalMethodArguments *a);
	static IOReturn sSetSwapInterval(OSObject *target, void *reference,
	                                 IOExternalMethodArguments *a);
	static IOReturn sSignalSharedEvent(OSObject *target, void *reference,
	                                   IOExternalMethodArguments *a);
	static IOReturn sLookup(OSObject *target, void *reference, IOExternalMethodArguments *a);

private:
	friend class RgfxGLSurfaceUserClient;

	// IOGPU's set_surface: attaches to the surface `in` names, or detaches for id 0.
	IOReturn setSurface(const RgfxGLSetSurface &in);
	// IOGPU's set_surface_get_config_status.
	IOReturn setSurfaceGetConfig(const RgfxGLSetSurface &in, RgfxGLConfigStatus &out);
	// Under the GL lock: off its surface, whose request bits follow, and its buffers too where
	// `prune`; the surface's reference, which the caller releases with the lock let go, in
	// `*held`.
	void detachLocked(RgfxGLSurfaceUserClient **held, bool prune);

	task_t task;
	// Under the GL lock: the surface, retained, and the slots this drawable asks for.
	RgfxGLSurfaceUserClient *surface;
	uint32_t requested;
	uint32_t face;
	uint32_t level;
	// IOGPU keeps the swap rect and interval, which nothing reads back.
	int16_t swapRect[4];
	int16_t swapInterval[2];
};

#endif // RGFX_GL_USER_CLIENTS_HPP_
