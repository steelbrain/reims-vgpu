// RgfxSubmitGate: which object-lifecycle message types the kext accepts on kRgfxAccelMethodSubmit
// (RgfxAcceleratorUserClient::submitMethod), and which of them change state the kext itself tracks
// and so must be routed rather than posted to the device raw.
//
// Submit carries a process's creates and destroys of its own objects onto the control ring. It
// accepts only the types strictly between QUEUE_CREATE and CONTEXT_CLOSE: a queue is made through
// its own selector (kRgfxAccelMethodQueueCreate, which checks its rings), and CONTEXT_CLOSE and
// the block grant/share/release messages (>= CONTEXT_CLOSE) are the kext's, never a process's. So
// a process cannot drive block tracking through submit.
//
// Of the accepted range, the command queue is the one object the kext tracks (it watches the
// queue's completion ring and lists it per client), so an OBJECT_DESTROY of a queue must stop the
// watch and drop the queue, not just reach the host: posted raw, it would leave the kext scanning
// a completion ring the host has torn down and whose block the process may then free and another
// take. Every other accepted type names an object only the host tracks, and is posted as it is.
// rgfxSubmitRoutesQueueDestroy names the one type to route; if a later increment adds another
// type that touches kext-tracked state, its entry here (and submit_gate_test) is where it is
// caught.
//
// Submit takes no lock across its post (a pipeline create waits up to a minute for its answer),
// so a thread in a submit may post its create after another thread's close of the client posted
// the context's CONTEXT_CLOSE, which the device answers having dropped every object of the
// context then. rgfxSubmitDestroysBehind says when the submit follows its create with a destroy.
//
// Header-only and free of the kernel's headers, so its native test (tests/submit_gate_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_SUBMIT_GATE_HPP_
#define RGFX_SUBMIT_GATE_HPP_

#include "reims_graphics_wire.h"

// Whether submitMethod accepts a message of `type`: an object create or destroy, strictly between
// QUEUE_CREATE and CONTEXT_CLOSE. (The caller also checks the obj_id lies in the sender's context.)
inline bool rgfxSubmitAdmitted(uint32_t type)
{
	return type > RGFX_CMD_QUEUE_CREATE && type < RGFX_CMD_CONTEXT_CLOSE;
}

// Whether an accepted `type`, when its obj_id is one of the client's command queues, must be
// routed through the queue-destroy path (stop watching the ring, drop the queue, post the
// destroy) rather than posted raw: only OBJECT_DESTROY, the one accepted type that ends a queue.
inline bool rgfxSubmitRoutesQueueDestroy(uint32_t type)
{
	return type == RGFX_CMD_OBJECT_DESTROY;
}

// Whether a message of an accepted `type` makes a host object under its obj_id, which an
// OBJECT_DESTROY of that id drops. The destroy, the queries (a heap texture's size, an argument
// layout) and the writes into an object already made (a Managed range or region, a shared
// event's value, an indirect command buffer's commands) make none. A type the wire does not name makes none: the device refuses it.
inline bool rgfxSubmitMakes(uint32_t type)
{
	switch (type) {
	case RGFX_CMD_HEAP_CREATE:
	case RGFX_CMD_HEAP_OVER_HOST_MEMORY:
	case RGFX_CMD_BUFFER_CREATE:
	case RGFX_CMD_BUFFER_OVER_HOST_MEMORY:
	case RGFX_CMD_BUFFER_AT_OFFSET:
	case RGFX_CMD_BUFFER_MANAGED_AT_OFFSET:
	case RGFX_CMD_HEAP_OVER_PAGES:
	case RGFX_CMD_TEXTURE_CREATE:
	case RGFX_CMD_TEXTURE_FROM_BUFFER:
	case RGFX_CMD_TEXTURE_FROM_IOSURFACE:
	case RGFX_CMD_SAMPLER_CREATE:
	case RGFX_CMD_DEPTH_STENCIL_CREATE:
	case RGFX_CMD_LIBRARY_CREATE:
	case RGFX_CMD_RENDER_PIPELINE_CREATE:
	case RGFX_CMD_COMPUTE_PIPELINE_CREATE:
	case RGFX_CMD_EVENT_CREATE:
	case RGFX_CMD_SHARED_EVENT_CREATE:
	case RGFX_CMD_FENCE_CREATE:
	case RGFX_CMD_LIBRARY_FROM_REGION:
	case RGFX_CMD_LIBRARY_LOOKUP:
	case RGFX_CMD_MANAGED_TEXTURE_CREATE:
	case RGFX_CMD_TEXTURE_VIEW:
	case RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER:
	case RGFX_CMD_BUFFER_OVER_SURFACE_PAGES:
	case RGFX_CMD_VIDEO_SESSION_CREATE:
	case RGFX_CMD_BUFFER_OVER_PAGES:
	case RGFX_CMD_ICB_CREATE:
		return true;
	default:
		return false;
	}
}

// Whether a submitted message of `type`, `posted` on the control ring, must be followed by an
// OBJECT_DESTROY of its obj_id, so the device keeps nothing of it: where it makes an object
// (rgfxSubmitMakes) and either its answer did not come in time (`timedOut`: the client takes the
// create as failed and may make it again under a new id), or the client's context closed between
// the submit's check and now (`closedSince`): a create posted after its context's CONTEXT_CLOSE
// is one the device's close never dropped. The destroy follows the create on the ring, which the
// device carries out in order, and context ids are never reused, so it drops that object alone;
// where the create came before the close after all, it finds nothing, which is no error.
inline bool rgfxSubmitDestroysBehind(uint32_t type, bool posted, bool timedOut, bool closedSince)
{
	return posted && rgfxSubmitMakes(type) && (timedOut || closedSince);
}

#endif // RGFX_SUBMIT_GATE_HPP_
