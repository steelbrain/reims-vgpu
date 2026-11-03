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

#endif // RGFX_SUBMIT_GATE_HPP_
