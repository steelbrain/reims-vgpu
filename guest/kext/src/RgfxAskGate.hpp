// RgfxAskGate: a process's asks of its context that the client's close must not overtake, and
// that wait for the device: a surface's block (kRgfxAccelMethodSurfaceBlock), a pages block
// (kRgfxAccelMethodPagesBlock), a chunk of resource ids (kRgfxAccelMethodResourceIds) and a
// command queue (kRgfxAccelMethodQueueCreate).
//
// Each makes something the close lets go of (a hold of a shared block, a block, a chunk, a watched
// queue), and an ask that began while the context was open must have made it, or given up, before
// the close lets go of what the context holds: else a hold or a block made after
// dropSurfaceBlocks, a chunk granted after the CONTEXT_CLOSE, a ring watched after the unwatch,
// outlives the context. So an ask begins only while the context is open and its close has not
// begun, and the close, once begun, waits for the asks in flight to end before it goes on.
//
// The asks themselves run beside each other: one that waits (wiring a surface's pages, then the
// host's answer, or a surface's write-back) holds up none of the process's other threads. One
// lock held across each ask ordered them against the close before, and so against each other.
//
// The caller holds the client's list lock around every call, and the close sleeps on the gate
// until rgfxAskEnd says the last ask ended. Header-only and free of the kernel's headers, so its
// native test (tests/ask_gate_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_ASK_GATE_HPP_
#define RGFX_ASK_GATE_HPP_

#include <stdint.h>

struct RgfxAskGate {
	// The asks begun and not yet ended, and the most at once (logged at the close).
	uint32_t inFlight;
	uint32_t most;
	// The close has begun: no ask begins from here.
	bool closing;
};

// An ask begins, the context `open`: false, nothing begun, where it is not, or its close has begun.
inline bool rgfxAskBegin(RgfxAskGate &g, bool open)
{
	if (!open || g.closing)
		return false;
	g.inFlight++;
	if (g.inFlight > g.most)
		g.most = g.inFlight;
	return true;
}

// An ask begun ends: true where it was the last in flight and the close waits for it, which the
// caller then wakes.
inline bool rgfxAskEnd(RgfxAskGate &g)
{
	if (g.inFlight)
		g.inFlight--;
	return g.closing && g.inFlight == 0;
}

// The close begins: no ask begins from here. How many are in flight, which it waits for.
inline uint32_t rgfxAskClose(RgfxAskGate &g)
{
	g.closing = true;
	return g.inFlight;
}

// Whether the close may go on: no ask is in flight.
inline bool rgfxAskDrained(const RgfxAskGate &g)
{
	return g.inFlight == 0;
}

#endif // RGFX_ASK_GATE_HPP_
