// RgfxQueueWakes: which looks at a watched completion ring wake its client's completion waiters
// (RgfxAccelerator::queueFilter and queueInterrupt). Header-only and free of the kernel's headers,
// so the native test (tests/queue_wakes_test.cpp) checks the rule as the kext keeps it.
//
// Each interrupt of the device's vector is looked at twice. The vector's filter looks first, at
// primary interrupt context as the interrupt arrives, so a client's completion thread is woken
// before the framebuffer's work loop thread is scheduled and has taken the control ring; the work
// loop's pass looks again after its take. A look wakes the client where the ring's producer moved
// since that look's last and the ring holds records the client has not taken. The work loop's
// look also skips a producer the filter has already looked at: the filter woke the client for
// it, or found every record taken, which the client's consumer stays past. So an interrupt for
// one completion wakes its client once, and a completion published after the filter's look is
// woken for by the work loop's, as before the filter.
//
// Our own code.
#ifndef RGFX_QUEUE_WAKES_HPP_
#define RGFX_QUEUE_WAKES_HPP_

#include <stdint.h>

// The filter's look, at the interrupt: `*seen` the producer its last look saw, which it updates
// (stored atomically, as the work loop's look reads it without the filter's lock), and
// `consumer()` the ring's consumer, read only where the producer moved.
template <class Consumer>
inline bool rgfxFilterWakes(uint64_t *seen, uint64_t producer, Consumer consumer)
{
	if (producer == __atomic_load_n(seen, __ATOMIC_RELAXED))
		return false;
	__atomic_store_n(seen, producer, __ATOMIC_RELAXED);
	return producer != consumer();
}

// The work loop's look, after its take of the control ring: `*seen` the producer its last look
// saw, which it updates, and `filterSeen` the filter's.
template <class Consumer>
inline bool rgfxLoopWakes(uint64_t *seen, const uint64_t *filterSeen, uint64_t producer,
                          Consumer consumer)
{
	if (producer == *seen)
		return false;
	*seen = producer;
	return producer != __atomic_load_n(filterSeen, __ATOMIC_RELAXED) && producer != consumer();
}

#endif
