// RgfxLifecycleAnswers: the events that answer a thread waiting in ReimsGraphics::postLifecycle,
// which the event handler hands to their waiters (answerLifecycle), and the only answer types a
// waiter may ask for.
//
// Most waiters wait with no deadline (RgfxAnswerWait.hpp), a POOL_TAKE's THREAD_UNINT, so an
// answer the handler drops leaves its thread waiting for good, unkillable for a take and ended
// only by its process's kill for the others: postLifecycle refuses at once a reply type this list
// does not hold, and the handler hands on every type it holds, from the one list.
//
// Header-only and free of the kernel's headers, so its native test
// (tests/lifecycle_answers_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_LIFECYCLE_ANSWERS_HPP_
#define RGFX_LIFECYCLE_ANSWERS_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// Whether an event of `type` answers a lifecycle waiter: COMPLETE, the creates' and queries'
// answers, POOL_TAKE's POOL_BLOCK, PRIVATE_TAKE's PRIVATE_BLOCK, and on an arm64 guest a
// VT_MESSAGE's VT_STATUS.
constexpr bool rgfxLifecycleAnswer(uint32_t type)
{
	switch (type) {
	case RGFX_EVT_COMPLETE:
	case RGFX_EVT_PIPELINE_REFLECTION:
	case RGFX_EVT_SIZE_AND_ALIGN:
	case RGFX_EVT_SAMPLER_STATE:
	case RGFX_EVT_ICB_STATE:
	case RGFX_EVT_SHARED_EVENT_VALUE:
	case RGFX_EVT_RATE_MAP_STATE:
	case RGFX_EVT_RATE_MAP_ANSWER:
	case RGFX_EVT_FUNCTION_POINTER_STATE:
	case RGFX_EVT_POOL_BLOCK:
	case RGFX_EVT_PRIVATE_BLOCK:
		return true;
	case RGFX_EVT_VT_STATUS:
		return RGFX_GUEST_ARM64 != 0;
	default:
		return false;
	}
}

#endif // RGFX_LIFECYCLE_ANSWERS_HPP_
