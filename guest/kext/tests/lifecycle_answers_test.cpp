// The native test of RgfxLifecycleAnswers.hpp (lifecycle_answers_test, and lifecycle_answers_arm64_test
// with RGFX_GUEST_ARM64=1): every answer type a kext thread waits for in postLifecycle is one the
// event handler hands to its waiter, POOL_TAKE's POOL_BLOCK among them (a boot whose handler
// dropped it left WindowServer's first take waiting for good), and the events the handler takes
// itself are not.
#include "RgfxLifecycleAnswers.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "lifecycle_answers_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// The answer types the kext's postLifecycle callers name.
	check(rgfxLifecycleAnswer(RGFX_EVT_COMPLETE), "COMPLETE");
	check(rgfxLifecycleAnswer(RGFX_EVT_PIPELINE_REFLECTION), "PIPELINE_REFLECTION");
	check(rgfxLifecycleAnswer(RGFX_EVT_SIZE_AND_ALIGN), "SIZE_AND_ALIGN");
	check(rgfxLifecycleAnswer(RGFX_EVT_SAMPLER_STATE), "SAMPLER_STATE");
	check(rgfxLifecycleAnswer(RGFX_EVT_ICB_STATE), "ICB_STATE");
	check(rgfxLifecycleAnswer(RGFX_EVT_SHARED_EVENT_VALUE), "SHARED_EVENT_VALUE");
	check(rgfxLifecycleAnswer(RGFX_EVT_RATE_MAP_STATE), "RATE_MAP_STATE");
	check(rgfxLifecycleAnswer(RGFX_EVT_RATE_MAP_ANSWER), "RATE_MAP_ANSWER");
	check(rgfxLifecycleAnswer(RGFX_EVT_FUNCTION_POINTER_STATE), "FUNCTION_POINTER_STATE");
	check(rgfxLifecycleAnswer(RGFX_EVT_POOL_BLOCK), "POOL_TAKE's POOL_BLOCK");
	// VT_MESSAGE's answer: the arm64e kext's alone.
	check(rgfxLifecycleAnswer(RGFX_EVT_VT_STATUS) == (RGFX_GUEST_ARM64 != 0), "VT_STATUS");

	// The events the handler takes itself, which no thread waits for.
	check(!rgfxLifecycleAnswer(RGFX_MSG_PAD), "padding");
	check(!rgfxLifecycleAnswer(RGFX_EVT_DISPLAYS), "DISPLAYS");
	check(!rgfxLifecycleAnswer(RGFX_EVT_DISPLAY_CHANGED), "DISPLAY_CHANGED");
	check(!rgfxLifecycleAnswer(RGFX_EVT_VBLANK), "VBLANK");
	check(!rgfxLifecycleAnswer(RGFX_EVT_POINTER_OVER), "POINTER_OVER");
	check(!rgfxLifecycleAnswer(RGFX_EVT_POINTER_MOTION), "POINTER_MOTION");
	check(!rgfxLifecycleAnswer(RGFX_EVT_SUBMIT_ROOM), "SUBMIT_ROOM");
	check(!rgfxLifecycleAnswer(RGFX_EVT_VT_MESSAGE), "VT_MESSAGE");
	check(!rgfxLifecycleAnswer(RGFX_EVT_VT_RING), "VT_RING");

	std::printf("lifecycle_answers_test: OK (%d checks)\n", checks);
	return 0;
}
