// RgfxAnswerLook: how a thread that waits for its lifecycle message's answer looks for it itself
// before it sleeps (ReimsGraphics::postLifecycle). The device answers most such messages within
// tens of microseconds of the post: a heap texture's size, a library's lookup, a surface's pages
// (60 to 110 us from the post on the RTX 3090 host, measured in a Vision request). The answer then
// waits for the interrupt, the work loop's wake and the waiter's own, about 25 us more (measured
// alike). So the waiter looks for it for at most kRgfxAnswerLookNs: where the completion ring
// holds a record the work loop has not taken, it takes the ring itself, in the work loop's gate,
// as the work loop would once woken; else it pauses the CPU and looks again. It never waits for
// the gate: where another thread holds it, the work loop most likely, which takes the record
// itself, the waiter pauses and looks again, so a holder that waits for something the waiter
// holds is bounded by the waiter's deadline, as before. It looks only while its own call waits
// for an answer, and only that long: an answer that comes later wakes it as before.
//
// Header-only and free of the kernel's headers, so its native test (tests/answer_look_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_ANSWER_LOOK_HPP_
#define RGFX_ANSWER_LOOK_HPP_

#include <stdint.h>

// The longest a waiter looks for its answer before it sleeps: 150 us, past a surface's pages'
// answer (above), so a pipeline create, whose compile takes longer, spends no more of its
// thread's CPU than that.
constexpr uint64_t kRgfxAnswerLookNs = 150 * 1000;

// A pause in a look: the CPU's spin-wait hint.
inline void rgfxAnswerLookPause()
{
#if defined(__x86_64__)
	__builtin_ia32_pause();
#elif defined(__arm64__) || defined(__aarch64__)
	__asm__ volatile("yield");
#endif
}

// Looks for an answer until `answered()`, or until `now()` reaches `deadline`: where `pending()`
// says the completion ring holds a record not yet taken, `take()` takes the ring where the work
// loop's gate is free, false where another holds it; where it holds none or the take did not
// run, `pause()`. Whether the answer came.
template <typename Now, typename Answered, typename Pending, typename Take, typename Pause>
bool rgfxLookForAnswer(uint64_t deadline, Now now, Answered answered, Pending pending, Take take,
                       Pause pause)
{
	while (!answered()) {
		if (now() >= deadline)
			return false;
		if (!pending() || !take())
			pause();
	}
	return true;
}

#endif // RGFX_ANSWER_LOOK_HPP_
