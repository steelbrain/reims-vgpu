// RgfxPipeNotifications: the words of the display pipe's notifications to its client
// (RgfxDisplayPipeUserClient), each sent with sendAsyncResult64. Header-only and free of the
// kernel's headers, so the native test (tests/user_client_words_test.cpp) checks the words the
// kext sends.
//
// A VBL request's answer, 6 words: the callback and refcon its registration named, then the
// refresh's time and the next refresh's, each a 64-bit value split over two words, low half
// first.
//
// A transaction's completion, 11 words: the cookie in word 0, the transaction's id in word 1's
// high half, the completion time in word 3 and the action, 1, in word 8; the others 0.
//
// Both are the layouts IOAccelerator's notification callbacks read (docs/graphics-kext/design.md,
// "The display pipe's notification words").
//
// Our own code.
#ifndef RGFX_PIPE_NOTIFICATIONS_HPP_
#define RGFX_PIPE_NOTIFICATIONS_HPP_

#include <stdint.h>

constexpr uint32_t kRgfxPipeVblWords = 6;
constexpr uint32_t kRgfxPipeCompletionWords = 11;

// A VBL request's answer into `words` (kRgfxPipeVblWords of them).
template <typename Word>
inline void rgfxPipeVblWords(Word *words, uint64_t callback, uint64_t refcon, uint64_t time,
                             uint64_t next)
{
	words[0] = callback;
	words[1] = refcon;
	words[2] = time & 0xffffffffu;
	words[3] = time >> 32;
	words[4] = next & 0xffffffffu;
	words[5] = next >> 32;
}

// Transaction `id`'s completion at `time` into `words` (kRgfxPipeCompletionWords of them).
template <typename Word>
inline void rgfxPipeCompletionWords(Word *words, uint64_t cookie, uint32_t id, uint64_t time)
{
	for (uint32_t i = 0; i < kRgfxPipeCompletionWords; i++)
		words[i] = 0;
	words[0] = cookie;
	words[1] = static_cast<uint64_t>(id) << 32;
	words[3] = time;
	words[8] = 1;
}

#endif // RGFX_PIPE_NOTIFICATIONS_HPP_
