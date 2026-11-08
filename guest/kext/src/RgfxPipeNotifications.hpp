// RgfxPipeNotifications: the words of the display pipe's notifications to its client
// (RgfxDisplayPipeUserClient), each sent with sendAsyncResult64. Header-only and free of the
// kernel's headers, so the native test (tests/user_client_words_test.cpp) checks the words the
// kext sends.
//
// A VBL request's answer, 6 words: the callback and refcon its registration named, then the
// refresh's time and the next refresh's, each a 64-bit value split over two words, low half
// first.
//
// A transaction's completion: the cookie in word 0, the transaction's id in word 1's high half,
// the completion time in word 3 and, where it has nine words or more, the action, 1, in word 8;
// the others 0. Its count is exactly the one the client's IOAccelerator expects: WindowServer's
// callback asserts it and aborts on any other (measured: macOS 11.7.11 and 12.7.6). That count
// follows the transaction the client submits and the family it runs with (rgfxPipeCompletionCount):
//   - 8 for a client that submits 0x108-byte transactions (macOS 11.7.11's WindowServer);
//   - for one that submits 0x118-byte transactions, 11 where IOAcceleratorFamily2 exports
//     IOAccelMemoryMap::getAssignedGPUVirtualAddressLength() const, and 9 where it does not
//     (macOS 12.7.6). The kernel collections of macOS 13.7.8, 14.8.9, 15.7.9, 15.8.2, 26.6.1 and
//     26.7.2 export it; those of 11.7.10, 11.7.11, 12.7.4 and 12.7.6 do not (read).
//
// Both are the layouts IOAccelerator's notification callbacks read (docs/graphics-kext/design.md,
// "The display pipe's notification words").
//
// Our own code.
#ifndef RGFX_PIPE_NOTIFICATIONS_HPP_
#define RGFX_PIPE_NOTIFICATIONS_HPP_

#include <stdint.h>

constexpr uint32_t kRgfxPipeVblWords = 6;
// The most words a completion has.
constexpr uint32_t kRgfxPipeCompletionWordsMax = 11;

// A transaction's sizes: 0x118 bytes, or 0x108 from macOS 11's WindowServer, the same layout
// without its last 16 bytes (which 12.7.6's and 26's send as zeros).
constexpr uint32_t kRgfxPipeTransactionBytes = 0x118;
constexpr uint32_t kRgfxPipeTransactionBytesShort = 0x108;

// The words of a completion to a client whose first transaction was `transactionBytes` long,
// where the family does (`assignedLengthExported`) or does not export
// IOAccelMemoryMap::getAssignedGPUVirtualAddressLength() const.
inline uint32_t rgfxPipeCompletionCount(uint32_t transactionBytes, bool assignedLengthExported)
{
	if (transactionBytes == kRgfxPipeTransactionBytesShort)
		return 8;
	return assignedLengthExported ? 11 : 9;
}

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

// Transaction `id`'s completion at `time` into `words` (`count` of them, 8 to
// kRgfxPipeCompletionWordsMax).
template <typename Word>
inline void rgfxPipeCompletionWords(Word *words, uint32_t count, uint64_t cookie, uint32_t id,
                                    uint64_t time)
{
	for (uint32_t i = 0; i < count; i++)
		words[i] = 0;
	words[0] = cookie;
	words[1] = static_cast<uint64_t>(id) << 32;
	words[3] = time;
	if (count > 8)
		words[8] = 1;
}

#endif // RGFX_PIPE_NOTIFICATIONS_HPP_
