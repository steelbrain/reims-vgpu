// RgfxLegacySwapInfo: the swap-info notifications an arm64 WindowServer learns its swaps' times
// from, as Apple's paravirtual display (AppleParavirtDisplay) sends them on each arm64 release the
// arm64e kext runs on (read from 14.6.1's, 15.6.1's, 26.6.2's and 27.0.1's AppleParavirtGPU and
// IOMobileGraphicsFamily):
// - IOMobileFramebufferLegacy's swap_complete_gated calls the display's notify_swap_done_gated with
//   the completed swap's ID. The paravirtual display makes the swap's record (RgfxSwapInfoRecord)
//   and hands it to the class's swap_info_notify_dispatch_gated, which calls the display's
//   do_swap_info_notify_gated for each notify request the class holds.
// - do_swap_info_notify_gated sends the request's client an async result (rgfxSwapInfoArgs): the
//   request's tag mask, the swap's ID, and the record's values the mask asks for.
// - set_notify_tag_mask_gated sets a request's tag mask, for the swap-info notification alone.
// Header-only and free of the kernel's headers, for the native tests.
//
// Our own code.

#ifndef RGFX_LEGACY_SWAP_INFO_HPP_
#define RGFX_LEGACY_SWAP_INFO_HPP_

#include <stddef.h>
#include <stdint.h>

// A swap's record, as the paravirtual display makes it in notify_swap_done_gated and the class's
// swap_info_notify_dispatch_gated passes it on: the swap's ID, then five doublewords. The fields'
// meanings are not stated; what the paravirtual display writes in each is
// (rgfxSwapInfoRecord): the class's last vblank time at 0x08, 0 at 0x10, the swap's copy of its
// IOMFBSwapRec's field_00 at 0x18, the swap's flags word at 0x20 and 0 at 0x28.
struct RgfxSwapInfoRecord {
	uint32_t field_00;
	uint32_t field_04;
	uint64_t field_08;
	uint64_t field_10;
	uint64_t field_18;
	uint64_t field_20;
	uint64_t field_28;
};
static_assert(sizeof(RgfxSwapInfoRecord) == 0x30, "a swap-info record is 0x30 bytes");
static_assert(offsetof(RgfxSwapInfoRecord, field_08) == 0x08, "its doublewords start at 8");

// In a notify request (IOMFBNotifyRequest), the same on each release: the async reference its
// client gave, which the result is sent to; the tag mask; and the field_08 of the last record
// sent, which the next one's second value is taken from.
constexpr uint32_t kRgfxNotifyRequestReference = 0x18;
constexpr uint32_t kRgfxNotifyRequestMask = 0x78;
constexpr uint32_t kRgfxNotifyRequestLast = 0x80;

// The notification type set_notify_tag_mask_gated takes a mask for (IOMFB_NotificationType).
constexpr uint32_t kRgfxSwapInfoNotification = 3;

// The most values an async result carries here, and the records the paravirtual display keeps
// while the class holds them.
constexpr uint32_t kRgfxSwapInfoMaxArgs = 12;
constexpr uint32_t kRgfxSwapInfoKept = 6;

// A swap's record: its ID, the class's last vblank time, and the swap's field_00 copy and flags.
inline RgfxSwapInfoRecord rgfxSwapInfoRecord(uint32_t swapId, uint64_t vblankTime,
                                             uint64_t swapField00, uint32_t swapFlags)
{
	RgfxSwapInfoRecord r = {};
	r.field_00 = swapId;
	r.field_08 = vblankTime;
	r.field_18 = swapField00;
	r.field_20 = swapFlags;
	return r;
}

// The async result's values for `record`, to a request whose tag mask is `mask` and whose last
// record's field_08 is `*last`, as the paravirtual display's do_swap_info_notify_gated makes them:
// the mask, the swap's ID, then for each of the record's five doublewords whose bit (0 to 4) the
// mask sets, the doubleword, the second one being field_08 less `*last` instead. With the mask's
// bit 31 a value is one argument, otherwise two, its low and then its high word. A value that
// would start at the twelfth argument is left out. `*last` becomes the record's field_08.
// Answers the count of `args` written.
inline uint32_t rgfxSwapInfoArgs(uint32_t mask, uint64_t *last, const RgfxSwapInfoRecord &record,
                                 uint64_t args[kRgfxSwapInfoMaxArgs + 1])
{
	const uint64_t values[5] = {record.field_08, record.field_08 - *last, record.field_18,
	                            record.field_20, record.field_28};
	uint32_t n = 0;
	args[n++] = mask;
	args[n++] = record.field_00;
	for (uint32_t i = 0; i < 5; i++) {
		if ((mask & (1u << i)) == 0 || n == kRgfxSwapInfoMaxArgs)
			continue;
		if (mask & 0x80000000u) {
			args[n++] = values[i];
		} else {
			args[n++] = static_cast<uint32_t>(values[i]);
			args[n++] = values[i] >> 32;
		}
	}
	*last = record.field_08;
	return n;
}

// The records a display keeps while the class holds them, and dispatches in order before the next
// one, in six slots written in turn as the paravirtual display keeps them: past six, the slots are
// overwritten and the ring reads as holding fewer.
struct RgfxSwapInfoKept {
	RgfxSwapInfoRecord records[kRgfxSwapInfoKept];
	uint32_t head;
	uint32_t tail;

	void keep(const RgfxSwapInfoRecord &record)
	{
		records[tail] = record;
		tail = (tail + 1) % kRgfxSwapInfoKept;
	}
	// Calls `dispatch` on each kept record, oldest first, and keeps none.
	template <typename Dispatch>
	void flush(Dispatch dispatch)
	{
		while (head != tail) {
			dispatch(records[head]);
			head = (head + 1) % kRgfxSwapInfoKept;
		}
	}
};

#endif // RGFX_LEGACY_SWAP_INFO_HPP_
