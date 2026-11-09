// RgfxLegacySwapInfo: the frame info an arm64 WindowServer learns its swaps' presentation times
// from, sent as Apple's paravirtual display (AppleParavirtDisplay) sends its swap-info
// notifications on each arm64 release the arm64e kext runs on (read from 14.6.1's, 15.6.1's,
// 26.6.2's and 27.0.1's AppleParavirtGPU and IOMobileGraphicsFamily):
// - IOMobileFramebufferLegacy's swap_complete_gated calls the display's notify_swap_done_gated with
//   the completed swap's ID. The paravirtual display makes the swap's record (RgfxSwapInfoRecord)
//   and hands it to the class's swap_info_notify_dispatch_gated, which calls the display's
//   do_swap_info_notify_gated for each notify request the class holds.
// - do_swap_info_notify_gated sends the request's client an async result (rgfxSwapInfoArgs): the
//   request's tag mask, the swap's ID, and the record's values the mask asks for.
// - set_notify_tag_mask_gated sets a request's tag mask, for the swap-info notification alone.
//
// The paravirtual display answers supported_frame_info kIOReturnUnsupported, and so a client never
// asks for its records: QuartzCore registers for frame info (IOMobileFramebufferFrameInfo, the
// notification type 3) only where IOMobileFramebufferSupportedFrameInfo answers a list of tags,
// and its frame info callback is the only source of a composited frame's presentation time, which
// SkyLight hands to each drawable the frame showed (its presentedTime; 0, dropped, without it).
// Read from 27.0.1's QuartzCore (CA::WindowServer::AccelServer, IOMFBServer::frame_info_callback),
// SkyLight (display_present_handler) and IOMobileFramebuffer.framework (InfoKeyInitialize,
// figure_frame_info_mask, FrameInfoNotifyFunc). ReimsGraphics answers the tags it fills
// (rgfxFrameInfoTag): the library asks each tag's name by its index, and a client's tag mask sets
// bit i for tag i; a result's values come in the tags' order, each one argument where the mask's
// bit 31 is clear (two, its low and then its high word, where it is set), and the library turns
// them into a dictionary of the tags' names. Its timestamps are mach_continuous_time's, which the
// library turns into mach_absolute_time's (kern_egest_timestamp), as the class's own
// get_current_absolute_time reads mach_continuous_time.
// Header-only and free of the kernel's headers, for the native tests.
//
// Our own code.

#ifndef RGFX_LEGACY_SWAP_INFO_HPP_
#define RGFX_LEGACY_SWAP_INFO_HPP_

#include <stddef.h>
#include <stdint.h>

// A swap's record, as the paravirtual display makes it in notify_swap_done_gated and the class's
// swap_info_notify_dispatch_gated passes it on: the swap's ID, then five doublewords. The fields'
// meanings are not stated; what the paravirtual display writes in each is the class's last vblank
// time at 0x08, 0 at 0x10, the swap's copy of its IOMFBSwapRec's field_00 at 0x18, the swap's
// flags word at 0x20 and 0 at 0x28. ReimsGraphics writes the swap's presentation time at 0x08
// (rgfxSwapInfoRecord), which its frame info's first two tags are made from.
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

// A swap's record: its ID, its presentation time (0 where it showed nothing), and the swap's
// field_00 copy and flags.
inline RgfxSwapInfoRecord rgfxSwapInfoRecord(uint32_t swapId, uint64_t presented,
                                             uint64_t swapField00, uint32_t swapFlags)
{
	RgfxSwapInfoRecord r = {};
	r.field_00 = swapId;
	r.field_08 = presented;
	r.field_18 = swapField00;
	r.field_20 = swapFlags;
	return r;
}

// The async result's values for `record`, to a request whose tag mask is `mask` and whose last
// record's field_08 is `*last`, as the paravirtual display's do_swap_info_notify_gated makes them:
// the mask, the swap's ID, then for each of the record's five doublewords whose bit (0 to 4) the
// mask sets, the doubleword, the second one being field_08 less `*last` instead. With the mask's
// bit 31 a value is two arguments, its low and then its high word, otherwise one, as the
// paravirtual display sends them and FrameInfoNotifyFunc reads them. A value that would start at
// the twelfth argument is left out. `*last` becomes the record's field_08.
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
			args[n++] = static_cast<uint32_t>(values[i]);
			args[n++] = values[i] >> 32;
		} else {
			args[n++] = values[i];
		}
	}
	*last = record.field_08;
	return n;
}

// The frame info tags ReimsGraphics answers (supported_frame_info), by index: those it fills from
// the device. A swap's presentation time, the refresh the host showed its layer-0 surface in,
// mapped from the host's stamp into the guest's clock (RgfxVblankClock), and the time since the
// one sent before: the record's field_08 and its difference (rgfxSwapInfoArgs's first two
// values). QuartzCore asks for nine tags; the others (Requested_presentation, Vbl_FrameTime,
// Min_FrameTime, Max_FrameTime, Performance_headroom, Thermal_throttle, Last_Fence_or_Event) are
// not answered: the device has none of their values.
constexpr uint32_t kRgfxFrameInfoTagCount = 2;

// Tag `index`'s name, null past the last.
inline const char *rgfxFrameInfoTag(uint32_t index)
{
	switch (index) {
	case 0:
		return "Presentation_time";
	case 1:
		return "Presentation_delta";
	default:
		return nullptr;
	}
}

// A swap's presentation time as the frame info carries it: `shown`, a time of mach_absolute_time's,
// as mach_continuous_time's, from the two clocks read together (`absolute`, `continuous`); 0
// stays 0, a swap that showed nothing.
inline uint64_t rgfxFrameInfoTime(uint64_t shown, uint64_t absolute, uint64_t continuous)
{
	return shown ? shown + (continuous - absolute) : 0;
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
