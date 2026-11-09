// RgfxLegacySwapInfo: the frame info an arm64 WindowServer learns its swaps' presentation times
// from, sent as Apple's paravirtual display (AppleParavirtDisplay) sends its swap-info
// notifications on each arm64 release the arm64e kext runs on (read from 14.6.1's, 15.6.1's,
// 26.6.2's and 27.0.1's AppleParavirtGPU and IOMobileGraphicsFamily):
// - IOMobileFramebufferLegacy's swap_complete_gated calls the display's notify_swap_done_gated with
//   the completed swap's ID. The paravirtual display makes the swap's record (RgfxSwapInfoRecord)
//   and hands it to the class's swap_info_notify_dispatch_gated, which calls the display's
//   do_swap_info_notify_gated for each notify request the class holds.
// - do_swap_info_notify_gated sends the request's client the record's values (rgfxSwapInfoArgs): the
//   request's tag mask, the swap's ID, and the record's values the mask asks for. The paravirtual
//   display sends them as an async result; ReimsGraphics enqueues them in the client's shared data
//   queue (below).
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
// bit i for tag i; a record's values come in the tags' order, each one word where the mask's
// bit 31 is clear (two, its low and then its high word, where it is set), and the library turns
// them into a dictionary of the tags' names. Its timestamps are mach_continuous_time's, which the
// library turns into mach_absolute_time's (kern_egest_timestamp), as the class's own
// get_current_absolute_time reads mach_continuous_time.
//
// The records go through the client's shared data queue, as on an Apple display: on every release
// the class adds a frame info request without an async reference (add_notification_gated, where
// its finfo_support_io_shared_queue answers true, as it does on each), so an async result would
// reach no port. supported_frame_info answers that the records come in a data queue, which
// IOMobileFramebufferUserClient keeps (s_supported_frame_info) and makes at the client's frame
// info port (registerNotificationPort), and which IOMobileFramebuffer.framework maps
// (clientMemoryForType 3) and reads (FrameInfoNotifyFuncIOShq, PackAndDispatchFrameInfoNotifictionData):
// each entry is a record's words, the same as an async result's arguments, copied into a buffer of
// 0x100 bytes. The display enqueues each through the request's user client
// (IOMobileFramebufferUserClient::enqueueInIOSharedDataQueue, which every release exports).
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

// In a notify request (IOMFBNotifyRequest), the same on each release: the user client it is for,
// the first word of the request's arguments, which add_notification_gated stores at 0x10 and reads
// the task of (its 0xe8, as IOMobileFramebufferUserClient's registerNotificationPort does); the tag
// mask; and the field_08 of the last record sent, which the next one's second value is taken from.
constexpr uint32_t kRgfxNotifyRequestClient = 0x10;
constexpr uint32_t kRgfxNotifyRequestMask = 0x78;
constexpr uint32_t kRgfxNotifyRequestLast = 0x80;

// The notification type set_notify_tag_mask_gated takes a mask for (IOMFB_NotificationType).
constexpr uint32_t kRgfxSwapInfoNotification = 3;

// The most words a record carries here, and the records the paravirtual display keeps while the
// class holds them.
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

// A record's words for `record`, to a request whose tag mask is `mask` and whose last
// record's field_08 is `*last`, as the paravirtual display's do_swap_info_notify_gated makes them:
// the mask, the swap's ID, then for each of the record's five doublewords whose bit (0 to 4) the
// mask sets, the doubleword, the second one being field_08 less `*last` instead. With the mask's
// bit 31 a value is two words, its low and then its high word, otherwise one, as the paravirtual
// display sends them and the library reads them (FrameInfoNotifyFunc from an async result,
// PackAndDispatchFrameInfoNotifictionData from a queue's entry). A value that would start at the
// twelfth word is left out. `*last` becomes the record's field_08. Answers the count of `args`
// written.
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

// The record do_swap_info_notify_gated enqueues for the notify request at `request`, whose tag mask
// and last time lie at their offsets (the last becomes the record's field_08): its words, in
// `words`, and their count (rgfxSwapInfoArgs).
inline uint32_t rgfxFrameInfoRecord(uint8_t *request, const RgfxSwapInfoRecord &record,
                                    uint64_t words[kRgfxSwapInfoMaxArgs + 1])
{
	const uint32_t mask = *reinterpret_cast<const uint32_t *>(request + kRgfxNotifyRequestMask);
	uint64_t *last = reinterpret_cast<uint64_t *>(request + kRgfxNotifyRequestLast);
	return rgfxSwapInfoArgs(mask, last, record, words);
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
// as mach_continuous_time's, `lead` ticks ahead of it (RgfxFrameInfoClock); 0 stays 0, a swap that
// showed nothing.
inline uint64_t rgfxFrameInfoTime(uint64_t shown, int64_t lead)
{
	return shown ? shown + static_cast<uint64_t>(lead) : 0;
}

// The presentation times a display's frame info carries, in mach_continuous_time's clock, as a
// Mac's are: each a refresh after the one given before. QuartzCore takes each back into
// mach_absolute_time's by the continuous clock's lead, which it reads with mach_get_times (its
// frame_info_callback, read from 27.0.1's), so a time given here reaches its kdebug trace
// (0x31ca002c) and its presentation timing as the shown refresh plus the error of the lead used
// here.
//
// The lead, the time the system has spent asleep, is read from the clocks: the continuous one
// read between two reads of the absolute one (rgfxReadClocks). A reading bounds it: it is at least
// the continuous read less the later absolute read, and exactly that where the two absolute reads
// agree. Taken from each swap's own reading, the lead came short by the ticks between the reads
// (24 MHz, 41.7 ns each) where none of the swap's tries agreed, as happens in an arm64 VM: a swap
// shown in the same refresh as the one before it got a time 1 or 2 ticks before that one's
// (Ambrosia's g147 boots of 14.6.1, 15.6.1 and 26.6.2: 2 to 5 of 180 swaps in 3 s). The lead never
// shrinks, as the system never gives back time it slept, so the clock keeps the largest lower bound
// it has read: exact from the first reading whose continuous read came in the later absolute
// read's tick, and never past the lead.
//
// A display shows one frame a refresh, so each swap's time is a refresh after the one given
// before. The refresh an answer maps to (RgfxVblankClock::shownAt) is the last VBLANK's taken, so
// an answer taken before its own refresh's VBLANK maps to the refresh before it, which the swap
// before may have been given; given that refresh's time, two swaps a refresh apart got one time
// (Ambrosia's g147 and g148 boots: 6 to 18 of 180 swaps in 3 s, one frame apart in wall time). A
// swap whose refresh is not after the last one given is given the refresh after that one, a
// period on, where that lies at most kAheadPeriods past its own; past that (swaps coming faster
// than the refreshes), the time given before, so their times do not run ahead of the refreshes.
// The refreshes given are kept in mach_absolute_time's clock, the one QuartzCore takes the times
// back into: a sleep between two swaps grows the lead the second is given with, and leaves them a
// period apart there all the same.
class RgfxFrameInfoClock
{
public:
	// How many periods past its own refresh a swap's time may lie.
	static constexpr uint64_t kAheadPeriods = 2;

	// A reading of the clocks: the absolute one, then the continuous one, then the absolute one
	// again.
	void read(uint64_t before, uint64_t continuous, uint64_t after)
	{
		(void)before;
		const int64_t atLeast = static_cast<int64_t>(continuous - after);
		if (!have_ || atLeast > lead_)
			lead_ = atLeast;
		have_ = true;
	}

	// The time of a swap shown at `shown`, mach_absolute_time's, in the continuous clock, on a
	// display whose refreshes are `period` ticks apart: its refresh, or a period after the refresh
	// given before where its own is not, if that lies at most kAheadPeriods past its own, else the
	// refresh given before; then the lead added. 0 for a swap that showed nothing, which changes
	// nothing.
	uint64_t time(uint64_t shown, uint64_t period)
	{
		if (!shown)
			return 0;
		uint64_t refresh = shown;
		if (haveGiven_ && refresh < given_ + period) {
			const uint64_t next = given_ + period;
			refresh = next - refresh <= kAheadPeriods * period ? next : given_;
		}
		haveGiven_ = true;
		given_ = refresh;
		return rgfxFrameInfoTime(refresh, lead_);
	}

	// The lead as read so far.
	int64_t lead() const { return lead_; }

private:
	int64_t lead_ = 0;
	bool have_ = false;
	bool haveGiven_ = false;
	uint64_t given_ = 0; // the refresh given last, mach_absolute_time's
};

// How many times rgfxReadClocks reads the clocks at most.
constexpr int kRgfxClockReads = 8;

// The absolute and the continuous clock read by `absolute` and `continuous` into `clock`: up to
// kRgfxClockReads readings, the continuous clock between two reads of the absolute one, until the
// two absolute reads agree.
template <typename Absolute, typename Continuous>
inline void rgfxReadClocks(Absolute absolute, Continuous continuous, RgfxFrameInfoClock &clock)
{
	for (int i = 0; i < kRgfxClockReads; i++) {
		const uint64_t before = absolute();
		const uint64_t c = continuous();
		const uint64_t after = absolute();
		clock.read(before, c, after);
		if (after == before)
			return;
	}
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
