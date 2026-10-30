/*
 * What the kext's classes log about their callers: the calling process, and a gate that lets
 * a site log only the first calls of each key, so a caller in a loop logs a line, not a flood.
 */

#pragma once

#include <IOKit/IOLib.h>
#include <sys/proc.h>

// The calling process, for a log line: its pid (0 for the kernel) and its name.
struct RgfxCaller {
	int pid;
	char name[32];
	RgfxCaller() : pid(proc_selfpid()) { proc_selfname(name, sizeof(name)); }
};

// Whether this call is among the first `limit` calls of `key` at `site` (the address of
// something of the site's own). The sites' keys share one table of kSlots; a key past it is
// never logged.
inline bool rgfxFirstCalls(const void *site, UInt64 key, UInt32 limit)
{
	static constexpr UInt32 kSlots = 512;
	struct Slot {
		UInt64 tag;
		UInt32 count;
	};
	static Slot slots[kSlots];
	UInt64 tag = (reinterpret_cast<uintptr_t>(site) * 0x9e3779b97f4a7c15ull) ^ key;
	tag |= 1; // 0 is a free slot
	for (UInt32 i = 0; i < kSlots; i++) {
		Slot &slot = slots[(tag + i) % kSlots];
		UInt64 seen = __atomic_load_n(&slot.tag, __ATOMIC_ACQUIRE);
		if (seen == 0) {
			UInt64 expected = 0;
			seen = __atomic_compare_exchange_n(&slot.tag, &expected, tag, false,
			                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
			         ? tag
			         : expected;
		}
		if (seen == tag)
			return __atomic_fetch_add(&slot.count, 1, __ATOMIC_RELAXED) < limit;
	}
	return false;
}

// A key for a string, for rgfxFirstCalls (FNV-1a).
inline UInt64 rgfxKeyOf(const char *s)
{
	UInt64 h = 0xcbf29ce484222325ull;
	for (; s && *s; s++)
		h = (h ^ static_cast<UInt8>(*s)) * 0x100000001b3ull;
	return h;
}

#if RGFX_FB_TRACE
// A trace build's line for a property a process (not the kernel) reads of `who`: the first
// two reads of each key, and whether the entry has it.
inline void rgfxTracePropertyRead(const char *who, const char *key, bool found)
{
	static char site;
	RgfxCaller caller;
	if (caller.pid == 0 || !rgfxFirstCalls(&site, rgfxKeyOf(who) ^ rgfxKeyOf(key), 2))
		return;
	IOLog("RGFX prop %s read '%s' by %s[%d]: %s\n", who, key ? key : "(null)", caller.name,
	      caller.pid, found ? "found" : "MISSING");
}
#endif
