// RgfxLogOnce: a gate that lets a site log each kind of failure once in a process
// (docs/conventions.md, "once per kind"): a caller that meets the same failure each frame, draw
// or command buffer logs one line for it, not one per call.
//
// A kind is the site (the address of something of the site's own, a `static char`) and, where
// the site has more than one, a key: a status, a selector, or a reason's text
// (rgfxLogKindOf). A site with one kind passes none.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <utility>

// The kinds a process keeps; a kind past them is never logged.
constexpr std::size_t kRgfxLogKindsKept = 256;

// Whether this is the first call of `kind` at `site` in this process.
inline bool rgfxLogFirst(const void *site, uint64_t kind = 0)
{
	static std::mutex lock;
	// Never destroyed: a thread may log while the process exits.
	static auto *seen = new std::set<std::pair<const void *, uint64_t>>();
	std::lock_guard<std::mutex> hold(lock);
	if (seen->size() >= kRgfxLogKindsKept)
		return false;
	return seen->emplace(site, kind).second;
}

// A key for a reason's text, for rgfxLogFirst (FNV-1a), as the kext's rgfxKeyOf.
inline uint64_t rgfxLogKindOf(const char *s)
{
	uint64_t h = 0xcbf29ce484222325ull;
	for (; s && *s; s++)
		h = (h ^ static_cast<uint8_t>(*s)) * 0x100000001b3ull;
	return h;
}
