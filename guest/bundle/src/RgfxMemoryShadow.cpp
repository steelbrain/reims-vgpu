/*
 * RgfxMemoryShadow: the counts of the process's shadows' syncs, and their report
 * (RgfxMemoryShadow.hpp).
 */

#include "RgfxMemoryShadow.hpp"

#include <dispatch/dispatch.h>
#include <os/log.h>

#include "RgfxLog.hpp"

namespace {

struct Totals {
	std::atomic<uint64_t> commits[RgfxShadowCounts::kKinds] = {};
	std::atomic<uint64_t> bytes[RgfxShadowCounts::kKinds] = {};
	std::atomic<uint64_t> ns[RgfxShadowCounts::kKinds] = {};
	std::atomic<uint64_t> writeBacks{0};
};

Totals &totals()
{
	static Totals *t = new Totals;
	return *t;
}

// A report is due within a minute of the first change since the last (docs/conventions.md: a
// summary at most once a minute).
std::atomic<bool> gReportDue{false};
const int64_t kReportNs = 60ll * 1000 * 1000 * 1000;

void report(void *)
{
	gReportDue.store(false);
	const RgfxShadowCounts c = rgfxShadowCounts();
	const unsigned w = RgfxShadowCounts::kWritable, r = RgfxShadowCounts::kReadOnly;
	// The process's id and name are the log's own.
	os_log_info(rgfxLog(),
	            "RGFX: memory shadows, totals: writable: %llu commits, %llu write-backs, %llu "
	            "KiB synced, %.3f ms; read-only: %llu commits, %llu KiB copied whole, %.3f ms",
	            (unsigned long long)c.commits[w], (unsigned long long)c.writeBacks,
	            (unsigned long long)(c.bytes[w] / 1024), c.ns[w] / 1e6,
	            (unsigned long long)c.commits[r], (unsigned long long)(c.bytes[r] / 1024),
	            c.ns[r] / 1e6);
}

} // namespace

void rgfxShadowsNote(const RgfxShadowCounts &delta)
{
	Totals &t = totals();
	for (unsigned k = 0; k < RgfxShadowCounts::kKinds; k++) {
		t.commits[k] += delta.commits[k];
		t.bytes[k] += delta.bytes[k];
		t.ns[k] += delta.ns[k];
	}
	t.writeBacks += delta.writeBacks;
	bool due = false;
	if (gReportDue.compare_exchange_strong(due, true))
		dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, kReportNs),
		                 dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), nullptr, &report);
}

RgfxShadowCounts rgfxShadowCounts()
{
	Totals &t = totals();
	RgfxShadowCounts c = {};
	for (unsigned k = 0; k < RgfxShadowCounts::kKinds; k++) {
		c.commits[k] = t.commits[k];
		c.bytes[k] = t.bytes[k];
		c.ns[k] = t.ns[k];
	}
	c.writeBacks = t.writeBacks;
	return c;
}
