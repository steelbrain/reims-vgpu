/*
 * RgfxInterop: see RgfxInterop.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxInterop.h"

#import <objc/runtime.h>

#include <atomic>
#include <mutex>
#include <set>
#include <utility>

namespace {

std::atomic<uint64_t> g_traceIds{0};

std::mutex g_lock;
std::set<std::pair<const void *, const void *>> *g_unrecognized; // (class, selector) pairs

} // namespace

uint64_t RgfxNextTraceObjectId(void)
{
	return g_traceIds.fetch_add(1, std::memory_order_relaxed) + 1;
}

void RgfxLogUnrecognized(Class cls, SEL sel)
{
	bool first;
	{
		std::lock_guard<std::mutex> hold(g_lock);
		if (g_unrecognized == nullptr)
			g_unrecognized = new std::set<std::pair<const void *, const void *>>();
		first = g_unrecognized->insert({(__bridge const void *)cls, (const void *)sel}).second;
	}
	if (first)
		NSLog(@"RGFX: unrecognized: -[%s %s]", class_getName(cls), sel_getName(sel));
}

BOOL RgfxUnrecognizedWasLogged(Class cls, SEL sel)
{
	std::lock_guard<std::mutex> hold(g_lock);
	return g_unrecognized != nullptr &&
	       g_unrecognized->count({(__bridge const void *)cls, (const void *)sel}) != 0;
}
