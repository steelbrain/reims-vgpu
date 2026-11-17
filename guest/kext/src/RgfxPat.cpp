/*
 * The guest PAT's entry 2 to write-back on every CPU, x86-64 only: why, and where it holds, are
 * RgfxPat.hpp's. ReimsGraphics' start runs it once its device answers as ours, and registers for
 * every wake, where it runs again.
 *
 * Each CPU changes its own PAT in an mp_rendezvous_no_intrs action: every online CPU enters with
 * its interrupts off, runs the action, and leaves once all have run it. The action is the SDM's
 * sequence for a PAT change (Intel SDM Vol. 3A, "Programming the PAT", by way of "MTRR
 * Considerations in MP Systems"): caches to no-fill (CR0.CD=1, NW=0), WBINVD, every TLB entry
 * dropped, the write, WBINVD and the TLBs again, caches back. Toggling CR4.PGE drops every TLB
 * entry, global ones and every PCID's. A CPU that holds the rewrite already is left alone. Timed
 * in guests of 8 CPUs, the rendezvous took 112 us on AMD KVM, and the sequence 10.7-22.6 us a CPU
 * on AMD KVM, Intel KVM and WHP. While some CPUs hold the rewrite and others not, the same pages
 * are write-back on some and UC- on others: in two-CPU tests on all three hypervisors (50,000
 * rounds each way of a store on one CPU and a load on the other, and GPU and host writes read by
 * both), no load saw stale data, so a CPU not yet rewritten is slow, not wrong.
 *
 * The kernel calls are com.apple.kpi.unsupported's on x86_64 (xnu's config/Unsupported.x86_64.exports),
 * which the kext links, and IOKit's registerPrioritySleepWakeInterest.
 *
 * Our own code.
 */

#include "RgfxPat.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMessage.h>
#include <IOKit/IOService.h>
#include <kern/clock.h>
#include <libkern/OSAtomic.h>

extern "C" {
// com.apple.kpi.unsupported, x86_64; no SDK header declares them.
void mp_rendezvous_no_intrs(void (*action)(void *), void *arg);

// IOKit's, with C linkage, as IOKit/pwr_mgt/RootDomain.h declares it. That header does not
// compile in the macOS 27.2 SDK: its IOPMAssertionTypes.h no longer defines the
// IOPMDriverAssertion types the header uses.
IONotifier *registerPrioritySleepWakeInterest(IOServiceInterestHandler handler, void *self,
                                              void *ref);
}

namespace {
constexpr uintptr_t kCr0CacheDisable = 1ul << 30; // CR0.CD
constexpr uintptr_t kCr0NotWriteThrough = 1ul << 29; // CR0.NW
constexpr uintptr_t kCr4GlobalPages = 1ul << 7; // CR4.PGE

inline uintptr_t readCr0()
{
	uintptr_t v;
	asm volatile("mov %%cr0, %0" : "=r"(v));
	return v;
}
inline void writeCr0(uintptr_t v) { asm volatile("mov %0, %%cr0" ::"r"(v) : "memory"); }
inline uintptr_t readCr4()
{
	uintptr_t v;
	asm volatile("mov %%cr4, %0" : "=r"(v));
	return v;
}
inline void writeCr4(uintptr_t v) { asm volatile("mov %0, %%cr4" ::"r"(v) : "memory"); }
inline void writeBackAndInvalidate() { asm volatile("wbinvd" ::: "memory"); }
inline uint64_t readPat()
{
	uint32_t lo, hi;
	asm volatile("rdmsr" : "=a"(lo), "=d"(hi) : "c"(RGFX_MSR_PAT));
	return (static_cast<uint64_t>(hi) << 32) | lo;
}
inline void writePat(uint64_t v)
{
	asm volatile("wrmsr" ::"c"(RGFX_MSR_PAT), "a"(static_cast<uint32_t>(v)),
	             "d"(static_cast<uint32_t>(v >> 32))
	             : "memory");
}
// Drops every TLB entry, global ones and every PCID's: a write of CR4 that changes PGE does, in
// either direction.
inline void dropTlbs(uintptr_t cr4)
{
	writeCr4(cr4 ^ kCr4GlobalPages);
	writeCr4(cr4);
}

struct Counts {
	uint32_t cpus;
	uint32_t changed;
};

// One CPU, its interrupts off.
void writeBackAction(void *arg)
{
	Counts *counts = static_cast<Counts *>(arg);
	const uint64_t pat = readPat();
	const uint64_t want = rgfxPatWithDeviceWriteBack(pat);
	if (pat != want) {
		const uintptr_t cr0 = readCr0(), cr4 = readCr4();
		writeCr0((cr0 | kCr0CacheDisable) & ~kCr0NotWriteThrough);
		writeBackAndInvalidate();
		dropTlbs(cr4);
		writePat(want);
		writeBackAndInvalidate();
		dropTlbs(cr4);
		writeCr0(cr0);
		__atomic_fetch_add(&counts->changed, 1u, __ATOMIC_RELAXED);
	}
	__atomic_fetch_add(&counts->cpus, 1u, __ATOMIC_RELAXED);
}

IONotifier *wakeInterest = nullptr;

// A wake: the rewrite runs at both of its messages, the second changing no CPU where the first
// found them all restarted. Measured on a wake from S3 (KVM, S3 on, which reims-ctl leaves off),
// xnu wrote its PAT on all 8 CPUs 87-101 ms after the wake, and this rewrite followed 3-19 ms
// after.
IOReturn wakeMessage(void *, void *, UInt32 type, IOService *, void *, vm_size_t)
{
	if (type == kIOMessageSystemWillPowerOn)
		rgfxPatWriteBackEverywhere("wake, will power on");
	else if (type == kIOMessageSystemHasPoweredOn)
		rgfxPatWriteBackEverywhere("wake, powered on");
	return kIOReturnSuccess;
}
} // namespace

RgfxPatRun rgfxPatWriteBackEverywhere(const char *why)
{
	Counts counts = {};
	const uint64_t start = mach_absolute_time();
	mp_rendezvous_no_intrs(writeBackAction, &counts);
	uint64_t ns = 0;
	absolutetime_to_nanoseconds(mach_absolute_time() - start, &ns);
	const RgfxPatRun run = {counts.cpus, counts.changed, ns};
	IOLog("ReimsGraphics: PAT entry 2 write-back on %u CPUs (%s): %u changed, %llu us\n", run.cpus,
	      why, run.changed, run.ns / 1000);
	return run;
}

void rgfxPatWatchWakes(void)
{
	if (__atomic_load_n(&wakeInterest, __ATOMIC_ACQUIRE))
		return;
	IONotifier *interest = registerPrioritySleepWakeInterest(wakeMessage, nullptr, nullptr);
	if (!interest) {
		IOLog("ReimsGraphics: no sleep/wake interest: a wake from S3 leaves PAT entry 2 UC-\n");
		return;
	}
	IONotifier *none = nullptr;
	if (!__atomic_compare_exchange_n(&wakeInterest, &none, interest, false, __ATOMIC_ACQ_REL,
	                                 __ATOMIC_ACQUIRE))
		interest->remove();
}

void rgfxPatStopWatching(void)
{
	if (IONotifier *interest = __atomic_exchange_n(&wakeInterest, nullptr, __ATOMIC_ACQ_REL))
		interest->remove();
}
