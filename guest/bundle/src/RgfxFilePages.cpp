/*
 * RgfxFilePages: the classification of read-only memory at a no-copy buffer's make, and its
 * counts (RgfxFilePages.hpp).
 */

#include "RgfxFilePages.hpp"

#include <dispatch/dispatch.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_region.h>
#include <mach/vm_statistics.h>
#include <os/log.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <vector>

#include "RgfxLog.hpp"

namespace {

RgfxFilePages neither(RgfxFileNeither why)
{
	RgfxFilePages out;
	out.why = why;
	return out;
}

// Whether every page of [start, end) resolves to an object with a pager of its own, the file's,
// below any private object over it (mach_vm_page_range_query's VM_PAGE_QUERY_PAGE_EXTERNAL).
bool everyPageExternal(mach_vm_address_t start, mach_vm_address_t end, uint64_t page)
{
	const uint64_t pages = (end - start) / page;
	std::vector<int> dispositions(pages);
	mach_vm_size_t count = pages;
	if (mach_vm_page_range_query(mach_task_self(), start, end - start,
	                             reinterpret_cast<mach_vm_address_t>(dispositions.data()),
	                             &count) != KERN_SUCCESS ||
	    count != pages)
		return false;
	for (int disposition : dispositions) {
		if ((disposition & VM_PAGE_QUERY_PAGE_EXTERNAL) == 0 ||
		    (disposition & (VM_PAGE_QUERY_PAGE_COPIED | VM_PAGE_QUERY_PAGE_PAGED_OUT)) != 0)
			return false;
	}
	return true;
}

// The clause of anonymous memory, the region `region` (RgfxFilePages.hpp, UNCHANGING ANONYMOUS
// MEMORY): unchanging where its maximum protection has no write and the kernel answers it SM_COW
// with no object below the region's own; else not a file.
RgfxFilePages anonymousPagesOf(const vm_region_submap_short_info_data_64_t &region)
{
	if ((region.max_protection & VM_PROT_WRITE) != 0 || region.share_mode != SM_COW ||
	    region.shadow_depth != 0)
		return neither(kRgfxNeitherNotAFile);
	RgfxFilePages out;
	out.clause = kRgfxAnonymousCopyOnWrite;
	out.why = kRgfxNeitherNone;
	return out;
}

// Whether the file system whose fsid is `fsid` is mounted read-only (getfsstat, which reads no
// path).
bool readOnlyVolume(const fsid_t &fsid)
{
	const int mounts = getfsstat(nullptr, 0, MNT_NOWAIT);
	if (mounts <= 0)
		return false;
	// Room for a few mounted meanwhile; one mounted past them is not looked at.
	std::vector<struct statfs> volumes(static_cast<size_t>(mounts) + 4);
	const int bytes = static_cast<int>(volumes.size() * sizeof(struct statfs));
	const int got = getfsstat(volumes.data(), bytes, MNT_NOWAIT);
	for (int i = 0; i < got; i++) {
		const struct statfs &v = volumes[static_cast<size_t>(i)];
		if (v.f_fsid.val[0] == fsid.val[0] && v.f_fsid.val[1] == fsid.val[1])
			return (v.f_flags & MNT_RDONLY) != 0;
	}
	return false;
}

} // namespace

RgfxFilePages rgfxFilePagesOf(const void *address, uint64_t length)
{
	const uint64_t page = vm_page_size;
	const uintptr_t at = reinterpret_cast<uintptr_t>(address);
	if (length == 0 || at + length < at)
		return neither(kRgfxNeitherOther);
	const mach_vm_address_t start = at & ~(page - 1);
	const mach_vm_address_t end = (at + length + page - 1) & ~(page - 1);

	// The VM region holding the range's first page, at the process's own map (no submap entered).
	mach_vm_address_t base = start;
	mach_vm_size_t size = 0;
	natural_t depth = 0;
	vm_region_submap_short_info_data_64_t region = {};
	mach_msg_type_number_t count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
	if (mach_vm_region_recurse(mach_task_self(), &base, &size, &depth,
	                           reinterpret_cast<vm_region_recurse_info_t>(&region),
	                           &count) != KERN_SUCCESS ||
	    base > start)
		return neither(kRgfxNeitherOther);
	if (base + size < end)
		return neither(kRgfxNeitherSpansRegions);
	if (region.is_submap)
		return neither(kRgfxNeitherNotAFile);
	if (!region.external_pager)
		return anonymousPagesOf(region);
	// The process can never write it: its maximum protection has no write.
	const bool copyOnWrite = region.share_mode == SM_COW;
	if ((region.max_protection & VM_PROT_WRITE) != 0)
		return neither(copyOnWrite ? kRgfxNeitherWritableMaxProtection
		                           : kRgfxNeitherWritableMapping);
	// Every page the process sees is the file's: the region's object is the file's, or each page
	// resolves to it below the private objects over it.
	if (region.shadow_depth != 0 && !everyPageExternal(start, end, page))
		return neither(kRgfxNeitherShadowed);

	// The vnode the kernel maps the region from, and its volume.
	struct proc_regionwithpathinfo info = {};
	const int got = proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, start, &info, sizeof(info));
	if (got != static_cast<int>(sizeof(info)))
		return neither(kRgfxNeitherOther);
	if (info.prp_prinfo.pri_address != base)
		return neither(kRgfxNeitherOther); // the region changed meanwhile
	if (info.prp_vip.vip_path[0] == '\0')
		return neither(kRgfxNeitherNotAFile);
	if (!readOnlyVolume(info.prp_vip.vip_vi.vi_fsid))
		return neither(kRgfxNeitherWritableVolume);
	RgfxFilePages out;
	out.clause = copyOnWrite ? kRgfxFileCopyOnWrite : kRgfxFileMaxNoWrite;
	out.why = kRgfxNeitherNone;
	return out;
}

namespace {

struct Totals {
	std::atomic<uint64_t> classified{0};
	std::atomic<uint64_t> count[kRgfxFileClauses] = {};
	std::atomic<uint64_t> bytes[kRgfxFileClauses] = {};
	std::atomic<uint64_t> neitherCount[kRgfxNeitherKinds] = {};
	std::atomic<uint64_t> neitherBytes[kRgfxNeitherKinds] = {};
	std::atomic<uint64_t> kextRefused{0};
	std::atomic<uint64_t> kextRefusedBytes{0};
	std::atomic<uint64_t> ns{0};
	std::atomic<uint64_t> buckets[RgfxFilePagesCounts::kBuckets] = {};
};

Totals &totals()
{
	static Totals *t = new Totals;
	return *t;
}

unsigned bucketOf(uint64_t ns)
{
	if (ns < 4)
		return static_cast<unsigned>(ns);
	const unsigned msb = 63u - static_cast<unsigned>(__builtin_clzll(ns));
	return msb * 4 + static_cast<unsigned>((ns >> (msb - 2)) & 3);
}

// A report is due within a minute of the first change since the last (docs/conventions.md: a
// summary at most once a minute).
std::atomic<bool> gReportDue{false};
const int64_t kReportNs = 60ll * 1000 * 1000 * 1000;

const char *const kClauseNames[kRgfxFileClauses] = {
    "max-no-write", "copy-on-write", "anonymous copy-on-write", "neither",
};

const char *const kNeitherNames[kRgfxNeitherKinds] = {
    "none",     "not a file",      "spans regions", "writable mapping", "writable max protection",
    "shadowed", "writable volume", "other",
};

// A line built in parts, cut at its end.
struct Line {
	char text[1536];
	size_t n = 0;
	__attribute__((format(printf, 2, 3))) void put(const char *format, ...)
	{
		if (n >= sizeof(text))
			return;
		va_list args;
		va_start(args, format);
		const int wrote = vsnprintf(text + n, sizeof(text) - n, format, args);
		va_end(args);
		if (wrote > 0)
			n += static_cast<size_t>(wrote);
	}
};

void report(void *)
{
	gReportDue.store(false);
	const RgfxFilePagesCounts c = rgfxFilePagesCounts();
	Line line;
	line.put("%llu classified, %.3f ms;", (unsigned long long)c.classified, c.ns / 1e6);
	for (unsigned i = 0; i < kRgfxFileClauses; i++)
		line.put("%s %s %llu %llu KiB", i == 0 ? "" : ",", kClauseNames[i],
		         (unsigned long long)c.count[i], (unsigned long long)(c.bytes[i] / 1024));
	line.put(" (");
	for (unsigned i = 1; i < kRgfxNeitherKinds; i++)
		line.put("%s%s %llu %llu KiB", i == 1 ? "" : ", ", kNeitherNames[i],
		         (unsigned long long)c.neitherCount[i],
		         (unsigned long long)(c.neitherBytes[i] / 1024));
	line.put("); the kext refused %llu %llu KiB; times, ns from: count,",
	         (unsigned long long)c.kextRefused, (unsigned long long)(c.kextRefusedBytes / 1024));
	for (unsigned i = 0; i < RgfxFilePagesCounts::kBuckets; i++) {
		if (c.buckets[i] != 0)
			line.put(" %llu: %llu", (unsigned long long)rgfxFilePagesBucketNs(i),
			         (unsigned long long)c.buckets[i]);
	}
	// The process's id and name are the log's own.
	os_log_info(rgfxLog(), "RGFX: file pages, totals: %{public}s", line.text);
}

} // namespace

uint64_t rgfxFilePagesBucketNs(unsigned i)
{
	if (i < 4)
		return i;
	if (i < 8)
		return 4;
	return static_cast<uint64_t>(4 + i % 4) << (i / 4 - 2);
}

void rgfxFilePagesNote(const RgfxFilePages &pages, uint64_t length, uint64_t ns, bool kextRefused)
{
	Totals &t = totals();
	t.classified++;
	t.count[pages.clause]++;
	t.bytes[pages.clause] += length;
	if (pages.clause == kRgfxFileNeither) {
		t.neitherCount[pages.why]++;
		t.neitherBytes[pages.why] += length;
	}
	if (kextRefused) {
		t.kextRefused++;
		t.kextRefusedBytes += length;
	}
	t.ns += ns;
	t.buckets[bucketOf(ns)]++;
	bool due = false;
	if (gReportDue.compare_exchange_strong(due, true))
		dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, kReportNs),
		                 dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), nullptr, &report);
}

RgfxFilePagesCounts rgfxFilePagesCounts()
{
	Totals &t = totals();
	RgfxFilePagesCounts c = {};
	c.classified = t.classified;
	for (unsigned i = 0; i < kRgfxFileClauses; i++) {
		c.count[i] = t.count[i];
		c.bytes[i] = t.bytes[i];
	}
	for (unsigned i = 0; i < kRgfxNeitherKinds; i++) {
		c.neitherCount[i] = t.neitherCount[i];
		c.neitherBytes[i] = t.neitherBytes[i];
	}
	c.kextRefused = t.kextRefused;
	c.kextRefusedBytes = t.kextRefusedBytes;
	c.ns = t.ns;
	for (unsigned i = 0; i < RgfxFilePagesCounts::kBuckets; i++)
		c.buckets[i] = t.buckets[i];
	return c;
}
