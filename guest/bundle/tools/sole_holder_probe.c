// sole_holder_probe: whether the guest's kernel answers anonymous memory as the bundle's clause for
// a sole holder's memory reads it (guest/bundle/src/RgfxFilePages.hpp, UNCHANGING ANONYMOUS
// MEMORY, kRgfxAnonymousPrivate), and whether the kext makes and refuses a sole holder's pages
// block as its RgfxPagesBlock.hpp says, on the kext's own selectors
// (guest/kext/src/RgfxUserClientABI.hpp), with no Metal. The bundle's clause rests on these
// answers; a release that answers otherwise is one to read the clause against again.
// Each step prints "sole_holder_probe: STEP ok|FAIL|SKIP detail", and the last line is
// "sole_holder_probe: OK N steps, S skipped" or "sole_holder_probe: FAIL F of N steps", exit 1:
//   alone       anonymous memory whose maximum protection was taken down to read reads SM_PRIVATE
//               at a shadow depth of 0, and mach_vm_page_info's VM_PAGE_INFO_BASIC answers a count
//               of 1 at depth 0, with the region's object id in its id's low 32 bits;
//   entry       a memory entry of it (mach_make_memory_entry_64, MAP_MEM_VM_SHARE) holds it too:
//               a count of 2, and 1 again once the entry goes;
//   alias       so does a second mapping of it (mach_vm_remap, sharing), whose maximum protection
//               has no write: a count of 2, and 1 once it goes;
//   no-write    mprotect cannot give the memory write back;
//   footprint   asked for its regions' footprint (vm.self_region_footprint), the process is not
//               answered a count of 1 with the region's id for the memory while the entry holds it
//               (SKIP where the process cannot ask);
//   wire-write  the kext's writable pages block of the memory is refused, as the kernel refuses its
//               wire for writing (kIOReturnBadArgument), while a read-only one is made;
//   sole-holder the kext makes a sole holder's block of it (asked again for 1 s while it refuses,
//               as another process's make of a writable block in flight refuses it too);
//   own-wire    memory a writable block of the kext's covers, its maximum protection taken down
//               after: a count of 1 still, as a wire holds no reference, and the kext refuses a
//               sole holder's block of it (kIOReturnNotPermitted) until that block is freed and
//               its release answered (3 s at most).
// The kext's steps are skipped, saying why, where the kext or the device does not take a sole
// holder's memory (kRgfxKextFeatureSoleHolderPages; RGFX_FEATURE_PAGES_BUFFERS,
// RGFX_FEATURE_READ_ONLY_PAGES and RGFX_FEATURE_UNCHANGING_PAGES). Each block made is freed.
// Build: xcrun clang -arch x86_64 -O2 -mmacosx-version-min=11.0 -Wall -Wextra -Werror
//        -framework IOKit sole_holder_probe.c -o sole_holder_probe
//        (-arch arm64 -mmacosx-version-min=13.0 for an arm64 guest)
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_region.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "../../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h"
#include "../../kext/src/RgfxUserClientABI.hpp"

static int steps, failed, skipped;

static void step(const char *name, int ok, const char *detail)
{
	steps++;
	failed += !ok;
	printf("sole_holder_probe: %s %s %s\n", name, ok ? "ok" : "FAIL", detail);
}

static void skip(const char *name, const char *why)
{
	steps++;
	skipped++;
	printf("sole_holder_probe: %s SKIP %s\n", name, why);
}

static int regionAt(mach_vm_address_t at, vm_region_submap_short_info_data_64_t *info)
{
	mach_vm_address_t base = at;
	mach_vm_size_t size = 0;
	natural_t depth = 0;
	mach_msg_type_number_t count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
	memset(info, 0, sizeof(*info));
	return mach_vm_region_recurse(mach_task_self(), &base, &size, &depth,
	                              (vm_region_recurse_info_t)info, &count) == KERN_SUCCESS &&
	       base <= at;
}

static int pageAt(mach_vm_address_t at, vm_page_info_basic_data_t *info)
{
	mach_msg_type_number_t count = VM_PAGE_INFO_BASIC_COUNT;
	memset(info, 0, sizeof(*info));
	return mach_vm_page_info(mach_task_self(), at, VM_PAGE_INFO_BASIC, (vm_page_info_t)info,
	                         &count) == KERN_SUCCESS &&
	       count == VM_PAGE_INFO_BASIC_COUNT;
}

// The object's count at `at`, as VM_PAGE_INFO_BASIC answers it; -1 where it does not.
static int countAt(mach_vm_address_t at)
{
	vm_page_info_basic_data_t info;
	return pageAt(at, &info) ? info.ref_count : -1;
}

// Anonymous memory of `bytes`, filled, its maximum protection taken down to read where `lower`.
static mach_vm_address_t anonymous(mach_vm_size_t bytes, int lower)
{
	mach_vm_address_t at = 0;
	if (mach_vm_allocate(mach_task_self(), &at, bytes, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
		return 0;
	memset((void *)(uintptr_t)at, 0x5a, bytes);
	if (lower &&
	    mach_vm_protect(mach_task_self(), at, bytes, TRUE, VM_PROT_READ) != KERN_SUCCESS) {
		mach_vm_deallocate(mach_task_self(), at, bytes);
		return 0;
	}
	return at;
}

struct Client {
	io_connect_t conn;
	uint64_t device;
	uint64_t kext;
};

static int open_(struct Client *c)
{
	memset(c, 0, sizeof(*c));
	io_service_t svc =
	    IOServiceGetMatchingService(MACH_PORT_NULL, IOServiceMatching("RgfxAccelerator"));
	if (!svc)
		return 0;
	kern_return_t kr = IOServiceOpen(svc, mach_task_self(), kRgfxAccelUserClientType, &c->conn);
	IOObjectRelease(svc);
	if (kr != KERN_SUCCESS)
		return 0;
	uint64_t features[3] = {0};
	uint32_t n = 3;
	if (IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodFeatures, NULL, 0, features, &n) !=
	    KERN_SUCCESS)
		return 0;
	c->device = features[0];
	c->kext = features[1];
	uint64_t ctx[5] = {0};
	n = 5;
	return IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodOpenContext, NULL, 0, ctx, &n) ==
	       KERN_SUCCESS;
}

static kern_return_t pagesBlock(struct Client *c, mach_vm_address_t at, uint64_t bytes,
                                uint64_t how, uint64_t *offset)
{
	const uint64_t in[3] = {at, bytes, how};
	uint64_t out[2] = {0};
	uint32_t n = 2;
	kern_return_t kr =
	    IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodPagesBlock, in, 3, out, &n);
	*offset = out[0];
	return kr;
}

static void freeBlock(struct Client *c, uint64_t offset)
{
	IOConnectCallScalarMethod(c->conn, kRgfxAccelMethodFree, &offset, 1, NULL, NULL);
}

int main(void)
{
	const mach_vm_size_t bytes = 16 * (mach_vm_size_t)vm_page_size;
	char say[256];

	mach_vm_address_t mem = anonymous(bytes, 1);
	vm_region_submap_short_info_data_64_t region;
	vm_page_info_basic_data_t page;
	const int got = mem && regionAt(mem, &region) && pageAt(mem, &page);
	snprintf(say, sizeof say, "share mode %d, shadow depth %d, count %d at depth %d, ids %s",
	         got ? region.share_mode : -1, got ? region.shadow_depth : -1,
	         got ? page.ref_count : -1, got ? page.depth : -1,
	         got && (uint32_t)page.object_id == region.object_id ? "match" : "differ");
	step("alone",
	     got && region.share_mode == SM_PRIVATE && region.shadow_depth == 0 &&
	         page.ref_count == 1 && page.depth == 0 &&
	         (uint32_t)page.object_id == region.object_id,
	     say);
	if (!mem) {
		printf("sole_holder_probe: FAIL %d of %d steps: no memory\n", failed, steps);
		return 1;
	}

	mach_port_t entry = MACH_PORT_NULL;
	memory_object_size_t entryBytes = bytes;
	const int made = mach_make_memory_entry_64(mach_task_self(), &entryBytes, mem,
	                                           VM_PROT_READ | MAP_MEM_VM_SHARE, &entry,
	                                           MACH_PORT_NULL) == KERN_SUCCESS;
	const int held = countAt(mem);
	// While the entry holds it: a process that asked for its footprint is answered otherwise.
	int asked = 1, footprintOk = 0, footprintRead = 0;
	const size_t askedBytes = sizeof asked;
	char footprintSay[96] = "";
	if (made && sysctlbyname("vm.self_region_footprint", NULL, NULL, &asked, askedBytes) == 0) {
		footprintRead = 1;
		vm_page_info_basic_data_t told = {0};
		const int toldGot = regionAt(mem, &region) && pageAt(mem, &told);
		footprintOk = toldGot &&
		              !(told.ref_count == 1 && (uint32_t)told.object_id == region.object_id);
		snprintf(footprintSay, sizeof footprintSay, "count %d, ids %s",
		         toldGot ? told.ref_count : -1,
		         toldGot && (uint32_t)told.object_id == region.object_id ? "match" : "differ");
		asked = 0;
		if (sysctlbyname("vm.self_region_footprint", NULL, NULL, &asked, askedBytes) != 0)
			footprintOk = 0;
	}
	if (made)
		mach_port_deallocate(mach_task_self(), entry);
	const int after = countAt(mem);
	char counts[96];
	snprintf(counts, sizeof counts, "count %d with the entry, %d after", held, after);
	step("entry", made && held == 2 && after == 1, made ? counts : "no entry");

	mach_vm_address_t alias = 0;
	vm_prot_t cur = VM_PROT_NONE, max = VM_PROT_NONE;
	const int aliased = mach_vm_remap(mach_task_self(), &alias, bytes, 0, VM_FLAGS_ANYWHERE,
	                                  mach_task_self(), mem, FALSE, &cur, &max,
	                                  VM_INHERIT_NONE) == KERN_SUCCESS;
	const int heldAlias = countAt(mem);
	if (aliased)
		mach_vm_deallocate(mach_task_self(), alias, bytes);
	const int afterAlias = countAt(mem);
	snprintf(say, sizeof say, "count %d with the alias, %d after; its maximum protection 0x%x",
	         heldAlias, afterAlias, max);
	step("alias",
	     aliased && heldAlias == 2 && afterAlias == 1 && (max & VM_PROT_WRITE) == 0,
	     aliased ? say : "no alias");

	step("no-write", mprotect((void *)(uintptr_t)mem, bytes, PROT_READ | PROT_WRITE) != 0,
	     "mprotect read-write refused");

	if (!footprintRead)
		skip("footprint", "the process could not ask for its footprint");
	else
		step("footprint", footprintOk, footprintSay);

	struct Client c;
	const char *why = NULL;
	if (!open_(&c))
		why = "no accelerator context";
	else if (!(c.kext & kRgfxKextFeatureSoleHolderPages))
		why = "the kext takes no sole holder's memory";
	else if ((c.device & (RGFX_FEATURE_PAGES_BUFFERS | RGFX_FEATURE_READ_ONLY_PAGES |
	                      RGFX_FEATURE_UNCHANGING_PAGES)) !=
	         (RGFX_FEATURE_PAGES_BUFFERS | RGFX_FEATURE_READ_ONLY_PAGES |
	          RGFX_FEATURE_UNCHANGING_PAGES))
		why = "the device took no unchanging pages";
	if (why) {
		skip("wire-write", why);
		skip("sole-holder", why);
		skip("own-wire", why);
	} else {
		uint64_t offset = 0;
		const kern_return_t write = pagesBlock(&c, mem, bytes, kRgfxPagesWritable, &offset);
		if (write == KERN_SUCCESS)
			freeBlock(&c, offset);
		const kern_return_t readOnly = pagesBlock(&c, mem, bytes, 0, &offset);
		if (readOnly == KERN_SUCCESS)
			freeBlock(&c, offset);
		snprintf(say, sizeof say, "writable 0x%x, read-only 0x%x", write, readOnly);
		step("wire-write", write == kIOReturnBadArgument && readOnly == KERN_SUCCESS, say);

		// Another process's make of a writable block, in flight at the call, refuses it too
		// (kIOReturnNotPermitted): asked again for up to 1 s.
		const uint64_t sole = kRgfxPagesUnchanging | kRgfxPagesSoleHolder;
		kern_return_t soleMade = kIOReturnError;
		int asks = 0;
		do {
			if (asks)
				usleep(50 * 1000);
			soleMade = pagesBlock(&c, mem, bytes, sole, &offset);
		} while (soleMade == kIOReturnNotPermitted && ++asks < 20);
		if (soleMade == KERN_SUCCESS)
			freeBlock(&c, offset);
		snprintf(say, sizeof say, "0x%x, %d refused before", soleMade, asks);
		step("sole-holder", soleMade == KERN_SUCCESS, say);

		// Writable memory a writable block covers, then its maximum protection taken down.
		mach_vm_address_t wired = anonymous(bytes, 0);
		uint64_t wiredAt = 0, soleAt = 0;
		const kern_return_t over =
		    wired ? pagesBlock(&c, wired, bytes, kRgfxPagesWritable, &wiredAt) : kIOReturnNoMemory;
		const int lowered = over == KERN_SUCCESS &&
		                    mach_vm_protect(mach_task_self(), wired, bytes, TRUE, VM_PROT_READ) ==
		                        KERN_SUCCESS;
		const int count = lowered ? countAt(wired) : -1;
		const kern_return_t refused = lowered ? pagesBlock(&c, wired, bytes, sole, &soleAt)
		                                      : kIOReturnError;
		if (refused == KERN_SUCCESS)
			freeBlock(&c, soleAt);
		if (over == KERN_SUCCESS)
			freeBlock(&c, wiredAt);
		// Made once the writable block's release is answered.
		kern_return_t later = kIOReturnError;
		int waitedMs = 0;
		if (lowered) {
			for (; waitedMs <= 3000; waitedMs += 50) {
				later = pagesBlock(&c, wired, bytes, sole, &soleAt);
				if (later != kIOReturnNotPermitted)
					break;
				usleep(50 * 1000);
			}
			if (later == KERN_SUCCESS)
				freeBlock(&c, soleAt);
		}
		snprintf(say, sizeof say,
		         "writable block 0x%x, then count %d, sole holder's 0x%x, after its free 0x%x in "
		         "%d ms",
		         over, count, refused, later, waitedMs);
		step("own-wire",
		     lowered && count == 1 && refused == kIOReturnNotPermitted && later == KERN_SUCCESS,
		     say);
		if (wired)
			mach_vm_deallocate(mach_task_self(), wired, bytes);
	}
	mach_vm_deallocate(mach_task_self(), mem, bytes);

	if (failed) {
		printf("sole_holder_probe: FAIL %d of %d steps\n", failed, steps);
		return 1;
	}
	printf("sole_holder_probe: OK %d steps, %d skipped\n", steps, skipped);
	return 0;
}
