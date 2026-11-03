// spguard: what the kernel's VM guard makes of the device's superpages (src/RgfxSuperpages.hpp), in
// a scratch process with its VM guard on, as Apple's platform binaries and WebKit's GPU process
// have it. The process turns its own guard on (task_set_exc_guard_behavior; a process may only add
// bits) and takes its EXC_GUARD exceptions on a port of its own, recording each one's codes and
// answering it handled, so a guard exception shows and the process lives on. To run in an x86-64
// guest; no Metal, no GPU.
//   spguard CASE...
// The cases:
//   behavior      the process's EXC_GUARD behaviour as it started, then with the VM guard on.
//   old.anywhere  car 8's call: one mach_vm_allocate of more superpages than the kernel has free
//                 RAM for (the free pages' bytes and 512 MiB, at most the RAM): its answer, and
//                 the guard exceptions it raised (code, subcode).
//   old.fixed     the same one call, at a fixed address of a free range of its own: the
//                 exception's subcode less the address, in superpages (the kernel's cleanup takes
//                 down past the end of the last superpage entry it made).
//   new.short     rgfxTakeSuperpages over as many bytes: each call the kernel was asked (the
//                 reservation, how many superpages it made over it before the first refusal and
//                 that refusal, the deallocation), the guard exceptions, and the process's
//                 IOAccelerator-tagged bytes after it (0: nothing left mapped).
//   new.room.MIB  rgfxTakeSuperpages of MIB MiB with room for them: its answer and time, the
//                 entries the range holds (one a superpage, each 2 MiB and tagged), the guard
//                 exceptions; then the same bytes in car 8's one call, for its time.
//   fatal.old     a child with the VM guard fatal (VM_DELIVER | VM_FATAL, WebKit's GPU process's)
//                 and no port of its own makes the old.anywhere call: how it ended.
//   fatal.new     the same child making the new.short call: how it ended.
// One line a case: "spg: NAME key=value ...". Build (on a Mac): `make -C guest/bundle spguard`,
// build/spguard-x86_64.
//
// Our own code; it names only the SDK's public Mach calls.

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <vector>

#include "RgfxSuperpages.hpp"

namespace {

constexpr uint64_t kMiB = 1ull << 20;

double nowUs()
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return static_cast<double>(mach_absolute_time()) * tb.numer / tb.denom / 1000.0;
}

// --- the process's own EXC_GUARD port --------------------------------------------------------

#pragma pack(push, 4)
struct ExcRequest {
	mach_msg_header_t head;
	mach_msg_body_t body;
	mach_msg_port_descriptor_t thread;
	mach_msg_port_descriptor_t task;
	NDR_record_t ndr;
	exception_type_t exception;
	mach_msg_type_number_t codeCount;
	int64_t code[2];
	mach_msg_trailer_t trailer;
};
struct ExcReply {
	mach_msg_header_t head;
	NDR_record_t ndr;
	kern_return_t result;
};
#pragma pack(pop)

struct Seen {
	uint64_t code;
	uint64_t subcode;
};
std::atomic<int> g_seenCount{0};
Seen g_seen[64];
mach_port_t g_port = MACH_PORT_NULL;

void *excServer(void *)
{
	for (;;) {
		ExcRequest req = {};
		if (mach_msg(&req.head, MACH_RCV_MSG, 0, sizeof(req), g_port, MACH_MSG_TIMEOUT_NONE,
		             MACH_PORT_NULL) != MACH_MSG_SUCCESS)
			continue;
		const int n = g_seenCount.load();
		if (n < 64 && req.exception == EXC_GUARD && req.codeCount >= 2) {
			g_seen[n] = {static_cast<uint64_t>(req.code[0]), static_cast<uint64_t>(req.code[1])};
			g_seenCount.store(n + 1);
		}
		mach_port_deallocate(mach_task_self(), req.thread.name);
		mach_port_deallocate(mach_task_self(), req.task.name);
		ExcReply reply = {};
		reply.head.msgh_bits = MACH_MSGH_BITS(MACH_MSGH_BITS_REMOTE(req.head.msgh_bits), 0);
		reply.head.msgh_remote_port = req.head.msgh_remote_port;
		reply.head.msgh_size = sizeof(reply);
		reply.head.msgh_id = req.head.msgh_id + 100;
		reply.ndr = NDR_record;
		reply.result = KERN_SUCCESS;
		mach_msg(&reply.head, MACH_SEND_MSG, sizeof(reply), 0, MACH_PORT_NULL,
		         MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
	}
	return nullptr;
}

bool takeGuardExceptions()
{
	if (mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &g_port) != KERN_SUCCESS ||
	    mach_port_insert_right(mach_task_self(), g_port, g_port, MACH_MSG_TYPE_MAKE_SEND) !=
	        KERN_SUCCESS ||
	    task_set_exception_ports(mach_task_self(), EXC_MASK_GUARD, g_port,
	                             EXCEPTION_DEFAULT | MACH_EXCEPTION_CODES,
	                             THREAD_STATE_NONE) != KERN_SUCCESS)
		return false;
	pthread_t t;
	return pthread_create(&t, nullptr, excServer, nullptr) == 0;
}

// The exceptions seen since `from`, as " guard=N codes=CODE/SUBCODE,...".
std::string seenSince(int from)
{
	const int n = g_seenCount.load();
	std::string s = " guard=" + std::to_string(n - from);
	char b[64];
	for (int i = from; i < n; i++) {
		snprintf(b, sizeof(b), "%s0x%llx/0x%llx", i == from ? " codes=" : ",",
		         (unsigned long long)g_seen[i].code, (unsigned long long)g_seen[i].subcode);
		s += b;
	}
	return s;
}

uint32_t guardBehavior()
{
	task_exc_guard_behavior_t b = 0;
	task_get_exc_guard_behavior(mach_task_self(), &b);
	return b;
}

// --- memory -----------------------------------------------------------------------------------

uint64_t memsize()
{
	uint64_t m = 0;
	size_t len = sizeof(m);
	sysctlbyname("hw.memsize", &m, &len, nullptr, 0);
	return m;
}

// More superpages than the kernel has free RAM for: the free and speculative pages' bytes and
// 512 MiB, in whole superpages, at most the RAM.
uint64_t pastFree(uint64_t *freeBytes)
{
	vm_statistics64_data_t s = {};
	mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
	host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&s), &n);
	*freeBytes = (static_cast<uint64_t>(s.free_count) + s.speculative_count) * vm_page_size;
	uint64_t want = *freeBytes + 512 * kMiB;
	const uint64_t ram = memsize() & ~(kRgfxSuperpage - 1);
	if (want > ram)
		want = ram;
	return want & ~(kRgfxSuperpage - 1);
}

// The process's bytes in entries tagged VM_MEMORY_IOACCELERATOR, and their count.
uint64_t taggedBytes(uint64_t *entries)
{
	uint64_t bytes = 0;
	*entries = 0;
	mach_vm_address_t at = 0;
	for (;;) {
		mach_vm_size_t size = 0;
		natural_t depth = 0;
		vm_region_submap_info_data_64_t info = {};
		mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
		if (mach_vm_region_recurse(mach_task_self(), &at, &size, &depth,
		                           reinterpret_cast<vm_region_recurse_info_t>(&info),
		                           &count) != KERN_SUCCESS)
			break;
		if (info.user_tag == VM_MEMORY_IOACCELERATOR) {
			bytes += size;
			(*entries)++;
		}
		at += size;
	}
	return bytes;
}

// The entries over [at, at + size): how many, and whether each is 2 MiB, tagged and resident.
void describeRange(uint64_t at, uint64_t size, uint64_t *entries, bool *twoMiBTagged,
                   bool *resident)
{
	*entries = 0;
	*twoMiBTagged = true;
	*resident = true;
	mach_vm_address_t a = at;
	while (a < at + size) {
		mach_vm_size_t s = 0;
		natural_t depth = 0;
		vm_region_submap_info_data_64_t info = {};
		mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
		if (mach_vm_region_recurse(mach_task_self(), &a, &s, &depth,
		                           reinterpret_cast<vm_region_recurse_info_t>(&info),
		                           &count) != KERN_SUCCESS ||
		    a >= at + size)
			break;
		(*entries)++;
		*twoMiBTagged = *twoMiBTagged && s == kRgfxSuperpage &&
		                info.user_tag == VM_MEMORY_IOACCELERATOR;
		*resident = *resident && info.pages_resident * vm_page_size == s;
		a += s;
	}
}

// The kernel's own calls, each one counted, for new.short's account of what was asked.
struct Counting {
	RgfxSuperpageVm kernel = rgfxKernelSuperpageVm();
	uint64_t reserved = 0, reserveAt = 0, made = 0, deallocated = 0;
	int refusal = 0;
	std::string order;

	static int superpages(void *c, uint64_t *at, uint64_t size, bool fixed)
	{
		Counting *k = static_cast<Counting *>(c);
		const int kr = k->kernel.superpages(k->kernel.context, at, size, fixed);
		if (kr == KERN_SUCCESS) {
			k->made++;
		} else if (k->refusal == 0) {
			k->refusal = kr;
			k->order += ",refused" + std::string(fixed ? "-fixed" : "");
		}
		return kr;
	}
	static int reserve(void *c, uint64_t *at, uint64_t size)
	{
		Counting *k = static_cast<Counting *>(c);
		const int kr = k->kernel.reserve(k->kernel.context, at, size);
		if (kr == KERN_SUCCESS) {
			k->reserved = size;
			k->reserveAt = *at;
		}
		k->order += "reserve";
		return kr;
	}
	static void deallocate(void *c, uint64_t at, uint64_t size)
	{
		Counting *k = static_cast<Counting *>(c);
		k->deallocated += size;
		k->order += ",deallocate";
		k->kernel.deallocate(k->kernel.context, at, size);
	}
	RgfxSuperpageVm calls()
	{
		RgfxSuperpageVm vm;
		vm.context = this;
		vm.superpages = &superpages;
		vm.reserve = &reserve;
		vm.deallocate = &deallocate;
		return vm;
	}
};

// --- the cases ---------------------------------------------------------------------------------

int oldCall(uint64_t size, mach_vm_address_t *at, bool fixed)
{
	return mach_vm_allocate(mach_task_self(), at, size,
	                        (fixed ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE) |
	                            VM_FLAGS_SUPERPAGE_SIZE_2MB | VM_MAKE_TAG(VM_MEMORY_IOACCELERATOR));
}

void oldAnywhere()
{
	uint64_t freeBytes = 0;
	const uint64_t size = pastFree(&freeBytes);
	const int from = g_seenCount.load();
	mach_vm_address_t at = 0;
	const double t = nowUs();
	const int kr = oldCall(size, &at, false);
	const double us = nowUs() - t;
	if (kr == KERN_SUCCESS)
		mach_vm_deallocate(mach_task_self(), at, size);
	printf("spg: old.anywhere free_mib=%llu asked_mib=%llu kr=%d us=%.0f%s\n",
	       (unsigned long long)(freeBytes / kMiB), (unsigned long long)(size / kMiB), kr, us,
	       seenSince(from).c_str());
}

void oldFixed()
{
	uint64_t freeBytes = 0;
	const uint64_t size = pastFree(&freeBytes);
	// A free range of its own: a reservation's, given back just before.
	mach_vm_address_t at = 0;
	if (mach_vm_map(mach_task_self(), &at, size, kRgfxSuperpage - 1, VM_FLAGS_ANYWHERE,
	                MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_DEFAULT,
	                VM_INHERIT_NONE) != KERN_SUCCESS) {
		printf("spg: old.fixed no range of %llu MiB\n", (unsigned long long)(size / kMiB));
		return;
	}
	mach_vm_deallocate(mach_task_self(), at, size);
	const int from = g_seenCount.load();
	mach_vm_address_t a = at;
	const int kr = oldCall(size, &a, true);
	if (kr == KERN_SUCCESS)
		mach_vm_deallocate(mach_task_self(), at, size);
	std::string where;
	const int n = g_seenCount.load();
	if (n > from) {
		const uint64_t sub = g_seen[from].subcode;
		char b[96];
		snprintf(b, sizeof(b), " at=0x%llx gap_at_superpage=%lld", (unsigned long long)at,
		         (long long)((int64_t)(sub - at) / (int64_t)kRgfxSuperpage));
		where = b;
	}
	printf("spg: old.fixed asked_mib=%llu kr=%d%s%s\n", (unsigned long long)(size / kMiB), kr,
	       where.c_str(), seenSince(from).c_str());
}

void newShort()
{
	uint64_t freeBytes = 0;
	const uint64_t size = pastFree(&freeBytes);
	uint64_t entriesBefore = 0, entriesAfter = 0;
	const uint64_t before = taggedBytes(&entriesBefore);
	const int from = g_seenCount.load();
	Counting k;
	uint8_t *bytes = nullptr;
	const double t = nowUs();
	const int kr = rgfxTakeSuperpages(k.calls(), size, &bytes);
	const double us = nowUs() - t;
	if (kr == KERN_SUCCESS)
		rgfxGiveSuperpages(k.calls(), bytes, size);
	const uint64_t after = taggedBytes(&entriesAfter);
	printf("spg: new.short free_mib=%llu asked_mib=%llu kr=%d us=%.0f calls=%s made=%llu "
	       "reserved_mib=%llu deallocated_mib=%llu tagged_mib_before=%llu tagged_mib_after=%llu%s\n",
	       (unsigned long long)(freeBytes / kMiB), (unsigned long long)(size / kMiB), kr, us,
	       k.order.c_str(), (unsigned long long)k.made, (unsigned long long)(k.reserved / kMiB),
	       (unsigned long long)(k.deallocated / kMiB), (unsigned long long)(before / kMiB),
	       (unsigned long long)(after / kMiB), seenSince(from).c_str());
}

void newRoom(uint64_t mib)
{
	const uint64_t size = (mib * kMiB + kRgfxSuperpage - 1) & ~(kRgfxSuperpage - 1);
	const int from = g_seenCount.load();
	uint8_t *bytes = nullptr;
	double t = nowUs();
	const int kr = rgfxTakeSuperpages(rgfxKernelSuperpageVm(), size, &bytes);
	const double us = nowUs() - t;
	uint64_t entries = 0;
	bool twoMiB = false, resident = false;
	if (kr == KERN_SUCCESS) {
		describeRange(reinterpret_cast<uintptr_t>(bytes), size, &entries, &twoMiB, &resident);
		rgfxGiveSuperpages(rgfxKernelSuperpageVm(), bytes, size);
	}
	mach_vm_address_t at = 0;
	t = nowUs();
	const int oldKr = oldCall(size, &at, false);
	const double oldUs = nowUs() - t;
	if (oldKr == KERN_SUCCESS)
		mach_vm_deallocate(mach_task_self(), at, size);
	printf("spg: new.room.%llu kr=%d us=%.0f entries=%llu each_2mib_tagged=%d resident=%d "
	       "old_kr=%d old_us=%.0f%s\n",
	       (unsigned long long)mib, kr, us, (unsigned long long)entries, twoMiB ? 1 : 0,
	       resident ? 1 : 0, oldKr, oldUs, seenSince(from).c_str());
}

void fatalChild(bool useNew)
{
	fflush(stdout);
	const pid_t pid = fork();
	if (pid == 0) {
		// No port of its own: the guard ends it, as it ends WebKit's GPU process.
		task_set_exception_ports(mach_task_self(), EXC_MASK_GUARD, MACH_PORT_NULL,
		                         EXCEPTION_DEFAULT, THREAD_STATE_NONE);
		const uint32_t want = guardBehavior() | TASK_EXC_GUARD_VM_DELIVER | TASK_EXC_GUARD_VM_FATAL;
		if (task_set_exc_guard_behavior(mach_task_self(), want) != KERN_SUCCESS)
			_exit(3);
		uint64_t freeBytes = 0;
		const uint64_t size = pastFree(&freeBytes);
		int kr;
		if (useNew) {
			uint8_t *bytes = nullptr;
			kr = rgfxTakeSuperpages(rgfxKernelSuperpageVm(), size, &bytes);
			if (kr == KERN_SUCCESS)
				rgfxGiveSuperpages(rgfxKernelSuperpageVm(), bytes, size);
		} else {
			mach_vm_address_t at = 0;
			kr = oldCall(size, &at, false);
			if (kr == KERN_SUCCESS)
				mach_vm_deallocate(mach_task_self(), at, size);
		}
		printf("spg: %s child kr=%d (alive after the call)\n", useNew ? "fatal.new" : "fatal.old",
		       kr);
		fflush(stdout);
		_exit(kr == KERN_SUCCESS ? 2 : 0);
	}
	int status = 0;
	waitpid(pid, &status, 0);
	if (WIFSIGNALED(status))
		printf("spg: %s ended=signal sig=%d\n", useNew ? "fatal.new" : "fatal.old",
		       WTERMSIG(status));
	else
		printf("spg: %s ended=exit code=%d\n", useNew ? "fatal.new" : "fatal.old",
		       WEXITSTATUS(status));
}

} // namespace

int main(int argc, char **argv)
{
	const uint32_t initial = guardBehavior();
	if (!takeGuardExceptions()) {
		printf("spg: no EXC_GUARD port\n");
		return 1;
	}
	// Delivered to the port above, not fatal: the process lives to say what it saw.
	const kern_return_t set = task_set_exc_guard_behavior(
	    mach_task_self(), initial | TASK_EXC_GUARD_VM_DELIVER);
	for (int i = 1; i < argc; i++) {
		const std::string c = argv[i];
		if (c == "behavior")
			printf("spg: behavior initial=0x%x set_kr=%d now=0x%x ram_mib=%llu\n", initial, set,
			       guardBehavior(), (unsigned long long)(memsize() / kMiB));
		else if (c == "old.anywhere")
			oldAnywhere();
		else if (c == "old.fixed")
			oldFixed();
		else if (c == "new.short")
			newShort();
		else if (c.rfind("new.room.", 0) == 0)
			newRoom(strtoull(c.c_str() + 9, nullptr, 10));
		else if (c == "fatal.old")
			fatalChild(false);
		else if (c == "fatal.new")
			fatalChild(true);
		else
			printf("spg: %s unknown\n", c.c_str());
		fflush(stdout);
	}
	return 0;
}
