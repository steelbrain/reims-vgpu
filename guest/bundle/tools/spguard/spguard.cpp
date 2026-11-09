// spguard: what the kernel's VM guard makes of one call for several superpages, the call the
// device's superpages never make (src/RgfxSuperpages.hpp, ONE SUPERPAGE A CALL), in a scratch
// process with its VM guard on, as Apple's platform binaries and WebKit's GPU process have it. The
// process turns its own guard on (task_set_exc_guard_behavior; a process may only add bits) and
// takes its EXC_GUARD exceptions on a port of its own, recording each one's codes and answering it
// handled, so a guard exception shows and the process lives on. To run in an x86-64 guest; no
// Metal, no GPU.
//   spguard CASE...
// The cases:
//   behavior          the process's EXC_GUARD behaviour as it started, then with the VM guard on.
//   several.anywhere  one mach_vm_allocate of more superpages than the kernel has free RAM for
//                     (the free pages' bytes and 512 MiB, at most the RAM): its answer, and the
//                     guard exceptions it raised (code, subcode).
//   several.fixed     the same one call, at a fixed address of a free range of its own: the
//                     exception's subcode less the address, in superpages (the kernel's cleanup
//                     takes down past the end of the last superpage entry it made).
//   several.fatal     a child with the VM guard fatal (VM_DELIVER | VM_FATAL, WebKit's GPU
//                     process's) and no port of its own makes the several.anywhere call: how it
//                     ended.
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

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kSuperpage = 2ull << 20;

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
	const uint64_t ram = memsize() & ~(kSuperpage - 1);
	if (want > ram)
		want = ram;
	return want & ~(kSuperpage - 1);
}

// --- the cases ---------------------------------------------------------------------------------

int severalCall(uint64_t size, mach_vm_address_t *at, bool fixed)
{
	return mach_vm_allocate(mach_task_self(), at, size,
	                        (fixed ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE) |
	                            VM_FLAGS_SUPERPAGE_SIZE_2MB | VM_MAKE_TAG(VM_MEMORY_IOACCELERATOR));
}

void severalAnywhere()
{
	uint64_t freeBytes = 0;
	const uint64_t size = pastFree(&freeBytes);
	const int from = g_seenCount.load();
	mach_vm_address_t at = 0;
	const double t = nowUs();
	const int kr = severalCall(size, &at, false);
	const double us = nowUs() - t;
	if (kr == KERN_SUCCESS)
		mach_vm_deallocate(mach_task_self(), at, size);
	printf("spg: several.anywhere free_mib=%llu asked_mib=%llu kr=%d us=%.0f%s\n",
	       (unsigned long long)(freeBytes / kMiB), (unsigned long long)(size / kMiB), kr, us,
	       seenSince(from).c_str());
}

void severalFixed()
{
	uint64_t freeBytes = 0;
	const uint64_t size = pastFree(&freeBytes);
	// A free range of its own: a reservation's, given back just before.
	mach_vm_address_t at = 0;
	if (mach_vm_map(mach_task_self(), &at, size, kSuperpage - 1, VM_FLAGS_ANYWHERE,
	                MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_DEFAULT,
	                VM_INHERIT_NONE) != KERN_SUCCESS) {
		printf("spg: several.fixed no range of %llu MiB\n", (unsigned long long)(size / kMiB));
		return;
	}
	mach_vm_deallocate(mach_task_self(), at, size);
	const int from = g_seenCount.load();
	mach_vm_address_t a = at;
	const int kr = severalCall(size, &a, true);
	if (kr == KERN_SUCCESS)
		mach_vm_deallocate(mach_task_self(), at, size);
	std::string where;
	const int n = g_seenCount.load();
	if (n > from) {
		const uint64_t sub = g_seen[from].subcode;
		char b[96];
		snprintf(b, sizeof(b), " at=0x%llx gap_at_superpage=%lld", (unsigned long long)at,
		         (long long)((int64_t)(sub - at) / (int64_t)kSuperpage));
		where = b;
	}
	printf("spg: several.fixed asked_mib=%llu kr=%d%s%s\n", (unsigned long long)(size / kMiB), kr,
	       where.c_str(), seenSince(from).c_str());
}

void severalFatal()
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
		mach_vm_address_t at = 0;
		const int kr = severalCall(size, &at, false);
		if (kr == KERN_SUCCESS)
			mach_vm_deallocate(mach_task_self(), at, size);
		printf("spg: several.fatal child kr=%d (alive after the call)\n", kr);
		fflush(stdout);
		_exit(kr == KERN_SUCCESS ? 2 : 0);
	}
	int status = 0;
	waitpid(pid, &status, 0);
	if (WIFSIGNALED(status))
		printf("spg: several.fatal ended=signal sig=%d\n", WTERMSIG(status));
	else
		printf("spg: several.fatal ended=exit code=%d\n", WEXITSTATUS(status));
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
		else if (c == "several.anywhere")
			severalAnywhere();
		else if (c == "several.fixed")
			severalFixed();
		else if (c == "several.fatal")
			severalFatal();
		else
			printf("spg: %s unknown\n", c.c_str());
		fflush(stdout);
	}
	return 0;
}
