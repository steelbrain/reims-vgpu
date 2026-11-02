/*
 * RgfxVideoDriver: the bundle's side of hardware video, as AppleGVA sees a GPU vendor's video
 * driver.
 *
 * AppleGVA (the framework under VideoToolbox's hardware decoders and encoders on an x86-64
 * Mac) reads the accelerator's IODVDBundleName, which the kext sets to this bundle's name,
 * loads /Library/GPUBundles/<name>.bundle/Contents/MacOS/<name>, and calls its
 * AVD_CreateAVDAccelerator for a table of sixteen entry points. It does so in
 * VTDecoderXPCService and VTEncoderXPCService, one of each per client process.
 *
 * This cut records: it answers the driver's identity, gives AppleGVA a capability block that
 * lists one engine and the functions in kFunctions, and writes every call AppleGVA makes, with
 * what its arguments point at, and every read AppleGVA makes of the capability block, with the
 * reading instruction, to rgfx-avd.<pid>.log in the process's temporary directory. Nothing is
 * decoded yet: every entry answers success and fills nothing in.
 *
 * AppleGVA hands this driver a session only where it picks the AMD driver for the flow: on a
 * board without an Intel GPU, or with com.apple.AppleGVA's gvaForceAMDKE set, which the install
 * sets.
 *
 * Layouts AppleGVA and the driver share are named by what they hold where that is stated by
 * the calls' use; a field whose meaning nothing states yet is field_xx, xx its byte offset.
 */

#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include <atomic>
#include <mutex>

namespace {

// The renderer ID the kext's IOVARendererID carries: the AMD Navi family's, as an iMac20,1's
// Radeon Pro answers. AppleGVA takes the driver's own answer from the identity below.
constexpr uint32_t kRendererID = 0x1020004;

// What AVD_CreateAVDAccelerator writes for AppleGVA.
struct DriverIdentity {
	uint32_t renderer_id;
	uint32_t field_04;
	const void *capabilities; // the capability block AppleGVA reads
	uint64_t field_10;
	uint32_t field_18;
	char name[16];
};
static_assert(sizeof(DriverIdentity) == 0x2c + 4, "identity layout");

// The capability block AppleGVA walks: the engines, then the functions (one per codec and
// direction the driver runs), each a count and an array of record pointers. The records lie in
// the block too, so the reads of them are recorded with the block's.
struct Engine {
	uint32_t type;
	char name[0x84];
	uint64_t field_88;
	uint32_t field_90;
	uint32_t field_94;
};
static_assert(sizeof(Engine) == 0x98, "engine layout");

struct Function {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint32_t field_10;
	uint32_t field_14;
	uint64_t field_18;
	uint64_t field_20;
	uint32_t field_28;
	uint32_t input_format; // the fourcc of what the function takes: 'rawb' for a bitstream
	uint32_t field_30;
	uint8_t field_34[0x22c - 0x34];
	uint32_t output_format_count;
	uint8_t output_formats[0x200];
	const Function *field_430;
};
static_assert(sizeof(Function) == 0x438, "function layout");

constexpr size_t kMaxEngines = 4;
constexpr size_t kMaxFunctions = 41;

struct CapabilityBlock {
	uint32_t engine_count;
	uint32_t field_04;
	const Engine *const *engines;
	uint32_t function_count;
	uint32_t field_14;
	const Function *const *functions;
	Engine engine[kMaxEngines];
	const Engine *engine_list[kMaxEngines];
	Function function[kMaxFunctions];
	const Function *function_list[kMaxFunctions];
};

// What the block lists while the driver records: one engine, and the functions below. Values
// that are not yet known stay zero, so the record shows which of them AppleGVA reads.
struct FunctionAnswer {
	uint32_t function_id;
	uint64_t sub_id;
	uint32_t input_format;
};
constexpr FunctionAnswer kFunctions[] = {
	{1, 0x8, 'rawb'}, // decode, H.264
};

void fillCapabilities(CapabilityBlock &b)
{
	memset(&b, 0, sizeof b);
	b.engine[0].type = 4;
	strcpy(b.engine[0].name, "video engine");
	b.engine_list[0] = &b.engine[0];
	b.engine_count = 1;
	b.engines = b.engine_list;
	for (const FunctionAnswer &a : kFunctions) {
		Function &f = b.function[b.function_count];
		f.function_id = a.function_id;
		f.sub_id = a.sub_id;
		f.input_format = a.input_format;
		f.field_28 = 1;
		f.field_30 = 1;
		b.function_list[b.function_count++] = &f;
	}
	b.functions = b.function_list;
}

constexpr size_t kCapabilityBytes = sizeof(CapabilityBlock);

// The sixteen entry points, in table order, by what AppleGVA asks of each.
constexpr const char *kEntryNames[16] = {
	"CreateDriver",
	"DeleteDriver",
	"CreateContexts",
	"DeleteContexts",
	"GetBitstreamBufferInfo",
	"AllocateBitstreamBufferPool",
	"FreeBitstreamBufferPool",
	"GetPixelBufferInfo",
	"AllocatePixelBufferPool",
	"FreePixelBufferPool",
	"ScheduleCommands",
	"QueryCommandStatus",
	"SetContextParameter",
	"GetContextParameter",
	"SetDriverParameter",
	"GetDriverParameter",
};

// The table AVD_CreateAVDAccelerator hands out: a driver pointer, then the entry points.
struct DriverTable {
	void *driver;
	uint64_t (*entry[16])(uint64_t, uint64_t, uint64_t);
};
static_assert(sizeof(DriverTable) == 0x88, "table layout");

int g_driver; // what the driver pointer AppleGVA passes back points at

// --- the record ----------------------------------------------------------------------------

std::mutex g_logLock;
FILE *g_log;
std::atomic<uint32_t> g_calls{0};
// Calls of each entry written out in full; later ones are counted only.
constexpr uint32_t kFullCalls = 40;
uint32_t g_perEntry[16];

FILE *recordFile()
{
	if (g_log)
		return g_log;
	char dir[512] = "/tmp/";
	confstr(_CS_DARWIN_USER_TEMP_DIR, dir, sizeof dir);
	char path[640];
	snprintf(path, sizeof path, "%srgfx-avd.%d.log", dir, getpid());
	g_log = fopen(path, "w");
	if (g_log)
		setvbuf(g_log, nullptr, _IOLBF, 0);
	return g_log;
}

// Copies n bytes at a, or answers false where they are not all mapped (an argument that is a
// number, not a pointer).
bool readable(uint64_t a, void *out, size_t n)
{
	if (a < 0x10000 || a > 0x7fffffffffffULL)
		return false;
	mach_vm_size_t got = 0;
	return mach_vm_read_overwrite(mach_task_self(), a, n, (mach_vm_address_t)out, &got) ==
		       KERN_SUCCESS &&
	       got == n;
}

// Writes n bytes at a as 32-bit words, and, depth levels further, what each pointer-sized
// value in them points at.
void dump(FILE *f, const char *tag, uint64_t a, size_t n, int depth)
{
	uint8_t bytes[512];
	if (n > sizeof bytes || !readable(a, bytes, n))
		return;
	fprintf(f, "  %s @0x%llx:\n", tag, (unsigned long long)a);
	for (size_t i = 0; i < n; i += 32) {
		fprintf(f, "    +%03zx:", i);
		for (size_t j = 0; j < 32; j += 4) {
			uint32_t w;
			memcpy(&w, bytes + i + j, 4);
			fprintf(f, " %08x", w);
		}
		fprintf(f, "\n");
	}
	if (depth == 0)
		return;
	int shown = 0;
	for (size_t i = 0; i + 8 <= n && shown < 12; i += 8) {
		uint64_t p;
		memcpy(&p, bytes + i, 8);
		uint8_t probe[8];
		if (p != a && readable(p, probe, sizeof probe)) {
			char inner[96];
			snprintf(inner, sizeof inner, "%s+%zx->", tag, i);
			dump(f, inner, p, 256, depth - 1);
			shown++;
		}
	}
}


// --- which capability bytes AppleGVA reads -------------------------------------------------
//
// The block lies on pages of its own with no access. A read faults; the handler notes the
// offset and the reading instruction (its image and offset), opens the page, and single-steps
// that one instruction, after which the trap closes the page again. The notes are written out
// at the next entry-point call, outside the signal handlers.

struct CapabilityRead {
	uint32_t offset;
	uint64_t pc;
};
constexpr size_t kReadsKept = 4096;
CapabilityRead g_reads[kReadsKept];
std::atomic<uint32_t> g_readCount{0};
uint32_t g_readsWritten;
uint8_t *g_capabilities;
size_t g_capabilityPages;
uint8_t *g_openPage;
struct sigaction g_oldSegv, g_oldBus, g_oldTrap;

size_t pageSize() { return (size_t)getpagesize(); }

void onFault(int sig, siginfo_t *info, void *context)
{
	uint8_t *at = (uint8_t *)info->si_addr;
	ucontext_t *uc = (ucontext_t *)context;
	if (g_capabilities && at >= g_capabilities && at < g_capabilities + g_capabilityPages) {
		uint32_t n = g_readCount.fetch_add(1);
		if (n < kReadsKept)
			g_reads[n] = {(uint32_t)(at - g_capabilities), uc->uc_mcontext->__ss.__rip};
		uint8_t *page = (uint8_t *)((uintptr_t)at & ~(uintptr_t)(pageSize() - 1));
		mprotect(page, pageSize(), PROT_READ | PROT_WRITE);
		g_openPage = page;
		uc->uc_mcontext->__ss.__rflags |= 0x100; // trap after the instruction
		return;
	}
	// Not ours: the fault the process would have taken.
	struct sigaction *old = sig == SIGBUS ? &g_oldBus : &g_oldSegv;
	sigaction(sig, old, nullptr);
}

void onTrap(int sig, siginfo_t *info, void *context)
{
	ucontext_t *uc = (ucontext_t *)context;
	if (g_openPage) {
		mprotect(g_openPage, pageSize(), PROT_NONE);
		g_openPage = nullptr;
		uc->uc_mcontext->__ss.__rflags &= ~(uint64_t)0x100;
		return;
	}
	sigaction(SIGTRAP, &g_oldTrap, nullptr);
}

void traceCapabilityReads()
{
	g_capabilityPages = (kCapabilityBytes + pageSize() - 1) & ~(pageSize() - 1);
	void *p = mmap(nullptr, g_capabilityPages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (p == MAP_FAILED)
		return;
	g_capabilities = (uint8_t *)p;
	fillCapabilities(*(CapabilityBlock *)p);
	struct sigaction sa = {};
	sa.sa_sigaction = onFault;
	sa.sa_flags = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, &g_oldSegv);
	sigaction(SIGBUS, &sa, &g_oldBus);
	sa.sa_sigaction = onTrap;
	sigaction(SIGTRAP, &sa, &g_oldTrap);
	mprotect(g_capabilities, g_capabilityPages, PROT_NONE);
}

// Writes the reads noted since the last call, each with the image and offset of its reader.
void writeReads(FILE *f)
{
	uint32_t n = g_readCount.load();
	if (n > kReadsKept)
		n = kReadsKept;
	for (; g_readsWritten < n; g_readsWritten++) {
		const CapabilityRead &r = g_reads[g_readsWritten];
		Dl_info di = {};
		const char *image = "?";
		uint64_t off = r.pc;
		if (dladdr((void *)r.pc, &di) && di.dli_fname) {
			image = strrchr(di.dli_fname, '/') ? strrchr(di.dli_fname, '/') + 1 : di.dli_fname;
			off = r.pc - (uint64_t)di.dli_fbase;
		}
		char where[64];
		const size_t engines = offsetof(CapabilityBlock, engine);
		const size_t functions = offsetof(CapabilityBlock, function);
		if (r.offset >= engines && r.offset < engines + sizeof(Engine) * kMaxEngines)
			snprintf(where, sizeof where, "engine[%zu]+0x%zx", (r.offset - engines) / sizeof(Engine),
				 (r.offset - engines) % sizeof(Engine));
		else if (r.offset >= functions && r.offset < functions + sizeof(Function) * kMaxFunctions)
			snprintf(where, sizeof where, "function[%zu]+0x%zx",
				 (r.offset - functions) / sizeof(Function), (r.offset - functions) % sizeof(Function));
		else
			snprintf(where, sizeof where, "block+0x%x", r.offset);
		fprintf(f, "CAPREAD %s by %s+0x%llx\n", where, image, (unsigned long long)off);
	}
}

void record(int entry, const char *when, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t result)
{
	std::lock_guard<std::mutex> hold(g_logLock);
	FILE *f = recordFile();
	if (!f)
		return;
	writeReads(f);
	if (g_perEntry[entry] > kFullCalls)
		return;
	fprintf(f, "%s #%u %s(%llx, %llx, %llx)", when, g_calls.load(), kEntryNames[entry],
		(unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
	if (when[0] == 'R')
		fprintf(f, " -> 0x%llx", (unsigned long long)result);
	fprintf(f, " tid %p\n", (void *)pthread_self());
	dump(f, "a1", a1, 512, 1);
	dump(f, "a2", a2, 256, 1);
}

template <int N>
uint64_t entryPoint(uint64_t a0, uint64_t a1, uint64_t a2)
{
	{
		std::lock_guard<std::mutex> hold(g_logLock);
		g_perEntry[N]++;
	}
	g_calls++;
	record(N, "CALL", a0, a1, a2, 0);
	// CreateDriver's first argument is where the driver pointer goes.
	uint8_t probe[8];
	if (N == 0 && readable(a0, probe, sizeof probe))
		*(void **)a0 = &g_driver;
	// Every entry answers success with nothing filled in, so the record shows how far
	// AppleGVA goes on an empty capability block.
	uint64_t result = 0;
	record(N, "RET ", a0, a1, a2, result);
	return result;
}

template <int... N>
struct Entries {
	static void fill(DriverTable &t) { ((t.entry[N] = &entryPoint<N>), ...); }
};

DriverTable g_table;

} // namespace

extern "C" __attribute__((visibility("default"))) int
AVD_CreateAVDAccelerator(void **out, const void *params, void *identity)
{
	if (!out || !params || !identity)
		return 5;
	{
		std::lock_guard<std::mutex> hold(g_logLock);
		if (FILE *f = recordFile()) {
			fprintf(f, "CREATE out %p params %p identity %p\n", (void *)out, params, identity);
			dump(f, "params", (uint64_t)params, 0x40, 1);
		}
	}
	Entries<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>::fill(g_table);
	g_table.driver = &g_driver;
	DriverIdentity id = {};
	id.renderer_id = kRendererID;
	if (!g_capabilities)
		traceCapabilityReads();
	id.capabilities = g_capabilities;
	id.field_10 = 1;
	memcpy(id.name, "RgfxVideo      ", 16);
	memcpy(identity, &id, offsetof(DriverIdentity, name) + sizeof id.name);
	*out = &g_table;
	return 0;
}

extern "C" __attribute__((visibility("default"))) int AVD_DestroyAVDAccelerator(void *table)
{
	std::lock_guard<std::mutex> hold(g_logLock);
	if (FILE *f = recordFile()) {
		writeReads(f);
		fprintf(f, "DESTROY %p calls %u:", table, g_calls.load());
		for (int i = 0; i < 16; i++)
			fprintf(f, " %s=%u", kEntryNames[i], g_perEntry[i]);
		fprintf(f, "\n");
	}
	return 0;
}
