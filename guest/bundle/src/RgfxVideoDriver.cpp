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
 * This first cut records: it answers the driver's identity, gives AppleGVA a capability block
 * that lists nothing, and writes every call AppleGVA makes, with what its arguments point at,
 * to rgfx-avd.<pid>.log in the process's temporary directory. Nothing is decoded yet; with no
 * capability listed, AppleGVA gives the session up and VideoToolbox falls back to its software
 * decoders.
 *
 * Layouts AppleGVA and the driver share are named by what they hold where that is stated by
 * the calls' use; a field whose meaning nothing states yet is field_xx, xx its byte offset.
 */

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
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

// The capability block's size. It is all zero: no engine and no function listed.
constexpr size_t kCapabilityBytes = 0xb378;
alignas(16) uint8_t g_capabilities[kCapabilityBytes];

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

void record(int entry, const char *when, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t result)
{
	std::lock_guard<std::mutex> hold(g_logLock);
	FILE *f = recordFile();
	if (!f)
		return;
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
		fprintf(f, "DESTROY %p calls %u:", table, g_calls.load());
		for (int i = 0; i < 16; i++)
			fprintf(f, " %s=%u", kEntryNames[i], g_perEntry[i]);
		fprintf(f, "\n");
	}
	return 0;
}
