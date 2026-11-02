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
 * This cut answers a whole H.264 decode session and records it. The capability block lists a
 * video engine and a graphics engine, the H.264 decode pipeline (head, decode, frame transfer)
 * and frame processing; the entry points answer AppleGVA's questions (the bitstream alignment,
 * the frames' layout, the capability parameters), hand out bitstream buffers of whole pages,
 * make a context per pipeline function, take each decode command with a token, and answer every
 * token done. Nothing is decoded yet: the frames hold what they held. Every call, what its
 * arguments point at, each decode command's parameters, slices and target surface, and every
 * read AppleGVA makes of the capability block go to rgfx-avd.<pid>.log in the process's
 * temporary directory.
 *
 * AppleGVA hands this driver a session only where it picks the AMD driver for the flow: on a
 * board without an Intel GPU, or with com.apple.AppleGVA's gvaForceAMDKE set.
 *
 * Layouts AppleGVA and the driver share are named by what they hold where that is stated by
 * the calls' use; a field whose meaning nothing states yet is field_xx, xx its byte offset.
 */

#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <objc/runtime.h>
#include <IOSurface/IOSurfaceRef.h>
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

// A format a function takes or gives: its fourcc ('rawb' for a bitstream, 'NV12' for 4:2:0
// frames).
struct Format {
	uint32_t fourcc;
	uint32_t field_04;
};

// One function: a codec and direction, or a step of a pipeline. AppleGVA looks a decoder up as a
// pipeline: a head (function 0x80), whose next record is the decode itself (function 1, its
// sub-ID the codec, a bitstream in and frames out), whose next is a frame transfer (function
// 0x10), all run by the same engine.
struct Function {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint32_t engine; // the type of the engine that runs it
	uint32_t field_14;
	uint64_t field_18;
	uint64_t field_20;
	uint32_t input_count;
	Format inputs[64];
	uint32_t output_count;
	Format outputs[64];
	const Function *next;
};
static_assert(offsetof(Function, inputs) == 0x2c, "function layout");
static_assert(offsetof(Function, output_count) == 0x22c, "function layout");
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

constexpr uint32_t kVideoEngine = 4;
// The engine that runs frame processing (scaling, format conversion): AppleGVA looks it up by the
// processing function's engine whenever it sets a decoder up.
constexpr uint32_t kGraphicsEngine = 1;
// What the engine and its functions can take at once, in AppleGVA's unit (pixels a second:
// AppleGVA asks 1920 x 1088 x 45 for a 1080p stream): room for any stream the host decodes. It is
// answered as 32 bits too, so it stays below 2^32.
constexpr uint64_t kEngineBudget = 0xf0000000ULL;

enum : uint32_t {
	kFunctionDecode = 1,
	kFunctionProcess = 2,
	kFunctionTransfer = 0x10,
	kFunctionPipeline = 0x80,
};
constexpr uint64_t kCodecH264 = 0x8;

Function &addFunction(CapabilityBlock &b, uint32_t function_id, uint64_t sub_id, uint32_t in,
		      uint32_t out, uint32_t engine = kVideoEngine)
{
	Function &f = b.function[b.function_count];
	f.function_id = function_id;
	f.sub_id = sub_id;
	f.engine = engine;
	f.field_18 = kEngineBudget;
	f.field_20 = kEngineBudget;
	f.input_count = 1;
	f.inputs[0] = {in, 1};
	f.output_count = 1;
	f.outputs[0] = {out, 1};
	b.function_list[b.function_count++] = &f;
	return f;
}

void fillCapabilities(CapabilityBlock &b)
{
	memset(&b, 0, sizeof b);
	b.engine[0].type = kVideoEngine;
	strcpy(b.engine[0].name, "video engine");
	b.engine[0].field_88 = kEngineBudget;
	b.engine[1].type = kGraphicsEngine;
	strcpy(b.engine[1].name, "graphics engine");
	b.engine[1].field_88 = kEngineBudget;
	b.engine_list[0] = &b.engine[0];
	b.engine_list[1] = &b.engine[1];
	b.engine_count = 2;
	b.engines = b.engine_list;
	// H.264 decode: the pipeline head, the decode, the transfer of its frames.
	Function &head = addFunction(b, kFunctionPipeline, 1, 'rawb', 'rawb');
	Function &decode = addFunction(b, kFunctionDecode, kCodecH264, 'rawb', 'NV12');
	Function &transfer = addFunction(b, kFunctionTransfer, 1, 'NV12', 'NV12');
	head.next = &decode;
	decode.next = &transfer;
	// Frame processing on the graphics engine.
	addFunction(b, kFunctionProcess, 0x10, 'NV12', 'NV12', kGraphicsEngine);
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

// What AppleGVA hands the entry points. Each entry's first argument is the driver handle
// CreateDriver wrote, its second the record below.

// CreateContexts: one context per function of a pipeline, each described so.
struct FunctionInfo {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint32_t engine;
	uint32_t field_14;
	uint32_t rate; // pixels a second the stream needs, against the engine's budget
	uint32_t field_1c;
	uint32_t width; // coded, in pixels
	uint32_t height;
	Format input;
	Format output;
	const FunctionInfo *next;
};
static_assert(sizeof(FunctionInfo) == 0x40, "function info layout");

struct ContextInfo {
	uint32_t count;
	uint32_t field_04;
	const FunctionInfo *const *functions;
	void **contexts; // out, one per function
};

struct ContextInit {
	ContextInfo *info;
	const void *capabilities; // out
};

struct ContextDelete {
	uint32_t count;
	uint32_t field_04;
	void *const *contexts;
};

// GetBitstreamBufferInfo: AppleGVA rounds its buffers' size up to the largest alignment any
// matching function answers.
struct BitstreamRequirement {
	uint32_t function_id;
	uint32_t sub_id;
	uint32_t engine;
	uint32_t field_0c;
	uint32_t size;
	Format format;
	uint32_t alignment[6]; // out
};
static_assert(sizeof(BitstreamRequirement) == 0x34, "bitstream requirement layout");

// AllocateBitstreamBufferPool: the driver hands out each buffer's address, and AppleGVA writes
// the compressed data there and names it in a command by (pool, index).
struct BitstreamPool {
	uint32_t pool;
	uint32_t field_04;
	uint32_t size;
	Format format;
	uint32_t count;
	uint64_t *addresses; // out, one per buffer
};
static_assert(offsetof(BitstreamPool, addresses) == 0x18, "bitstream pool layout");

// GetPixelBufferInfo: the driver answers each plane's size, row pitch and alignment, and
// AppleGVA makes its frames' IOSurfaces so.
struct PixelRequirement {
	uint32_t function_id;
	uint32_t sub_id;
	uint32_t field_08;
	uint32_t field_0c;
	uint32_t width;
	uint32_t height;
	uint32_t fourcc;
	uint32_t field_1c;
	uint32_t planes;
	uint32_t field_24;
	uint32_t *widths; // out, per plane
	uint32_t *heights;
	uint32_t *pitches;
	uint32_t *alignments;
};
static_assert(offsetof(PixelRequirement, alignments) == 0x40, "pixel requirement layout");

// ScheduleAVDCommands: a list of commands, and a list the driver fills with a token per command,
// which AppleGVA hands back in its status queries.
struct Command {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint64_t field_10;
	void *context;
	const uint8_t *parameters; // decode: the picture's
	const void *field_28;
	const uint8_t *output; // decode: what the picture is decoded into
};
static_assert(offsetof(Command, output) == 0x30, "command layout");

struct CommandList {
	uint32_t count;
	uint32_t field_04;
	const Command *const *commands;
};

struct TokenList {
	uint32_t count;
	uint32_t field_04;
	void **tokens; // out, one per command
};

struct Query {
	void *token;
	uint32_t field_08;
	uint32_t detail;
	uint32_t status; // out: 1 done, 2 running
	uint32_t field_14;
};
static_assert(sizeof(Query) == 0x18, "query layout");

struct QueryList {
	uint32_t count;
	uint32_t field_04;
	Query *queries;
};

// Get/SetDriverParameter: a list of parameters, each an ID and its value or where it goes.
struct DriverParam {
	uint32_t id;
	uint32_t field_04;
	void *data;
};
static_assert(sizeof(DriverParam) == 0x10, "driver parameter layout");

struct DriverParamList {
	uint32_t count;
	uint32_t field_04;
	DriverParam *params;
};

enum : uint32_t {
	// Whether a function (its ID, sub-ID and engine) is listed, and its budget.
	kParamFunctionAvailable = 9,
	// The decoders' pixel-rate limit.
	kParamDecodeLimit = 0xd,
	// The capability block, current: AppleGVA asks it before it sizes a session's buffers.
	kParamCapabilities = 0xe,
};

struct FunctionAvailable {
	uint32_t function_id;
	uint32_t sub_id;
	uint32_t engine;
	uint32_t budget; // out, 0 where the function is not listed
};

int g_driver; // what the driver pointer AppleGVA passes back points at

// Metal's default device in this process: the bundle's own device where Metal loaded this bundle
// for the accelerator, whose connection (region, rings, blocks) video work is to share.
extern "C" void *MTLCreateSystemDefaultDevice(void);

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

// --- the decode commands, recorded ---------------------------------------------------------

uint32_t u32At(const uint8_t *base, size_t off)
{
	uint32_t v = 0;
	readable((uint64_t)(base + off), &v, sizeof v);
	return v;
}

uint64_t u64At(const uint8_t *base, size_t off)
{
	uint64_t v = 0;
	readable((uint64_t)(base + off), &v, sizeof v);
	return v;
}

const uint8_t *bitstreamBuffer(uint32_t pool, uint32_t index);

void recordSurface(FILE *f, IOSurfaceRef surface)
{
	if (!surface)
		return;
	OSType format = IOSurfaceGetPixelFormat(surface);
	fprintf(f, "  surface id %u %.4s %zux%zu alloc %zu planes %zu", IOSurfaceGetID(surface),
		(const char *)&format, IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface),
		IOSurfaceGetAllocSize(surface), IOSurfaceGetPlaneCount(surface));
	for (size_t p = 0; p < IOSurfaceGetPlaneCount(surface); p++)
		fprintf(f, " [p%zu %zux%zu bpr %zu off %zu]", p, IOSurfaceGetWidthOfPlane(surface, p),
			IOSurfaceGetHeightOfPlane(surface, p), IOSurfaceGetBytesPerRowOfPlane(surface, p),
			(size_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, p) -
				 (uint8_t *)IOSurfaceGetBaseAddress(surface)));
	fprintf(f, "\n");
}

void recordDecode(FILE *f, const Command *c, int depth)
{
	Command cmd = {};
	if (depth > 3 || !readable((uint64_t)c, &cmd, sizeof cmd))
		return;
	fprintf(f, "  command function 0x%x sub 0x%llx context %p parameters %p field_28 %p output %p\n",
		cmd.function_id, (unsigned long long)cmd.sub_id, cmd.context, (const void *)cmd.parameters,
		cmd.field_28, (const void *)cmd.output);
	if (cmd.function_id == kFunctionPipeline) {
		recordDecode(f, (const Command *)cmd.output, depth + 1);
		return;
	}
	if (cmd.function_id != kFunctionDecode || !cmd.parameters)
		return;
	const uint8_t *P = cmd.parameters;
	dump(f, "picture", (uint64_t)P, 0x200, 0);
	dump(f, "picture+200", (uint64_t)P + 0x200, 0xc0, 0);
	const uint8_t *sps = (const uint8_t *)u64At(P, 0x260);
	const uint8_t *pps = (const uint8_t *)u64At(P, 0x268);
	if (sps)
		fprintf(f, "  sps profile %u level %u chroma %u depth %u/%u frame_num_log2-4 %u poc_type %u "
			   "poc_lsb_log2-4 %u max_refs %u mbs %ux%u frame_mbs_only %u\n",
			u32At(sps, 0x4), u32At(sps, 0x24), u32At(sps, 0x2c), u32At(sps, 0x34),
			u32At(sps, 0x38), u32At(sps, 0x404), u32At(sps, 0x408), u32At(sps, 0x40c),
			u32At(sps, 0x820), u32At(sps, 0x828) + 1, u32At(sps, 0x82c) + 1, u32At(sps, 0x830));
	if (pps)
		fprintf(f, "  pps cabac %u refs_l0-1 %u refs_l1-1 %u wp %u/%u qp-26 %d transform8x8 %u\n",
			u32At(pps, 0xc), u32At(pps, 0x90), u32At(pps, 0x94), u32At(pps, 0x98),
			u32At(pps, 0x9c), (int32_t)u32At(pps, 0xa0), u32At(pps, 0xb8));
	uint32_t pool = u32At(P, 0x270), index = u32At(P, 0x274), slices = u32At(P, 0x278);
	fprintf(f, "  poc %d/%d bitstream pool %u index %u slices %u\n", (int32_t)u32At(P, 0x14),
		(int32_t)u32At(P, 0x18), pool, index, slices);
	const uint8_t *entries = (const uint8_t *)u64At(P, 0x290);
	const uint8_t *headers = (const uint8_t *)u64At(P, 0x288);
	const uint8_t *buffer = bitstreamBuffer(pool, index);
	for (uint32_t i = 0; i < slices && i < 8 && entries; i++) {
		const uint8_t *e = (const uint8_t *)u64At(entries, i * 8);
		uint32_t size = u32At(e, 0), offset = u32At(e, 4);
		fprintf(f, "  slice %u size %u offset %u", i, size, offset);
		uint8_t head[16] = {};
		if (buffer && readable((uint64_t)(buffer + offset), head, sizeof head)) {
			fprintf(f, " bytes");
			for (uint8_t b : head)
				fprintf(f, " %02x", b);
		}
		const uint8_t *h = headers ? (const uint8_t *)u64At(headers, i * 8) : nullptr;
		if (h)
			fprintf(f, " header frame_num %u field_pic %u", u32At(h, 0x10), u32At(h, 0x14));
		fprintf(f, "\n");
	}
	if (headers)
		dump(f, "slice header 0", u64At(headers, 0), 0x100, 0);
	const uint8_t *out = (const uint8_t *)u64At(cmd.output, 0x20);
	if (cmd.output)
		dump(f, "output", (uint64_t)cmd.output, 0x40, 1);
	if (out)
		recordSurface(f, (IOSurfaceRef)u64At(out, 0x10));
}

// --- the answers -----------------------------------------------------------------------------

struct Context {
	uint32_t function_id;
	uint64_t sub_id;
	uint32_t width;
	uint32_t height;
};

// Bitstream buffers, by AppleGVA's pool type. Each is whole pages of its own.
struct BitstreamBuffers {
	uint32_t pool;
	uint32_t count;
	size_t bytes;
	uint8_t *buffers[32];
};
constexpr size_t kMaxPools = 8;
std::mutex g_poolLock;
BitstreamBuffers g_pools[kMaxPools];

const uint8_t *bitstreamBuffer(uint32_t pool, uint32_t index)
{
	std::lock_guard<std::mutex> hold(g_poolLock);
	for (const BitstreamBuffers &b : g_pools)
		if (b.count && b.pool == pool && index < b.count)
			return b.buffers[index];
	return nullptr;
}

void freePool(BitstreamBuffers &b)
{
	for (uint32_t i = 0; i < b.count; i++)
		munmap(b.buffers[i], b.bytes);
	b = {};
}

// Tokens of scheduled commands; AppleGVA holds a token only until its command is done.
struct Token {
	uint64_t seqno;
};
constexpr size_t kTokens = 1024;
Token g_tokens[kTokens];
std::atomic<uint64_t> g_seqno{0};

constexpr uint32_t kOk = 0, kUnsupported = 1, kUnsupportedFormat = 3, kBadArgument = 5,
		   kNoMemory = 7;

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

uint64_t answer(int entry, uint64_t a0, uint64_t a1, uint64_t a2)
{
	switch (entry) {
	case 0: { // CreateDriver
		uint8_t probe[8];
		if (!readable(a0, probe, sizeof probe))
			return kBadArgument;
		*(void **)a0 = &g_driver;
		void *device = MTLCreateSystemDefaultDevice();
		std::lock_guard<std::mutex> hold(g_logLock);
		if (FILE *f = recordFile())
			fprintf(f, "DEVICE %s\n", device ? object_getClassName((id)device) : "none");
		return kOk;
	}
	case 2: { // CreateContexts
		ContextInit *ci = (ContextInit *)a1;
		if (!ci || !ci->info)
			return kBadArgument;
		for (uint32_t i = 0; i < ci->info->count; i++) {
			const FunctionInfo *fi = ci->info->functions[i];
			ci->info->contexts[i] = fi ? new Context{fi->function_id, fi->sub_id, fi->width, fi->height}
						   : nullptr;
		}
		ci->capabilities = g_capabilities;
		return kOk;
	}
	case 3: { // DeleteContexts
		const ContextDelete *cd = (const ContextDelete *)a1;
		if (!cd)
			return kBadArgument;
		for (uint32_t i = 0; i < cd->count; i++)
			delete (Context *)cd->contexts[i];
		return kOk;
	}
	case 4: { // GetBitstreamBufferInfo
		BitstreamRequirement *r = (BitstreamRequirement *)a1;
		if (!r)
			return kBadArgument;
		memset(r->alignment, 0, sizeof r->alignment);
		r->alignment[0] = 0x100;
		return kOk;
	}
	case 5: { // AllocateBitstreamBufferPool
		BitstreamPool *p = (BitstreamPool *)a1;
		if (!p || !p->addresses || p->count == 0 || p->count > 32 || p->size == 0)
			return kBadArgument;
		std::lock_guard<std::mutex> hold(g_poolLock);
		BitstreamBuffers *slot = nullptr;
		for (BitstreamBuffers &b : g_pools)
			if (b.count && b.pool == p->pool)
				freePool(*(slot = &b));
		for (BitstreamBuffers &b : g_pools)
			if (!slot && !b.count)
				slot = &b;
		if (!slot)
			return kNoMemory;
		slot->pool = p->pool;
		slot->bytes = (p->size + pageSize() - 1) & ~(pageSize() - 1);
		for (uint32_t i = 0; i < p->count; i++) {
			void *m = mmap(nullptr, slot->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
			if (m == MAP_FAILED) {
				freePool(*slot);
				return kNoMemory;
			}
			slot->buffers[slot->count++] = (uint8_t *)m;
			p->addresses[i] = (uint64_t)m;
		}
		return kOk;
	}
	case 6: { // FreeBitstreamBufferPool
		std::lock_guard<std::mutex> hold(g_poolLock);
		for (BitstreamBuffers &b : g_pools)
			if (b.count && b.pool == (uint32_t)a1)
				freePool(b);
		return kOk;
	}
	case 7: { // GetPixelBufferInfo
		PixelRequirement *r = (PixelRequirement *)a1;
		if (!r || !r->widths || !r->heights || !r->pitches || !r->alignments)
			return kBadArgument;
		if (r->fourcc != 'NV12' || r->planes != 2)
			return kUnsupportedFormat;
		uint32_t height = alignUp(r->height, 16), pitch = alignUp(r->width, 0x100);
		r->widths[0] = r->width;
		r->heights[0] = height;
		r->widths[1] = r->width / 2;
		r->heights[1] = height / 2;
		for (int p = 0; p < 2; p++) {
			r->pitches[p] = pitch;
			r->alignments[p] = 0x100;
		}
		return kOk;
	}
	case 10: { // ScheduleCommands
		const CommandList *cl = (const CommandList *)a1;
		TokenList *tl = (TokenList *)a2;
		if (!cl || !tl || cl->count != tl->count || !tl->tokens)
			return kBadArgument;
		for (uint32_t i = 0; i < cl->count; i++) {
			uint64_t n = g_seqno++;
			Token *t = &g_tokens[n % kTokens];
			t->seqno = n;
			tl->tokens[i] = t;
			std::lock_guard<std::mutex> hold(g_logLock);
			if (FILE *f = recordFile()) {
				fprintf(f, "DECODE seqno %llu\n", (unsigned long long)n);
				if (n < kFullCalls)
					recordDecode(f, cl->commands[i], 0);
			}
		}
		return kOk;
	}
	case 11: { // QueryCommandStatus
		QueryList *ql = (QueryList *)a1;
		if (!ql || !ql->queries)
			return kBadArgument;
		for (uint32_t i = 0; i < ql->count; i++) {
			if (!ql->queries[i].token)
				return kBadArgument;
			ql->queries[i].status = 1;
			ql->queries[i].field_14 = 0;
		}
		return kOk;
	}
	case 8: // AllocatePixelBufferPool: not on the decode path
		return kUnsupported;
	case 15: { // GetDriverParameter
		DriverParamList *l = (DriverParamList *)a1;
		if (!l || !l->params)
			return kBadArgument;
		for (uint32_t i = 0; i < l->count; i++) {
			DriverParam &param = l->params[i];
			if (param.id == kParamCapabilities) {
				param.data = g_capabilities;
			} else if (param.id == kParamFunctionAvailable && param.data) {
				FunctionAvailable *fa = (FunctionAvailable *)param.data;
				const CapabilityBlock *b = (const CapabilityBlock *)g_capabilities;
				fa->budget = 0;
				for (uint32_t f = 0; f < b->function_count; f++)
					if (b->function[f].function_id == fa->function_id &&
					    b->function[f].sub_id == fa->sub_id && b->function[f].engine == fa->engine)
						fa->budget = (uint32_t)b->function[f].field_18;
				if (!fa->budget)
					return kUnsupported;
			} else if (param.id == kParamDecodeLimit && param.data) {
				*(uint32_t *)param.data = (uint32_t)(kEngineBudget / 0x5a000 * 0x5a000);
			} else {
				return kUnsupported;
			}
		}
		return kOk;
	}
	default: // DeleteDriver, FreePixelBufferPool, the parameters: recorded, answered success
		return kOk;
	}
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
	uint64_t result = answer(N, a0, a1, a2);
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
