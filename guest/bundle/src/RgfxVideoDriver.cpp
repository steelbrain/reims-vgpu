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
 * The driver decodes H.264 on the host's hardware, where the host's video caps list it
 * (RgfxVideoChannel.hpp). The capability block lists a video engine and a graphics engine, the
 * H.264 decode pipeline (head, decode, frame transfer) and frame processing; the entry points
 * answer AppleGVA's questions (the bitstream alignment, the frames' layout, the capability
 * parameters), hand out bitstream buffers of whole pages, open a host session per decode
 * context, and send each decode command to it as a command buffer of the session's queue, its
 * access unit copied out of AppleGVA's buffer, with the SPS and PPS written back out of
 * AppleGVA's records where they changed (RgfxVideoH264.hpp). A command's token is answered done
 * once its command buffer completed, which is once the picture is in the output surface. The host keeps the decoded
 * picture buffer itself, from the stream: AppleGVA's commands name no references.
 *
 * AppleGVA hands this driver a session only where it picks the AMD driver for the flow: on a
 * board without an Intel GPU, or with com.apple.AppleGVA's gvaForceAMDKE set.
 *
 * Layouts AppleGVA and the driver share are named by what they hold where that is stated by
 * the calls' use; a field whose meaning nothing states yet is field_xx, xx its byte offset.
 */

#include <IOSurface/IOSurfaceRef.h>
#include <os/log.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "RgfxVideoChannel.hpp"
#include "RgfxVideoH264.hpp"

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
// the block too.
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
	// H.264 decode, where the host's hardware takes it: the pipeline head, the decode, the
	// transfer of its frames.
	if (rgfx::video::hostCodec(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE)) {
		Function &head = addFunction(b, kFunctionPipeline, 1, 'rawb', 'rawb');
		Function &decode = addFunction(b, kFunctionDecode, kCodecH264, 'rawb', 'NV12');
		Function &transfer = addFunction(b, kFunctionTransfer, 1, 'NV12', 'NV12');
		head.next = &decode;
		decode.next = &transfer;
	}
	// Frame processing on the graphics engine.
	addFunction(b, kFunctionProcess, 0x10, 'NV12', 'NV12', kGraphicsEngine);
	b.functions = b.function_list;
}

// The block AppleGVA reads, filled once, as AVD_CreateAVDAccelerator is first called.
CapabilityBlock g_block;
void *g_capabilities;

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

size_t pageSize() { return (size_t)getpagesize(); }

// The value at `off` in a record AppleGVA hands the driver.
uint32_t u32At(const uint8_t *base, size_t off)
{
	uint32_t v;
	memcpy(&v, base + off, sizeof v);
	return v;
}

uint64_t u64At(const uint8_t *base, size_t off)
{
	uint64_t v;
	memcpy(&v, base + off, sizeof v);
	return v;
}

// --- the answers -----------------------------------------------------------------------------

// A context AppleGVA made for one function of a pipeline; the decode's holds the session on the
// host and the parameter sets it last sent there.
struct Context {
	uint32_t function_id;
	uint64_t sub_id;
	uint32_t width;
	uint32_t height;
	rgfx::video::Session *session;
	std::vector<uint8_t> sent;
};

// The live sessions, whose output textures go when AppleGVA frees its frames' pool.
std::mutex g_sessionLock;
std::vector<Context *> g_sessions;

// Bitstream buffers, by AppleGVA's pool type: whole pages, which a decode copies its access unit
// out of (RgfxVideoChannel.hpp).
struct BitstreamBuffers {
	uint32_t pool;
	uint32_t count;
	size_t bytes;
	uint8_t *buffers[32];
};
constexpr size_t kMaxPools = 8;
std::mutex g_poolLock;
BitstreamBuffers g_pools[kMaxPools];

// The buffer AppleGVA names by (pool, index): its bytes and their count; false where none is.
bool bitstreamBuffer(uint32_t pool, uint32_t index, const uint8_t **bytes, size_t *size)
{
	std::lock_guard<std::mutex> hold(g_poolLock);
	for (const BitstreamBuffers &b : g_pools)
		if (b.count && b.pool == pool && index < b.count) {
			*bytes = b.buffers[index];
			*size = b.bytes;
			return true;
		}
	return false;
}

void freePool(BitstreamBuffers &b)
{
	for (uint32_t i = 0; i < b.count; i++)
		munmap(b.buffers[i], b.bytes);
	b = {};
}

// Tokens of scheduled commands; AppleGVA holds a token only until its command is done. Each
// holds its work on the host, if any, until it is answered done or the slot comes round again.
struct Token {
	uint64_t seqno;
	void *work;
	bool failed;
};
constexpr size_t kTokens = 1024;
std::mutex g_tokenLock;
std::mutex g_scheduleLock;
Token g_tokens[kTokens];
uint64_t g_seqno;

constexpr uint32_t kOk = 0, kUnsupported = 1, kUnsupportedFormat = 3, kBadArgument = 5,
		   kNoMemory = 7;

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// The decode command of `c`, following a pipeline's head to it; null where there is none.
const Command *decodeCommand(const Command *c)
{
	for (int depth = 0; c && depth < 4; depth++) {
		if (c->function_id == kFunctionDecode)
			return c;
		if (c->function_id != kFunctionPipeline)
			return nullptr;
		c = (const Command *)c->output;
	}
	return nullptr;
}

// Logs, once a process, why a picture was not sent.
void notSent(std::atomic<bool> &said, const char *why)
{
	if (!said.exchange(true))
		os_log_error(OS_LOG_DEFAULT, "RGFX: video: a picture not sent: %{public}s", why);
}

// Commits the picture `c` decodes on its context's session: its work on the host, or null
// where it could not be sent.
void *decodePicture(const Command *c)
{
	Context *context = (Context *)c->context;
	const uint8_t *P = c->parameters;
	if (!context || !context->session || !P || !c->output)
		return nullptr;
	// The access unit: the slices, which lie one after another in one bitstream buffer.
	const uint8_t *bytes;
	size_t bufferBytes;
	const uint32_t pool = u32At(P, 0x270), index = u32At(P, 0x274), slices = u32At(P, 0x278);
	const uint8_t *entries = (const uint8_t *)u64At(P, 0x290);
	if (!slices || !entries || !bitstreamBuffer(pool, index, &bytes, &bufferBytes))
		return nullptr;
	uint64_t first = UINT64_MAX, end = 0;
	for (uint32_t i = 0; i < slices; i++) {
		const uint8_t *e = (const uint8_t *)u64At(entries, i * 8);
		if (!e)
			return nullptr;
		const uint64_t size = u32At(e, 0), offset = u32At(e, 4);
		first = offset < first ? offset : first;
		end = offset + size > end ? offset + size : end;
	}
	if (end > bufferBytes)
		return nullptr;
	// The parameter sets, sent where they differ from the last ones.
	std::vector<uint8_t> sets;
	const auto *sps = (const rgfx::video::h264::Sps *)u64At(P, 0x260);
	const auto *pps = (const rgfx::video::h264::Pps *)u64At(P, 0x268);
	if (!sps || !pps || !rgfx::video::h264::writeSps(*sps, sets) ||
	    !rgfx::video::h264::writePps(*pps, *sps, sets)) {
		static std::atomic<bool> said;
		notSent(said, "its parameter sets are refused");
		return nullptr;
	}
	const bool changed = sets != context->sent;
	rgfx::video::DecodeH264 d = {};
	d.bytes = bytes + first;
	d.size = end - first;
	d.parameterSets = changed ? sets.data() : nullptr;
	d.parameterSetsSize = changed ? (uint32_t)sets.size() : 0;
	const uint8_t *out = (const uint8_t *)u64At(c->output, 0x20);
	d.output = out ? (IOSurfaceRef)u64At(out, 0x10) : nullptr;
	void *work = rgfx::video::decodeH264(context->session, d);
	if (work && changed)
		context->sent = std::move(sets);
	if (!work) {
		static std::atomic<bool> said;
		notSent(said, "no texture over its output, or no command buffer");
	}
	return work;
}

uint64_t answer(int entry, uint64_t a0, uint64_t a1, uint64_t a2)
{
	switch (entry) {
	case 0: { // CreateDriver
		if (!a0)
			return kBadArgument;
		*(void **)a0 = &g_driver;
		return kOk;
	}
	case 2: { // CreateContexts
		ContextInit *ci = (ContextInit *)a1;
		if (!ci || !ci->info)
			return kBadArgument;
		for (uint32_t i = 0; i < ci->info->count; i++) {
			const FunctionInfo *fi = ci->info->functions[i];
			Context *context = fi ? new Context{fi->function_id, fi->sub_id, fi->width,
							    fi->height, nullptr, {}}
					      : nullptr;
			if (context && fi->function_id == kFunctionDecode && fi->sub_id == kCodecH264) {
				context->session =
				    rgfx::video::openSession(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE);
				if (!context->session) {
					for (uint32_t j = 0; j < i; j++) {
						Context *made = (Context *)ci->info->contexts[j];
						delete made;
						ci->info->contexts[j] = nullptr;
					}
					delete context;
					return kUnsupported;
				}
				std::lock_guard<std::mutex> hold(g_sessionLock);
				g_sessions.push_back(context);
			}
			ci->info->contexts[i] = context;
		}
		ci->capabilities = g_capabilities;
		return kOk;
	}
	case 3: { // DeleteContexts
		const ContextDelete *cd = (const ContextDelete *)a1;
		if (!cd)
			return kBadArgument;
		for (uint32_t i = 0; i < cd->count; i++) {
			Context *context = (Context *)cd->contexts[i];
			if (context && context->session) {
				{
					std::lock_guard<std::mutex> hold(g_sessionLock);
					for (auto it = g_sessions.begin(); it != g_sessions.end(); ++it)
						if (*it == context) {
							g_sessions.erase(it);
							break;
						}
				}
				rgfx::video::closeSession(context->session);
			}
			delete context;
		}
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
			void *cpu = mmap(nullptr, slot->bytes, PROT_READ | PROT_WRITE,
					 MAP_ANON | MAP_PRIVATE, -1, 0);
			if (cpu == MAP_FAILED) {
				freePool(*slot);
				return kNoMemory;
			}
			slot->buffers[slot->count++] = (uint8_t *)cpu;
			p->addresses[i] = (uint64_t)cpu;
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
	case 9: { // FreePixelBufferPool: the frames go, so do the textures over them
		std::lock_guard<std::mutex> hold(g_sessionLock);
		for (Context *context : g_sessions)
			rgfx::video::forgetOutputs(context->session);
		return kOk;
	}
	case 10: { // ScheduleCommands
		const CommandList *cl = (const CommandList *)a1;
		TokenList *tl = (TokenList *)a2;
		if (!cl || !tl || cl->count != tl->count || !tl->tokens)
			return kBadArgument;
		// One list at a time, so pictures reach their sessions in AppleGVA's order.
		std::lock_guard<std::mutex> hold(g_scheduleLock);
		for (uint32_t i = 0; i < cl->count; i++) {
			const Command *decode = decodeCommand(cl->commands[i]);
			void *work = decode ? decodePicture(decode) : nullptr;
			std::lock_guard<std::mutex> tokens(g_tokenLock);
			const uint64_t n = g_seqno++;
			Token *t = &g_tokens[n % kTokens];
			rgfx::video::releaseToken(t->work);
			*t = {n, work, decode && !work};
			tl->tokens[i] = t;
		}
		return kOk;
	}
	case 11: { // QueryCommandStatus
		QueryList *ql = (QueryList *)a1;
		if (!ql || !ql->queries)
			return kBadArgument;
		for (uint32_t i = 0; i < ql->count; i++) {
			Token *t = (Token *)ql->queries[i].token;
			if (!t)
				return kBadArgument;
			std::lock_guard<std::mutex> hold(g_tokenLock);
			int status = t->work ? rgfx::video::tokenStatus(t->work) : 1;
			if (status == -1)
				t->failed = true;
			if (status != 0) {
				rgfx::video::releaseToken(t->work);
				t->work = nullptr;
			}
			ql->queries[i].status = status == 0 ? 2 : 1;
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
	default: // DeleteDriver, the parameters: answered success
		return kOk;
	}
}

template <int N>
uint64_t entryPoint(uint64_t a0, uint64_t a1, uint64_t a2)
{
	return answer(N, a0, a1, a2);
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
	Entries<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>::fill(g_table);
	g_table.driver = &g_driver;
	DriverIdentity id = {};
	id.renderer_id = kRendererID;
	static std::once_flag filled;
	std::call_once(filled, [] {
		fillCapabilities(g_block);
		g_capabilities = &g_block;
	});
	id.capabilities = g_capabilities;
	id.field_10 = 1;
	memcpy(id.name, "RgfxVideo      ", 16);
	memcpy(identity, &id, offsetof(DriverIdentity, name) + sizeof id.name);
	*out = &g_table;
	return 0;
}

extern "C" __attribute__((visibility("default"))) int AVD_DestroyAVDAccelerator(void *table)
{
	return 0;
}
