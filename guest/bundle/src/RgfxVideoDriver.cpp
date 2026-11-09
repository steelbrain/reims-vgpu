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
 * The driver decodes H.264 and HEVC on the host's hardware, each where the host's video caps
 * list it (RgfxVideoChannel.hpp), HEVC in the profiles they list: Main into NV12, Main 10 into
 * P010. The capability block lists a video engine and a graphics engine, each codec's decode
 * pipeline (head, decode, frame transfer) and frame processing; the entry points answer
 * AppleGVA's questions (the bitstream alignment, the frames' layout, the capability
 * parameters), hand out bitstream buffers of whole pages, open a host session per decode
 * context, and send each decode command to it as a command buffer of the session's queue, its
 * access unit copied out of AppleGVA's buffer, with the SPS and PPS written back out of
 * AppleGVA's records where they changed (RgfxVideoH264.hpp, RgfxVideoH265.hpp). A command's
 * token is answered done once its command buffer completed, which is once the picture is in
 * the output surface. The host keeps the decoded picture buffer itself, from the stream:
 * AppleGVA's commands name no references.
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

#include <algorithm>
#include <atomic>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <vector>

#include "RgfxVideoChannel.hpp"
#include "RgfxVideoH264.hpp"
#include "RgfxVideoH265.hpp"
#include "RgfxVideoStatus.hpp"

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
constexpr uint64_t kCodecHevc = 0x800;

Function &addFunction(CapabilityBlock &b, uint32_t function_id, uint64_t sub_id, uint32_t in,
		      std::initializer_list<uint32_t> outs, uint32_t engine = kVideoEngine)
{
	Function &f = b.function[b.function_count];
	f.function_id = function_id;
	f.sub_id = sub_id;
	f.engine = engine;
	f.field_18 = kEngineBudget;
	f.field_20 = kEngineBudget;
	f.input_count = 1;
	f.inputs[0] = {in, 1};
	for (uint32_t out : outs)
		f.outputs[f.output_count++] = {out, 1};
	b.function_list[b.function_count++] = &f;
	return f;
}

// A decoder as AppleGVA looks it up: a pipeline head, whose next is the decode of `codec`, a
// bitstream in and frames of `outs` out, whose next is a transfer of its frames.
void addDecoder(CapabilityBlock &b, uint64_t codec, std::initializer_list<uint32_t> outs)
{
	Function &head = addFunction(b, kFunctionPipeline, 1, 'rawb', {'rawb'});
	Function &decode = addFunction(b, kFunctionDecode, codec, 'rawb', outs);
	Function &transfer = addFunction(b, kFunctionTransfer, 1, 'NV12', {'NV12'});
	head.next = &decode;
	decode.next = &transfer;
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
	// H.264 decode, where the host's hardware takes it, into NV12.
	if (rgfx::video::hostCodec(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE))
		addDecoder(b, kCodecH264, {'NV12'});
	// HEVC decode, where the host's hardware takes it: Main into NV12 (video or full range),
	// Main 10 into P010 ('x420', 'xf20').
	if (const RgfxVideoCodecCaps *hevc =
	        rgfx::video::hostCodec(RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE)) {
		const bool main = (hevc->profiles & RGFX_VIDEO_HEVC_MAIN) != 0;
		const bool main10 = (hevc->profiles & RGFX_VIDEO_HEVC_MAIN_10) != 0;
		if (main && main10)
			addDecoder(b, kCodecHevc, {'NV12', 'x420', '420f', 'xf20'});
		else if (main)
			addDecoder(b, kCodecHevc, {'NV12', '420f'});
		else if (main10)
			addDecoder(b, kCodecHevc, {'x420', 'xf20'});
	}
	// Frame processing on the graphics engine.
	addFunction(b, kFunctionProcess, 0x10, 'NV12', {'NV12'}, kGraphicsEngine);
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
// matching function answers. It reads six alignments; the driver answers each with the host
// decoder's, none 0, as AppleGVA may divide by any of them.
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

// GetPixelBufferInfo: the driver answers each plane's size, row pitch and alignment, as the host
// lands a picture, and AppleGVA makes its frames' IOSurfaces so. Each answer is whole, none 0, as
// AppleGVA may divide by any of them; a request the driver does not answer whole is refused, its
// record untouched.
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

// Get/SetContextParameter: a list of parameters, each of a function's context.
struct ContextParam {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint32_t engine;
	uint32_t field_14;
	void *context;
	uint32_t id;
	uint32_t field_24;
	void *data;
};
static_assert(sizeof(ContextParam) == 0x30, "context parameter layout");

struct ContextParamList {
	uint32_t count;
	uint32_t field_04;
	ContextParam *params;
};

// A decoder's tile parameter: what VideoToolbox's tile decoder session asks of an HEIF image's
// tiles and of the canvas it lays them in (its TileDecoderRequirements). AppleGVA starts that
// session only where max_canvas_row_bytes is not 0, and CMPhoto, above it, divides by the
// alignments: an HEIC decode died of SIGFPE in CMPhotoDecompressionContainerCreateImageForIndex
// while the parameter was answered success with nothing written (measured on the RTX 5080 host,
// 2026-11-03). So it is answered whole or refused, never success with the record as it came.
constexpr uint32_t kParamTileInfo = 0xb;

struct TileInfo {
	uint32_t width_alignment;  // TileWidthPixelAlignment
	uint32_t height_alignment; // TileHeightPixelAlignment
	uint32_t max_canvas_row_bytes;
	uint32_t min_width;
	uint32_t min_height;
	uint32_t max_width;
	uint32_t max_height;
};

// The tile parameter of the host's decoder of `codec` (RGFX_VIDEO_CODEC_*), from its caps, into
// `out`; false where the host has no such decoder. A tile is a picture the host decodes, of the
// sizes it takes (its caps' least and largest coded size). Where Core Animation draws an HEIF
// grid, CMPhoto has VideoToolbox's tile decoder lay each HEVC tile straight into one YCbCr canvas
// of the whole image, and AppleGVA names the tile's place in it in the picture's parameters
// (hevcTilePlace): the picture lands there, each plane's first byte at the alignment the host's
// textures and copy take (outputPlaneAlignment). So a tile's place is of whole chroma samples
// (2 by 2 in 4:2:0) and, across, of samples whose bytes keep that alignment in every format
// the host puts out; the canvas's row is at most the widest the host writes, its largest width
// in samples of its widest format. An H.264 picture's parameters name no place the driver knows,
// so its decoder answers a largest canvas row of 0, with which AppleGVA starts no tile decoder,
// and its tiles go through frames of their own, which VideoToolbox lays in the canvas (measured
// on the RTX 5080 host with BGRA canvases: a 4096x4096 image's 64 tiles of 512 went through four
// 512x512 frames). None of its answers is 0 but that row, as CMPhoto divides by the alignments.
bool tileInfo(uint32_t codec, TileInfo *out)
{
	const RgfxVideoCodecCaps *caps = rgfx::video::hostCodec(codec, RGFX_VIDEO_DECODE);
	if (!caps || !caps->max_size[0] || !caps->max_size[1])
		return false;
	uint32_t across = 2, widest = 1;
	for (uint32_t sampleBytes : {1u, 2u}) {
		if (!(caps->formats & (sampleBytes == 1 ? RGFX_VIDEO_FORMAT_NV12 : RGFX_VIDEO_FORMAT_P010)))
			continue;
		const uint32_t alignment = rgfx::video::outputPlaneAlignment(sampleBytes);
		if (alignment == 0 || alignment % sampleBytes != 0)
			return false;
		across = std::max(across, alignment / sampleBytes);
		widest = sampleBytes;
	}
	const uint32_t least[2] = {caps->min_size[0] ? caps->min_size[0] : 1,
	                           caps->min_size[1] ? caps->min_size[1] : 1};
	const uint32_t canvasRow = codec == RGFX_VIDEO_CODEC_HEVC ? caps->max_size[0] * widest : 0;
	*out = {across, 2, canvasRow, least[0], least[1], caps->max_size[0], caps->max_size[1]};
	return true;
}

struct FunctionAvailable {
	uint32_t function_id;
	uint32_t sub_id;
	uint32_t engine;
	uint32_t budget; // out, 0 where the function is not listed
};

int g_tableDriver; // what the table's driver pointer points at

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
struct Driver;

struct Context {
	// The driver AppleGVA made the context on, whose bitstream buffers its pictures name.
	Driver *driver;
	uint32_t function_id;
	uint64_t sub_id;
	uint32_t width;
	uint32_t height;
	rgfx::video::Session *session;
	std::vector<uint8_t> sent;
	// Where its HEVC pictures' frame records describe their surfaces, once a picture's has shown
	// it (FrameDescription::packed), and 0 until then.
	std::atomic<uint32_t> hevc_description{0};
	// The layout its H.264 pictures' parsed SPS comes in (rgfx::video::h264::SpsLayout), once a
	// picture's has shown it, and 0 until then.
	std::atomic<uint8_t> h264_sps_layout{0};
	// Held across a schedule of its pictures (ScheduleCommands): one list of the context's at a
	// time, so they reach its session in AppleGVA's order, while other contexts' lists go on.
	std::mutex schedule{};
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

// A driver AppleGVA made (CreateDriver), one for each of its decoders, with bitstream pools of its
// own, as AMD's driver keeps them. A pool allocated again replaces its driver's pool of that type
// alone: with the pools a process's, a second decoder's pool (HEVC with alpha decodes its two
// layers in two in one process) freed the buffers the first was still writing access units in.
struct Driver {
	BitstreamBuffers pools[kMaxPools];
};

// The live drivers, and their pools, under g_poolLock.
std::mutex g_poolLock;
std::vector<Driver *> g_drivers;

// `handle` as a live driver, under g_poolLock; null where it names none.
Driver *liveDriver(uint64_t handle)
{
	for (Driver *d : g_drivers)
		if ((uint64_t)d == handle)
			return d;
	return nullptr;
}

// The buffer `driver` names by (pool, index): its bytes and their count; false where none is.
bool bitstreamBuffer(Driver *driver, uint32_t pool, uint32_t index, const uint8_t **bytes, size_t *size)
{
	std::lock_guard<std::mutex> hold(g_poolLock);
	if (!liveDriver((uint64_t)driver))
		return false;
	for (const BitstreamBuffers &b : driver->pools)
		if (b.count && b.pool == pool && index < b.count) {
			*bytes = b.buffers[index];
			*size = b.bytes;
			return true;
		}
	return false;
}

// The buffer of index `index` of the pool AppleGVA names an HEVC picture's slice segments in,
// which they name by index alone: `driver`'s pool of type 0, else the one pool it has.
bool hevcBitstreamBuffer(Driver *driver, uint32_t index, const uint8_t **bytes, size_t *size)
{
	if (bitstreamBuffer(driver, 0, index, bytes, size))
		return true;
	std::lock_guard<std::mutex> hold(g_poolLock);
	if (!liveDriver((uint64_t)driver))
		return false;
	const BitstreamBuffers *only = nullptr;
	for (const BitstreamBuffers &b : driver->pools)
		if (b.count) {
			if (only)
				return false;
			only = &b;
		}
	if (!only || index >= only->count)
		return false;
	*bytes = only->buffers[index];
	*size = only->bytes;
	return true;
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
};
constexpr size_t kTokens = 1024;
std::mutex g_tokenLock;
Token g_tokens[kTokens];
uint64_t g_seqno;

constexpr uint32_t kOk = 0, kUnsupported = 1, kUnsupportedFormat = 3, kBadArgument = 5,
		   kNoMemory = 7;

// The alignment the host's decoder of AppleGVA's function sub-ID `sub` reads an access unit in
// place at (RgfxVideoCodecCaps::bitstream_alignment), the larger of the two decoders' for a
// function of neither; 1 where the host answers none.
uint32_t bitstreamAlignment(uint64_t sub)
{
	uint32_t alignment = 1;
	for (uint32_t codec : {RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC}) {
		const bool its = (codec == RGFX_VIDEO_CODEC_H264 && sub == kCodecH264) ||
		                 (codec == RGFX_VIDEO_CODEC_HEVC && sub == kCodecHevc);
		if ((sub == kCodecH264 || sub == kCodecHevc) && !its)
			continue;
		const RgfxVideoCodecCaps *caps = rgfx::video::hostCodec(codec, RGFX_VIDEO_DECODE);
		if (caps && caps->bitstream_alignment > alignment)
			alignment = caps->bitstream_alignment;
	}
	return alignment;
}

// The largest frame side the driver answers a layout for, past any the host decodes, so that a
// plane's pitch and size stay within 32 bits.
constexpr uint32_t kMaxFrameSide = 16384;

// Deletes `context`: its session closed and out of the live ones.
void deleteContext(Context *context)
{
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

// Logs, once a process, that a schedule failed for a decode (RgfxVideoStatus.hpp), `why`.
void scheduleFailed(const char *why)
{
	static std::atomic<bool> said;
	if (!said.exchange(true))
		os_log_error(OS_LOG_DEFAULT, "RGFX: video: a decode reported to AppleGVA as failed: %{public}s",
		             why);
}

// The surface AppleGVA decodes command `c`'s picture into.
IOSurfaceRef outputSurface(const Command *c)
{
	const uint8_t *out = (const uint8_t *)u64At(c->output, 0x20);
	return out ? (IOSurfaceRef)u64At(out, 0x10) : nullptr;
}

// Where an HEVC frame record of AppleGVA's describes its surface (hevcOutputSurface). Every
// release measured describes it with the same seven fields, in order: the surface's global id, a
// size, its width, its height, a pitch, 8 bytes and its pixel format. Where the fields start in
// the frame and how wide each is differ by release; each is read as a u32 at its start:
// - fields of 4 bytes from +0x70 on macOS 11.7.10 and 11.7.11, from +0x78 on 12.7.6 and from
//   +0x80 on 13.7.8 (measured on 11 and 12 on the RTX 5080, 2026-11-08, for 1024x1024 '420f' and
//   320x240 '420v' surfaces; on 13.7.8 on the 780M, ids 6, 7, 9 and 15 for 6016x6016 and
//   1024x1024 '420f' and 320x240 '420v' surfaces);
// - fields of 8 bytes from +0x80 on macOS 14.8.9 (measured on the RTX 5080, 2026-11-08: ids 10
//   and 33 for 3840x2160 'x420' and 320x240 '420v' surfaces);
// - fields of 8 bytes from +0x88 on macOS 15.8.2 and 26.7.2 (measured on the RTX 5080 and the
//   780M, 2026-11-08: ids 14 and 22 on 26.7.2 for 3840x2160 'x420' and 320x240 '420v'
//   surfaces; and on the RTX 5080, 2026-11-03: the id is that of the IOSurface VideoToolbox hands
//   back for the picture, 7 for picture 0, 27 for picture 1).
// So the driver finds the layout instead of listing them: every start from kDescriptionFirst to
// kDescriptionLast in steps of 4, with fields of 4 or 8 bytes, the farthest read the pixel format
// of 8-byte fields from kDescriptionLast, at +0xd0. The window starts two 8-byte steps below the
// lowest start measured (11's +0x70), for the 11 releases before 11.7.10, which no base boots. In
// the frames of every release measured, no start and width but the release's own reads the
// surface's width, height and pixel format (their words from +0x60 to +0xdc, read on each release
// above but 11.7.10).
struct FrameDescription {
	uint32_t start, fieldBytes;
	size_t field(size_t index) const { return start + index * fieldBytes; }
	size_t id() const { return field(0); }
	size_t width() const { return field(2); }
	size_t height() const { return field(3); }
	size_t format() const { return field(6); }
	// As Context::hevc_description keeps it, never 0.
	uint32_t packed() const { return start << 8 | fieldBytes; }
	static FrameDescription unpacked(uint32_t p) { return {p >> 8, p & 0xff}; }
};
constexpr uint32_t kDescriptionFirst = 0x60, kDescriptionLast = 0xa0;
constexpr uint32_t kFieldBytes[] = {4, 8};

// The frame AppleGVA's decoded-picture record names for an HEVC picture: the record pointer is at
// P+0x10 and its current frame at the record's +0x08 (`field_08`). Null where either is.
const uint8_t *hevcFrame(const uint8_t *P)
{
	const uint8_t *record = (const uint8_t *)u64At(P, 0x10);
	return record ? (const uint8_t *)u64At(record, 0x08) : nullptr;
}

// The surface description `d` of `frame` names, where it is one whose width, height and pixel
// format `d` gives: retained, or null.
IOSurfaceRef describedSurface(const uint8_t *frame, FrameDescription d)
{
	IOSurfaceRef surface = IOSurfaceLookup(u32At(frame, d.id()));
	if (!surface)
		return nullptr;
	if (IOSurfaceGetWidth(surface) == u32At(frame, d.width()) &&
	    IOSurfaceGetHeight(surface) == u32At(frame, d.height()) &&
	    IOSurfaceGetPixelFormat(surface) == u32At(frame, d.format()))
		return surface;
	CFRelease(surface);
	return nullptr;
}

// Why an HEVC picture's frame record gave no output surface.
enum class NoOutput { kNone, kMany, kNotInSessionLayout };

// The surface AppleGVA decodes an HEVC picture into. An HEVC decode command carries no output on
// the command (c->output is null and there is no following transfer command, unlike H.264); the
// output frame is named in the decode's parameters, in AppleGVA's own decoded-picture record
// (hevcFrame), which describes its surface (FrameDescription). The first picture of `context`'s
// whose frame describes a surface in exactly one layout of the search window shows the session's
// layout, which `context` keeps; a later picture's surface is the one its frame describes in that
// layout. A frame that describes none, or describes one in more than one layout, gives none, and
// *why says which. The returned surface is retained; the caller releases it.
IOSurfaceRef hevcOutputSurface(Context *context, const uint8_t *P, NoOutput *why)
{
	*why = NoOutput::kNone;
	const uint8_t *frame = hevcFrame(P);
	if (!frame)
		return nullptr;
	if (const uint32_t known = context->hevc_description.load(std::memory_order_relaxed)) {
		IOSurfaceRef surface = describedSurface(frame, FrameDescription::unpacked(known));
		if (!surface)
			*why = NoOutput::kNotInSessionLayout;
		return surface;
	}
	IOSurfaceRef only = nullptr;
	FrameDescription found = {};
	for (uint32_t bytes : kFieldBytes)
		for (uint32_t start = kDescriptionFirst; start <= kDescriptionLast; start += 4) {
			IOSurfaceRef surface = describedSurface(frame, {start, bytes});
			if (!surface)
				continue;
			if (only) {
				CFRelease(only);
				CFRelease(surface);
				*why = NoOutput::kMany;
				return nullptr;
			}
			only = surface;
			found = {start, bytes};
		}
	if (only)
		context->hevc_description.store(found.packed(), std::memory_order_relaxed);
	return only;
}

// Where in its output surface an HEVC picture lands, in luma samples: its tile's place in an HEIF
// grid's canvas, which AppleGVA names in the picture's parameters at P+0x58 (across) and P+0x5c
// (down), and (0, 0) for a frame of its own. Measured on the RTX 5080 host (2026-11-03): for the
// 12, 12, 32 and 2 tiles of 2048x1536, 2000x1400, 4096x2048 and 1024x1024 grids of 512x512 tiles
// drawn through Core Animation, each into one canvas of its image, the two read each tile's
// column and row times 512; for every picture decoded into a frame of its own, both read 0.
struct Place {
	uint32_t x, y;
};

Place hevcTilePlace(const uint8_t *P) { return {u32At(P, 0x58), u32At(P, 0x5c)}; }

// The picture's size as the SPS crops it (its conformance window, in chroma samples of 4:2:0):
// what the host's copy lands.
void hevcPictureSize(const rgfx::video::h265::Sps &sps, uint32_t *width, uint32_t *height)
{
	const bool crops = sps.conformance_window_flag;
	const uint32_t cropX = crops ? 2 * (sps.conf_win_left_offset + sps.conf_win_right_offset) : 0;
	const uint32_t cropY = crops ? 2 * (sps.conf_win_top_offset + sps.conf_win_bottom_offset) : 0;
	*width = sps.pic_width_in_luma_samples > cropX ? sps.pic_width_in_luma_samples - cropX : 0;
	*height = sps.pic_height_in_luma_samples > cropY ? sps.pic_height_in_luma_samples - cropY : 0;
}

// The part of a picture `width` by `height` that lies in a surface `surfaceWidth` by
// `surfaceHeight` from `place`, a place inside it, into `width` and `height`, in whole chroma
// samples: an odd remainder loses its last column or row. A grid's tiles are all of one size, so
// a tile of its last column or row reaches past a canvas whose size is no whole number of tiles;
// its surface's memory holds whole tiles, but its planes, and the device's copy of them
// (RgfxIOSurface.h), hold the canvas alone (measured on the RTX 5080 host: a 2000x1400 grid of
// 512x512 tiles drawn through Core Animation lands in a canvas of planes 2000x1400 and 1000x700,
// 2048 bytes a row, chroma at byte 3145728 of 4718592, the bytes of 2048x1536).
void clipToSurface(Place place, size_t surfaceWidth, size_t surfaceHeight, uint32_t *width,
                   uint32_t *height)
{
	const auto clip = [](uint32_t at, size_t side, uint32_t *size) {
		if (at + (uint64_t)*size > side)
			*size = (uint32_t)(side - at) & ~1u;
	};
	clip(place.x, surfaceWidth, width);
	clip(place.y, surfaceHeight, height);
}

// Commits `sets` and the access unit at `bytes` on `context`'s session, into `output` at `place`
// (`width` by `height` of it there; RgfxVideoChannel.hpp), the sets sent only where they changed
// since the session's last: the work on the host, or null.
void *commitPicture(Context *context, std::vector<uint8_t> &sets, const uint8_t *bytes,
                    size_t size, IOSurfaceRef output, Place place = {0, 0}, uint32_t width = 0,
                    uint32_t height = 0)
{
	const bool changed = sets != context->sent;
	rgfx::video::Decode d = {};
	d.bytes = bytes;
	d.size = size;
	d.parameterSets = changed ? sets.data() : nullptr;
	d.parameterSetsSize = changed ? (uint32_t)sets.size() : 0;
	d.output = output;
	d.x = place.x;
	d.y = place.y;
	d.width = width;
	d.height = height;
	void *work = rgfx::video::decode(context->session, d);
	if (work && changed)
		context->sent = std::move(sets);
	if (!work) {
		static std::atomic<bool> said;
		notSent(said, "no texture over its output, or no command buffer");
	}
	return work;
}

// Whether the host takes the HEVC picture whose slice segments are the `size` bytes at `au`, of
// `sps` and `pps`, by what its caps say it decodes. It refuses, saying why once a reason: its
// coded size outside the caps' least and largest; a tile column of the PPS narrower than the
// caps' narrowest (RgfxVideoDecodeCaps); or, where its first slice segment's header gives
// NumPicTotalCurr (h265::currentReferences), more current references than the caps'
// max_references. A bound the caps leave 0 refuses nothing, and a header or PPS the driver cannot
// read leaves the verdict to the host. AppleGVA's HEVC decoder hears of a refusal only as a failed
// schedule (RgfxVideoStatus.hpp), so a picture the host would refuse fails its own: its frame and
// every one after it carry kVTVideoDecoderMalfunctionErr, where a schedule that went ahead would
// deliver it and the next one undecoded.
bool hevcTaken(const rgfx::video::h265::Sps &sps, const rgfx::video::h265::Pps &pps,
               const uint8_t *au, size_t size)
{
	const RgfxVideoCodecCaps *caps = rgfx::video::hostCodec(RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE);
	if (!caps)
		return true;
	const uint32_t coded[2] = {sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples};
	for (int i = 0; i < 2; i++)
		if ((caps->min_size[i] && coded[i] < caps->min_size[i]) ||
		    (caps->max_size[i] && coded[i] > caps->max_size[i])) {
			static std::atomic<bool> said;
			notSent(said, "its coded size is outside the sizes the host decodes");
			return false;
		}
	const RgfxVideoDecodeCaps bounds = rgfx::video::decodeCaps(RGFX_VIDEO_CODEC_HEVC);
	uint32_t narrowest;
	if (bounds.min_tile_width && rgfx::video::h265::narrowestTileColumn(sps, pps, &narrowest) &&
	    narrowest != 0 && narrowest < bounds.min_tile_width) {
		static std::atomic<bool> said;
		notSent(said, "a tile column of its PPS is narrower than the host decodes");
		return false;
	}
	uint32_t current;
	if (caps->max_references && rgfx::video::h265::currentReferences(sps, pps, au, size, &current) &&
	    current > caps->max_references) {
		static std::atomic<bool> said;
		notSent(said, "its current reference sets hold more pictures than the host decodes");
		return false;
	}
	return true;
}

// Commits the HEVC picture `c` decodes on its context's session: its work on the host, or null
// where it could not be sent. Its slice segments lie one after another in one bitstream buffer,
// each named by a record of the picture's.
void *decodeHevcPicture(const Command *c)
{
	Context *context = (Context *)c->context;
	const uint8_t *P = c->parameters;
	if (!context || !context->session || !P) {
		static std::atomic<bool> said;
		notSent(said, "its command names no decode session or no picture record");
		return nullptr;
	}
	constexpr size_t kSliceRecord = 0x328;
	const uint32_t slices = u32At(P, 0x28);
	const uint8_t *records = (const uint8_t *)u64At(P, 0x30);
	const uint8_t *bytes;
	size_t bufferBytes;
	if (!slices || !records) {
		static std::atomic<bool> said;
		notSent(said, "its picture record holds no slice count (+0x28) or no slice records (+0x30)");
		return nullptr;
	}
	if (!hevcBitstreamBuffer(context->driver, u32At(records, 0x8), &bytes, &bufferBytes)) {
		static std::atomic<bool> said;
		notSent(said, "its first slice record names (+0x8) no bitstream buffer of the driver's");
		return nullptr;
	}
	uint64_t first = UINT64_MAX, end = 0;
	for (uint32_t i = 0; i < slices; i++) {
		const uint8_t *r = records + i * kSliceRecord;
		const uint64_t offset = u32At(r, 0x10), size = u32At(r, 0x18);
		first = offset < first ? offset : first;
		end = offset + size > end ? offset + size : end;
	}
	if (end > bufferBytes || first >= end) {
		static std::atomic<bool> said;
		notSent(said, "its slice segments are empty or run past their bitstream buffer");
		return nullptr;
	}
	std::vector<uint8_t> sets;
	const auto *sps = (const rgfx::video::h265::Sps *)u64At(P, 0x18);
	const auto *pps = (const rgfx::video::h265::Pps *)u64At(P, 0x20);
	if (!sps || !pps || !rgfx::video::h265::writeSps(*sps, sets) ||
	    !rgfx::video::h265::writePps(*pps, sets)) {
		static std::atomic<bool> said;
		notSent(said, "its parameter sets are refused");
		return nullptr;
	}
	if (!hevcTaken(*sps, *pps, bytes + first, end - first))
		return nullptr;
	NoOutput why;
	IOSurfaceRef output = hevcOutputSurface(context, P, &why);
	if (!output) {
		static_assert(kDescriptionFirst == 0x60 && kDescriptionLast == 0xa0,
		              "the lines below name the search window");
		static std::atomic<bool> none, many, session;
		if (why == NoOutput::kMany)
			notSent(many, "its frame record describes a surface in more than one layout "
			              "from +0x60 to +0xa0");
		else if (why == NoOutput::kNotInSessionLayout)
			notSent(session, "its frame record describes no surface of its own "
			                 "in its session's layout");
		else
			notSent(none, "its frame record describes no surface of its own "
			              "from +0x60 to +0xa0");
		return nullptr;
	}
	// A tile's place lies inside its canvas, of whole chroma samples, and the part of it that
	// lands there is what the SPS's conformance window crops it to, which is what the host's copy
	// lands.
	const Place place = hevcTilePlace(P);
	uint32_t width, height;
	hevcPictureSize(*sps, &width, &height);
	const uint32_t whole[2] = {width, height};
	const size_t across = IOSurfaceGetWidth(output), down = IOSurfaceGetHeight(output);
	if (place.x < across && place.y < down)
		clipToSurface(place, across, down, &width, &height);
	if (place.x % 2 || place.y % 2 || place.x >= across || place.y >= down || width == 0 ||
	    height == 0) {
		static std::atomic<bool> said;
		notSent(said, "its place lies off its output surface");
		CFRelease(output);
		return nullptr;
	}
	// The SPS sent crops the picture to that part: its window grown on the right and the bottom,
	// in chroma samples.
	if (width != whole[0] || height != whole[1]) {
		auto cropped = std::make_unique<rgfx::video::h265::Sps>(*sps);
		if (!sps->conformance_window_flag) {
			cropped->conformance_window_flag = 1;
			cropped->conf_win_left_offset = cropped->conf_win_right_offset = 0;
			cropped->conf_win_top_offset = cropped->conf_win_bottom_offset = 0;
		}
		cropped->conf_win_right_offset += (whole[0] - width) / 2;
		cropped->conf_win_bottom_offset += (whole[1] - height) / 2;
		sets.clear();
		if (!rgfx::video::h265::writeSps(*cropped, sets) ||
		    !rgfx::video::h265::writePps(*pps, sets)) {
			static std::atomic<bool> said;
			notSent(said, "its parameter sets are refused");
			CFRelease(output);
			return nullptr;
		}
	}
	void *work =
	    commitPicture(context, sets, bytes + first, end - first, output, place, width, height);
	CFRelease(output);
	return work;
}

// The SPS of an H.264 picture of `context`'s that AppleGVA decodes into a `width` x `height`
// surface, its parsed form at `record` and its PPS's seq_parameter_set_id `ppsSps`, in `sps`: read
// in the session's layout, or in the one layout it fits where the session has none yet
// (rgfx::video::h264::chooseSpsLayout), which the session then keeps. False, logged, where it fits
// no layout, more than one, or not the session's.
bool h264Sps(Context *context, const uint8_t *record, uint32_t ppsSps, uint32_t width,
             uint32_t height, rgfx::video::h264::Sps *sps)
{
	using namespace rgfx::video::h264;
	const uint8_t kept = context->h264_sps_layout.load();
	if (kept) {
		*sps = spsIn(record, (SpsLayout)kept);
		if (spsFits(*sps, ppsSps, width, height))
			return true;
		static std::atomic<bool> said;
		notSent(said, "its SPS does not describe its picture in its session's layout");
		return false;
	}
	SpsLayout layout;
	NoLayout why;
	if (chooseSpsLayout(record, ppsSps, width, height, &layout, sps, &why)) {
		context->h264_sps_layout.store((uint8_t)layout);
		return true;
	}
	static std::atomic<bool> none, many;
	if (why == NoLayout::kMany)
		notSent(many, "its SPS describes its picture in both layouts of AppleGVA's");
	else
		notSent(none, "its SPS describes its picture in no layout of AppleGVA's");
	return false;
}

// Commits the H.264 picture `c` decodes on its context's session: its work on the host, or null
// where it could not be sent.
void *decodeH264Picture(const Command *c)
{
	Context *context = (Context *)c->context;
	const uint8_t *P = c->parameters;
	if (!context || !context->session || !P || !c->output) {
		static std::atomic<bool> said;
		notSent(said, "its command names no decode session, no picture record or no output");
		return nullptr;
	}
	// The access unit: the slices, which lie one after another in one bitstream buffer.
	const uint8_t *bytes;
	size_t bufferBytes;
	const uint32_t pool = u32At(P, 0x270), index = u32At(P, 0x274), slices = u32At(P, 0x278);
	const uint8_t *entries = (const uint8_t *)u64At(P, 0x290);
	if (!slices || !entries) {
		static std::atomic<bool> said;
		notSent(said, "its picture record holds no slice count (+0x278) or no slice entries (+0x290)");
		return nullptr;
	}
	if (!bitstreamBuffer(context->driver, pool, index, &bytes, &bufferBytes)) {
		static std::atomic<bool> said;
		notSent(said, "its picture record names (+0x270, +0x274) no bitstream buffer of the driver's");
		return nullptr;
	}
	uint64_t first = UINT64_MAX, end = 0;
	for (uint32_t i = 0; i < slices; i++) {
		const uint8_t *e = (const uint8_t *)u64At(entries, i * 8);
		if (!e) {
			static std::atomic<bool> said;
			notSent(said, "one of its slice entries is null");
			return nullptr;
		}
		const uint64_t size = u32At(e, 0), offset = u32At(e, 4);
		first = offset < first ? offset : first;
		end = offset + size > end ? offset + size : end;
	}
	if (end > bufferBytes) {
		static std::atomic<bool> said;
		notSent(said, "its slices run past their bitstream buffer");
		return nullptr;
	}
	// The parameter sets, sent where they differ from the last ones.
	const uint8_t *spsRecord = (const uint8_t *)u64At(P, 0x260);
	const auto *pps = (const rgfx::video::h264::Pps *)u64At(P, 0x268);
	IOSurfaceRef output = outputSurface(c);
	if (!spsRecord || !pps || !output) {
		static std::atomic<bool> said;
		notSent(said, "its picture record holds no SPS (+0x260) or no PPS (+0x268), or its "
		              "command no output surface");
		return nullptr;
	}
	rgfx::video::h264::Sps sps;
	if (!h264Sps(context, spsRecord, pps->seq_parameter_set_id, (uint32_t)IOSurfaceGetWidth(output),
	             (uint32_t)IOSurfaceGetHeight(output), &sps))
		return nullptr;
	std::vector<uint8_t> sets;
	if (!rgfx::video::h264::writeSps(sps, sets) || !rgfx::video::h264::writePps(*pps, sps, sets)) {
		static std::atomic<bool> said;
		notSent(said, "its parameter sets are refused");
		return nullptr;
	}
	return commitPicture(context, sets, bytes + first, end - first, output);
}

// Commits the picture `c` decodes, of its context's codec.
void *decodePicture(const Command *c)
{
	const Context *context = (const Context *)c->context;
	if (!context)
		return nullptr;
	return context->sub_id == kCodecHevc ? decodeHevcPicture(c) : decodeH264Picture(c);
}

uint64_t answer(int entry, uint64_t a0, uint64_t a1, uint64_t a2)
{
	switch (entry) {
	case 0: { // CreateDriver
		if (!a0)
			return kBadArgument;
		Driver *driver = new Driver{};
		std::lock_guard<std::mutex> hold(g_poolLock);
		g_drivers.push_back(driver);
		*(void **)a0 = driver;
		return kOk;
	}
	case 2: { // CreateContexts
		ContextInit *ci = (ContextInit *)a1;
		if (!ci || !ci->info)
			return kBadArgument;
		for (uint32_t i = 0; i < ci->info->count; i++) {
			const FunctionInfo *fi = ci->info->functions[i];
			Context *context = fi ? new Context{(Driver *)a0, fi->function_id, fi->sub_id,
							    fi->width, fi->height, nullptr, {}}
					      : nullptr;
			const uint32_t codec = !context || fi->function_id != kFunctionDecode ? 0
			                       : fi->sub_id == kCodecH264                       ? RGFX_VIDEO_CODEC_H264
			                       : fi->sub_id == kCodecHevc                       ? RGFX_VIDEO_CODEC_HEVC
			                                                                        : 0;
			if (codec != 0) {
				// An H.264 stream of a coded size the host does not decode is refused here, where
				// VideoToolbox hears of it and turns to its software decoder: AppleGVA's H.264
				// decoder hears of no failure later (RgfxVideoStatus.hpp).
				if (codec == RGFX_VIDEO_CODEC_H264 &&
				    !rgfx::video::sizeTaken(rgfx::video::hostCodec(codec, RGFX_VIDEO_DECODE),
				                            fi->width, fi->height)) {
					static std::atomic<bool> said;
					if (!said.exchange(true))
						os_log_error(OS_LOG_DEFAULT,
						             "RGFX: video: an H.264 decode of %ux%u refused at its create: "
						             "the host decodes no picture of its coded size",
						             fi->width, fi->height);
				} else {
					context->session = rgfx::video::openSession(codec, RGFX_VIDEO_DECODE);
				}
				if (!context->session) {
					// The contexts made before it go with their sessions, which leave the
					// live ones.
					for (uint32_t j = 0; j < i; j++) {
						deleteContext((Context *)ci->info->contexts[j]);
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
		for (uint32_t i = 0; i < cd->count; i++)
			deleteContext((Context *)cd->contexts[i]);
		return kOk;
	}
	case 4: { // GetBitstreamBufferInfo
		BitstreamRequirement *r = (BitstreamRequirement *)a1;
		if (!r)
			return kBadArgument;
		const uint32_t alignment = bitstreamAlignment(r->sub_id);
		for (uint32_t &a : r->alignment)
			a = alignment;
		return kOk;
	}
	case 5: { // AllocateBitstreamBufferPool
		BitstreamPool *p = (BitstreamPool *)a1;
		if (!p || !p->addresses || p->count == 0 || p->count > 32 || p->size == 0)
			return kBadArgument;
		std::lock_guard<std::mutex> hold(g_poolLock);
		Driver *driver = liveDriver(a0);
		if (!driver)
			return kBadArgument;
		BitstreamBuffers *slot = nullptr;
		for (BitstreamBuffers &b : driver->pools)
			if (b.count && b.pool == p->pool)
				freePool(*(slot = &b));
		for (BitstreamBuffers &b : driver->pools)
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
		Driver *driver = liveDriver(a0);
		if (!driver)
			return kBadArgument;
		for (BitstreamBuffers &b : driver->pools)
			if (b.count && b.pool == (uint32_t)a1)
				freePool(b);
		return kOk;
	}
	case 7: { // GetPixelBufferInfo
		PixelRequirement *r = (PixelRequirement *)a1;
		if (!r || !r->widths || !r->heights || !r->pitches || !r->alignments)
			return kBadArgument;
		// Two planes, of a byte a sample (NV12) or two (P010).
		uint32_t sampleBytes;
		switch (r->fourcc) {
		case 'NV12':
		case '420v':
		case '420f':
			sampleBytes = 1;
			break;
		case 'x420':
		case 'xf20':
			sampleBytes = 2;
			break;
		default:
			return kUnsupportedFormat;
		}
		if (r->planes != 2)
			return kUnsupportedFormat;
		if (r->width == 0 || r->height == 0 || r->width > kMaxFrameSide || r->height > kMaxFrameSide)
			return kBadArgument;
		// What the host's copy writes, a picture's crop at each plane's origin (a frame's; a tile
		// in a canvas lands at its place, hevcTilePlace): luma of the frame's size, chroma of
		// half, in whole pairs; each plane's first byte and pitch at the alignment its texture and
		// the copy take, one pitch for both planes.
		uint32_t alignment = rgfx::video::outputPlaneAlignment(sampleBytes);
		if (alignment == 0)
			return kUnsupported;
		// Where the host decodes the codec in place (outputLayout), what the decode writes: the
		// coded picture's whole rows and columns, in the host's units, at its planes' alignment,
		// as AMD's driver on a Mac answers (rows of whole macroblocks, at 256 bytes): the planes'
		// rows those, the frame's width as it is, and AppleGVA extends the planes below the
		// frame by the rows past it.
		const uint32_t codec = r->sub_id == kCodecHevc   ? RGFX_VIDEO_CODEC_HEVC
		                       : r->sub_id == kCodecH264 ? RGFX_VIDEO_CODEC_H264
		                                                 : 0;
		const RgfxVideoOutputLayout layout = rgfx::video::outputLayout(codec);
		uint32_t codedWidth = r->width, codedHeight = r->height;
		if (layout.granularity[0] != 0 && layout.granularity[1] != 0 && layout.alignment != 0) {
			const auto roundUp = [](uint32_t v, uint32_t unit) { return (v + unit - 1) / unit * unit; };
			codedWidth = roundUp(r->width, layout.granularity[0]);
			codedHeight = roundUp(r->height, layout.granularity[1]);
			// A multiple of both: the textures' and the host's.
			uint32_t a = alignment, b = layout.alignment;
			while (b != 0) {
				const uint32_t t = a % b;
				a = b;
				b = t;
			}
			alignment = alignment / a * layout.alignment;
		}
		const uint32_t chromaWidth = (r->width + 1) / 2, chromaHeight = (codedHeight + 1) / 2;
		const uint32_t codedChroma = (codedWidth + 1) / 2;
		const uint32_t row = codedWidth * sampleBytes > codedChroma * 2 * sampleBytes
		                         ? codedWidth * sampleBytes
		                         : codedChroma * 2 * sampleBytes;
		const uint32_t pitch = (row + alignment - 1) / alignment * alignment;
		r->widths[0] = r->width;
		r->heights[0] = codedHeight;
		r->widths[1] = chromaWidth;
		r->heights[1] = chromaHeight;
		for (int p = 0; p < 2; p++) {
			r->pitches[p] = pitch;
			r->alignments[p] = alignment;
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
		// The work of the list's HEVC decodes, held for the schedule to wait on its verdict
		// (RgfxVideoStatus.hpp).
		std::vector<void *> verdicts;
		bool sent = true;
		{
			// One list of a context's at a time, so its pictures reach its session in AppleGVA's
			// order: the schedule locks of the contexts the list names, taken in their order in
			// memory. Another context's list goes on meanwhile, as a picture's work here can take
			// milliseconds (the first textures over a new output surface wait for the host).
			std::vector<Context *> contexts;
			for (uint32_t i = 0; i < cl->count; i++) {
				const Command *decode = decodeCommand(cl->commands[i]);
				Context *context = decode ? (Context *)decode->context : nullptr;
				if (context &&
				    std::find(contexts.begin(), contexts.end(), context) == contexts.end())
					contexts.push_back(context);
			}
			std::sort(contexts.begin(), contexts.end(), std::less<Context *>());
			std::vector<std::unique_lock<std::mutex>> holds;
			holds.reserve(contexts.size());
			for (Context *context : contexts)
				holds.emplace_back(context->schedule);
			for (uint32_t i = 0; i < cl->count && sent; i++) {
				const Command *decode = decodeCommand(cl->commands[i]);
				const Context *context = decode ? (const Context *)decode->context : nullptr;
				void *work = decode ? decodePicture(decode) : nullptr;
				{
					std::lock_guard<std::mutex> tokens(g_tokenLock);
					const uint64_t n = g_seqno++;
					Token *t = &g_tokens[n % kTokens];
					rgfx::video::releaseToken(t->work);
					*t = {n, work};
					tl->tokens[i] = t;
				}
				sent = !decode || work;
				if (work && context && rgfx::video::waitsForHost(context->sub_id == kCodecHevc))
					verdicts.push_back(rgfx::video::retainToken(work));
			}
		}
		// Out of the lock, so other sessions' pictures go to the host meanwhile.
		bool hostFailed = false;
		for (void *work : verdicts) {
			hostFailed = rgfx::video::waitToken(work) == -1 || hostFailed;
			rgfx::video::releaseToken(work);
		}
		if (rgfx::video::scheduleFails(sent, hostFailed)) {
			scheduleFailed(sent ? "the host failed its decode" : "the driver could not send it");
			return rgfx::video::kNotSent;
		}
		return kOk;
	}
	case 11: { // QueryCommandStatus
		QueryList *ql = (QueryList *)a1;
		if (!ql || !ql->queries)
			return kBadArgument;
		for (uint32_t i = 0; i < ql->count; i++) {
			Query &q = ql->queries[i];
			Token *t = (Token *)q.token;
			if (!t)
				return kBadArgument;
			std::lock_guard<std::mutex> hold(g_tokenLock);
			const int status = t->work ? rgfx::video::tokenStatus(t->work) : 1;
			if (status != 0) {
				rgfx::video::releaseToken(t->work);
				t->work = nullptr;
			}
			q.status = rgfx::video::queryStatus(status);
			q.field_14 = 0;
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
	case 13: { // GetContextParameter
		ContextParamList *l = (ContextParamList *)a1;
		if (!l || !l->params)
			return kBadArgument;
		for (uint32_t i = 0; i < l->count; i++) {
			ContextParam &param = l->params[i];
			// A parameter the driver does not know is left as it came, success, as AMD's driver
			// answers it.
			if (param.id != kParamTileInfo)
				continue;
			const uint32_t codec = param.sub_id == kCodecH264   ? RGFX_VIDEO_CODEC_H264
			                       : param.sub_id == kCodecHevc ? RGFX_VIDEO_CODEC_HEVC
			                                                    : 0;
			// The tile parameter is refused where the driver cannot write it whole: not a
			// decoder's, no record, or no decoder of the codec on the host (no device of ours in
			// the process, or caps without it).
			TileInfo tile;
			if (param.function_id != kFunctionDecode || !param.data || !codec ||
			    !tileInfo(codec, &tile))
				return kUnsupported;
			*(TileInfo *)param.data = tile;
		}
		return kOk;
	}
	case 1: { // DeleteDriver: its pools go with it
		std::lock_guard<std::mutex> hold(g_poolLock);
		Driver *driver = liveDriver(a0);
		if (!driver)
			return kBadArgument;
		for (BitstreamBuffers &b : driver->pools)
			if (b.count)
				freePool(b);
		g_drivers.erase(std::find(g_drivers.begin(), g_drivers.end(), driver));
		delete driver;
		return kOk;
	}
	default: // Set{Context,Driver}Parameter: answered success
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
	g_table.driver = &g_tableDriver;
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
