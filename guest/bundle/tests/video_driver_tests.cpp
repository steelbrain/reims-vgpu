/*
 * video_driver_tests: the video driver's entry points (src/RgfxVideoDriver.cpp) as AppleGVA calls
 * them, run on the Mac (guest/bundle/Makefile `test`), over a stand-in for the host's sessions
 * (src/RgfxVideoChannel.hpp) that counts what the driver opens, closes, forgets and sends. It
 * checks that a context the driver deletes, by DeleteContexts or by a CreateContexts that fails
 * part way, takes its session with it, closed and out of the live sessions, that a decode the
 * driver cannot send fails its schedule, that an HEIF grid's HEVC tile lands at its place in its
 * canvas, cropped to the canvas, that no answer of the driver AppleGVA may divide by is 0 (an
 * alignment, a plane's size or pitch, a budget, a limit, a decoder's tile parameter), that a tile
 * parameter it cannot write whole is refused rather than answered unwritten, and that each driver
 * AppleGVA makes keeps bitstream pools of its own.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the first failure.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "RgfxVideoChannel.hpp"
#include "RgfxVideoH265.hpp"

extern "C" int AVD_CreateAVDAccelerator(void **out, const void *params, void *identity);

namespace {

int g_checks = 0;

#define CHECK(cond, ...)                                                                  \
	do {                                                                              \
		g_checks++;                                                               \
		if (!(cond)) {                                                            \
			std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
			std::fprintf(stderr, __VA_ARGS__);                                \
			std::fprintf(stderr, "\n");                                       \
			std::abort();                                                     \
		}                                                                         \
	} while (0)

// The records AppleGVA hands the entry points, as the driver lays them out.
struct Format {
	uint32_t fourcc;
	uint32_t field_04;
};
struct FunctionInfo {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint32_t engine;
	uint32_t field_14;
	uint32_t rate;
	uint32_t field_1c;
	uint32_t width;
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
	void **contexts;
};
struct ContextInit {
	ContextInfo *info;
	const void *capabilities;
};
struct ContextDelete {
	uint32_t count;
	uint32_t field_04;
	void *const *contexts;
};
struct Command {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint64_t field_10;
	void *context;
	const uint8_t *parameters;
	const void *field_28;
	const uint8_t *output;
};
static_assert(sizeof(Command) == 0x38, "command layout");
struct CommandList {
	uint32_t count;
	uint32_t field_04;
	const Command *const *commands;
};
struct TokenList {
	uint32_t count;
	uint32_t field_04;
	void **tokens;
};
struct DriverTable {
	void *driver;
	uint64_t (*entry[16])(uint64_t, uint64_t, uint64_t);
};
struct BitstreamRequirement {
	uint32_t function_id;
	uint32_t sub_id;
	uint32_t engine;
	uint32_t field_0c;
	uint32_t size;
	Format format;
	uint32_t alignment[6];
};
static_assert(sizeof(BitstreamRequirement) == 0x34, "bitstream requirement layout");
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
	uint32_t *widths;
	uint32_t *heights;
	uint32_t *pitches;
	uint32_t *alignments;
};
static_assert(offsetof(PixelRequirement, alignments) == 0x40, "pixel requirement layout");
struct DriverParam {
	uint32_t id;
	uint32_t field_04;
	void *data;
};
struct DriverParamList {
	uint32_t count;
	uint32_t field_04;
	DriverParam *params;
};
struct FunctionAvailable {
	uint32_t function_id;
	uint32_t sub_id;
	uint32_t engine;
	uint32_t budget;
};
// The capability block's engines and functions, as far as the checks read them.
struct Engine {
	uint32_t type;
	char name[0x84];
	uint64_t budget; // field_88
	uint32_t field_90;
	uint32_t field_94;
};
static_assert(sizeof(Engine) == 0x98, "engine layout");
struct Function {
	uint32_t function_id;
	uint32_t field_04;
	uint64_t sub_id;
	uint32_t engine;
	uint32_t field_14;
	uint64_t budget; // field_18
	uint64_t field_20;
	uint32_t input_count;
	Format inputs[64];
	uint32_t output_count;
	Format outputs[64];
	const Function *next;
};
static_assert(sizeof(Function) == 0x438, "function layout");
struct CapabilityBlock {
	uint32_t engine_count;
	uint32_t field_04;
	const Engine *const *engines;
	uint32_t function_count;
	uint32_t field_14;
	const Function *const *functions;
};

constexpr uint32_t kFunctionDecode = 1;
constexpr uint64_t kCodecH264 = 0x8, kCodecHevc = 0x800;
constexpr uint64_t kOk = 0, kUnsupported = 1, kNotSent = 5;
enum {
	kCreateContexts = 2,
	kDeleteContexts = 3,
	kCreateDriver = 0,
	kDeleteDriver = 1,
	kGetBitstreamBufferInfo = 4,
	kAllocateBitstreamBufferPool = 5,
	kFreeBitstreamBufferPool = 6,
	kGetPixelBufferInfo = 7,
	kFreePixelBufferPool = 9,
	kSchedule = 10,
	kGetContextParameter = 13,
	kGetDriverParameter = 15,
};
constexpr uint32_t kParamFunctionAvailable = 9, kParamDecodeLimit = 0xd, kParamCapabilities = 0xe;
constexpr uint32_t kParamTileInfo = 0xb;

// The stand-in's sessions: each one opened, whether it was closed, and how often its outputs
// were forgotten; and how many opens in a row succeed before one fails (-1: all).
struct FakeSession {
	bool closed = false;
	int forgotten = 0;
};
std::vector<FakeSession *> g_opened;
int g_opensBeforeFailure = -1;
// Whether the stand-in host has its decoders: false answers no codec's caps, as a process with no
// device of ours, or a host with no hardware video, does.
bool g_hostDecodes = true;
// Each decode the driver sends, as it sent it: its surface's id, its place and its crop, and the
// parameter sets it sent (none where the session already had them).
struct SentDecode {
	uint32_t surface, x, y, width, height;
	std::vector<uint8_t> sets;
};
std::vector<SentDecode> g_sent;
int g_work;

} // namespace

namespace rgfx::video {

struct Session : FakeSession {};

// A host of H.264 decode and HEVC decode of Main and Main 10, of pictures of 144x144 to 8192x8192
// read in place at 256 bytes, so the capability block lists every decoder the driver has.
const RgfxVideoCodecCaps *hostCodec(uint32_t codec, uint32_t direction)
{
	static RgfxVideoCodecCaps h264 = {}, hevc = {};
	h264.codec = RGFX_VIDEO_CODEC_H264;
	h264.direction = RGFX_VIDEO_DECODE;
	h264.profiles = RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH;
	hevc.codec = RGFX_VIDEO_CODEC_HEVC;
	hevc.direction = RGFX_VIDEO_DECODE;
	hevc.profiles = RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10;
	for (RgfxVideoCodecCaps *c : {&h264, &hevc}) {
		c->min_size[0] = c->min_size[1] = 144;
		c->max_size[0] = c->max_size[1] = 8192;
		c->bitstream_alignment = 256;
	}
	h264.formats = RGFX_VIDEO_FORMAT_NV12;
	hevc.formats = RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010;
	if (direction != RGFX_VIDEO_DECODE || !g_hostDecodes)
		return nullptr;
	return codec == RGFX_VIDEO_CODEC_H264 ? &h264 : codec == RGFX_VIDEO_CODEC_HEVC ? &hevc : nullptr;
}

// A device whose linear textures of the planes' formats take 64 bytes.
uint32_t outputPlaneAlignment(uint32_t sampleBytes) { return sampleBytes == 1 || sampleBytes == 2 ? 64 : 0; }

// The host's layout in place for every codec: none, but where a check sets it.
RgfxVideoOutputLayout g_layout = {};
RgfxVideoOutputLayout outputLayout(uint32_t codec)
{
	return codec == RGFX_VIDEO_CODEC_H264 || codec == RGFX_VIDEO_CODEC_HEVC ? g_layout
	                                                                        : RgfxVideoOutputLayout{};
}

Session *openSession(uint32_t, uint32_t)
{
	if (g_opensBeforeFailure == 0)
		return nullptr;
	if (g_opensBeforeFailure > 0)
		g_opensBeforeFailure--;
	Session *s = new Session;
	g_opened.push_back(s);
	return s;
}

void closeSession(Session *session)
{
	if (session) {
		CHECK(!session->closed, "a session closed twice");
		session->closed = true;
	}
}

void forgetOutputs(Session *session)
{
	CHECK(session && !session->closed, "the outputs of a closed session forgotten");
	session->forgotten++;
}

// Each decode is sent, as work that is done at once.
void *decode(Session *, const Decode &d)
{
	const uint8_t *sets = d.parameterSets;
	g_sent.push_back({d.output ? IOSurfaceGetID(d.output) : 0, d.x, d.y, d.width, d.height,
	                  std::vector<uint8_t>(sets, sets + d.parameterSetsSize)});
	return &g_work;
}
int tokenStatus(void *) { return 1; }
void releaseToken(void *) {}

} // namespace rgfx::video

namespace {

DriverTable *g_table;
// The driver CreateDriver made, which the checks call the entries on.
void *g_driver;

uint64_t callOn(void *driver, int entry, const void *record, const void *out = nullptr)
{
	return g_table->entry[entry]((uint64_t)driver, (uint64_t)record, (uint64_t)out);
}

uint64_t call(int entry, const void *record, const void *out = nullptr)
{
	return callOn(g_driver, entry, record, out);
}

// A driver made as AppleGVA makes one.
void *createDriver()
{
	void *driver = nullptr;
	uint8_t create[0x20] = {};
	CHECK(g_table->entry[kCreateDriver]((uint64_t)&driver, (uint64_t)create, 0) == kOk && driver,
	      "CreateDriver");
	return driver;
}

// AllocateBitstreamBufferPool's record.
struct BitstreamPool {
	uint32_t pool;
	uint32_t field_04;
	uint32_t size;
	Format format;
	uint32_t count;
	uint64_t *addresses;
};
static_assert(offsetof(BitstreamPool, addresses) == 0x18, "bitstream pool layout");

// One buffer of `bytes` in `driver`'s pool `pool`: its address, 0 where refused.
uint64_t allocate(void *driver, uint32_t pool, uint32_t bytes)
{
	uint64_t address = 0;
	BitstreamPool p = {pool, 0, bytes, {'rawb', 1}, 1, &address};
	return callOn(driver, kAllocateBitstreamBufferPool, &p) == kOk ? address : 0;
}

// Whether the `bytes` at `address` are mapped and hold what was written there.
bool holds(uint64_t address, uint32_t bytes, uint8_t value)
{
	unsigned char present = 0;
	const size_t page = (size_t)getpagesize();
	for (uint64_t at = address & ~(uint64_t)(page - 1); at < address + bytes; at += page)
		if (mincore((void *)at, page, (char *)&present) != 0)
			return false;
	const uint8_t *b = (const uint8_t *)address;
	for (uint32_t i = 0; i < bytes; i++)
		if (b[i] != value)
			return false;
	return true;
}

// Each driver keeps bitstream pools of its own: a second driver's pool of the same type leaves
// the first's buffers mapped and theirs, a driver's pool allocated again replaces its own, a
// driver deleted takes its pools with it alone, and a handle no CreateDriver made is refused.
void checkPoolsPerDriver()
{
	constexpr uint32_t kBytes = 0x10000;
	void *first = createDriver(), *second = createDriver();
	CHECK(first != second, "two drivers share a handle");
	const uint64_t a = allocate(first, 0, kBytes);
	CHECK(a != 0, "the first driver's pool refused");
	std::memset((void *)a, 0x11, kBytes);
	const uint64_t b = allocate(second, 0, kBytes);
	CHECK(b != 0 && b != a, "the second driver's pool took the first's buffer");
	std::memset((void *)b, 0x22, kBytes);
	CHECK(holds(a, kBytes, 0x11), "the second driver's pool freed or wrote the first's buffer");
	const uint64_t again = allocate(second, 0, kBytes);
	CHECK(again != 0, "a pool allocated again refused");
	std::memset((void *)again, 0x33, kBytes);
	CHECK(holds(a, kBytes, 0x11), "a pool allocated again freed another driver's buffer");
	CHECK(callOn(second, kDeleteDriver, nullptr) == kOk, "DeleteDriver");
	CHECK(holds(a, kBytes, 0x11), "a deleted driver freed another driver's buffer");
	CHECK(allocate(second, 0, kBytes) == 0, "a pool allocated on a deleted driver");
	int notADriver = 0;
	CHECK(allocate(&notADriver, 0, kBytes) == 0, "a pool allocated on no driver");
	CHECK(callOn(first, kFreeBitstreamBufferPool, (const void *)0) == kOk, "FreeBitstreamBufferPool");
	CHECK(callOn(first, kDeleteDriver, nullptr) == kOk, "DeleteDriver");
}

FunctionInfo decodeH264()
{
	FunctionInfo f = {};
	f.function_id = kFunctionDecode;
	f.sub_id = kCodecH264;
	f.width = 1280;
	f.height = 720;
	return f;
}

// Makes the contexts of `n` decodes of `codec` (H.264's unless named): CreateContexts' answer,
// the contexts in `contexts`.
uint64_t makeContexts(uint32_t n, std::vector<void *> &contexts, uint64_t codec = kCodecH264)
{
	static std::vector<FunctionInfo> infos;
	static std::vector<const FunctionInfo *> list;
	FunctionInfo f = decodeH264();
	f.sub_id = codec;
	infos.assign(n, f);
	list.clear();
	for (const FunctionInfo &f : infos)
		list.push_back(&f);
	contexts.assign(n, nullptr);
	ContextInfo info = {n, 0, list.data(), contexts.data()};
	ContextInit init = {&info, nullptr};
	return call(kCreateContexts, &init);
}

// A CreateContexts whose second session the host refuses: the first context's session is closed
// and leaves the live sessions, so a later FreePixelBufferPool forgets nothing of it.
void checkFailedCreate()
{
	g_opened.clear();
	g_opensBeforeFailure = 1;
	std::vector<void *> contexts;
	CHECK(makeContexts(2, contexts) == kUnsupported, "a create whose second session is refused");
	g_opensBeforeFailure = -1;
	CHECK(g_opened.size() == 1, "%zu sessions opened", g_opened.size());
	CHECK(g_opened[0]->closed, "the first context's session left open");
	CHECK(contexts[0] == nullptr && contexts[1] == nullptr, "a context handed back");
	CHECK(call(kFreePixelBufferPool, nullptr) == kOk, "FreePixelBufferPool");
	CHECK(g_opened[0]->forgotten == 0, "the closed session's outputs forgotten");
}

// DeleteContexts closes each context's session, which a later FreePixelBufferPool no longer
// reaches; the contexts still open are reached.
void checkDelete()
{
	g_opened.clear();
	std::vector<void *> contexts;
	CHECK(makeContexts(2, contexts) == kOk, "a create of two decodes");
	CHECK(g_opened.size() == 2, "%zu sessions opened", g_opened.size());
	ContextDelete del = {1, 0, &contexts[0]};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(g_opened[0]->closed && !g_opened[1]->closed, "the deleted context's session");
	CHECK(call(kFreePixelBufferPool, nullptr) == kOk, "FreePixelBufferPool");
	CHECK(g_opened[0]->forgotten == 0 && g_opened[1]->forgotten == 1, "the outputs forgotten");
	ContextDelete rest = {1, 0, &contexts[1]};
	CHECK(call(kDeleteContexts, &rest) == kOk, "DeleteContexts");
	CHECK(g_opened[1]->closed, "the second context's session left open");
}

// A decode the driver cannot send (here: no picture parameters) fails its schedule, the one
// failure AppleGVA reports (RgfxVideoStatus.hpp), and the token it hands back names no work.
void checkUnsentDecode()
{
	g_opened.clear();
	std::vector<void *> contexts;
	CHECK(makeContexts(1, contexts) == kOk, "a create of one decode");
	Command c = {};
	c.function_id = kFunctionDecode;
	c.sub_id = kCodecH264;
	c.context = contexts[0];
	const Command *list[1] = {&c};
	void *tokens[1] = {};
	CommandList cl = {1, 0, list};
	TokenList tl = {1, 0, tokens};
	CHECK(call(kSchedule, &cl, &tl) == kNotSent, "a decode not sent scheduled");
	CHECK(tokens[0] != nullptr, "no token for the command");
	ContextDelete del = {1, 0, &contexts[0]};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
}

// An HEVC picture's parameters as AppleGVA lays them out, as far as the driver reads them: its
// output frame's surface id (P+0x10, the record's +0x08, the frame's +0x88), its SPS and PPS
// (P+0x18, P+0x20), its one slice segment in buffer 0 of its pool (P+0x28 and P+0x30, the
// record's +0x08, +0x10 and +0x18) and its tile's place (P+0x58, P+0x5c).
struct HevcPicture {
	uint8_t frame[0x90] = {};
	const uint8_t *record[2] = {nullptr, frame};
	uint8_t slice[0x328] = {};
	uint8_t parameters[0x60] = {};

	HevcPicture(IOSurfaceRef output, const rgfx::video::h265::Sps &sps,
	            const rgfx::video::h265::Pps &pps, uint32_t x, uint32_t y, uint32_t bytes)
	{
		const uint32_t id = IOSurfaceGetID(output), slices = 1, index = 0, offset = 0;
		const void *at[3] = {record, &sps, &pps};
		const uint8_t *segments = slice;
		std::memcpy(frame + 0x88, &id, 4);
		std::memcpy(slice + 0x08, &index, 4);
		std::memcpy(slice + 0x10, &offset, 4);
		std::memcpy(slice + 0x18, &bytes, 4);
		std::memcpy(parameters + 0x10, at, sizeof at);
		std::memcpy(parameters + 0x28, &slices, 4);
		std::memcpy(parameters + 0x30, &segments, 8);
		std::memcpy(parameters + 0x58, &x, 4);
		std::memcpy(parameters + 0x5c, &y, 4);
	}
};

// Schedules the HEVC picture `picture` on `context`: Schedule's answer.
uint64_t scheduleHevc(void *context, const HevcPicture &picture)
{
	Command c = {};
	c.function_id = kFunctionDecode;
	c.sub_id = kCodecHevc;
	c.context = context;
	c.parameters = picture.parameters;
	const Command *list[1] = {&c};
	void *tokens[1] = {};
	CommandList cl = {1, 0, list};
	TokenList tl = {1, 0, tokens};
	return call(kSchedule, &cl, &tl);
}

// The SPS and PPS `sps` and `pps` as the driver sends them.
std::vector<uint8_t> setsOf(const rgfx::video::h265::Sps &sps, const rgfx::video::h265::Pps &pps)
{
	std::vector<uint8_t> sets;
	CHECK(rgfx::video::h265::writeSps(sps, sets) && rgfx::video::h265::writePps(pps, sets),
	      "the test's parameter sets refused");
	return sets;
}

// Each HEVC tile of an HEIF grid lands at the place AppleGVA names (hevcTilePlace) in its canvas,
// cropped to the canvas: a tile of the last column or row of a canvas of no whole number of tiles
// is sent with an SPS whose conformance window crops it there, a canvas's odd remainder losing
// its last column, an SPS's own window kept; the next whole tile has its own SPS sent again. A
// frame of its own lands at (0, 0) whole. A place of odd samples, or off the surface, is not sent.
void checkTileLanding()
{
	g_opened.clear();
	g_sent.clear();
	std::vector<void *> contexts;
	CHECK(makeContexts(1, contexts, kCodecHevc) == kOk, "a create of one HEVC decode");
	constexpr uint32_t kBytes = 0x1000;
	const uint64_t bitstream = allocate(g_driver, 0, 0x10000);
	CHECK(bitstream != 0, "the bitstream pool refused");
	std::memset((void *)bitstream, 0x5a, kBytes);
	// A canvas of 2000x1400 for tiles of 512x512, as the RTX 5080 host's measured one; and one of
	// an odd width.
	auto canvas = [](uint32_t width, uint32_t height) {
		CFMutableDictionaryRef props = CFDictionaryCreateMutable(
		    nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
		const int32_t values[3] = {(int32_t)width, (int32_t)height, 1};
		const CFStringRef keys[3] = {kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement};
		for (int i = 0; i < 3; i++) {
			CFNumberRef n = CFNumberCreate(nullptr, kCFNumberSInt32Type, &values[i]);
			CFDictionarySetValue(props, keys[i], n);
			CFRelease(n);
		}
		IOSurfaceRef s = IOSurfaceCreate(props);
		CFRelease(props);
		CHECK(s != nullptr, "a %ux%u surface", width, height);
		return s;
	};
	IOSurfaceRef surface = canvas(2000, 1400), odd = canvas(2001, 1400);
	static rgfx::video::h265::Sps sps, cropped, windowed, windowedCropped;
	static rgfx::video::h265::Pps pps;
	sps.chroma_format_idc = 1;
	sps.pic_width_in_luma_samples = sps.pic_height_in_luma_samples = 512;
	const std::vector<uint8_t> whole = setsOf(sps, pps);
	struct Want {
		IOSurfaceRef surface;
		const rgfx::video::h265::Sps *sps;
		uint32_t x, y, width, height;
		// The SPS the decode sends, the tile's own (sps) or one cropped, or null for none.
		const rgfx::video::h265::Sps *sent;
	};
	cropped = sps;
	cropped.conformance_window_flag = 1;
	cropped.conf_win_right_offset = (512 - 464) / 2;
	cropped.conf_win_bottom_offset = (512 - 376) / 2;
	windowed = sps;
	windowed.conformance_window_flag = 1;
	windowed.conf_win_left_offset = windowed.conf_win_right_offset = 2;
	windowedCropped = windowed;
	windowedCropped.conf_win_right_offset = 2 + (504 - 464) / 2;
	static rgfx::video::h265::Sps lastColumn, lastRow;
	lastColumn = lastRow = sps;
	lastColumn.conformance_window_flag = lastRow.conformance_window_flag = 1;
	lastColumn.conf_win_right_offset = (512 - 464) / 2;
	lastRow.conf_win_bottom_offset = (512 - 376) / 2;
	for (const Want &w : {
	         Want{surface, &sps, 512, 512, 512, 512, &sps},
	         Want{surface, &sps, 1024, 512, 512, 512, nullptr},
	         // The canvas's last tile: 464 of its columns and 376 of its rows lie in the canvas.
	         Want{surface, &sps, 1536, 1024, 464, 376, &cropped},
	         Want{surface, &sps, 512, 1024, 512, 376, &lastRow},
	         Want{surface, &sps, 0, 0, 512, 512, &sps},
	         // 465 columns of 2001 from 1536: whole chroma pairs, 464.
	         Want{odd, &sps, 1536, 0, 464, 512, &lastColumn},
	         // A tile 504 wide by its own window (4 columns off each side): 464 of it, its window
	         // grown on the right.
	         Want{surface, &windowed, 1536, 0, 464, 512, &windowedCropped},
	         Want{surface, &windowed, 1024, 0, 504, 512, &windowed},
	     }) {
		g_sent.clear();
		const HevcPicture picture(w.surface, *w.sps, pps, w.x, w.y, kBytes);
		CHECK(scheduleHevc(contexts[0], picture) == kOk, "a tile at (%u, %u) not sent", w.x, w.y);
		CHECK(g_sent.size() == 1, "%zu decodes for one tile", g_sent.size());
		const SentDecode &d = g_sent[0];
		CHECK(d.surface == IOSurfaceGetID(w.surface) && d.x == w.x && d.y == w.y &&
		          d.width == w.width && d.height == w.height,
		      "a tile at (%u, %u) sent at (%u, %u), %ux%u, not %ux%u", w.x, w.y, d.x, d.y, d.width,
		      d.height, w.width, w.height);
		CHECK(w.sent ? d.sets == setsOf(*w.sent, pps) : d.sets.empty(),
		      "the tile at (%u, %u): sets of %zu bytes", w.x, w.y, d.sets.size());
	}
	CHECK(setsOf(sps, pps) == whole, "the picture's own SPS written");
	struct Off {
		uint32_t x, y;
	};
	for (const Off &o : {Off{513, 0}, Off{0, 511}, Off{2000, 0}, Off{0, 1400}, Off{4096, 4096}}) {
		g_sent.clear();
		const HevcPicture picture(surface, sps, pps, o.x, o.y, kBytes);
		CHECK(scheduleHevc(contexts[0], picture) == kNotSent, "a tile at (%u, %u) sent", o.x, o.y);
		CHECK(g_sent.empty(), "a tile at (%u, %u) reached the host", o.x, o.y);
	}
	ContextDelete del = {1, 0, &contexts[0]};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(surface);
	CFRelease(odd);
}

// The answers are the host's: each bitstream alignment its decoders' (256), and a frame's planes
// as its copy lands a picture, luma of the frame's size and chroma of half in whole pairs, at the
// device's linear texture alignment (64), one pitch for both, unpadded.
void checkLayoutFromHost()
{
	BitstreamRequirement br = {};
	br.function_id = kFunctionDecode;
	br.sub_id = (uint32_t)kCodecHevc;
	br.size = 0x100000;
	br.format = {'rawb', 1};
	CHECK(call(kGetBitstreamBufferInfo, &br) == kOk, "GetBitstreamBufferInfo");
	for (uint32_t a : br.alignment)
		CHECK(a == 256, "a bitstream alignment of %u, not the host's 256", a);
	struct Want {
		uint32_t fourcc, width, height, chromaWidth, chromaHeight, pitch;
	};
	for (const Want &w : {Want{'NV12', 1920, 1080, 960, 540, 1920}, Want{'NV12', 1367, 767, 684, 384, 1408},
	                      Want{'x420', 1367, 767, 684, 384, 2752}, Want{'420f', 1, 1, 1, 1, 64}}) {
		uint32_t widths[2] = {}, heights[2] = {}, pitches[2] = {}, alignments[2] = {};
		PixelRequirement pr = {};
		pr.function_id = kFunctionDecode;
		pr.sub_id = (uint32_t)kCodecHevc;
		pr.width = w.width;
		pr.height = w.height;
		pr.fourcc = w.fourcc;
		pr.planes = 2;
		pr.widths = widths;
		pr.heights = heights;
		pr.pitches = pitches;
		pr.alignments = alignments;
		CHECK(call(kGetPixelBufferInfo, &pr) == kOk, "GetPixelBufferInfo of %.4s", (const char *)&w.fourcc);
		CHECK(widths[0] == w.width && heights[0] == w.height && widths[1] == w.chromaWidth &&
		          heights[1] == w.chromaHeight,
		      "%.4s %ux%u: planes %ux%u and %ux%u", (const char *)&w.fourcc, w.width, w.height, widths[0],
		      heights[0], widths[1], heights[1]);
		for (int p = 0; p < 2; p++)
			CHECK(pitches[p] == w.pitch && alignments[p] == 64, "%.4s %ux%u plane %d: pitch %u alignment %u",
			      (const char *)&w.fourcc, w.width, w.height, p, pitches[p], alignments[p]);
	}
}

// Where the host decodes in place (a layout of 16 by 16 at 256 bytes, RADV's), a surface is
// asked for as AMD's driver on a Mac answers (measured on a Radeon Pro 5300M, macOS 26.7.1):
// the planes' rows the frame's height rounded up to whole macroblocks (chroma half), the widths
// the frame's, one pitch for both planes from the width rounded up likewise, at the larger of
// the host's 256 bytes and the textures' 64, each plane at that alignment. P010 takes two bytes
// a sample. A layout of an H.264 session holds for HEVC alike; a function of another codec, and
// a host with no layout, keeps the frame's own rows.
void checkLayoutInPlace()
{
	rgfx::video::g_layout = {{16, 16}, 256, 0};
	struct Want {
		uint64_t sub;
		uint32_t fourcc, width, height, chromaWidth, rows, pitch;
	};
	for (const Want &w : {Want{kCodecH264, 'NV12', 1920, 1080, 960, 1088, 2048},
	                      Want{kCodecHevc, 'NV12', 1920, 1080, 960, 1088, 2048},
	                      Want{kCodecH264, 'NV12', 1918, 1078, 959, 1088, 2048},
	                      Want{kCodecH264, 'NV12', 1280, 720, 640, 720, 1280},
	                      Want{kCodecH264, 'NV12', 640, 360, 320, 368, 768},
	                      Want{kCodecHevc, 'NV12', 1000, 562, 500, 576, 1024},
	                      Want{kCodecHevc, 'NV12', 3840, 2160, 1920, 2160, 3840},
	                      Want{kCodecHevc, 'NV12', 176, 144, 88, 144, 256},
	                      Want{kCodecHevc, 'x420', 1920, 1080, 960, 1088, 3840},
	                      Want{kCodecHevc, 'xf20', 1367, 767, 684, 768, 2816},
	                      Want{0x10, 'NV12', 1920, 1080, 960, 1080, 1920}}) {
		uint32_t widths[2] = {}, heights[2] = {}, pitches[2] = {}, alignments[2] = {};
		PixelRequirement pr = {};
		pr.function_id = kFunctionDecode;
		pr.sub_id = (uint32_t)w.sub;
		pr.width = w.width;
		pr.height = w.height;
		pr.fourcc = w.fourcc;
		pr.planes = 2;
		pr.widths = widths;
		pr.heights = heights;
		pr.pitches = pitches;
		pr.alignments = alignments;
		CHECK(call(kGetPixelBufferInfo, &pr) == kOk, "GetPixelBufferInfo of %.4s", (const char *)&w.fourcc);
		CHECK(widths[0] == w.width && heights[0] == w.rows && widths[1] == w.chromaWidth &&
		          heights[1] == w.rows / 2,
		      "%.4s %ux%u in place: planes %ux%u and %ux%u", (const char *)&w.fourcc, w.width, w.height,
		      widths[0], heights[0], widths[1], heights[1]);
		const uint32_t alignment = w.sub == 0x10 ? 64 : 256;
		for (int p = 0; p < 2; p++)
			CHECK(pitches[p] == w.pitch && alignments[p] == alignment,
			      "%.4s %ux%u in place, plane %d: pitch %u alignment %u", (const char *)&w.fourcc, w.width,
			      w.height, p, pitches[p], alignments[p]);
	}
	rgfx::video::g_layout = {};
}

// Get/SetContextParameter's records.
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

// Each decoder's tile parameter (0xb) is answered whole, from the host decoder's caps and the
// device's plane alignment, with no alignment or size 0 (H.264's canvas row alone is 0); a
// parameter the driver does not know is left as it came, as AMD's driver leaves it.
void checkTileInfo()
{
	for (uint64_t codec : {kCodecH264, kCodecHevc}) {
		uint32_t tile[8];
		for (uint32_t &t : tile)
			t = 0xabcd;
		ContextParam param = {kFunctionDecode, 0, codec, 0, 0, nullptr, kParamTileInfo, 0, tile};
		ContextParamList list = {1, 0, &param};
		CHECK(call(kGetContextParameter, &list) == kOk, "the tile parameter of 0x%llx refused",
		      (unsigned long long)codec);
		// The stand-in host's: tiles of 144x144 to 8192x8192, placed across at 64 samples, whose
		// bytes keep the device's 64 in NV12 and P010, and down in whole chroma pairs; for HEVC in
		// a canvas whose row is at most its widest, 8192 samples of 2 bytes for P010, and for
		// H.264 none, so no tile decoder.
		const uint32_t want[7] = {64, 2, codec == kCodecHevc ? 16384u : 0u, 144, 144, 8192, 8192};
		for (int i = 0; i < 7; i++)
			CHECK(tile[i] == want[i], "codec 0x%llx: tile field %d is %u, not %u",
			      (unsigned long long)codec, i, tile[i], want[i]);
		for (int i = 0; i < 7; i++)
			CHECK(tile[i] != 0 || i == 2, "codec 0x%llx: tile field %d is 0",
			      (unsigned long long)codec, i);
		CHECK(tile[3] <= tile[5] && tile[4] <= tile[6], "a least tile past the largest");
		CHECK(tile[7] == 0xabcd, "the tile parameter wrote past its 28 bytes");
	}
	uint32_t other[8];
	for (uint32_t &t : other)
		t = 0xabcd;
	ContextParam param = {kFunctionDecode, 0, kCodecHevc, 0, 0, nullptr, 5, 0, other};
	ContextParamList list = {1, 0, &param};
	CHECK(call(kGetContextParameter, &list) == kOk, "an unknown context parameter refused");
	for (uint32_t t : other)
		CHECK(t == 0xabcd, "an unknown context parameter written");
}

// The tile parameter the driver cannot write whole is refused, its record as it came, never
// answered success unwritten (which CMPhoto divided by): with no decoder of the codec on the host,
// of a codec the driver has no decoder of, of a function other than decode, and with no record.
void checkTileInfoRefused()
{
	uint32_t tile[8];
	auto refused = [&](uint32_t function, uint64_t codec, void *data, const char *what) {
		for (uint32_t &t : tile)
			t = 0xabcd;
		ContextParam param = {function, 0, codec, 0, 0, nullptr, kParamTileInfo, 0, data};
		ContextParamList list = {1, 0, &param};
		CHECK(call(kGetContextParameter, &list) == kUnsupported,
		      "the tile parameter %s answered", what);
		for (uint32_t t : tile)
			CHECK(t == 0xabcd, "the tile parameter %s written", what);
	};
	g_hostDecodes = false;
	refused(kFunctionDecode, kCodecH264, tile, "of H.264 with no host decoder");
	refused(kFunctionDecode, kCodecHevc, tile, "of HEVC with no host decoder");
	g_hostDecodes = true;
	refused(kFunctionDecode, 0x7777, tile, "of an unknown codec");
	refused(kFunctionDecode + 1, kCodecHevc, tile, "of another function");
	refused(kFunctionDecode, kCodecHevc, nullptr, "with no record");
}

// The capability block, by GetDriverParameter.
const CapabilityBlock *capabilities()
{
	DriverParam param = {kParamCapabilities, 0, nullptr};
	DriverParamList list = {1, 0, &param};
	CHECK(call(kGetDriverParameter, &list) == kOk, "the capability block asked");
	CHECK(param.data != nullptr, "no capability block");
	return (const CapabilityBlock *)param.data;
}

// No answer AppleGVA may divide by is 0: each engine's and function's budget, the decoders'
// limit, each listed function's availability, all six alignments of each function's bitstream
// buffers, and every plane's size, pitch and alignment of each format a function puts out, at
// sizes even, odd and least. A layout the driver does not answer whole (a format it does not
// put out, a plane count other than two, an empty frame) is refused and leaves the record as it
// was.
void checkNoZeroAnswers()
{
	const CapabilityBlock *b = capabilities();
	CHECK(b->engine_count > 0 && b->function_count > 0, "an empty capability block");
	for (uint32_t e = 0; e < b->engine_count; e++)
		CHECK(b->engines[e]->budget != 0, "engine %u's budget is 0", b->engines[e]->type);
	uint32_t limit = 0;
	DriverParam limitParam = {kParamDecodeLimit, 0, &limit};
	DriverParamList limitList = {1, 0, &limitParam};
	CHECK(call(kGetDriverParameter, &limitList) == kOk && limit != 0, "the decoders' limit is 0");
	for (uint32_t f = 0; f < b->function_count; f++) {
		const Function &fn = *b->functions[f];
		CHECK(fn.budget != 0, "function 0x%x's budget is 0", fn.function_id);
		FunctionAvailable fa = {fn.function_id, (uint32_t)fn.sub_id, fn.engine, 0};
		DriverParam available = {kParamFunctionAvailable, 0, &fa};
		DriverParamList availableList = {1, 0, &available};
		CHECK(call(kGetDriverParameter, &availableList) == kOk && fa.budget != 0,
		      "function 0x%x sub 0x%llx answered unavailable", fn.function_id,
		      (unsigned long long)fn.sub_id);
		BitstreamRequirement br = {};
		br.function_id = fn.function_id;
		br.sub_id = (uint32_t)fn.sub_id;
		br.engine = fn.engine;
		br.size = 0x100000;
		br.format = {'rawb', 1};
		CHECK(call(kGetBitstreamBufferInfo, &br) == kOk, "GetBitstreamBufferInfo");
		for (uint32_t a : br.alignment)
			CHECK(a != 0 && (a & (a - 1)) == 0, "function 0x%x: a bitstream alignment of %u",
			      fn.function_id, a);
		for (uint32_t o = 0; o < fn.output_count; o++) {
			const uint32_t fourcc = fn.outputs[o].fourcc;
			if (fourcc == 'rawb')
				continue;
			static const uint32_t kSizes[][2] = {{1920, 1080}, {1367, 767}, {1, 1}};
			for (const uint32_t *size : kSizes) {
				uint32_t widths[2] = {}, heights[2] = {}, pitches[2] = {}, alignments[2] = {};
				PixelRequirement pr = {};
				pr.function_id = fn.function_id;
				pr.sub_id = (uint32_t)fn.sub_id;
				pr.width = size[0];
				pr.height = size[1];
				pr.fourcc = fourcc;
				pr.planes = 2;
				pr.widths = widths;
				pr.heights = heights;
				pr.pitches = pitches;
				pr.alignments = alignments;
				CHECK(call(kGetPixelBufferInfo, &pr) == kOk, "GetPixelBufferInfo of %.4s %ux%u",
				      (const char *)&fourcc, size[0], size[1]);
				for (int p = 0; p < 2; p++)
					CHECK(widths[p] && heights[p] && pitches[p] && alignments[p] &&
					          pitches[p] % alignments[p] == 0,
					      "%.4s %ux%u plane %d: %u x %u, pitch %u, alignment %u", (const char *)&fourcc,
					      size[0], size[1], p, widths[p], heights[p], pitches[p], alignments[p]);
			}
		}
	}
	struct Refused {
		uint32_t fourcc, planes, width, height;
	};
	for (const Refused &r : {Refused{'BGRA', 1, 512, 512}, Refused{'L008', 1, 512, 512},
	                         Refused{'NV12', 3, 512, 512}, Refused{'NV12', 2, 0, 512},
	                         Refused{'x420', 2, 512, 0}}) {
		uint32_t widths[4], heights[4], pitches[4], alignments[4];
		for (uint32_t *a : {widths, heights, pitches, alignments})
			for (int k = 0; k < 4; k++)
				a[k] = 0xabcd;
		PixelRequirement pr = {};
		pr.function_id = 1;
		pr.width = r.width;
		pr.height = r.height;
		pr.fourcc = r.fourcc;
		pr.planes = r.planes;
		pr.widths = widths;
		pr.heights = heights;
		pr.pitches = pitches;
		pr.alignments = alignments;
		CHECK(call(kGetPixelBufferInfo, &pr) != kOk, "%.4s of %u planes, %ux%u answered",
		      (const char *)&r.fourcc, r.planes, r.width, r.height);
		for (uint32_t *a : {widths, heights, pitches, alignments})
			for (int k = 0; k < 4; k++)
				CHECK(a[k] == 0xabcd, "a refused layout's record written");
	}
}

} // namespace

int main()
{
	void *table = nullptr;
	uint8_t params[64] = {}, identity[64] = {};
	CHECK(AVD_CreateAVDAccelerator(&table, params, identity) == 0 && table, "the driver's table");
	g_table = (DriverTable *)table;
	g_driver = createDriver();
	checkPoolsPerDriver();
	checkFailedCreate();
	checkDelete();
	checkUnsentDecode();
	checkTileLanding();
	checkNoZeroAnswers();
	checkTileInfo();
	checkTileInfoRefused();
	checkLayoutFromHost();
	checkLayoutInPlace();
	std::printf("video_driver_tests: %d checks passed\n", g_checks);
	return 0;
}
