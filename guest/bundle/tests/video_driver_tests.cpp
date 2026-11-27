/*
 * video_driver_tests: the video driver's entry points (src/RgfxVideoDriver.cpp) as AppleGVA calls
 * them, run on the Mac (guest/bundle/Makefile `test`), over a stand-in for the host's sessions
 * (src/RgfxVideoChannel.hpp) that counts what the driver opens, closes, forgets and sends. It
 * checks that a context the driver deletes, by DeleteContexts or by a CreateContexts that fails
 * part way, takes its session with it, closed and out of the live sessions, that a decode the
 * driver cannot send fails its schedule, that an HEIF grid's HEVC tile lands at its place in its
 * canvas, cropped to the canvas, that no answer of the driver AppleGVA may divide by is 0 (an
 * alignment, a plane's size or pitch, a budget, a limit, a decoder's tile parameter), that a tile
 * parameter it cannot write whole is refused rather than answered unwritten, that each driver
 * AppleGVA makes keeps bitstream pools of its own, and that a context's schedule waits only for
 * its own context's, not for another's held in its work. Of the encoder: that H.264 encode is
 * listed and its contexts made only where the host encodes it, that its pixel pools are given and
 * freed by kind, and that a frame is sent with the settings AppleGVA's records give and its access
 * unit published in AppleGVA's ring as it is queried done, a failed one failing the next schedule.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the first failure.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "RgfxVideoBits.hpp"
#include "RgfxVideoChannel.hpp"
#include "RgfxVideoH264.hpp"
#include "RgfxVideoH265.hpp"

extern "C" int AVD_CreateAVDAccelerator(void **out, const void *params, void *identity);
extern "C" char **environ;

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
// The stand-in host's HEVC decode's most current references, and its bounds beyond its caps: none,
// but where a check sets them.
uint32_t g_hevcMaxReferences = 0;
RgfxVideoDecodeCaps g_decodeCaps = {};
// Each decode the driver sends, as it sent it: its surface's id, its place and its crop, and the
// parameter sets it sent (none where the session already had them).
struct SentDecode {
	uint32_t surface, x, y, width, height;
	std::vector<uint8_t> sets;
};
std::vector<SentDecode> g_sent;
int g_work;
// The host's verdict on each decode a schedule waits for, and how many verdicts schedules waited
// for since a check cleared the count.
int g_verdict = 1;
int g_waits;
// The stand-in's decodes run on the schedules' threads, so what they record is under g_sentLock;
// a decode on the session g_holdOn waits there until g_holdOn is cleared, as the driver's work for
// a picture waits for the host (its output's first textures), g_held saying it is waiting.
std::mutex g_sentLock;
std::condition_variable g_sentChanged;
FakeSession *g_holdOn = nullptr;
bool g_held = false;
std::vector<FakeSession *> g_sentOn;
// Whether the stand-in host has its encoders, H.264's and HEVC's, as the hosts' arms have (the
// run's host has; a check may take them away), and the rate controls they take (none: an encoder
// the host lists without one, which the driver reads as none).
bool g_hostEncodes = true;
uint32_t g_encodeRates = 1u << RGFX_VIDEO_RATE_CQP | 1u << RGFX_VIDEO_RATE_VBR;
// The profiles the stand-in host's HEVC encoder takes, each from its format (Main from NV12, Main
// 10 from P010): both, as the hosts' arms have them, where a check does not change them.
uint32_t g_hevcEncodeProfiles = RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10;
// The stand-in host's encoders' flags (RgfxVideoEncodeCaps::flags): frames coded in place, as the
// hosts' arms code them, where a check does not change them.
uint32_t g_encodeFlags = RGFX_VIDEO_ENCODE_CAP_IN_PLACE;
// Each encode the driver sends, as it sent it.
struct SentEncode {
	FakeSession *session;
	uint32_t surface;
	RgfxVideoEncodeSettings settings;
	uint64_t frameIndex;
	bool forceKeyframe;
	uint64_t capacity;
	bool inPlace;
};
std::vector<SentEncode> g_encodes;
// Each encode's work, done at once: what the host answers of it, the status and access unit the
// stand-in answers the next encodes.
struct EncodeWork {
	RgfxVideoEncodeResult result;
	std::vector<uint8_t> bytes;
};
std::deque<EncodeWork> g_encodeWork;
uint32_t g_encodeStatus = RGFX_VIDEO_ENCODE_OK;
std::vector<uint8_t> g_accessUnit;

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
	hevc.max_references = g_hevcMaxReferences;
	// And, where a check has them, H.264 encode of every profile and HEVC encode of the profiles
	// g_hevcEncodeProfiles names, of 144x144 to 4096x4096.
	static RgfxVideoCodecCaps h264Encode = {}, hevcEncode = {};
	h264Encode.codec = RGFX_VIDEO_CODEC_H264;
	h264Encode.profiles = RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH;
	h264Encode.max_level = 52;
	hevcEncode.codec = RGFX_VIDEO_CODEC_HEVC;
	hevcEncode.profiles = g_hevcEncodeProfiles;
	hevcEncode.max_level = 186;
	for (RgfxVideoCodecCaps *c : {&h264Encode, &hevcEncode}) {
		c->direction = RGFX_VIDEO_ENCODE;
		c->formats = RGFX_VIDEO_FORMAT_NV12;
		c->min_size[0] = c->min_size[1] = 144;
		c->max_size[0] = c->max_size[1] = 4096;
	}
	hevcEncode.formats = (g_hevcEncodeProfiles & RGFX_VIDEO_HEVC_MAIN ? RGFX_VIDEO_FORMAT_NV12 : 0) |
	                     (g_hevcEncodeProfiles & RGFX_VIDEO_HEVC_MAIN_10 ? RGFX_VIDEO_FORMAT_P010 : 0);
	if (direction == RGFX_VIDEO_ENCODE)
		return !g_hostEncodes                 ? nullptr
		       : codec == RGFX_VIDEO_CODEC_H264 ? &h264Encode
		       : codec == RGFX_VIDEO_CODEC_HEVC ? &hevcEncode
		                                        : nullptr;
	if (direction != RGFX_VIDEO_DECODE || !g_hostDecodes)
		return nullptr;
	return codec == RGFX_VIDEO_CODEC_H264 ? &h264 : codec == RGFX_VIDEO_CODEC_HEVC ? &hevc : nullptr;
}

RgfxVideoDecodeCaps decodeCaps(uint32_t codec)
{
	return codec == RGFX_VIDEO_CODEC_HEVC && g_hostDecodes ? g_decodeCaps : RgfxVideoDecodeCaps{};
}

RgfxVideoEncodeCaps encodeCaps(uint32_t codec)
{
	RgfxVideoEncodeCaps c = {};
	if (g_hostEncodes && (codec == RGFX_VIDEO_CODEC_H264 || codec == RGFX_VIDEO_CODEC_HEVC)) {
		c.rate_controls = g_encodeRates;
		c.flags = g_encodeFlags;
	}
	return c;
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

// Each decode is sent, as work that is done at once; one on g_holdOn's session waits first.
void *decode(Session *session, const Decode &d)
{
	std::unique_lock<std::mutex> hold(g_sentLock);
	if (session == g_holdOn) {
		g_held = true;
		g_sentChanged.notify_all();
		g_sentChanged.wait(hold, [&] { return g_holdOn != session; });
		g_held = false;
	}
	const uint8_t *sets = d.parameterSets;
	g_sent.push_back({d.output ? IOSurfaceGetID(d.output) : 0, d.x, d.y, d.width, d.height,
	                  std::vector<uint8_t>(sets, sets + d.parameterSetsSize)});
	g_sentOn.push_back(session);
	g_sentChanged.notify_all();
	return &g_work;
}
// Each encode is sent, as work that is done at once with the status and access unit the stand-in
// answers.
void *encode(Session *session, const Encode &e)
{
	g_encodes.push_back({session, e.source ? IOSurfaceGetID(e.source) : 0, e.settings, e.frameIndex,
	                     e.forceKeyframe, e.capacity, e.inPlace});
	EncodeWork &work = g_encodeWork.emplace_back();
	work.result.status = g_encodeStatus;
	work.result.bitstream_size = g_accessUnit.size();
	work.result.frame_index = e.frameIndex;
	work.bytes = g_accessUnit;
	return &work;
}

bool encodeResult(void *token, RgfxVideoEncodeResult *result, const uint8_t **bytes)
{
	for (EncodeWork &w : g_encodeWork)
		if (&w == token) {
			*result = w.result;
			*bytes = w.bytes.data();
			return true;
		}
	return false;
}

int tokenStatus(void *) { return 1; }
int waitToken(void *)
{
	g_waits++;
	return g_verdict;
}
void *retainToken(void *token) { return token; }
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

// Whether the `bytes` at `address` are mapped and hold what was written there. Whether they are
// mapped is msync's answer (ENOMEM for pages not mapped): macOS's mincore answers success for
// pages not mapped, and reading them would fault.
bool holds(uint64_t address, uint32_t bytes, uint8_t value)
{
	const size_t page = (size_t)getpagesize();
	const uint64_t first = address & ~(uint64_t)(page - 1);
	if (msync((void *)first, (size_t)(address + bytes - first), MS_ASYNC) != 0)
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

// Makes the contexts of `n` decodes of `codec` (H.264's unless named), of `width` x `height`
// coded samples (1280x720 unless named): CreateContexts' answer, the contexts in `contexts`.
uint64_t makeContexts(uint32_t n, std::vector<void *> &contexts, uint64_t codec = kCodecH264,
                      uint32_t width = 1280, uint32_t height = 720)
{
	static std::vector<FunctionInfo> infos;
	static std::vector<const FunctionInfo *> list;
	FunctionInfo f = decodeH264();
	f.sub_id = codec;
	f.width = width;
	f.height = height;
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

// A decode of a coded size outside the host's (144 to 8192 samples a side here) is refused at its
// create, H.264's and HEVC's alike, with no session opened, as are the contexts made before it in
// the same create; one of the host's least and largest sizes is made. HEVC's is refused there
// too, where a picture refused at its schedule would end the session's frames in -12911 and leave
// VideoToolbox no software decode to turn to.
void checkDecodeSizeAtCreate()
{
	const struct {
		uint32_t width, height;
		uint64_t want;
	} cases[] = {
	    {8193, 720, kUnsupported}, {1280, 8208, kUnsupported}, {128, 720, kUnsupported},
	    {1280, 128, kUnsupported}, {64, 64, kUnsupported},     {144, 144, kOk},
	    {8192, 8192, kOk},
	};
	const struct {
		uint64_t sub;
		const char *name;
	} codecs[] = {{kCodecH264, "H.264"}, {kCodecHevc, "HEVC"}};
	for (const auto &codec : codecs)
		for (const auto &c : cases) {
			g_opened.clear();
			std::vector<void *> contexts;
			CHECK(makeContexts(1, contexts, codec.sub, c.width, c.height) == c.want,
			      "an %s create of %ux%u", codec.name, c.width, c.height);
			CHECK(g_opened.size() == (c.want == kOk ? 1u : 0u), "%zu %s sessions opened for %ux%u",
			      g_opened.size(), codec.name, c.width, c.height);
			CHECK((contexts[0] != nullptr) == (c.want == kOk), "the %s context of %ux%u", codec.name,
			      c.width, c.height);
			if (contexts[0]) {
				ContextDelete del = {1, 0, &contexts[0]};
				CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
			}
		}
	// A create of two decodes whose second is of a size the host does not decode: the first's
	// session goes with it.
	for (const auto &codec : codecs) {
		static FunctionInfo infos[2];
		infos[0] = infos[1] = decodeH264();
		infos[0].sub_id = infos[1].sub_id = codec.sub;
		infos[1].width = 8193;
		const FunctionInfo *list[2] = {&infos[0], &infos[1]};
		void *contexts[2] = {};
		ContextInfo info = {2, 0, list, contexts};
		ContextInit init = {&info, nullptr};
		g_opened.clear();
		CHECK(call(kCreateContexts, &init) == kUnsupported,
		      "an %s create whose second decode is too wide", codec.name);
		CHECK(g_opened.size() == 1 && g_opened[0]->closed, "the first %s decode's session left open",
		      codec.name);
		CHECK(contexts[0] == nullptr && contexts[1] == nullptr, "an %s context handed back",
		      codec.name);
	}
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

// Where an output frame describes its surface: its fields' start and their width, the id at the
// start, the width, height and pixel format the third, fourth and seventh fields. The releases
// measured: macOS 26's and 15's record (8-byte fields from +0x88), 14's (8 from +0x80), 13's (4
// from +0x80), 12's (4 from +0x78) and 11's (4 from +0x70).
struct Described {
	size_t start, fieldBytes;
	size_t id() const { return start; }
	size_t width() const { return start + 2 * fieldBytes; }
	size_t height() const { return start + 3 * fieldBytes; }
	size_t format() const { return start + 6 * fieldBytes; }
};
constexpr Described kMacOS26 = {0x88, 8}, kMacOS14 = {0x80, 8}, kMacOS13 = {0x80, 4},
                    kMacOS12 = {0x78, 4}, kMacOS11 = {0x70, 4};

// An HEVC picture's parameters as AppleGVA lays them out, as far as the driver reads them: its
// output frame's description of its surface (P+0x10, the record's +0x08, then as `described`
// places it), its SPS and PPS (P+0x18, P+0x20), its one slice segment in buffer 0 of its pool
// (P+0x28 and P+0x30, the record's +0x08, +0x10 and +0x18) and its tile's place (P+0x58,
// P+0x5c).
struct HevcPicture {
	// As far as the driver's search reads (to +0xd4).
	uint8_t frame[0xd8] = {};
	const uint8_t *record[2] = {nullptr, frame};
	uint8_t slice[0x328] = {};
	uint8_t parameters[0x60] = {};

	HevcPicture(IOSurfaceRef output, const rgfx::video::h265::Sps &sps,
	            const rgfx::video::h265::Pps &pps, uint32_t x, uint32_t y, uint32_t bytes,
	            const Described &described = kMacOS26)
	{
		const uint32_t slices = 1, index = 0, offset = 0;
		const void *at[3] = {record, &sps, &pps};
		const uint8_t *segments = slice;
		describe(output, described);
		std::memcpy(slice + 0x08, &index, 4);
		std::memcpy(slice + 0x10, &offset, 4);
		std::memcpy(slice + 0x18, &bytes, 4);
		std::memcpy(parameters + 0x10, at, sizeof at);
		std::memcpy(parameters + 0x28, &slices, 4);
		std::memcpy(parameters + 0x30, &segments, 8);
		std::memcpy(parameters + 0x58, &x, 4);
		std::memcpy(parameters + 0x5c, &y, 4);
	}

	// Describes `output` in the frame as `described` places it.
	void describe(IOSurfaceRef output, const Described &described)
	{
		const uint32_t id = IOSurfaceGetID(output), width = (uint32_t)IOSurfaceGetWidth(output),
		               height = (uint32_t)IOSurfaceGetHeight(output),
		               format = IOSurfaceGetPixelFormat(output);
		std::memcpy(frame + described.id(), &id, 4);
		std::memcpy(frame + described.width(), &width, 4);
		std::memcpy(frame + described.height(), &height, 4);
		std::memcpy(frame + described.format(), &format, 4);
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

// An H.264 picture's parameters as AppleGVA lays them out, as far as the driver reads them: its
// parsed SPS and PPS (P+0x260, P+0x268), its one slice in buffer 0 of pool 0 (P+0x270, P+0x274;
// the slice count at P+0x278 and the entries at P+0x290, each its size and offset), and its
// command's output (its +0x20, whose +0x10 is the surface).
struct H264Picture {
	std::vector<uint8_t> sps;
	rgfx::video::h264::Pps pps;
	uint32_t entry[2] = {};
	const uint32_t *entries[1] = {entry};
	uint8_t parameters[0x298] = {};
	uint8_t frame[0x18] = {};
	uint8_t output[0x28] = {};

	H264Picture(std::vector<uint8_t> record, const rgfx::video::h264::Pps &p, IOSurfaceRef surface,
	            uint32_t bytes)
	    : sps(std::move(record)), pps(p)
	{
		const uint32_t slices = 1;
		const void *sets[2] = {sps.data(), &pps};
		const void *list = entries, *outputAt = frame;
		entry[0] = bytes;
		std::memcpy(parameters + 0x260, sets, sizeof sets);
		std::memcpy(parameters + 0x278, &slices, 4);
		std::memcpy(parameters + 0x290, &list, 8);
		std::memcpy(frame + 0x10, &surface, 8);
		std::memcpy(output + 0x20, &outputAt, 8);
	}
};

// Schedules the H.264 picture `picture` on `context`: Schedule's answer.
uint64_t scheduleH264(void *context, const H264Picture &picture)
{
	Command c = {};
	c.function_id = kFunctionDecode;
	c.sub_id = kCodecH264;
	c.context = context;
	c.parameters = picture.parameters;
	c.output = picture.output;
	const Command *list[1] = {&c};
	void *tokens[1] = {};
	CommandList cl = {1, 0, list};
	TokenList tl = {1, 0, tokens};
	return call(kSchedule, &cl, &tl);
}

// The record of `s` as macOS 11's AppleGVA lays out its parsed SPS (as video_h264_tests' of the
// same name): four constraint flags, one field of the four header bits after them, then Sps's
// fields from level_idc on, 8 bytes earlier.
std::vector<uint8_t> spsRecordOn11(const rgfx::video::h264::Sps &s)
{
	using rgfx::video::h264::Sps;
	std::vector<uint8_t> r(sizeof(Sps), 0);
	std::memcpy(r.data(), &s, 0x18);
	const uint32_t bits =
	    s.constraint_set_flag[4] << 3 | s.constraint_set_flag[5] << 2 | s.reserved_zero_2bits;
	std::memcpy(r.data() + 0x18, &bits, 4);
	std::memcpy(r.data() + 0x1c, reinterpret_cast<const uint8_t *>(&s) + offsetof(Sps, level_idc),
	            sizeof(Sps) - offsetof(Sps, level_idc));
	return r;
}

// The record of `s` as every later release measured lays it out.
std::vector<uint8_t> spsRecordOn12(const rgfx::video::h264::Sps &s)
{
	std::vector<uint8_t> r(sizeof s);
	std::memcpy(r.data(), &s, sizeof s);
	return r;
}

// The SPS and PPS `sps` and `pps` as the driver sends them.
std::vector<uint8_t> setsOf(const rgfx::video::h265::Sps &sps, const rgfx::video::h265::Pps &pps)
{
	std::vector<uint8_t> sets;
	CHECK(rgfx::video::h265::writeSps(sps, sets) && rgfx::video::h265::writePps(pps, sets),
	      "the test's parameter sets refused");
	return sets;
}

// A surface of `width` by `height` samples of a byte each, of pixel format `format` where it is
// not 0.
IOSurfaceRef canvas(uint32_t width, uint32_t height, OSType format = 0)
{
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(
	    nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	const int32_t values[4] = {(int32_t)width, (int32_t)height, 1, (int32_t)format};
	const CFStringRef keys[4] = {kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement,
	                             kIOSurfacePixelFormat};
	for (int i = 0; i < (format ? 4 : 3); i++) {
		CFNumberRef n = CFNumberCreate(nullptr, kCFNumberSInt32Type, &values[i]);
		CFDictionarySetValue(props, keys[i], n);
		CFRelease(n);
	}
	IOSurfaceRef s = IOSurfaceCreate(props);
	CFRelease(props);
	CHECK(s != nullptr, "a %ux%u surface", width, height);
	return s;
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

// An HEVC picture the host's caps say it refuses is not sent, so its schedule fails: one of a
// coded size outside the caps' least and largest; one whose PPS has a tile column narrower than
// the caps' narrowest; and one whose slice header holds more current references than the caps'
// most. A bound the caps leave 0 refuses nothing, and a picture at a bound is sent.
void checkHostBounds()
{
	g_opened.clear();
	std::vector<void *> contexts;
	CHECK(makeContexts(1, contexts, kCodecHevc) == kOk, "a create of one HEVC decode");
	const uint64_t bitstream = allocate(g_driver, 0, 0x10000);
	CHECK(bitstream != 0, "the bitstream pool refused");
	IOSurfaceRef surface = canvas(8320, 720);
	// The picture's one slice segment in the pool, `slice` or bytes with no start code.
	auto sends = [&](const rgfx::video::h265::Sps &sps, const rgfx::video::h265::Pps &pps,
	                 const std::vector<uint8_t> &slice = {}) {
		uint32_t bytes = 0x100;
		std::memset((void *)bitstream, 0x5a, bytes);
		if (!slice.empty()) {
			std::memcpy((void *)bitstream, slice.data(), slice.size());
			bytes = (uint32_t)slice.size();
		}
		g_sent.clear();
		const uint64_t answer = scheduleHevc(contexts[0], HevcPicture(surface, sps, pps, 0, 0, bytes));
		CHECK(answer == kOk || answer == kNotSent, "a schedule answered %llu", answer);
		CHECK((answer == kOk) == (g_sent.size() == 1), "%zu decodes sent of a schedule of %llu",
		      g_sent.size(), answer);
		return answer == kOk;
	};
	static rgfx::video::h265::Sps sps;
	static rgfx::video::h265::Pps pps;
	sps = {};
	pps = {};
	sps.chroma_format_idc = 1;
	sps.log2_diff_max_min_luma_coding_block_size = 3; // CTBs of 64
	struct Size {
		uint32_t width, height;
		bool sent;
	};
	// The stand-in's caps: 144x144 to 8192x8192.
	for (const Size &z : {Size{1280, 720, true}, Size{144, 144, true}, Size{8192, 720, true},
	                      Size{128, 720, false}, Size{1280, 128, false}, Size{8320, 720, false}}) {
		sps.pic_width_in_luma_samples = z.width;
		sps.pic_height_in_luma_samples = z.height;
		CHECK(sends(sps, pps) == z.sent, "a %ux%u picture %s", z.width, z.height,
		      z.sent ? "not sent" : "sent");
	}
	sps.pic_width_in_luma_samples = 1280;
	sps.pic_height_in_luma_samples = 720;

	// Tile columns of 64, 64, 1088 and 64 samples; and of 128, 192, 640 and 320.
	pps.tiles_enabled_flag = 1;
	pps.num_tile_columns_minus1 = 3;
	const uint32_t narrow[3] = {0, 0, 16}, wide[3] = {1, 2, 9};
	auto columns = [&](const uint32_t (&minus1)[3]) {
		std::memcpy(pps.column_width_minus1, minus1, sizeof minus1);
	};
	columns(narrow);
	CHECK(sends(sps, pps), "columns of 64 not sent with no bound");
	g_decodeCaps.min_tile_width = 128;
	CHECK(!sends(sps, pps), "columns of 64 sent under a bound of 128");
	columns(wide);
	CHECK(sends(sps, pps), "columns of 128 and more not sent under a bound of 128");
	pps.uniform_spacing_flag = 1;
	pps.num_tile_columns_minus1 = 19;
	CHECK(!sends(sps, pps), "20 uniform columns of 64 sent under a bound of 128");
	pps.num_tile_columns_minus1 = 9;
	CHECK(sends(sps, pps), "10 uniform columns of 128 not sent under a bound of 128");
	g_decodeCaps = {};
	pps = {};

	// A P slice using the SPS's set 0, of 8 current references, or set 1, of 9.
	sps.log2_max_pic_order_cnt_lsb_minus4 = 4;
	sps.num_short_term_ref_pic_sets = 2;
	for (uint32_t set = 0; set < 2; set++) {
		rgfx::video::h265::ShortTermRps &r = sps.st_ref_pic_set[set];
		r.num_negative_pics = r.num_delta_pocs = 8 + set;
		for (uint32_t i = 0; i < r.num_negative_pics; i++) {
			r.delta_poc_s0[i] = -(int32_t)(i + 1);
			r.used_by_curr_pic_s0[i] = 1;
		}
	}
	auto slice = [](uint32_t set) {
		rgfx::video::BitWriter w;
		w.flag(1);    // first_slice_segment_in_pic_flag
		w.ue(0);      // slice_pic_parameter_set_id
		w.ue(1);      // slice_type P
		w.u(8, 9);    // slice_pic_order_cnt_lsb
		w.flag(1);    // short_term_ref_pic_set_sps_flag
		w.u(1, set);  // short_term_ref_pic_set_idx
		for (int i = 0; i < 16; i++)
			w.u(8, 0xa5);
		std::vector<uint8_t> out;
		w.appendNal({1 << 1, 1}, out); // TRAIL_R
		return out;
	};
	CHECK(sends(sps, pps, slice(1)), "9 current references not sent with no bound");
	g_hevcMaxReferences = 8;
	CHECK(sends(sps, pps, slice(0)), "8 current references not sent under a bound of 8");
	CHECK(!sends(sps, pps, slice(1)), "9 current references sent under a bound of 8");
	// A slice the driver cannot read leaves it to the host.
	CHECK(sends(sps, pps), "a slice of no start code not sent");
	g_hevcMaxReferences = 0;

	ContextDelete del = {1, 0, &contexts[0]};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(surface);
}

// The schedule of an HEVC picture waits for the host's verdict on it, and fails where the host
// failed it, the one failure AppleGVA's HEVC decoder reports for the picture itself
// (RgfxVideoStatus.hpp): a picture the host decodes is scheduled, one it fails is not, and the
// session's next picture is scheduled again; each was sent, and its verdict waited for once.
void checkHostVerdict()
{
	g_opened.clear();
	std::vector<void *> contexts;
	CHECK(makeContexts(1, contexts, kCodecHevc) == kOk, "a create of one HEVC decode");
	const uint64_t bitstream = allocate(g_driver, 0, 0x10000);
	CHECK(bitstream != 0, "the bitstream pool refused");
	std::memset((void *)bitstream, 0x5a, 0x100);
	IOSurfaceRef surface = canvas(1280, 720);
	static rgfx::video::h265::Sps sps;
	static rgfx::video::h265::Pps pps;
	sps = {};
	pps = {};
	sps.chroma_format_idc = 1;
	sps.log2_diff_max_min_luma_coding_block_size = 3; // CTBs of 64
	sps.pic_width_in_luma_samples = 1280;
	sps.pic_height_in_luma_samples = 720;
	auto schedule = [&](int verdict) {
		g_verdict = verdict;
		g_waits = 0;
		g_sent.clear();
		const uint64_t answer = scheduleHevc(contexts[0], HevcPicture(surface, sps, pps, 0, 0, 0x100));
		CHECK(g_sent.size() == 1 && g_waits == 1, "%zu decodes sent and %d verdicts waited for",
		      g_sent.size(), g_waits);
		return answer;
	};
	CHECK(schedule(1) == kOk, "a picture the host decoded not scheduled");
	CHECK(schedule(-1) == kNotSent, "a picture the host failed scheduled");
	CHECK(schedule(1) == kOk, "the picture after a failed one not scheduled");
	g_verdict = 1;

	ContextDelete del = {1, 0, &contexts[0]};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(surface);
}

// Contexts schedule apart: while the work of one context's picture waits in the driver (as the
// first textures over a new output surface wait for the host), another context's schedule goes on
// and its picture reaches its session; the same context's next schedule waits for the first, so
// the context's pictures reach its session in AppleGVA's order.
void checkContextsScheduleApart()
{
	using namespace std::chrono_literals;
	g_opened.clear();
	g_sentOn.clear();
	std::vector<void *> contexts;
	CHECK(makeContexts(2, contexts, kCodecHevc) == kOk, "a create of two HEVC decodes");
	const uint64_t bitstream = allocate(g_driver, 0, 0x10000);
	CHECK(bitstream != 0, "the bitstream pool refused");
	std::memset((void *)bitstream, 0x5a, 0x100);
	IOSurfaceRef surface = canvas(1280, 720);
	static rgfx::video::h265::Sps sps;
	static rgfx::video::h265::Pps pps;
	sps = {};
	pps = {};
	sps.chroma_format_idc = 1;
	sps.log2_diff_max_min_luma_coding_block_size = 3; // CTBs of 64
	sps.pic_width_in_luma_samples = 1280;
	sps.pic_height_in_luma_samples = 720;
	const HevcPicture picture(surface, sps, pps, 0, 0, 0x100);
	FakeSession *held = g_opened[0], *other = g_opened[1];
	std::atomic<uint64_t> heldAnswer{~0ull}, otherAnswer{~0ull}, againAnswer{~0ull};
	std::unique_lock<std::mutex> lock(g_sentLock);
	g_holdOn = held;
	std::thread first([&] { heldAnswer = scheduleHevc(contexts[0], picture); });
	CHECK(g_sentChanged.wait_for(lock, 5s, [] { return g_held; }), "the held decode never began");
	lock.unlock();
	std::thread second([&] { otherAnswer = scheduleHevc(contexts[1], picture); });
	std::thread again([&] { againAnswer = scheduleHevc(contexts[0], picture); });
	lock.lock();
	CHECK(g_sentChanged.wait_for(lock, 5s, [&] { return g_sentOn.size() == 1; }),
	      "another context's schedule waited for a context's held decode");
	CHECK(g_sentOn[0] == other, "a decode of the held context's sent while it was held");
	// The held context's next schedule stays behind its first.
	CHECK(!g_sentChanged.wait_for(lock, 200ms, [&] { return g_sentOn.size() > 1; }),
	      "a context's second schedule went ahead of its first");
	g_holdOn = nullptr;
	g_sentChanged.notify_all();
	lock.unlock();
	first.join();
	second.join();
	again.join();
	CHECK(heldAnswer == kOk && otherAnswer == kOk && againAnswer == kOk,
	      "schedules answered %llu, %llu and %llu", (unsigned long long)heldAnswer.load(),
	      (unsigned long long)otherAnswer.load(), (unsigned long long)againAnswer.load());
	CHECK(g_sentOn.size() == 3 && g_sentOn[1] == held && g_sentOn[2] == held,
	      "%zu decodes sent, the held context's not after the other's", g_sentOn.size());
	ContextDelete del = {2, 0, contexts.data()};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(surface);
}

// An HEVC picture's frame describes its surface where its release lays the description out
// (hevcOutputSurface): the driver searches the frame for the one layout, of fields of 4 or 8 bytes
// from +0x60 to +0xa0, that names its surface, and its session keeps the layout its first picture
// shows. Each release's measured layout lands its picture in its surface, as do the window's
// ends; a description from outside the window, or whose width, height or pixel format is not its
// surface's, names none, before the session has a layout and after. A frame that describes its
// surface in two layouts names none and shows its session no layout. A session keeps the layout
// its first picture showed.
void checkFrameDescriptions()
{
	g_sent.clear();
	constexpr uint32_t kBytes = 0x1000;
	const uint64_t bitstream = allocate(g_driver, 0, 0x10000);
	CHECK(bitstream != 0, "the bitstream pool refused");
	std::memset((void *)bitstream, 0x5a, kBytes);
	IOSurfaceRef surface = canvas(512, 512);
	static rgfx::video::h265::Sps sps;
	static rgfx::video::h265::Pps pps;
	sps.chroma_format_idc = 1;
	sps.pic_width_in_luma_samples = sps.pic_height_in_luma_samples = 512;
	auto session = [] {
		std::vector<void *> contexts;
		CHECK(makeContexts(1, contexts, kCodecHevc) == kOk, "a create of one HEVC decode");
		return contexts.empty() ? nullptr : contexts[0];
	};
	auto finish = [](void *context) {
		ContextDelete del = {1, 0, &context};
		CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	};
	auto lands = [&](void *context, const HevcPicture &picture, const char *what) {
		g_sent.clear();
		CHECK(scheduleHevc(context, picture) == kOk, "%s not sent", what);
		CHECK(g_sent.size() == 1 && g_sent[0].surface == IOSurfaceGetID(surface),
		      "%s sent %zu times", what, g_sent.size());
	};
	auto refused = [&](void *context, const HevcPicture &picture, const char *what) {
		g_sent.clear();
		CHECK(scheduleHevc(context, picture) == kNotSent, "%s sent", what);
		CHECK(g_sent.empty(), "%s reached the host", what);
	};
	char what[128];
	for (const Described &described : {kMacOS26, kMacOS14, kMacOS13, kMacOS12, kMacOS11}) {
		const size_t fields[] = {described.width(), described.height(), described.format()};
		void *context = session();
		for (int known = 0; known < 2; known++) {
			for (size_t field : fields) {
				HevcPicture wrong(surface, sps, pps, 0, 0, kBytes, described);
				wrong.frame[field] ^= 1;
				snprintf(what, sizeof what,
				         "a description in %zu-byte fields from +0x%zx, its +0x%zx not its surface's, "
				         "%s",
				         described.fieldBytes, described.start, field,
				         known ? "on its layout's session" : "first");
				refused(context, wrong, what);
			}
			snprintf(what, sizeof what, "a picture described in %zu-byte fields from +0x%zx, %s",
			         described.fieldBytes, described.start, known ? "again" : "first");
			lands(context, HevcPicture(surface, sps, pps, 0, 0, kBytes, described), what);
		}
		finish(context);
	}
	for (const Described &described : {Described{0x60, 4}, Described{0x60, 8},
	                                   Described{0xa0, 4}, Described{0xa0, 8}}) {
		void *context = session();
		snprintf(what, sizeof what, "a picture described in %zu-byte fields from +0x%zx",
		         described.fieldBytes, described.start);
		lands(context, HevcPicture(surface, sps, pps, 0, 0, kBytes, described), what);
		finish(context);
	}
	for (const Described &described : {Described{0x5c, 8}, Described{0xa4, 4}}) {
		void *context = session();
		snprintf(what, sizeof what, "a picture described in %zu-byte fields from +0x%zx",
		         described.fieldBytes, described.start);
		refused(context, HevcPicture(surface, sps, pps, 0, 0, kBytes, described), what);
		finish(context);
	}
	{
		void *context = session();
		HevcPicture twice(surface, sps, pps, 0, 0, kBytes, kMacOS14);
		twice.describe(surface, Described{0x70, 4});
		refused(context, twice, "a picture described in two layouts");
		lands(context, HevcPicture(surface, sps, pps, 0, 0, kBytes, kMacOS13),
		      "macOS 13's description after one in two layouts");
		finish(context);
	}
	{
		void *context = session();
		lands(context, HevcPicture(surface, sps, pps, 0, 0, kBytes, kMacOS26),
		      "macOS 26's description first");
		refused(context, HevcPicture(surface, sps, pps, 0, 0, kBytes, kMacOS13),
		        "macOS 13's description on a session of 26's layout");
		finish(context);
	}
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(surface);
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

// A session's H.264 pictures are read in the one layout of AppleGVA's parsed SPS in which the
// first describes its picture, 11's or 12's, and their SPS is sent as the stream holds it; a
// later picture in the other layout, a record that fits neither, and a surface the SPS's size
// does not round up to are refused, and nothing of them reaches the host. A picture the host
// fails is scheduled, its schedule waiting for no verdict.
void checkH264SpsLayouts()
{
	using namespace rgfx::video::h264;
	constexpr uint32_t kBytes = 0x1000;
	const uint64_t bitstream = allocate(g_driver, 0, 0x10000);
	CHECK(bitstream != 0, "the bitstream pool refused");
	std::memset((void *)bitstream, 0x5a, kBytes);
	IOSurfaceRef surface = canvas(640, 480), narrower = canvas(624, 480);
	// A 640x480 High stream's, as x264 writes it.
	Sps sps = {};
	sps.field_000 = 1;
	sps.profile_idc = 100;
	sps.level_idc = 30;
	sps.chroma_format_idc = 1;
	sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	sps.max_num_ref_frames = 4;
	sps.pic_width_in_mbs_minus1 = 39;
	sps.pic_height_in_map_units_minus1 = 29;
	sps.frame_mbs_only_flag = 1;
	sps.direct_8x8_inference_flag = 1;
	Pps pps = {};
	pps.field_000 = 1;
	pps.entropy_coding_mode_flag = 1;
	pps.num_ref_idx_l0_default_active_minus1 = 2;
	pps.weighted_pred_flag = 1;
	pps.weighted_bipred_idc = 2;
	pps.transform_8x8_mode_flag = 1;
	std::vector<uint8_t> sets;
	CHECK(writeSps(sps, sets) && writePps(pps, sps, sets), "the test's parameter sets refused");
	Sps none = sps;
	none.level_idc = 0;
	const H264Picture on11(spsRecordOn11(sps), pps, surface, kBytes),
	    on12(spsRecordOn12(sps), pps, surface, kBytes),
	    neither(spsRecordOn12(none), pps, surface, kBytes),
	    tooNarrow(spsRecordOn12(sps), pps, narrower, kBytes);
	auto session = [] {
		std::vector<void *> contexts;
		CHECK(makeContexts(1, contexts) == kOk, "a create of one H.264 decode");
		return contexts.empty() ? nullptr : contexts[0];
	};
	auto finish = [](void *context) {
		ContextDelete del = {1, 0, &context};
		CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	};
	auto lands = [&](void *context, const H264Picture &picture, const char *what) {
		g_sent.clear();
		CHECK(scheduleH264(context, picture) == kOk, "%s not sent", what);
		CHECK(g_sent.size() == 1 && g_sent[0].surface == IOSurfaceGetID(surface) &&
		          g_sent[0].sets == sets,
		      "%s sent %zu times, or with other parameter sets", what, g_sent.size());
	};
	auto refused = [&](void *context, const H264Picture &picture, const char *what) {
		g_sent.clear();
		CHECK(scheduleH264(context, picture) == kNotSent, "%s sent", what);
		CHECK(g_sent.empty(), "%s reached the host", what);
	};
	void *first = session();
	lands(first, on11, "11's record");
	refused(first, on12, "12's record on a session that read 11's");
	finish(first);
	void *second = session();
	lands(second, on12, "12's record");
	refused(second, on11, "11's record on a session that read 12's");
	finish(second);
	void *third = session();
	refused(third, neither, "a record fitting no layout");
	refused(third, tooNarrow, "a surface a macroblock narrower than the SPS's size");
	lands(third, on12, "12's record after two refused");
	finish(third);
	// AppleGVA's H.264 decoder reports no failed schedule, so a schedule waits for no verdict of
	// the host's (RgfxVideoStatus.hpp).
	void *fourth = session();
	g_verdict = -1;
	g_waits = 0;
	lands(fourth, on12, "a picture the host fails");
	CHECK(g_waits == 0, "an H.264 schedule waited for %d verdicts of the host's", g_waits);
	g_verdict = 1;
	finish(fourth);
	CFRelease(surface);
	CFRelease(narrower);
}


// --- encode ----------------------------------------------------------------------------------

constexpr uint32_t kFunctionEncode = 0x20;
constexpr uint64_t kEncodeH264 = 0x800, kEncodeHevc = 0x1000;
constexpr uint32_t kFunctionProcess = 2;
constexpr uint64_t kProcessFrameStatistics = 0x1000;
constexpr uint64_t kBadArgument = 5;
enum { kAllocatePixelBufferPool = 8, kQueryCommandStatus = 11 };
constexpr uint32_t kQueryDone = 1;

struct Query {
	void *token;
	uint32_t field_08;
	uint32_t detail;
	uint32_t status;
	uint32_t field_14;
};
struct QueryList {
	uint32_t count;
	uint32_t field_04;
	Query *queries;
};
struct PixelPoolRecord {
	Format format;
	uint64_t field_08, field_10, field_18, field_20, field_28;
	uint64_t *address;
};
static_assert(sizeof(PixelPoolRecord) == 0x38, "pixel pool record layout");
struct PixelPool {
	uint32_t pool;
	uint32_t field_04;
	uint32_t size_factors[2];
	uint32_t count;
	uint32_t field_14;
	PixelPoolRecord *records;
};

// The functions of the capability block of `id` and `sub`.
int listed(uint32_t id, uint64_t sub)
{
	const CapabilityBlock *b = capabilities();
	int n = 0;
	for (uint32_t f = 0; f < b->function_count; f++)
		n += b->functions[f]->function_id == id && b->functions[f]->sub_id == sub;
	return n;
}

// The test binary's path, which checkListedOn runs again.
const char *g_self;

// In a process of its own, as the driver fills its capability block once a process (the test
// binary run again, `listed ENCODES RATES WANT WANT_HEVC HEVC_PROFILES`, which exits 0 where the
// block lists H.264 encode `want` times and HEVC encode, with the frame statistics beside it,
// `wantHevc` times): a host that `encodes` H.264 and HEVC with rate controls `rates`, its HEVC
// encoder of `hevcProfiles`.
void checkListedOn(bool encodes, uint32_t rates, int want, int wantHevc, const char *what,
                   uint32_t hevcProfiles = RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10,
                   uint32_t flags = RGFX_VIDEO_ENCODE_CAP_IN_PLACE)
{
	char encodesArg[4], ratesArg[16], wantArg[4], wantHevcArg[4], profilesArg[16], flagsArg[16];
	std::snprintf(encodesArg, sizeof encodesArg, "%d", encodes ? 1 : 0);
	std::snprintf(ratesArg, sizeof ratesArg, "%u", rates);
	std::snprintf(wantArg, sizeof wantArg, "%d", want);
	std::snprintf(wantHevcArg, sizeof wantHevcArg, "%d", wantHevc);
	std::snprintf(profilesArg, sizeof profilesArg, "%u", hevcProfiles);
	std::snprintf(flagsArg, sizeof flagsArg, "%u", flags);
	char *const argv[] = {const_cast<char *>(g_self), const_cast<char *>("listed"), encodesArg,
	                      ratesArg, wantArg, wantHevcArg, profilesArg, flagsArg, nullptr};
	pid_t pid = 0;
	CHECK(posix_spawn(&pid, g_self, nullptr, nullptr, argv, environ) == 0, "%s: no child", what);
	int status = 0;
	CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
	      "%s (the child's status %#x)", what, status);
}

// The child of checkListedOn: whether the block of its host lists H.264 encode `want` times, and
// HEVC encode and the frame statistics `wantHevc` times each.
int listedChild(char **argv)
{
	g_hostEncodes = std::atoi(argv[2]) != 0;
	g_encodeRates = (uint32_t)std::strtoul(argv[3], nullptr, 10);
	g_hevcEncodeProfiles = (uint32_t)std::strtoul(argv[6], nullptr, 10);
	g_encodeFlags = (uint32_t)std::strtoul(argv[7], nullptr, 10);
	void *table = nullptr;
	uint8_t params[64] = {}, identity[64] = {};
	if (AVD_CreateAVDAccelerator(&table, params, identity) != 0 || !table)
		return 2;
	g_table = (DriverTable *)table;
	g_driver = createDriver();
	const int wantHevc = std::atoi(argv[5]);
	if (listed(kFunctionEncode, kEncodeHevc) != wantHevc ||
	    listed(kFunctionProcess, kProcessFrameStatistics) != wantHevc)
		return 3;
	return listed(kFunctionEncode, kEncodeH264) == std::atoi(argv[4]) ? 0 : 1;
}

// The capability block lists H.264 encode (function 0x20, sub-ID 0x800, NV12 in and a bitstream
// out) where the host has an H.264 encoder with a rate control, and HEVC encode (sub-ID 0x1000),
// with the frame statistics the HEVC encoder plug-in looks up beside it (function 2, sub-ID
// 0x1000), where the host has an HEVC encoder of Main from NV12 and Main 10 from P010 with a
// constant QP, at which the driver codes the plug-in's frames, that codes a frame in place of the
// last; a host with no encoder, or one it lists with no rate control, lists neither, and one with
// no constant QP, of one of the two profiles alone, or that codes no frame in place, lists no HEVC
// encode, and H.264 encode as before. The rest of the checks run on a host
// that encodes, so every answer of the block with the encoders listed is one AppleGVA may divide
// by (checkNoZeroAnswers).
void checkEncodeListedApart()
{
	checkListedOn(false, 1u << RGFX_VIDEO_RATE_CQP, 0, 0, "encode listed with no host encoder");
	checkListedOn(true, 0, 0, 0, "encode listed with no rate control");
	checkListedOn(true, 1u << RGFX_VIDEO_RATE_VBR, 1, 0,
	              "H.264 encode not listed once, or HEVC encode listed, with no constant QP");
	checkListedOn(true, 1u << RGFX_VIDEO_RATE_CQP, 1, 1, "H.264 or HEVC encode not listed once");
	checkListedOn(true, 1u << RGFX_VIDEO_RATE_CQP, 1, 0, "HEVC encode listed of Main alone",
	              RGFX_VIDEO_HEVC_MAIN);
	checkListedOn(true, 1u << RGFX_VIDEO_RATE_CQP, 1, 0, "HEVC encode listed of Main 10 alone",
	              RGFX_VIDEO_HEVC_MAIN_10);
	checkListedOn(true, 1u << RGFX_VIDEO_RATE_CQP, 1, 0,
	              "HEVC encode listed, or H.264 encode not, on a host that codes no frame in place",
	              RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10, 0);
}

void checkEncodeListed()
{
	CHECK(listed(kFunctionEncode, kEncodeH264) == 1, "H.264 encode not listed once");
	CHECK(listed(kFunctionEncode, kEncodeHevc) == 1, "HEVC encode not listed once");
	CHECK(listed(kFunctionProcess, kProcessFrameStatistics) == 1,
	      "the frame statistics not listed once");
	const CapabilityBlock *b = capabilities();
	for (uint32_t f = 0; f < b->function_count; f++) {
		const Function &fn = *b->functions[f];
		if (fn.function_id != kFunctionEncode)
			continue;
		CHECK(fn.input_count == 1 && fn.inputs[0].fourcc == 'NV12' && fn.output_count == 1 &&
		          fn.outputs[0].fourcc == 'rawb',
		      "the encode's formats");
	}
}

// The stand-in host's rate controls where a check does not change them.
constexpr uint32_t kCqpVbr = 1u << RGFX_VIDEO_RATE_CQP | 1u << RGFX_VIDEO_RATE_VBR;

// Makes one encode context of `sub` of `width` x `height`: CreateContexts' answer, the context in
// `context`.
uint64_t makeEncode(void **context, uint32_t width, uint32_t height, uint64_t sub = kEncodeH264)
{
	static FunctionInfo f;
	f = {};
	f.function_id = kFunctionEncode;
	f.sub_id = sub;
	f.width = width;
	f.height = height;
	static const FunctionInfo *list[1];
	list[0] = &f;
	*context = nullptr;
	ContextInfo info = {1, 0, list, context};
	ContextInit init = {&info, nullptr};
	return call(kCreateContexts, &init);
}

// An H.264 or HEVC encode context opens an encode session on the host where the host encodes its
// size; one of a size past the host's, one on a host with no encoder, and an HEVC encode on a host
// whose encoder has no constant QP, encodes Main alone, or codes no frame in place, are refused at
// their create, with no session left open, so VideoToolbox takes its software encoder where the
// app allows it; an H.264 encode on that last host opens as before.
void checkEncodeCreate()
{
	g_hostEncodes = true;
	const struct {
		uint32_t width, height;
		uint64_t sub;
		bool encodes;
		uint32_t rates;
		uint64_t want;
	} cases[] = {
	    {640, 480, kEncodeH264, true, kCqpVbr, kOk},
	    {4096, 4096, kEncodeH264, true, kCqpVbr, kOk},
	    {4097, 480, kEncodeH264, true, kCqpVbr, kUnsupported},
	    {128, 480, kEncodeH264, true, kCqpVbr, kUnsupported},
	    {640, 480, kEncodeH264, false, kCqpVbr, kUnsupported},
	    {640, 480, kEncodeHevc, true, kCqpVbr, kOk},
	    {1920, 1088, kEncodeHevc, true, kCqpVbr, kOk},
	    {4097, 480, kEncodeHevc, true, kCqpVbr, kUnsupported},
	    {640, 480, kEncodeHevc, false, kCqpVbr, kUnsupported},
	    {640, 480, kEncodeHevc, true, 1u << RGFX_VIDEO_RATE_VBR, kUnsupported},
	};
	for (const auto &c : cases) {
		g_hostEncodes = c.encodes;
		g_encodeRates = c.rates;
		g_opened.clear();
		void *context = nullptr;
		CHECK(makeEncode(&context, c.width, c.height, c.sub) == c.want,
		      "an encode of sub 0x%llx of %ux%u", (unsigned long long)c.sub, c.width, c.height);
		CHECK(g_opened.size() == (c.want == kOk ? 1u : 0u), "%zu sessions for %ux%u",
		      g_opened.size(), c.width, c.height);
		CHECK((context != nullptr) == (c.want == kOk), "the context of %ux%u", c.width, c.height);
		if (context) {
			ContextDelete del = {1, 0, &context};
			CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
			CHECK(g_opened[0]->closed, "a deleted encode's session left open");
		}
	}
	g_hostEncodes = true;
	g_encodeRates = kCqpVbr;
	g_hevcEncodeProfiles = RGFX_VIDEO_HEVC_MAIN;
	g_opened.clear();
	void *mainOnly = nullptr;
	CHECK(makeEncode(&mainOnly, 640, 480, kEncodeHevc) == kUnsupported && !mainOnly &&
	          g_opened.empty(),
	      "an HEVC encode on a host of Main alone not refused");
	g_hevcEncodeProfiles = RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10;
	g_encodeFlags = 0;
	void *notInPlace = nullptr;
	CHECK(makeEncode(&notInPlace, 640, 480, kEncodeHevc) == kUnsupported && !notInPlace &&
	          g_opened.empty(),
	      "an HEVC encode on a host that codes no frame in place not refused");
	void *h264 = nullptr;
	CHECK(makeEncode(&h264, 640, 480, kEncodeH264) == kOk && h264 && g_opened.size() == 1,
	      "an H.264 encode on a host that codes no frame in place refused");
	ContextDelete del = {1, 0, &h264};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	g_encodeFlags = RGFX_VIDEO_ENCODE_CAP_IN_PLACE;
}

// AllocatePixelBufferPool gives an encoder's pools of kinds 0xe and 0x1b: a buffer of the product
// of the size factors' bytes for each record, its address written where the record points, each
// its own. A pool of another kind, of no buffer or more than 32, or of no size, is refused, and so
// is a record with nowhere to write. FreePixelBufferPool of a kind frees that kind's buffers alone,
// a pool allocated again replaces the kind's, and DeleteDriver frees the rest.
void checkPixelPools()
{
	void *driver = createDriver();
	auto pool = [&](uint32_t kind, uint32_t count, uint32_t a, uint32_t b,
	                std::vector<uint64_t> &addresses) {
		addresses.assign(count, 0);
		std::vector<PixelPoolRecord> records(count);
		for (uint32_t i = 0; i < count; i++)
			records[i].address = &addresses[i];
		PixelPool p = {kind, 0, {a, b}, count, 0, records.data()};
		return callOn(driver, kAllocatePixelBufferPool, &p);
	};
	std::vector<uint64_t> qp, other;
	CHECK(pool(0xe, 3, 128, 64, qp) == kOk, "a pool of kind 0xe");
	CHECK(qp[0] && qp[1] && qp[2] && qp[0] != qp[1] && qp[1] != qp[2] && qp[0] != qp[2],
	      "the pool's addresses");
	for (int i = 0; i < 3; i++)
		std::memset((void *)qp[i], 0x40 + i, 128 * 64);
	CHECK(pool(0x1b, 2, 4096, 3, other) == kOk, "a pool of kind 0x1b");
	for (int i = 0; i < 2; i++)
		std::memset((void *)other[i], 0x50 + i, 4096 * 3);
	std::vector<uint64_t> none;
	CHECK(pool(0x7, 1, 64, 64, none) == kBadArgument, "a pool of kind 7");
	CHECK(pool(0xe, 0, 64, 64, none) == kBadArgument, "a pool of no buffer");
	CHECK(pool(0xe, 33, 64, 64, none) == kBadArgument, "a pool of 33 buffers");
	CHECK(pool(0xe, 1, 0, 64, none) == kBadArgument, "a pool of no size");
	PixelPoolRecord nowhere = {};
	PixelPool p = {0xe, 0, {64, 64}, 1, 0, &nowhere};
	CHECK(callOn(driver, kAllocatePixelBufferPool, &p) == kBadArgument, "a record with no slot");
	CHECK(holds(qp[2], 128 * 64, 0x42) && holds(other[1], 4096 * 3, 0x51),
	      "a refused pool freed or wrote the buffers given");
	CHECK(callOn(driver, kFreePixelBufferPool, (const void *)0xe) == kOk, "FreePixelBufferPool");
	CHECK(!holds(qp[0], 128 * 64, 0x40), "kind 0xe's buffers kept after its free");
	CHECK(holds(other[0], 4096 * 3, 0x50), "kind 0x1b's buffers freed with kind 0xe's");
	std::vector<uint64_t> again;
	CHECK(pool(0x1b, 1, 64, 64, again) == kOk && again[0], "a pool of kind 0x1b allocated again");
	CHECK(!holds(other[0], 4096 * 3, 0x50), "the replaced pool's buffers kept");
	CHECK(callOn(driver, kDeleteDriver, nullptr) == kOk, "DeleteDriver");
	CHECK(!holds(again[0], 64, 0), "a deleted driver's pixel pool kept");
}

// An H.264 encode command's parameters as AppleGVA fills them, as far as the driver reads them:
// the bitstream ring (buffer 0 of pool 0, its capacity, the access unit's start, and the end the
// driver writes, P+0x0 to P+0x10), the source surface (P+0x20), its SPS and PPS records (P+0x2f0,
// P+0x2f8), its NAL header (P+0x310, nal_unit_type at +0xc), its rate control (P+0x328: the mode,
// the QP at +0x10 and the frame rate at +0x24) and the record the driver writes the access unit's
// length in bits in (P+0x330, at +0x6c, and at +0x3c for AppleGVA's rate control).
struct H264Encode {
	std::vector<uint8_t> sps;
	rgfx::video::h264::Pps pps;
	uint8_t nal[0x20] = {};
	uint8_t rate[0x40] = {};
	uint8_t length[0x80] = {};
	uint8_t parameters[0x340] = {};

	H264Encode(std::vector<uint8_t> record, const rgfx::video::h264::Pps &p, IOSurfaceRef source,
	           uint32_t capacity, uint32_t start, uint32_t nalType, uint32_t mode, uint32_t qp)
	    : sps(std::move(record)), pps(p)
	{
		const uint32_t ring[4] = {0, 0, capacity, start};
		std::memcpy(parameters, ring, sizeof ring);
		std::memcpy(parameters + 0x20, &source, 8);
		const void *records[2] = {sps.data(), &pps};
		std::memcpy(parameters + 0x2f0, records, sizeof records);
		const void *at[] = {nal, rate, length};
		std::memcpy(parameters + 0x310, &at[0], 8);
		std::memcpy(parameters + 0x328, &at[1], 8);
		std::memcpy(parameters + 0x330, &at[2], 8);
		std::memcpy(nal + 0xc, &nalType, 4);
		const uint32_t fps[2] = {30, 1};
		std::memcpy(rate, &mode, 4);
		std::memcpy(rate + 0x10, &qp, 4);
		std::memcpy(rate + 0x24, fps, sizeof fps);
	}
	uint32_t end() const { return u32(parameters + 0x10); }
	uint32_t bits() const { return u32(length + 0x6c); }
	uint32_t field3c() const { return u32(length + 0x3c); }
	static uint32_t u32(const uint8_t *at)
	{
		uint32_t v;
		std::memcpy(&v, at, 4);
		return v;
	}
};

// Schedules `e` on `context`: Schedule's answer, its token in `token`.
uint64_t scheduleEncode(void *context, const H264Encode &e, void **token)
{
	Command c = {};
	c.function_id = kFunctionEncode;
	c.sub_id = kEncodeH264;
	c.context = context;
	c.parameters = e.parameters;
	const Command *list[1] = {&c};
	void *tokens[1] = {};
	CommandList cl = {1, 0, list};
	TokenList tl = {1, 0, tokens};
	const uint64_t answer = call(kSchedule, &cl, &tl);
	*token = tokens[0];
	return answer;
}

// QueryCommandStatus of `token`, its detail set as AppleGVA's encoder sets it: the status.
uint32_t queryEncode(void *token)
{
	Query q = {token, 0, 1, 0, 0};
	QueryList ql = {1, 0, &q};
	CHECK(call(kQueryCommandStatus, &ql) == kOk, "QueryCommandStatus");
	return q.status;
}

// An encode is sent with the settings AppleGVA's records give, and published to AppleGVA as it is
// queried done:
// - the settings: the SPS's profile and level, the PPS's entropy coder, the source's size, the
//   rate control's QP as a constant QP and its frame rate, no keyframes of the host's own; an IDR
//   NAL header forces a keyframe, another does not; the frames count up in display order; the
//   access unit takes the ring's capacity but a byte;
// - the access unit lands in the ring at its start, wrapping at the ring's end, with the end
//   cursor past it and its length in bits; a second query publishes nothing again;
// - a frame the host failed, or whose access unit fills the ring, publishes none (the end at the
//   start, a length of 0) and fails the context's next schedule, sending nothing; the schedule
//   after it goes on;
// - a rate control of a mode the driver does not read is not sent.
void checkEncodeFrames()
{
	using namespace rgfx::video::h264;
	g_hostEncodes = true;
	g_encodeRates = 1u << RGFX_VIDEO_RATE_CQP;
	g_encodes.clear();
	g_encodeWork.clear();
	constexpr uint32_t kRing = 0x1000;
	const uint64_t ringAt = allocate(g_driver, 0, kRing);
	CHECK(ringAt != 0, "the bitstream ring refused");
	uint8_t *ring = (uint8_t *)ringAt;
	IOSurfaceRef source = canvas(640, 480);
	Sps sps = {};
	sps.field_000 = 1;
	sps.profile_idc = 100;
	sps.level_idc = 30;
	sps.chroma_format_idc = 1;
	sps.log2_max_pic_order_cnt_lsb_minus4 = 2;
	sps.max_num_ref_frames = 1;
	sps.pic_width_in_mbs_minus1 = 39;
	sps.pic_height_in_map_units_minus1 = 29;
	sps.frame_mbs_only_flag = 1;
	sps.direct_8x8_inference_flag = 1;
	Pps pps = {};
	pps.field_000 = 1;
	pps.entropy_coding_mode_flag = 1;
	void *context = nullptr;
	CHECK(makeEncode(&context, 640, 480) == kOk, "a create of one H.264 encode");
	g_accessUnit.assign(300, 0);
	for (size_t i = 0; i < g_accessUnit.size(); i++)
		g_accessUnit[i] = (uint8_t)(i * 7 + 1);
	auto published = [&](uint32_t start) {
		for (size_t i = 0; i < g_accessUnit.size(); i++)
			if (ring[(start + i) % kRing] != g_accessUnit[i])
				return false;
		return true;
	};
	void *token = nullptr;
	// An IDR frame at the ring's start.
	std::memset(ring, 0, kRing);
	H264Encode idr(spsRecordOn12(sps), pps, source, kRing, 0, 5, 0, 26);
	CHECK(scheduleEncode(context, idr, &token) == kOk && token, "an IDR frame not scheduled");
	CHECK(g_encodes.size() == 1, "%zu encodes sent", g_encodes.size());
	const SentEncode &first = g_encodes[0];
	const RgfxVideoEncodeSettings &st = first.settings;
	CHECK(first.surface == IOSurfaceGetID(source) && first.frameIndex == 0 && first.forceKeyframe &&
	          first.capacity == kRing - 1,
	      "the IDR frame's encode: surface %u, frame %llu, keyframe %d, capacity %llu", first.surface,
	      (unsigned long long)first.frameIndex, first.forceKeyframe,
	      (unsigned long long)first.capacity);
	CHECK(st.width == 640 && st.height == 480 && st.format == RGFX_VIDEO_FORMAT_NV12 &&
	          st.profile == RGFX_VIDEO_H264_HIGH && st.level == 30 &&
	          st.entropy == RGFX_VIDEO_ENTROPY_CABAC && st.rate_control == RGFX_VIDEO_RATE_CQP &&
	          st.qp_i == 26 && st.qp_p == 26 && st.qp_b == 26 &&
	          st.key_frame_interval == UINT32_MAX && st.frame_rate_num == 30 &&
	          st.frame_rate_den == 1 && st.full_range == 0,
	      "the IDR frame's settings: profile %#x level %u entropy %u rate %u qp %u keys %u fps %u/%u",
	      st.profile, st.level, st.entropy, st.rate_control, st.qp_i, st.key_frame_interval,
	      st.frame_rate_num, st.frame_rate_den);
	CHECK(queryEncode(token) == kQueryDone, "the IDR frame not done");
	CHECK(published(0) && idr.end() == 300 && idr.bits() == 2400,
	      "the IDR frame's access unit: end %u, %u bits", idr.end(), idr.bits());
	// The rate control's word takes the same bits.
	CHECK(idr.field3c() == 2400, "the IDR frame's +0x3c: %u", idr.field3c());
	std::memset(ring, 0, kRing);
	CHECK(queryEncode(token) == kQueryDone && ring[0] == 0, "a second query published again");
	// A P frame near the ring's end: its access unit wraps.
	H264Encode p(spsRecordOn12(sps), pps, source, kRing, 0xf00, 1, 0, 30);
	CHECK(scheduleEncode(context, p, &token) == kOk, "a P frame not scheduled");
	CHECK(g_encodes.size() == 2 && g_encodes[1].frameIndex == 1 && !g_encodes[1].forceKeyframe &&
	          g_encodes[1].settings.qp_p == 30,
	      "the P frame's encode");
	CHECK(queryEncode(token) == kQueryDone, "the P frame not done");
	CHECK(published(0xf00) && p.end() == (0xf00 + 300) % kRing && p.bits() == 2400,
	      "the wrapped access unit: end %#x, %u bits", p.end(), p.bits());
	// A frame the host failed.
	g_encodeStatus = RGFX_VIDEO_ENCODE_FAILED;
	H264Encode failed(spsRecordOn12(sps), pps, source, kRing, 0x100, 1, 0, 30);
	CHECK(scheduleEncode(context, failed, &token) == kOk, "the failing frame not scheduled");
	g_encodeStatus = RGFX_VIDEO_ENCODE_OK;
	CHECK(queryEncode(token) == kQueryDone, "the failed frame not done");
	CHECK(failed.end() == 0x100 && failed.bits() == 0, "the failed frame published: end %#x, %u bits",
	      failed.end(), failed.bits());
	CHECK(failed.field3c() == 0, "the failed frame's +0x3c: %u", failed.field3c());
	const size_t before = g_encodes.size();
	H264Encode after(spsRecordOn12(sps), pps, source, kRing, 0x100, 1, 0, 30);
	CHECK(scheduleEncode(context, after, &token) == kNotSent, "the frame after a failed one scheduled");
	CHECK(g_encodes.size() == before, "the frame after a failed one sent");
	CHECK(scheduleEncode(context, after, &token) == kOk, "the next frame not scheduled");
	CHECK(queryEncode(token) == kQueryDone && after.end() == 0x100 + 300, "the next frame not published");
	// An access unit that fills the ring is not published.
	g_accessUnit.assign(kRing, 0x77);
	H264Encode full(spsRecordOn12(sps), pps, source, kRing, 0, 1, 0, 30);
	CHECK(scheduleEncode(context, full, &token) == kOk, "the full frame not scheduled");
	CHECK(queryEncode(token) == kQueryDone && full.end() == 0 && full.bits() == 0,
	      "an access unit that fills the ring published");
	CHECK(scheduleEncode(context, after, &token) == kNotSent, "the frame after a full one scheduled");
	g_accessUnit.assign(16, 0x11);
	// A rate control of mode 1.
	const size_t sent = g_encodes.size();
	H264Encode bitrate(spsRecordOn12(sps), pps, source, kRing, 0, 1, 1, 0);
	CHECK(scheduleEncode(context, bitrate, &token) == kNotSent, "a frame of rate mode 1 scheduled");
	CHECK(g_encodes.size() == sent, "a frame of rate mode 1 sent");
	// Constrained Baseline with CAVLC.
	sps.profile_idc = 66;
	pps.entropy_coding_mode_flag = 0;
	void *baseline = nullptr;
	CHECK(makeEncode(&baseline, 640, 480) == kOk, "a create of a Baseline encode");
	H264Encode cavlc(spsRecordOn12(sps), pps, source, kRing, 0, 5, 0, 26);
	CHECK(scheduleEncode(baseline, cavlc, &token) == kOk, "a Baseline frame not scheduled");
	CHECK(g_encodes.back().settings.profile == RGFX_VIDEO_H264_BASELINE &&
	          g_encodes.back().settings.entropy == RGFX_VIDEO_ENTROPY_CAVLC &&
	          g_encodes.back().frameIndex == 0,
	      "the Baseline frame's settings");
	CHECK(queryEncode(token) == kQueryDone, "the Baseline frame not done");
	for (void *c : {context, baseline}) {
		ContextDelete del = {1, 0, &c};
		CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	}
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(source);
	g_encodeRates = 1u << RGFX_VIDEO_RATE_CQP | 1u << RGFX_VIDEO_RATE_VBR;
}

// An HEVC encode command's parameters as the HEVC encoder plug-in's Intel driver class fills
// them, as far as the driver reads and writes them: the source surface (P+0x10), the NAL unit type
// the plug-in chose (P+0x33c), its slice QP (P+0x3d8), and the words the access unit's length is
// published as (P+0x3a0, P+0x3a8), which the plug-in zeroes before each encode.
struct HevcEncode {
	uint8_t parameters[0x800] = {};

	HevcEncode(IOSurfaceRef source, uint32_t nalType, uint32_t qp, uint32_t poc)
	{
		std::memcpy(parameters + 0x4, &poc, 4);
		std::memcpy(parameters + 0x10, &source, 8);
		std::memcpy(parameters + 0x33c, &nalType, 4);
		std::memcpy(parameters + 0x3d8, &qp, 4);
	}
	uint64_t begin() const { return u64(parameters + 0x3a0); }
	uint64_t size() const { return u64(parameters + 0x3a8) - begin(); }
	static uint64_t u64(const uint8_t *at)
	{
		uint64_t v;
		std::memcpy(&v, at, 8);
		return v;
	}
};

// Schedules `e` on `context`: Schedule's answer, its token in `token`.
uint64_t scheduleHevcEncode(void *context, const HevcEncode &e, void **token)
{
	Command c = {};
	c.function_id = kFunctionEncode;
	c.sub_id = kEncodeHevc;
	c.context = context;
	c.parameters = e.parameters;
	const Command *list[1] = {&c};
	void *tokens[1] = {};
	CommandList cl = {1, 0, list};
	TokenList tl = {1, 0, tokens};
	const uint64_t answer = call(kSchedule, &cl, &tl);
	*token = tokens[0];
	return answer;
}

// An HEVC encode is sent with the settings the source and the plug-in's choices give, and
// published where the plug-in reads it as it is queried done:
// - the settings: the source's size, Main of NV12 for a '420v' or '420f' source, its range, the
//   plug-in's slice QP as a constant QP, no keyframes of the host's own; an IRAP NAL unit type
//   (BLA, IDR, CRA) forces a keyframe, another does not; the frames count up; the access unit
//   takes at most half the bitstream pool's buffer;
// - the access unit lands at the start of the buffer, each NAL unit after a 4-byte start code: a
//   3-byte start code made 4, trailing zero bytes left out, and the NAL units the plug-in drops
//   (an access unit delimiter, SEI) left out; its length is published as the words at +0x3a0 (0)
//   and +0x3a8;
// - a frame the host failed, an access unit of no slice segment, or one past the buffer, publishes
//   a length of 0 and fails the context's next schedule, sending nothing; the schedule after it
//   goes on;
// - an 'x420' source is sent as Main 10 of P010 and an 'xf20' one as Main 10 at full range; a
//   source of another format (BGRA) is not sent.
void checkHevcEncodeFrames()
{
	g_hostEncodes = true;
	g_encodeRates = 1u << RGFX_VIDEO_RATE_CQP;
	g_encodes.clear();
	g_encodeWork.clear();
	constexpr uint32_t kBuffer = 0x1000;
	const uint64_t bufferAt = allocate(g_driver, 0, kBuffer);
	CHECK(bufferAt != 0, "the bitstream buffer refused");
	const uint8_t *buffer = (const uint8_t *)bufferAt;
	IOSurfaceRef source = canvas(640, 480, '420v');
	void *context = nullptr;
	CHECK(makeEncode(&context, 640, 480, kEncodeHevc) == kOk, "a create of one HEVC encode");
	// An IDR access unit as a host writes one: an access unit delimiter, then VPS, SPS and PPS,
	// an SEI and the IDR slice; the SPS after a 3-byte start code, trailing zero bytes at the end.
	const std::vector<uint8_t> aud = {0x46, 0x01, 0x10}, vps = {0x40, 0x01, 0x0c, 0x01, 0xff},
	                           sps = {0x42, 0x01, 0x01, 0x01, 0x60, 0x03},
	                           pps = {0x44, 0x01, 0xc1, 0x72}, sei = {0x4e, 0x01, 0x05, 0x02, 0x80},
	                           idr = {0x28, 0x01, 0xaf, 0x13, 0x77, 0x03, 0x11},
	                           trail = {0x02, 0x01, 0xd0, 0x2a, 0x05};
	auto unit = [](std::vector<uint8_t> &au, const std::vector<uint8_t> &nal, bool four) {
		const uint8_t code[4] = {0, 0, 0, 1};
		au.insert(au.end(), four ? code : code + 1, code + 4);
		au.insert(au.end(), nal.begin(), nal.end());
	};
	g_accessUnit.clear();
	unit(g_accessUnit, aud, true);
	unit(g_accessUnit, vps, true);
	unit(g_accessUnit, sps, false);
	unit(g_accessUnit, pps, true);
	unit(g_accessUnit, sei, true);
	unit(g_accessUnit, idr, true);
	g_accessUnit.insert(g_accessUnit.end(), {0, 0});
	std::vector<uint8_t> want;
	for (const auto *nal : {&vps, &sps, &pps, &idr})
		unit(want, *nal, true);
	void *token = nullptr;
	HevcEncode first(source, 20, 24, 0);
	CHECK(scheduleHevcEncode(context, first, &token) == kOk && token, "an IDR frame not scheduled");
	CHECK(g_encodes.size() == 1, "%zu encodes sent", g_encodes.size());
	const SentEncode &sent = g_encodes[0];
	const RgfxVideoEncodeSettings &st = sent.settings;
	CHECK(sent.surface == IOSurfaceGetID(source) && sent.frameIndex == 0 && sent.forceKeyframe &&
	          sent.capacity == kBuffer / 2,
	      "the IDR frame's encode: surface %u, frame %llu, keyframe %d, capacity %llu", sent.surface,
	      (unsigned long long)sent.frameIndex, sent.forceKeyframe,
	      (unsigned long long)sent.capacity);
	CHECK(st.width == 640 && st.height == 480 && st.format == RGFX_VIDEO_FORMAT_NV12 &&
	          st.profile == RGFX_VIDEO_HEVC_MAIN && st.level == 0 &&
	          st.rate_control == RGFX_VIDEO_RATE_CQP && st.qp_i == 24 && st.qp_p == 24 &&
	          st.qp_b == 24 && st.key_frame_interval == UINT32_MAX && st.full_range == 0,
	      "the IDR frame's settings: profile %#x level %u rate %u qp %u keys %u range %u",
	      st.profile, st.level, st.rate_control, st.qp_i, st.key_frame_interval, st.full_range);
	CHECK(queryEncode(token) == kQueryDone, "the IDR frame not done");
	CHECK(first.begin() == 0 && first.size() == want.size() &&
	          std::memcmp(buffer, want.data(), want.size()) == 0,
	      "the IDR frame's access unit: %llu bytes from %llu", (unsigned long long)first.size(),
	      (unsigned long long)first.begin());
	// A TRAIL_R frame of another QP, after a 3-byte start code.
	g_accessUnit.clear();
	unit(g_accessUnit, trail, false);
	HevcEncode p(source, 1, 31, 1);
	CHECK(scheduleHevcEncode(context, p, &token) == kOk, "a P frame not scheduled");
	CHECK(g_encodes.size() == 2 && g_encodes[1].frameIndex == 1 && !g_encodes[1].forceKeyframe &&
	          g_encodes[1].settings.qp_p == 31,
	      "the P frame's encode");
	CHECK(queryEncode(token) == kQueryDone && p.size() == trail.size() + 4 && buffer[3] == 1 &&
	          std::memcmp(buffer + 4, trail.data(), trail.size()) == 0,
	      "the P frame's access unit: %llu bytes", (unsigned long long)p.size());
	// A CRA picture is a keyframe; a QP past 51 is sent as 51.
	HevcEncode cra(source, 21, 60, 2);
	CHECK(scheduleHevcEncode(context, cra, &token) == kOk && g_encodes.back().forceKeyframe &&
	          g_encodes.back().settings.qp_i == 51,
	      "a CRA frame's encode");
	CHECK(queryEncode(token) == kQueryDone, "the CRA frame not done");
	// A frame the host failed publishes no access unit, and fails the next schedule.
	g_encodeStatus = RGFX_VIDEO_ENCODE_FAILED;
	HevcEncode failed(source, 1, 31, 3);
	CHECK(scheduleHevcEncode(context, failed, &token) == kOk, "the failing frame not scheduled");
	g_encodeStatus = RGFX_VIDEO_ENCODE_OK;
	CHECK(queryEncode(token) == kQueryDone && failed.size() == 0, "the failed frame published");
	const size_t before = g_encodes.size();
	HevcEncode after(source, 1, 31, 4);
	CHECK(scheduleHevcEncode(context, after, &token) == kNotSent && g_encodes.size() == before,
	      "the frame after a failed one sent");
	CHECK(scheduleHevcEncode(context, after, &token) == kOk, "the next frame not scheduled");
	CHECK(queryEncode(token) == kQueryDone && after.size() == trail.size() + 4,
	      "the next frame not published");
	// An access unit of no slice segment, and one past the buffer, publish none.
	g_accessUnit.clear();
	unit(g_accessUnit, sei, true);
	HevcEncode noSlice(source, 1, 31, 5);
	CHECK(scheduleHevcEncode(context, noSlice, &token) == kOk, "the frame of no slice not scheduled");
	CHECK(queryEncode(token) == kQueryDone && noSlice.size() == 0,
	      "an access unit of no slice published");
	CHECK(scheduleHevcEncode(context, after, &token) == kNotSent,
	      "the frame after one of no slice sent");
	g_accessUnit.clear();
	unit(g_accessUnit, std::vector<uint8_t>(kBuffer, 0x26), true);
	HevcEncode past(source, 1, 31, 6);
	CHECK(scheduleHevcEncode(context, past, &token) == kOk, "the frame past the buffer not scheduled");
	CHECK(queryEncode(token) == kQueryDone && past.size() == 0,
	      "an access unit past the buffer published");
	CHECK(scheduleHevcEncode(context, after, &token) == kNotSent,
	      "the frame after one past the buffer sent");
	// A full-range source is sent of its range, a 10-bit one as Main 10 of P010; a BGRA one is
	// not sent.
	g_accessUnit.clear();
	unit(g_accessUnit, trail, true);
	IOSurfaceRef full = canvas(640, 480, '420f'), deep = canvas(640, 480, 'x420'),
	             deepFull = canvas(640, 480, 'xf20'), bgra = canvas(640, 480, 'BGRA');
	const struct {
		IOSurfaceRef source;
		uint32_t profile, format, fullRange;
	} sources[] = {
	    {full, RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_FORMAT_NV12, 1},
	    {deep, RGFX_VIDEO_HEVC_MAIN_10, RGFX_VIDEO_FORMAT_P010, 0},
	    {deepFull, RGFX_VIDEO_HEVC_MAIN_10, RGFX_VIDEO_FORMAT_P010, 1},
	};
	for (const auto &c : sources) {
		HevcEncode frame(c.source, 1, 31, 7);
		CHECK(scheduleHevcEncode(context, frame, &token) == kOk, "a frame of format %#x not sent",
		      (unsigned)IOSurfaceGetPixelFormat(c.source));
		const RgfxVideoEncodeSettings &fs = g_encodes.back().settings;
		CHECK(fs.profile == c.profile && fs.format == c.format && fs.full_range == c.fullRange &&
		          fs.qp_p == 31,
		      "a frame of format %#x: profile %#x format %u range %u",
		      (unsigned)IOSurfaceGetPixelFormat(c.source), fs.profile, fs.format, fs.full_range);
		CHECK(queryEncode(token) == kQueryDone, "a frame of format %#x not done",
		      (unsigned)IOSurfaceGetPixelFormat(c.source));
	}
	const size_t sentBefore = g_encodes.size();
	HevcEncode refused(bgra, 1, 31, 8);
	CHECK(scheduleHevcEncode(context, refused, &token) == kNotSent &&
	          g_encodes.size() == sentBefore,
	      "a BGRA frame sent");
	ContextDelete del = {1, 0, &context};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	for (IOSurfaceRef s : {source, full, deep, deepFull, bgra})
		CFRelease(s);
	g_accessUnit.clear();
	g_encodeRates = kCqpVbr;
}

// The plug-in's frame coded again (hevcCodesAgain): a command of the last frame's order count and
// source, of a P picture, is sent in place of it (RGFX_VIDEO_ENCODE_IN_PLACE), with its frame
// index, at its own QP. Not in place: a new order count from the same surface, the same order
// count from another surface, and a keyframe, which is sent as a new frame: an IDR picture coded
// again right after itself, and each of keyframes from one reused surface, of order count 0 each.
// Both of two retries in a row are in place, and so is the retry of the P picture right after an
// IDR picture. After a frame the host failed, the next frame sent is a new one, whatever its order
// count and surface.
void checkHevcCodesAgain()
{
	g_hostEncodes = true;
	g_encodeRates = 1u << RGFX_VIDEO_RATE_CQP;
	g_encodes.clear();
	g_encodeWork.clear();
	CHECK(allocate(g_driver, 0, 0x1000) != 0, "the bitstream buffer refused");
	IOSurfaceRef a = canvas(640, 480, '420v'), b = canvas(640, 480, '420v');
	void *context = nullptr;
	CHECK(makeEncode(&context, 640, 480, kEncodeHevc) == kOk, "a create of one HEVC encode");
	const uint8_t trail[] = {0, 0, 0, 1, 0x02, 0x01, 0xd0, 0x2a, 0x05};
	g_accessUnit.assign(trail, trail + sizeof trail);
	constexpr uint32_t kIdr = 19, kTrail = 1;
	// Sends a frame and has it done: the encode sent for it.
	auto send = [&](IOSurfaceRef source, uint32_t nal, uint32_t qp, uint32_t poc) {
		HevcEncode e(source, nal, qp, poc);
		void *token = nullptr;
		CHECK(scheduleHevcEncode(context, e, &token) == kOk && token,
		      "a frame of order count %u not scheduled", poc);
		CHECK(queryEncode(token) == kQueryDone && e.size() == sizeof trail,
		      "a frame of order count %u not published", poc);
		return g_encodes.back();
	};
	auto is = [&](const SentEncode &e, bool inPlace, uint64_t index, bool key, uint32_t qp,
	              const char *what) {
		CHECK(e.inPlace == inPlace && e.frameIndex == index && e.forceKeyframe == key &&
		          e.settings.qp_i == qp,
		      "%s: in place %d, frame %llu, keyframe %d, QP %u", what, e.inPlace,
		      (unsigned long long)e.frameIndex, e.forceKeyframe, e.settings.qp_i);
	};
	// An IDR picture, and the same again at a higher QP right after it: a new IDR picture.
	is(send(a, kIdr, 30, 0), false, 0, true, 30, "the IDR picture");
	is(send(a, kIdr, 33, 0), false, 1, true, 33, "the IDR picture again");
	// The P picture after it, and that again: in place, of its index.
	is(send(a, kTrail, 30, 1), false, 2, false, 30, "the P picture after the IDR picture");
	is(send(a, kTrail, 33, 1), true, 2, false, 33, "the P picture after the IDR picture again");
	// A new order count from the same surface: a new frame.
	is(send(a, kTrail, 30, 2), false, 3, false, 30, "a new order count from the same surface");
	// That frame again twice: in place both times, of its index.
	is(send(a, kTrail, 33, 2), true, 3, false, 33, "a P picture again");
	is(send(a, kTrail, 36, 2), true, 3, false, 36, "a P picture again, twice");
	// The same order count from another surface: a new frame.
	is(send(b, kTrail, 30, 2), false, 4, false, 30, "the same order count from another surface");
	is(send(b, kTrail, 30, 3), false, 5, false, 30, "the next P picture");
	// Keyframes from one reused surface, each of order count 0: each a new frame.
	for (uint64_t n = 6; n < 9; n++)
		is(send(a, kIdr, 30, 0), false, n, true, 30, "a keyframe from one surface");
	// A frame the host fails, then the same frame: its schedule fails, and the one after it is a
	// new frame.
	g_encodeStatus = RGFX_VIDEO_ENCODE_FAILED;
	HevcEncode failed(a, kTrail, 30, 1);
	void *token = nullptr;
	CHECK(scheduleHevcEncode(context, failed, &token) == kOk, "the failing frame not scheduled");
	g_encodeStatus = RGFX_VIDEO_ENCODE_OK;
	CHECK(queryEncode(token) == kQueryDone && failed.size() == 0, "the failed frame published");
	HevcEncode again(a, kTrail, 33, 1);
	CHECK(scheduleHevcEncode(context, again, &token) == kNotSent, "the frame after a failed one sent");
	is(send(a, kTrail, 33, 1), false, 10, false, 33, "the frame after a failed one's refusal");
	ContextDelete del = {1, 0, &context};
	CHECK(call(kDeleteContexts, &del) == kOk, "DeleteContexts");
	CHECK(callOn(g_driver, kFreeBitstreamBufferPool, (const void *)0) == kOk,
	      "FreeBitstreamBufferPool");
	CFRelease(a);
	CFRelease(b);
	g_accessUnit.clear();
	g_encodeRates = kCqpVbr;
}

// Runs a check, naming it first where VIDEO_TESTS_TRACE is set, so a crash names its check.
#define RUN(check)                                       \
	do {                                             \
		if (std::getenv("VIDEO_TESTS_TRACE"))    \
			std::fprintf(stderr, "%s\n", #check); \
		check();                                 \
	} while (0)

int main(int argc, char **argv)
{
	if (argc == 8 && std::strcmp(argv[1], "listed") == 0)
		return listedChild(argv);
	g_self = argv[0];
	RUN(checkEncodeListedApart);
	void *table = nullptr;
	uint8_t params[64] = {}, identity[64] = {};
	CHECK(AVD_CreateAVDAccelerator(&table, params, identity) == 0 && table, "the driver's table");
	g_table = (DriverTable *)table;
	g_driver = createDriver();
	RUN(checkPoolsPerDriver);
	RUN(checkFailedCreate);
	RUN(checkDecodeSizeAtCreate);
	RUN(checkDelete);
	RUN(checkUnsentDecode);
	RUN(checkTileLanding);
	RUN(checkHostBounds);
	RUN(checkHostVerdict);
	RUN(checkContextsScheduleApart);
	RUN(checkFrameDescriptions);
	RUN(checkH264SpsLayouts);
	RUN(checkNoZeroAnswers);
	RUN(checkTileInfo);
	RUN(checkTileInfoRefused);
	RUN(checkLayoutFromHost);
	RUN(checkLayoutInPlace);
	RUN(checkEncodeListed);
	RUN(checkEncodeCreate);
	RUN(checkPixelPools);
	RUN(checkEncodeFrames);
	RUN(checkHevcEncodeFrames);
	RUN(checkHevcCodesAgain);
	std::printf("video_driver_tests: %d checks passed\n", g_checks);
	return 0;
}
