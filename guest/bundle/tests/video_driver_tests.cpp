/*
 * video_driver_tests: the video driver's entry points (src/RgfxVideoDriver.cpp) as AppleGVA calls
 * them, run on the Mac (guest/bundle/Makefile `test`), over a stand-in for the host's sessions
 * (src/RgfxVideoChannel.hpp) that counts what the driver opens, closes and forgets. It checks that
 * a context the driver deletes, by DeleteContexts or by a CreateContexts that fails part way,
 * takes its session with it, closed and out of the live sessions, and that a decode the driver
 * cannot send fails its schedule.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the first failure.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "RgfxVideoChannel.hpp"

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

constexpr uint32_t kFunctionDecode = 1;
constexpr uint64_t kCodecH264 = 0x8;
constexpr uint64_t kOk = 0, kUnsupported = 1, kNotSent = 5;
enum { kCreateContexts = 2, kDeleteContexts = 3, kFreePixelBufferPool = 9, kSchedule = 10 };

// The stand-in's sessions: each one opened, whether it was closed, and how often its outputs
// were forgotten; and how many opens in a row succeed before one fails (-1: all).
struct FakeSession {
	bool closed = false;
	int forgotten = 0;
};
std::vector<FakeSession *> g_opened;
int g_opensBeforeFailure = -1;

} // namespace

namespace rgfx::video {

struct Session : FakeSession {};

const RgfxVideoCodecCaps *hostCodec(uint32_t, uint32_t) { return nullptr; }

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

void *decode(Session *, const Decode &) { return nullptr; }
int tokenStatus(void *) { return 1; }
void releaseToken(void *) {}

} // namespace rgfx::video

namespace {

DriverTable *g_table;

uint64_t call(int entry, const void *record, const void *out = nullptr)
{
	return g_table->entry[entry]((uint64_t)g_table->driver, (uint64_t)record, (uint64_t)out);
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

// Makes the contexts of `n` H.264 decodes: CreateContexts' answer, the contexts in `contexts`.
uint64_t makeContexts(uint32_t n, std::vector<void *> &contexts)
{
	static std::vector<FunctionInfo> infos;
	static std::vector<const FunctionInfo *> list;
	infos.assign(n, decodeH264());
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

} // namespace

int main()
{
	void *table = nullptr;
	uint8_t params[64] = {}, identity[64] = {};
	CHECK(AVD_CreateAVDAccelerator(&table, params, identity) == 0 && table, "the driver's table");
	g_table = (DriverTable *)table;
	checkFailedCreate();
	checkDelete();
	checkUnsentDecode();
	std::printf("video_driver_tests: %d checks passed\n", g_checks);
	return 0;
}
