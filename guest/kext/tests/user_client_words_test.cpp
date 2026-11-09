// What the kext's user clients answer, word for word, built natively: the display pipe's VBL and
// completion notifications and the completions' count (RgfxPipeNotifications.hpp), the replies to the IOGPU base's device
// calls and which calls it answers (RgfxIOGPUReplies.hpp), and the user-client protocol the
// bundle shares (RgfxUserClientABI.hpp): its type, its selectors, each a number every WebKit
// web-content sandbox allows, the bundle's calls of them, and the features call with the kext's
// own feature bits; and the superpage record the kext keeps for every process
// (RgfxSuperpageRecord.hpp).
#include <cctype>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <string>

#include "RgfxIOGPUReplies.hpp"
#include "RgfxPipeNotifications.hpp"
#include "RgfxSuperpageRecord.hpp"
#include "RgfxUserClientABI.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

static uint64_t rd64(const uint8_t *p)
{
	uint64_t v = 0;
	for (int i = 7; i >= 0; i--)
		v = (v << 8) | p[i];
	return v;
}

// A VBL answer: the registration's callback and refcon, then the two times split low half first.
static void vbl()
{
	uint64_t w[kRgfxPipeVblWords + 1];
	for (uint64_t &x : w)
		x = 0xa5a5a5a5a5a5a5a5;
	rgfxPipeVblWords(w, 0x7ff812345678, 0x600000aabbcc, 0x0000001234567890, 0x00000012abcdef01);
	CHECK(kRgfxPipeVblWords == 6);
	CHECK(w[0] == 0x7ff812345678);
	CHECK(w[1] == 0x600000aabbcc);
	CHECK(w[2] == 0x34567890 && w[3] == 0x12);
	CHECK(w[4] == 0xabcdef01 && w[5] == 0x12);
	CHECK(w[6] == 0xa5a5a5a5a5a5a5a5); // nothing past the answer
}

// A completion of each count: every one of its words written, the cookie, the id in word 1's
// high half, the time and, from nine words, the action, the rest 0, and nothing past its count.
static void completion()
{
	CHECK(kRgfxPipeCompletionWordsMax == 11);
	const uint32_t counts[] = {8, 9, 11};
	for (uint32_t count : counts) {
		uint64_t w[kRgfxPipeCompletionWordsMax + 1];
		for (uint64_t &x : w)
			x = 0xa5a5a5a5a5a5a5a5;
		rgfxPipeCompletionWords(w, count, 0x6000002de3a0, 4461, 0x0000001234567890);
		CHECK(w[0] == 0x6000002de3a0);
		CHECK(w[1] == (uint64_t(4461) << 32));
		CHECK(w[3] == 0x0000001234567890);
		const uint32_t zero[] = {2, 4, 5, 6, 7, 9, 10};
		for (uint32_t i : zero)
			if (i < count)
				CHECK(w[i] == 0);
		if (count > 8)
			CHECK(w[8] == 1);
		for (uint32_t i = count; i <= kRgfxPipeCompletionWordsMax; i++)
			CHECK(w[i] == 0xa5a5a5a5a5a5a5a5);
	}
}

// The count a client's completions take: 8 after a 0x108-byte transaction (macOS 11), and after a
// 0x118-byte one, 11 where the family exports the assigned length (macOS 13 on) and 9 where it
// does not (macOS 12).
static void completionCount()
{
	CHECK(kRgfxPipeTransactionBytes == 0x118 && kRgfxPipeTransactionBytesShort == 0x108);
	CHECK(rgfxPipeCompletionCount(0x108, false) == 8);
	CHECK(rgfxPipeCompletionCount(0x108, true) == 8);
	CHECK(rgfxPipeCompletionCount(0x118, false) == 9);
	CHECK(rgfxPipeCompletionCount(0x118, true) == 11);
}

// The IOGPU base's two device calls are answered at their exact shapes only, and the replies
// are their sizes: the configuration's dedicated memory and the region's size, every other byte 0.
static void iogpu()
{
	CHECK(rgfxIOGPUDeviceReplySize(0, 0, 0, 0, 64) == 64);
	CHECK(rgfxIOGPUDeviceReplySize(2, 0, 0, 0, 536) == 536);
	CHECK(rgfxIOGPUDeviceReplySize(0, 0, 0, 0, 536) == 0);
	CHECK(rgfxIOGPUDeviceReplySize(2, 0, 0, 0, 64) == 0);
	CHECK(rgfxIOGPUDeviceReplySize(1, 0, 0, 0, 64) == 0);
	CHECK(rgfxIOGPUDeviceReplySize(0, 1, 0, 0, 64) == 0);
	CHECK(rgfxIOGPUDeviceReplySize(0, 0, 1, 0, 64) == 0);
	CHECK(rgfxIOGPUDeviceReplySize(0, 0, 0, 8, 64) == 0);

	uint8_t reply[kRgfxIOGPUDeviceInfoSize + 1];
	std::memset(reply, 0xa5, sizeof(reply));
	rgfxIOGPUDeviceReply(kRgfxIOGPUSelConfig, reply, 17094934528ull, 1ull << 30);
	CHECK(rd64(reply + 0x10) == 17094934528ull);
	CHECK(rd64(reply + 0x20) == 1ull << 30);
	bool rest = true;
	for (uint32_t i = 0; i < kRgfxIOGPUConfigSize; i++)
		if ((i < 0x10 || i >= 0x18) && (i < 0x20 || i >= 0x28))
			rest = rest && reply[i] == 0;
	CHECK(rest);
	CHECK(reply[kRgfxIOGPUConfigSize] == 0xa5); // nothing past the configuration

	std::memset(reply, 0xa5, sizeof(reply));
	rgfxIOGPUDeviceReply(kRgfxIOGPUSelDeviceInfo, reply, 17094934528ull, 1ull << 30);
	bool zero = true;
	for (uint32_t i = 0; i < kRgfxIOGPUDeviceInfoSize; i++)
		zero = zero && reply[i] == 0;
	CHECK(zero);
	CHECK(reply[kRgfxIOGPUDeviceInfoSize] == 0xa5);
}

// Every selector of ours, by its name in RgfxUserClientABI.hpp, and the client type it is
// called on: the bundle names them so, and the source check below finds them by name.
struct Selector {
	const char *name;
	unsigned number;
	bool port; // on a shared event's port ('RGSE'), else on the accelerator's 'RGFX' client
};
static const Selector kSelectors[] = {
	{"kRgfxAccelMethodDoorbell", kRgfxAccelMethodDoorbell, false},
	{"kRgfxAccelMethodFeatures", kRgfxAccelMethodFeatures, false},
	{"kRgfxAccelMethodPagesBlock", kRgfxAccelMethodPagesBlock, false},
	{"kRgfxAccelMethodTakeAnswers", kRgfxAccelMethodTakeAnswers, false},
	{"kRgfxAccelMethodSurfaceBlockRelease", kRgfxAccelMethodSurfaceBlockRelease, false},
	{"kRgfxAccelMethodSurfaceRetire", kRgfxAccelMethodSurfaceRetire, false},
	{"kRgfxAccelMethodSuperpages", kRgfxAccelMethodSuperpages, false},
	{"kRgfxAccelMethodOpenContext", kRgfxAccelMethodOpenContext, false},
	{"kRgfxAccelMethodAlloc", kRgfxAccelMethodAlloc, false},
	{"kRgfxAccelMethodFree", kRgfxAccelMethodFree, false},
	{"kRgfxAccelMethodQueueCreate", kRgfxAccelMethodQueueCreate, false},
	{"kRgfxAccelMethodQueueDestroy", kRgfxAccelMethodQueueDestroy, false},
	{"kRgfxAccelMethodSubmit", kRgfxAccelMethodSubmit, false},
	{"kRgfxAccelMethodResourceIds", kRgfxAccelMethodResourceIds, false},
	{"kRgfxAccelMethodWaitCompletions", kRgfxAccelMethodWaitCompletions, false},
	{"kRgfxAccelMethodSurfaceScanout", kRgfxAccelMethodSurfaceScanout, false},
	{"kRgfxAccelMethodSurfaceBlock", kRgfxAccelMethodSurfaceBlock, false},
	{"kRgfxAccelMethodDeviceInfo", kRgfxAccelMethodDeviceInfo, false},
	{"kRgfxAccelMethodSurfaceWrites", kRgfxAccelMethodSurfaceWrites, false},
	{"kRgfxSharedEventMethodExport", kRgfxSharedEventMethodExport, true},
	{"kRgfxSharedEventMethodImport", kRgfxSharedEventMethodImport, true},
};
static const unsigned kSelectorCount = sizeof(kSelectors) / sizeof(kSelectors[0]);
// The numbers kept for calls not yet carried out: Consumed, SubmitBatch and WakeCompletions.
static const unsigned kKept[] = {1, 5, 6};

// The external-method numbers every WebKit web-content sandbox lets its process call on an
// IOAccelerator connection, on every release from macOS 11 to 26. WebKit's web-content profile
// (com.apple.WebProcess.sb, its IOAcceleratorMessageFilter) lists numbers on macOS 12 alone, and
// refuses any other there before the kext sees it: these are the ones every list from Safari 15.1
// (WebKit 612.2) to 17.6 (WebKit 7618.3) holds, read from WebKit's source at each release branch
// and on a macOS 12.7.6 guest (Safari 17.6's, which adds 41, 43, 46, 49, 266, 269 and 513). On
// macOS 11 (Safari 14 to 16) and from 13 on it lists none, allowing every number; from Safari 18.4
// (WebKit 7622) on, no release lists any. Its asynchronous methods (0 and 47) and traps the bundle
// makes none of.
static bool sandboxAllows(unsigned n)
{
	static const unsigned kAllowed[] = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,
	                                    11,  12,  13,  15,  16,  17,  18,  22,  23,  24,  28,
	                                    29,  30,  31,  40,  42,  48,  256, 257, 258, 259, 260,
	                                    261, 262, 263, 264, 265, 267, 268, 512};
	for (unsigned a : kAllowed)
		if (a == n)
			return true;
	return false;
}

// The protocol both sides build from: 'RGFX'; every selector, and every number kept, in the
// sandbox's numbers; no two of a client's selectors, or of the numbers kept, alike; and the kext's
// routing: every selector of 'RGFX' but the doorbell goes to its context methods, and no number
// kept, no port selector and no number of the map before this one (0x10a, 0x10d to 0x114, which
// a bundle of that map may still call) does.
static void protocol()
{
	CHECK(kRgfxAccelUserClientType == 0x52474658);
	CHECK(kRgfxAccelMethodDoorbell == 0);
	CHECK(kRgfxSelectorMap == 1);
	for (unsigned i = 0; i < kSelectorCount; i++) {
		const Selector &s = kSelectors[i];
		if (!sandboxAllows(s.number))
			std::fprintf(stderr, "%s, %u, is not a number the sandbox allows\n", s.name,
			             s.number);
		CHECK(sandboxAllows(s.number));
		CHECK(rgfxAccelContextSelector(s.number) ==
		      (!s.port && s.number != kRgfxAccelMethodDoorbell));
		for (unsigned j = 0; j < i; j++)
			CHECK(kSelectors[j].number != s.number);
		for (unsigned k : kKept)
			CHECK(k != s.number);
	}
	for (unsigned k : kKept) {
		CHECK(sandboxAllows(k));
		CHECK(!rgfxAccelContextSelector(k));
	}
	CHECK(!rgfxAccelContextSelector(2));
	for (unsigned n : {0x10au, 0x10du, 0x10eu, 0x10fu, 0x110u, 0x111u, 0x112u, 0x113u, 0x114u})
		CHECK(!rgfxAccelContextSelector(n));
	CHECK(kRgfxSurfaceCpuGenOffset == 16);
	CHECK(kRgfxSurfaceClientsOffset == 12);
}

// Every IOKit call the bundle's sources make names a selector of the table above, so that each
// lies in the sandbox's numbers: the second argument of each IOConnectCall*Method, by name. A
// literal number, or a name the table lacks, fails here. The bundle makes no asynchronous call and
// no trap. Run from guest/kext, as `make test` runs it.
static std::string secondArgument(const std::string &text, size_t open)
{
	int depth = 0;
	size_t i = open, from = std::string::npos;
	for (; i < text.size(); i++) {
		const char c = text[i];
		if (c == '(' || c == '[' || c == '{')
			depth++;
		else if (c == ')' || c == ']' || c == '}') {
			if (--depth == 0)
				break;
		} else if (c == ',' && depth == 1) {
			if (from != std::string::npos)
				break;
			from = i + 1;
		}
	}
	if (from == std::string::npos)
		return "";
	std::string arg = text.substr(from, i - from);
	const size_t a = arg.find_first_not_of(" \t\n");
	const size_t b = arg.find_last_not_of(" \t\n");
	return a == std::string::npos ? "" : arg.substr(a, b - a + 1);
}

static void bundleCalls()
{
	const char *dir = "../bundle/src";
	DIR *d = opendir(dir);
	CHECK(d != nullptr);
	if (!d)
		return;
	unsigned calls = 0;
	while (struct dirent *e = readdir(d)) {
		const std::string name = e->d_name;
		const size_t dot = name.rfind('.');
		const std::string ext = dot == std::string::npos ? "" : name.substr(dot);
		if (ext != ".mm" && ext != ".cpp" && ext != ".hpp" && ext != ".h" && ext != ".m")
			continue;
		std::ifstream in(std::string(dir) + "/" + name);
		const std::string text((std::istreambuf_iterator<char>(in)),
		                       std::istreambuf_iterator<char>());
		for (const char *banned : {"IOConnectCallAsync", "IOConnectTrap"}) {
			if (text.find(banned) != std::string::npos)
				std::fprintf(stderr, "%s: calls %s\n", name.c_str(), banned);
			CHECK(text.find(banned) == std::string::npos);
		}
		for (size_t at = text.find("IOConnectCall"); at != std::string::npos;
		     at = text.find("IOConnectCall", at + 1)) {
			size_t end = at;
			while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) ||
			                             text[end] == '_'))
				end++;
			const std::string fn = text.substr(at, end - at);
			const size_t open = text.find_first_not_of(" \t\n", end);
			if (fn.size() < 6 || fn.compare(fn.size() - 6, 6, "Method") != 0 ||
			    open == std::string::npos || text[open] != '(')
				continue;
			const std::string selector = secondArgument(text, open);
			bool known = false;
			for (const Selector &s : kSelectors)
				known = known || selector == s.name;
			if (!known)
				std::fprintf(stderr, "%s: %s calls selector '%s', which is not one of ours\n",
				             name.c_str(), fn.c_str(), selector.c_str());
			CHECK(known);
			calls++;
		}
	}
	closedir(d);
	// The device's setup, its blocks, queues, submits and shared events: tens of calls.
	CHECK(calls >= 20);
}

// The features call and the kext's own feature bits, as design.md's registry gives them out:
// Features at 3, answered by every map at that number, and one bit for each call or flag a kext
// may lack, none shared.
static void features()
{
	CHECK(kRgfxAccelMethodFeatures == 3);
	CHECK(kRgfxAccelMethodPagesBlock == 4);
	const uint64_t bits[] = {
		kRgfxKextFeaturePagesBlock,
		kRgfxKextFeatureSubmitBatch,
		kRgfxKextFeatureWakeCompletions,
		kRgfxKextFeatureConsumedDoorbell,
		kRgfxKextFeatureDeferredAnswers,
		kRgfxKextFeatureContextSpan,
		kRgfxKextFeatureSharedEvents,
		kRgfxKextFeatureWriteCombined,
		kRgfxKextFeatureSuperpages,
	};
	for (unsigned i = 0; i < sizeof(bits) / sizeof(bits[0]); i++)
		CHECK(bits[i] == uint64_t(1) << i);
	// TakeAnswers past the numbers SubmitBatch and WakeCompletions keep, and Submit's one flag.
	CHECK(kRgfxAccelMethodTakeAnswers == 7);
	CHECK(kRgfxSubmitDeferred == 1);
	CHECK(kRgfxAccelMethodSurfaceBlockRelease == 8);
	CHECK(kRgfxAccelMethodSurfaceRetire == 9);
	// A shared event's port: 'RGSE', its two selectors past the accelerator's low ones, so a call
	// on a connection of another type finds no selector of the accelerator's of that number.
	CHECK(kRgfxSharedEventUserClientType == 0x52475345);
	CHECK(kRgfxSharedEventMethodExport == 10);
	CHECK(kRgfxSharedEventMethodImport == 11);
	// The superpage record past the port's numbers, and its notes.
	CHECK(kRgfxAccelMethodSuperpages == 12);
	CHECK(kRgfxSuperpagesAsk == 0 && kRgfxSuperpagesRefused == 1 && kRgfxSuperpagesGivenBack == 2);
}

// The guest's superpage record (RgfxSuperpageRecord.hpp): open at the kext's start; a refusal
// closes it for every asker, an ask leaves it as it is, and a slab of superpages given back opens
// it again; a note that is none of its own is refused, the record unchanged.
static void superpageRecord()
{
	bool none = false;
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesAsk) == 1);
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesRefused) == 0 && none);
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesAsk) == 0);
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesAsk) == 0);
	CHECK(rgfxSuperpagesNote(&none, 3) == -1 && none);
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesGivenBack) == 1 && !none);
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesAsk) == 1);
	CHECK(rgfxSuperpagesNote(&none, ~0ull) == -1 && !none);
	CHECK(rgfxSuperpagesNote(&none, kRgfxSuperpagesGivenBack) == 1 && !none);
}

int main()
{
	vbl();
	completion();
	completionCount();
	iogpu();
	protocol();
	bundleCalls();
	features();
	superpageRecord();
	if (g_failed) {
		std::fprintf(stderr, "user_client_words_test: %d of %d checks failed\n", g_failed,
		             g_checks);
		return 1;
	}
	std::printf("user_client_words_test: OK (%d checks)\n", g_checks);
	return 0;
}
