// What the kext's user clients answer, word for word, built natively: the display pipe's VBL and
// completion notifications (RgfxPipeNotifications.hpp), the replies to the IOGPU base's device
// calls and which calls it answers (RgfxIOGPUReplies.hpp), and the user-client protocol the
// bundle shares (RgfxUserClientABI.hpp): its type, its context selectors without a gap, and the
// features call with the kext's own feature bits.
#include <cstdio>
#include <cstring>

#include "RgfxIOGPUReplies.hpp"
#include "RgfxPipeNotifications.hpp"
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

// A completion: every one of its words written, the cookie, the id in word 1's high half, the
// time and the action, the rest 0.
static void completion()
{
	uint64_t w[kRgfxPipeCompletionWords + 1];
	for (uint64_t &x : w)
		x = 0xa5a5a5a5a5a5a5a5;
	rgfxPipeCompletionWords(w, 0x6000002de3a0, 4461, 0x0000001234567890);
	CHECK(kRgfxPipeCompletionWords == 11);
	CHECK(w[0] == 0x6000002de3a0);
	CHECK(w[1] == (uint64_t(4461) << 32));
	CHECK(w[3] == 0x0000001234567890);
	CHECK(w[8] == 1);
	const int zero[] = {2, 4, 5, 6, 7, 9, 10};
	for (int i : zero)
		CHECK(w[i] == 0);
	CHECK(w[11] == 0xa5a5a5a5a5a5a5a5);
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

// The protocol both sides build from: 'RGFX', the doorbell below the context selectors, and
// those from the first to the last with no gap, the range the kext routes to its context.
static void protocol()
{
	CHECK(kRgfxAccelUserClientType == 0x52474658);
	CHECK(kRgfxAccelMethodDoorbell == 0);
	const unsigned context[] = {
		kRgfxAccelMethodOpenContext,
		kRgfxAccelMethodAlloc,
		kRgfxAccelMethodFree,
		kRgfxAccelMethodQueueCreate,
		kRgfxAccelMethodQueueDestroy,
		kRgfxAccelMethodSubmit,
		kRgfxAccelMethodResourceIds,
		kRgfxAccelMethodWaitCompletions,
		kRgfxAccelMethodSurfaceScanout,
		kRgfxAccelMethodSurfaceBlock,
		kRgfxAccelMethodSurfaceBlockRelease,
		kRgfxAccelMethodDeviceInfo,
		kRgfxAccelMethodSurfaceWrites,
		kRgfxAccelMethodSurfaceRetire,
		kRgfxAccelMethodPagesBlock,
	};
	CHECK(context[0] == kRgfxAccelMethodContextFirst && context[0] == 0x100);
	for (unsigned i = 1; i < sizeof(context) / sizeof(context[0]); i++)
		CHECK(context[i] == context[i - 1] + 1);
	CHECK(context[sizeof(context) / sizeof(context[0]) - 1] == kRgfxAccelMethodContextLast);
	CHECK(kRgfxSurfaceCpuGenOffset == 16);
	CHECK(kRgfxSurfaceClientsOffset == 12);
}

// The features call and the kext's own feature bits, as design.md's registry gives them out:
// Features at 0x10f, past PagesBlock at 0x10e, and one bit for each call or flag a kext may lack,
// none shared.
static void features()
{
	CHECK(kRgfxAccelMethodPagesBlock == 0x10e);
	CHECK(kRgfxAccelMethodFeatures == 0x10f);
	CHECK(kRgfxAccelMethodFeatures == kRgfxAccelMethodContextLast + 1);
	const uint64_t bits[] = {
		kRgfxKextFeaturePagesBlock,
		kRgfxKextFeatureSubmitBatch,
		kRgfxKextFeatureWakeCompletions,
		kRgfxKextFeatureConsumedDoorbell,
		kRgfxKextFeatureDeferredAnswers,
		kRgfxKextFeatureContextSpan,
	};
	for (unsigned i = 0; i < sizeof(bits) / sizeof(bits[0]); i++)
		CHECK(bits[i] == uint64_t(1) << i);
	// TakeAnswers past the numbers SubmitBatch and WakeCompletions keep, and Submit's one flag.
	CHECK(kRgfxAccelMethodTakeAnswers == 0x112);
	CHECK(kRgfxAccelMethodTakeAnswers == kRgfxAccelMethodFeatures + 3);
	CHECK(kRgfxSubmitDeferred == 1);
}

int main()
{
	vbl();
	completion();
	iogpu();
	protocol();
	features();
	if (g_failed) {
		std::fprintf(stderr, "user_client_words_test: %d of %d checks failed\n", g_failed,
		             g_checks);
		return 1;
	}
	std::printf("user_client_words_test: OK (%d checks)\n", g_checks);
	return 0;
}
