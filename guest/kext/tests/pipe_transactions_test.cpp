// A display pipe client's transactions (RgfxPipeTransactions.hpp), built natively: answers
// complete transactions oldest first, a transaction with nothing to show rides on the one before
// it, and only transactions with a cookie are notified, as at WindowServer's hand-over of the
// display to the CPU at its shutdown.
#include <cstdio>

#include "RgfxPipeTransactions.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

typedef RgfxPipeTransactions<2> Transactions;

static void rule()
{
	CHECK(rgfxPipeNotifies(0x6000002de3a0)); // a cookie, as WindowServer's transactional frames carry
	CHECK(!rgfxPipeNotifies(0));            // none, as its shutdown's frames carry
}

static void answers()
{
	Transactions t;
	Transactions::Completion out[2] = {};

	// One sent, answered: it completes, notified, with its cookie.
	CHECK(t.hasRoom() && t.outstanding() == 0);
	t.sent(10, 1, 0xc001, true);
	CHECK(t.outstanding() == 1 && t.hasRoom());
	CHECK(t.answered(99, out) == 0); // another client's seqno: nothing
	CHECK(t.answered(10, out) == 1 && out[0].id == 1 && out[0].cookie == 0xc001 && out[0].notified);
	CHECK(t.outstanding() == 0 && t.answered(10, out) == 0); // answered once

	// Two sent: no room. The second's answer completes both, oldest first.
	t.sent(11, 2, 0xc002, true);
	t.sent(12, 3, 0xc003, true);
	CHECK(!t.hasRoom() && t.outstanding() == 2);
	CHECK(t.answered(12, out) == 2 && out[0].id == 2 && out[1].id == 3);
	CHECK(t.outstanding() == 0 && t.hasRoom());

	// The first's answer completes it alone; the second stays.
	t.sent(13, 4, 0xc004, true);
	t.sent(14, 5, 0xc005, true);
	CHECK(t.answered(13, out) == 1 && out[0].id == 4);
	CHECK(t.outstanding() == 1 && t.answered(14, out) == 1 && out[0].id == 5);
}

static void nothingToShow()
{
	Transactions t;
	Transactions::Completion now = {}, out[2] = {};

	// None outstanding: it completes now.
	CHECK(t.nothingToShow(1, 0xa, true, &now) && now.id == 1 && now.cookie == 0xa && now.notified);
	CHECK(t.outstanding() == 0);

	// One outstanding: it completes with it, at that one's answer, oldest first.
	t.sent(20, 2, 0xb, true);
	CHECK(!t.nothingToShow(3, 0xc, true, &now) && t.outstanding() == 2);
	CHECK(t.answered(20, out) == 2 && out[0].id == 2 && out[1].id == 3 && out[1].cookie == 0xc);
	CHECK(t.outstanding() == 0);
}

// WindowServer's shutdown at a session restart, as measured on a macOS 26 guest: transaction 770
// with a cookie, then 'wsrv' 0x2 and transactions 771-774 with none (options 0x801, 0x801,
// 0x40801, 0x801), each waited on with selector 9. 770 is notified, also where its answer comes
// after the hand-over; 771-774 are not.
static void handOver()
{
	Transactions t;
	Transactions::Completion out[2] = {};

	t.sent(100, 770, 0x6000002e9fe0, rgfxPipeNotifies(0x6000002e9fe0));
	CHECK(t.answered(100, out) == 1 && out[0].id == 770 && out[0].notified);
	for (uint32_t id = 771; id <= 774; id++) {
		t.sent(100 + id, id, 0, rgfxPipeNotifies(0));
		CHECK(t.answered(100 + id, out) == 1 && out[0].id == id && !out[0].notified);
	}

	// One with a cookie and one with none outstanding together: one answer completes both, only
	// the first notified.
	t.sent(200, 21, 0x21, rgfxPipeNotifies(0x21));
	t.sent(201, 22, 0, rgfxPipeNotifies(0));
	CHECK(t.answered(201, out) == 2);
	CHECK(out[0].id == 21 && out[0].notified && out[1].id == 22 && !out[1].notified);

	// A frame with nothing to show and no cookie, none outstanding: done now, unnotified.
	Transactions::Completion now = {};
	CHECK(t.nothingToShow(23, 0, rgfxPipeNotifies(0), &now) && now.id == 23 && !now.notified);

	// The next WindowServer's first frame carries a cookie again: notified.
	t.sent(300, 1, 0x6000002de3a0, rgfxPipeNotifies(0x6000002de3a0));
	CHECK(t.answered(300, out) == 1 && out[0].notified);
}

int main()
{
	rule();
	answers();
	nothingToShow();
	handOver();
	if (g_failed) {
		std::fprintf(stderr, "pipe_transactions_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("pipe_transactions_test: OK (%d checks)\n", g_checks);
	return 0;
}
