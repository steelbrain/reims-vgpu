// oracle_waits: whether waitUntilCompleted and waitUntilScheduled return before, while or only
// after a command buffer's completed and scheduled handlers run, whether a handler that blocks
// holds them, and whether a later buffer of the queue waits for an earlier buffer's handler.
//   oracle_waits DEVICE [RUNS]
// Each buffer is a tiny blit (a 256-byte fill). A handler that blocks sleeps BLOCK ms (20) and
// stamps its start and end. A thread polls the buffer's status from before its commit and stamps
// the first read of each status. RUNS runs of each case (default 20; case 7, three times that).
// Cases:
//   1  a completed handler that blocks: waitUntilCompleted's return against the handler's run
//      (before it starts, while it runs, after it ends); the status read in the handler and once
//      the wait returns; when the polling thread first reads Completed, against the handler.
//   2  two completed handlers, the first quick, the second blocks: the wait against the second.
//   3  a scheduled handler that blocks, and a completed handler: waitUntilScheduled against the
//      scheduled handler; when the completed handler starts and waitUntilCompleted returns,
//      against the scheduled handler's end; when the polling thread first reads Scheduled and
//      Completed.
//   4  a buffer with a completed handler that blocks, then a second with no handler, committed
//      right after: the second's waitUntilCompleted, and its Completed status, against the
//      first's handler.
//   5  the same, the second with a completed handler of its own: the second's handler start and
//      its wait against the first's handler.
//   6  a buffer with a completed handler that blocks; once that handler runs, a second buffer
//      with no handler is committed: its waitUntilScheduled and waitUntilCompleted, and its
//      Scheduled and Completed status, against the first's handler.
//   7  no handler blocks, an idle GPU: commit to waitUntilCompleted's return with no handler;
//      with an empty completed handler, commit to the handler's start and to the wait's return,
//      and how often the wait returned before the handler started or ended (median ms).
// Times are ms; "from the end" is the wait's return less the handler's end. Build: README.md.
#import "oracle_common.h"
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
static id<MTLBuffer> B;
static int RUNS = 20;
static const useconds_t kBlockUs = 20000;

static id<MTLCommandBuffer> tiny(void)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e fillBuffer:B range:NSMakeRange(0, 256) value:1];
	[e endEncoding];
	return cb;
}

// A thread that polls a buffer's status and stamps the first read of each value.
typedef struct {
	void *cb; // the buffer, held by the caller until join
	_Atomic int stop;
	_Atomic double first[8];
	pthread_t thread;
} Watch;

static void *watch_main(void *p)
{
	Watch *w = p;
	id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)w->cb;
	while (!atomic_load(&w->stop)) {
		long s = (long)cb.status;
		if (s >= 0 && s < 8 && atomic_load(&w->first[s]) == 0)
			atomic_store(&w->first[s], oracle_now_ms());
		if (s >= MTLCommandBufferStatusCompleted)
			break;
	}
	return NULL;
}

static void watch_start(Watch *w, id<MTLCommandBuffer> cb)
{
	memset(w, 0, sizeof(*w));
	w->cb = (__bridge void *)cb;
	pthread_create(&w->thread, NULL, watch_main, w);
}

static void watch_join(Watch *w)
{
	// Let it see the final status, then stop it.
	usleep(2000);
	atomic_store(&w->stop, 1);
	pthread_join(w->thread, NULL);
}

static void spin_until(_Atomic double *t, double ms)
{
	double end = oracle_now_ms() + ms;
	while (atomic_load(t) == 0 && oracle_now_ms() < end)
		usleep(100);
}

typedef struct {
	int before, during, after, never;
	double v[256];
	int n;
} Tally;

static void tally(Tally *t, double at, double start, double end)
{
	if (start == 0 || end == 0)
		t->never++;
	else if (at < start)
		t->before++;
	else if (at < end)
		t->during++;
	else
		t->after++;
	if (end != 0 && t->n < 256)
		t->v[t->n++] = at - end;
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static void tally_print(const char *what, Tally *t)
{
	qsort(t->v, t->n, sizeof(double), cmpd);
	printf("    %s: before %d, while %d, after %d", what, t->before, t->during, t->after);
	if (t->never)
		printf(", handler not seen %d", t->never);
	if (t->n)
		printf("; from the end: median %+.3f, min %+.3f, max %+.3f", t->v[t->n / 2], t->v[0],
		       t->v[t->n - 1]);
	printf("\n");
}

static const char *status_name(long s)
{
	static const char *const names[] = {"NotEnqueued", "Enqueued", "Committed",
	                                    "Scheduled",   "Completed", "Error"};
	return s >= 0 && s < 6 ? names[s] : "?";
}

static void case1(void)
{
	printf("  1. a completed handler that blocks %u ms\n", kBlockUs / 1000);
	Tally wait = {0}, polled = {0};
	int inHandler[8] = {0}, afterWait[8] = {0};
	for (int r = 0; r < RUNS; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = tiny();
			__block _Atomic double hs = 0, he = 0;
			__block long st = -1;
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&hs, oracle_now_ms());
				st = (long)b.status;
				usleep(kBlockUs);
				atomic_store(&he, oracle_now_ms());
			}];
			Watch w;
			watch_start(&w, cb);
			[cb commit];
			[cb waitUntilCompleted];
			double ret = oracle_now_ms();
			long after = (long)cb.status;
			spin_until(&he, 1000);
			watch_join(&w);
			tally(&wait, ret, hs, he);
			tally(&polled, atomic_load(&w.first[MTLCommandBufferStatusCompleted]), hs, he);
			if (st >= 0 && st < 8)
				inHandler[st]++;
			if (after >= 0 && after < 8)
				afterWait[after]++;
		}
	}
	tally_print("waitUntilCompleted returned, against the handler", &wait);
	tally_print("the polling thread first read Completed, against the handler", &polled);
	printf("    status read in the handler:");
	for (int s = 0; s < 8; s++)
		if (inHandler[s])
			printf(" %s %d", status_name(s), inHandler[s]);
	printf("; once the wait returned:");
	for (int s = 0; s < 8; s++)
		if (afterWait[s])
			printf(" %s %d", status_name(s), afterWait[s]);
	printf("\n");
}

static void case2(void)
{
	printf("  2. two completed handlers, the first quick, the second blocks %u ms\n",
	       kBlockUs / 1000);
	Tally wait = {0}, first = {0};
	for (int r = 0; r < RUNS; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = tiny();
			__block _Atomic double h1 = 0, hs = 0, he = 0;
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&h1, oracle_now_ms());
			}];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&hs, oracle_now_ms());
				usleep(kBlockUs);
				atomic_store(&he, oracle_now_ms());
			}];
			[cb commit];
			[cb waitUntilCompleted];
			double ret = oracle_now_ms();
			spin_until(&he, 1000);
			tally(&wait, ret, hs, he);
			tally(&first, atomic_load(&h1), hs, he);
		}
	}
	tally_print("waitUntilCompleted returned, against the second handler", &wait);
	tally_print("the first handler ran, against the second", &first);
}

static void case3(void)
{
	printf("  3. a scheduled handler that blocks %u ms, and a completed handler\n", kBlockUs / 1000);
	Tally wsched = {0}, chandler = {0}, wdone = {0}, psched = {0}, pdone = {0};
	for (int r = 0; r < RUNS; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = tiny();
			__block _Atomic double ss = 0, se = 0, hs = 0;
			[cb addScheduledHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&ss, oracle_now_ms());
				usleep(kBlockUs);
				atomic_store(&se, oracle_now_ms());
			}];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&hs, oracle_now_ms());
			}];
			Watch w;
			watch_start(&w, cb);
			[cb commit];
			[cb waitUntilScheduled];
			double rs = oracle_now_ms();
			[cb waitUntilCompleted];
			double rc = oracle_now_ms();
			spin_until(&se, 1000);
			spin_until(&hs, 1000);
			watch_join(&w);
			tally(&wsched, rs, ss, se);
			tally(&chandler, atomic_load(&hs), ss, se);
			tally(&wdone, rc, ss, se);
			tally(&psched, atomic_load(&w.first[MTLCommandBufferStatusScheduled]), ss, se);
			tally(&pdone, atomic_load(&w.first[MTLCommandBufferStatusCompleted]), ss, se);
		}
	}
	tally_print("waitUntilScheduled returned, against the scheduled handler", &wsched);
	tally_print("the completed handler started, against the scheduled handler", &chandler);
	tally_print("waitUntilCompleted returned, against the scheduled handler", &wdone);
	tally_print("the polling thread first read Scheduled, against the scheduled handler", &psched);
	tally_print("the polling thread first read Completed, against the scheduled handler", &pdone);
}

// Cases 4 and 5: a first buffer whose completed handler blocks, and a second committed right
// after it, with a quick completed handler of its own or none.
static void case45(int second_handler)
{
	printf("  %d. a buffer whose completed handler blocks %u ms, then a second %s, committed "
	       "right after\n",
	       second_handler ? 5 : 4, kBlockUs / 1000,
	       second_handler ? "with a quick completed handler" : "with no handler");
	Tally wait = {0}, polled = {0}, h2 = {0};
	for (int r = 0; r < RUNS; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> a = tiny(), b = tiny();
			__block _Atomic double hs = 0, he = 0, bh = 0;
			[a addCompletedHandler:^(id<MTLCommandBuffer> x) {
				atomic_store(&hs, oracle_now_ms());
				usleep(kBlockUs);
				atomic_store(&he, oracle_now_ms());
			}];
			if (second_handler)
				[b addCompletedHandler:^(id<MTLCommandBuffer> x) {
					atomic_store(&bh, oracle_now_ms());
				}];
			Watch w;
			watch_start(&w, b);
			[a commit];
			[b commit];
			[b waitUntilCompleted];
			double ret = oracle_now_ms();
			spin_until(&he, 1000);
			if (second_handler)
				spin_until(&bh, 1000);
			watch_join(&w);
			[a waitUntilCompleted];
			tally(&wait, ret, hs, he);
			tally(&polled, atomic_load(&w.first[MTLCommandBufferStatusCompleted]), hs, he);
			if (second_handler)
				tally(&h2, atomic_load(&bh), hs, he);
		}
	}
	tally_print("the second's waitUntilCompleted returned, against the first's handler", &wait);
	tally_print("the polling thread first read the second Completed, against the first's handler",
	            &polled);
	if (second_handler)
		tally_print("the second's handler started, against the first's handler", &h2);
}

static void case6(void)
{
	printf("  6. a buffer whose completed handler blocks %u ms; once it runs, a second with no "
	       "handler is committed\n",
	       kBlockUs / 1000);
	Tally wsched = {0}, wdone = {0}, psched = {0}, pdone = {0};
	for (int r = 0; r < RUNS; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> a = tiny();
			__block _Atomic double hs = 0, he = 0;
			[a addCompletedHandler:^(id<MTLCommandBuffer> x) {
				atomic_store(&hs, oracle_now_ms());
				usleep(kBlockUs);
				atomic_store(&he, oracle_now_ms());
			}];
			[a commit];
			spin_until(&hs, 1000);
			id<MTLCommandBuffer> b = tiny();
			Watch w;
			watch_start(&w, b);
			[b commit];
			[b waitUntilScheduled];
			double rs = oracle_now_ms();
			[b waitUntilCompleted];
			double rc = oracle_now_ms();
			spin_until(&he, 1000);
			watch_join(&w);
			[a waitUntilCompleted];
			tally(&wsched, rs, hs, he);
			tally(&wdone, rc, hs, he);
			tally(&psched, atomic_load(&w.first[MTLCommandBufferStatusScheduled]), hs, he);
			tally(&pdone, atomic_load(&w.first[MTLCommandBufferStatusCompleted]), hs, he);
		}
	}
	tally_print("the second's waitUntilScheduled returned, against the first's handler", &wsched);
	tally_print("the second's waitUntilCompleted returned, against the first's handler", &wdone);
	tally_print("the polling thread first read the second Scheduled, against the first's handler",
	            &psched);
	tally_print("the polling thread first read the second Completed, against the first's handler",
	            &pdone);
}

static double median(double *v, int n)
{
	qsort(v, n, sizeof(double), cmpd);
	return n ? v[n / 2] : 0;
}

static void case7(void)
{
	const int n = RUNS * 3 > 256 ? 256 : RUNS * 3;
	printf("  7. no handler blocks, an idle GPU, %d runs each\n", n);
	double none[256], start[256], ret[256], end[256];
	for (int r = 0; r < n; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = tiny();
			usleep(2000);
			double c = oracle_now_ms();
			[cb commit];
			[cb waitUntilCompleted];
			none[r] = oracle_now_ms() - c;
		}
	}
	int before = 0, during = 0;
	for (int r = 0; r < n; r++) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = tiny();
			__block _Atomic double hs = 0, he = 0;
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&hs, oracle_now_ms());
				atomic_store(&he, oracle_now_ms());
			}];
			usleep(2000);
			double c = oracle_now_ms();
			[cb commit];
			[cb waitUntilCompleted];
			double w = oracle_now_ms();
			spin_until(&he, 1000);
			start[r] = atomic_load(&hs) - c;
			end[r] = atomic_load(&he) - c;
			ret[r] = w - c;
			if (w < atomic_load(&hs))
				before++;
			else if (w < atomic_load(&he))
				during++;
		}
	}
	printf("    no handler: commit to waitUntilCompleted's return, median %.3f\n", median(none, n));
	printf("    an empty completed handler: commit to its start, median %.3f; to the wait's "
	       "return, median %.3f; the wait returned before the handler started %d, before it "
	       "ended %d, of %d\n",
	       median(start, n), median(ret, n), before, during, n);
	(void)end;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE [RUNS]\n", argv[0]);
			return 2;
		}
		if (argc > 2)
			RUNS = atoi(argv[2]) > 0 && atoi(argv[2]) <= 256 ? atoi(argv[2]) : 20;
		D = oracle_device(argv[1]);
		oracle_header("oracle_waits", D);
		Q = [D newCommandQueue];
		B = [D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
		// Warm the queue up.
		for (int i = 0; i < 5; i++)
			oracle_commit_wait(tiny(), NULL, NULL);
		case1();
		case2();
		case3();
		case45(0);
		case45(1);
		case6();
		case7();
	}
	return 0;
}
