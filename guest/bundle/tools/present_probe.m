// present_probe: what an on-screen CAMetalLayer app sees of the display's presentation, as an app
// on a Mac sees it: each drawable's presented handler and its presentedTime, how long nextDrawable
// blocks, and the display link's rate. It renders as little as it can (one clear a frame, its
// colour changing), so what it measures is the present path, not the GPU.
//   present_probe [-t SECONDS] [-w WIDTHxHEIGHT] [-n] [-c COUNT] [-f FPS] [-a MS]
//     -t SECONDS  how long to measure, after a second of warm-up (default 10).
//     -w WxH      a titled window of that size in points, in place of a full-screen borderless
//                 one at the status level.
//     -n          displaySyncEnabled NO (the default is YES, CAMetalLayer's).
//     -c COUNT    maximumDrawableCount (default 3, CAMetalLayer's).
//     -f FPS      submits at most FPS frames a second (default: as fast as nextDrawable gives).
//     -a MS       presents each drawable at its commit's time plus MS milliseconds
//                 (presentDrawable:atTime:), in place of at once (presentDrawable:).
// One line: "present_probe: OK key=value ..." or "present_probe: FAIL why":
//   submitted_fps   frames committed a second;
//   presented, dropped, unanswered  drawables whose presented handler gave a nonzero
//                   presentedTime, gave 0, or had not run by the end;
//   presented_fps   presented frames a second, from the span of their presentedTimes;
//   gap_ms_p50/max  between consecutive presentedTimes;
//   latency_ms_p10/p50/p90  from a frame's commit to its presentedTime;
//   at_ms_p05/p50/p95  with -a, from the time asked for to the presentedTime (0 without);
//   wait_ms_p50/max how long nextDrawable blocked;
//   link_hz         CVDisplayLink callbacks a second, link_period_ms the period it states;
//   vsync_ms_p05/p50/p95/absmax  from each presentedTime to the nearest refresh the display link
//                   gave (its callbacks' current time), signed, and the largest distance.
// Then one line "present_probe-timeline: key=value ...", where a run that falls short shows when:
//   occlusion       each change of the window's occlusion state, as seconds:visible (1) or not
//                   (0), from the run's start (the window's ordering front);
//   committed, presented, dropped, wait_ms_max  per half second of the measured window, by
//                   commit time: frames committed, presented, dropped, and nextDrawable's longest
//                   block;
//   first_waits_ms  the first 12 blocks of nextDrawable from the run's start, warm-up included,
//                   as seconds:milliseconds; long_waits_ms the first 20 of 100 ms or more;
//   presented_first_s, presented_last_s  the first and last presentedTime, from the run's start.
// It needs the logged-in user's window server session (launchctl asuser).
// Build (-arch arm64 for an Apple-silicon Mac):
//   xcrun clang -arch x86_64 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-parameter
//       -framework AppKit -framework Metal -framework QuartzCore -framework CoreVideo
//       present_probe.m -o present_probe
#import <AppKit/AppKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <math.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CVDisplayLink is deprecated from macOS 15, and apps still pace themselves by it: it is the link
// this measures.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static double gSeconds = 10;
static double gWidth = 0, gHeight = 0;
static BOOL gSync = YES;
static int gCount = 3;
static double gFps = 0;
static double gAt = 0;

static double now(void)
{
	return CACurrentMediaTime();
}

static double percentile(NSArray<NSNumber *> *values, double p)
{
	if (values.count == 0)
		return 0;
	NSArray *s = [values sortedArrayUsingSelector:@selector(compare:)];
	NSUInteger i = (NSUInteger)(p * (double)(s.count - 1) + 0.5);
	return [s[i] doubleValue];
}

// The timeline line's series, from the run's start (gT0).
static double gT0;
static NSString *gTimeline;
enum { kBucketsMax = 512 };
static const double kBucket = 0.5;

static int gCommittedB[kBucketsMax], gPresentedB[kBucketsMax], gDroppedB[kBucketsMax];
static double gWaitMaxB[kBucketsMax];

static NSString *joinInts(const int *v, int n)
{
	NSMutableArray *a = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
	for (int i = 0; i < n; i++)
		[a addObject:[NSString stringWithFormat:@"%d", v[i]]];
	return [a componentsJoinedByString:@","];
}

static _Atomic int gLinkTicks;
static _Atomic int64_t gLinkPeriodNs;
// The refreshes the display link gave, in mach time, for the presentedTimes' distance to them.
enum { kMaxLinkTimes = 1 << 15 };
static uint64_t gLinkTimes[kMaxLinkTimes];

static CVReturn linkTick(CVDisplayLinkRef link, const CVTimeStamp *nowTs, const CVTimeStamp *out,
                         CVOptionFlags flagsIn, CVOptionFlags *flagsOut, void *context)
{
	const int n = atomic_fetch_add(&gLinkTicks, 1);
	if (n < kMaxLinkTimes)
		gLinkTimes[n] = nowTs->hostTime;
	if (out->videoRefreshPeriod && out->videoTimeScale)
		atomic_store(&gLinkPeriodNs,
		             (int64_t)((double)out->videoRefreshPeriod * 1e9 / (double)out->videoTimeScale));
	return kCVReturnSuccess;
}

static NSString *run(void)
{
	[NSApplication sharedApplication];
	[NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
	NSScreen *screen = NSScreen.mainScreen;
	if (screen == nil)
		return @"FAIL no screen (run it in the logged-in user's session)";
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (device == nil)
		return @"FAIL no Metal device";
	const BOOL windowed = gWidth > 0 && gHeight > 0;
	const NSRect frame = windowed ? NSMakeRect(100, 100, gWidth, gHeight) : screen.frame;
	const CGFloat scale = screen.backingScaleFactor;
	NSWindow *win = [[NSWindow alloc]
	    initWithContentRect:frame
	              styleMask:windowed ? NSWindowStyleMaskTitled : NSWindowStyleMaskBorderless
	                backing:NSBackingStoreBuffered
	                  defer:NO
	                 screen:screen];
	if (!windowed)
		win.level = NSStatusWindowLevel;
	win.opaque = YES;
	CAMetalLayer *layer = [CAMetalLayer layer];
	layer.device = device;
	layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
	layer.framebufferOnly = YES;
	layer.opaque = YES;
	layer.displaySyncEnabled = gSync;
	layer.maximumDrawableCount = (NSUInteger)gCount;
	layer.contentsScale = scale;
	layer.drawableSize = CGSizeMake(frame.size.width * scale, frame.size.height * scale);
	NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, frame.size.width, frame.size.height)];
	view.layer = layer;
	view.wantsLayer = YES;
	win.contentView = view;
	NSMutableArray<NSString *> *occlusion = [NSMutableArray array];
	id occlusionObserver = [NSNotificationCenter.defaultCenter
	    addObserverForName:NSWindowDidChangeOcclusionStateNotification
	                object:win
	                 queue:NSOperationQueue.mainQueue
	            usingBlock:^(NSNotification *note) {
		            const BOOL visible = (win.occlusionState & NSWindowOcclusionStateVisible) != 0;
		            @synchronized(occlusion) {
			            [occlusion addObject:[NSString stringWithFormat:@"%.2f:%d", now() - gT0, visible]];
		            }
	            }];
	gT0 = now();
	[win orderFrontRegardless];

	CVDisplayLinkRef link = NULL;
	CVDisplayLinkCreateWithActiveCGDisplays(&link);
	if (link) {
		CVDisplayLinkSetOutputCallback(link, linkTick, NULL);
		CVDisplayLinkStart(link);
	}

	id<MTLCommandQueue> queue = [device newCommandQueue];
	__block NSString *result = nil;
	NSThread *render = [[NSThread alloc] initWithBlock:^{
		NSMutableArray<NSNumber *> *presented = [NSMutableArray array], *latency = [NSMutableArray array],
		                           *waits = [NSMutableArray array], *late = [NSMutableArray array];
		NSLock *lock = [NSLock new];
		__block atomic_int dropped = 0, answered = 0;
		int submitted = 0, nils = 0, linkStart = 0;
		const double start = now(), measure = start + 1, end = measure + gSeconds;
		const int buckets = MIN(kBucketsMax, (int)ceil(gSeconds / kBucket));
		int (^bucketOf)(double) = ^int(double t) {
			return MAX(0, MIN(buckets - 1, (int)floor((t - measure) / kBucket)));
		};
		NSMutableArray<NSString *> *firstWaits = [NSMutableArray array], *longWaits = [NSMutableArray array];
		double next = start;
		BOOL counting = NO;
		id<MTLCommandBuffer> last = nil;
		while (now() < end) {
			@autoreleasepool {
				if (gFps > 0) {
					while (now() < next)
						usleep(200);
					next += 1.0 / gFps;
				}
				const double asked = now();
				id<CAMetalDrawable> d = [layer nextDrawable];
				const double got = now();
				if (firstWaits.count < 12)
					[firstWaits addObject:[NSString stringWithFormat:@"%.2f:%.1f", asked - gT0, (got - asked) * 1e3]];
				if (got - asked >= 0.1 && longWaits.count < 20)
					[longWaits addObject:[NSString stringWithFormat:@"%.2f:%.1f", asked - gT0, (got - asked) * 1e3]];
				if (d == nil) {
					nils++;
					continue;
				}
				if (!counting && got >= measure) {
					counting = YES;
					linkStart = atomic_load(&gLinkTicks);
				}
				MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
				pass.colorAttachments[0].texture = d.texture;
				pass.colorAttachments[0].loadAction = MTLLoadActionClear;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				const double phase = got - start;
				pass.colorAttachments[0].clearColor =
				    MTLClearColorMake(0.5 + 0.5 * sin(phase * 3), 0.5 + 0.5 * sin(phase * 5), 0.5, 1);
				id<MTLCommandBuffer> cb = [queue commandBuffer];
				[[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
				const double committed = now(), target = committed + gAt / 1e3;
				if (counting) {
					submitted++;
					[waits addObject:@((got - asked) * 1e3)];
					const int b = bucketOf(committed);
					gCommittedB[b]++;
					gWaitMaxB[b] = MAX(gWaitMaxB[b], (got - asked) * 1e3);
					[d addPresentedHandler:^(id<MTLDrawable> dr) {
						atomic_fetch_add(&answered, 1);
						const double t = dr.presentedTime;
						if (t > 0) {
							[lock lock];
							gPresentedB[b]++;
							[presented addObject:@(t)];
							[latency addObject:@((t - committed) * 1e3)];
							if (gAt > 0)
								[late addObject:@((t - target) * 1e3)];
							[lock unlock];
						} else {
							atomic_fetch_add(&dropped, 1);
							[lock lock];
							gDroppedB[b]++;
							[lock unlock];
						}
					}];
				}
				if (gAt > 0)
					[cb presentDrawable:d atTime:target];
				else
					[cb presentDrawable:d];
				[cb commit];
				last = cb;
			}
		}
		const int linkTicks = atomic_load(&gLinkTicks) - linkStart;
		[last waitUntilCompleted];
		[NSThread sleepForTimeInterval:0.5];
		[lock lock];
		NSArray *ts = [presented sortedArrayUsingSelector:@selector(compare:)];
		const double span = ts.count > 1 ? [ts.lastObject doubleValue] - [ts.firstObject doubleValue] : 0;
		NSMutableArray *gaps = [NSMutableArray array];
		for (NSUInteger i = 1; i < ts.count; i++)
			[gaps addObject:@(([ts[i] doubleValue] - [ts[i - 1] doubleValue]) * 1e3)];
		// Each presentedTime's distance to the nearest refresh of the display link's.
		mach_timebase_info_data_t tb;
		mach_timebase_info(&tb);
		const int links = MIN(atomic_load(&gLinkTicks), kMaxLinkTimes);
		NSMutableArray *offs = [NSMutableArray array], *absOffs = [NSMutableArray array];
		for (NSNumber *t in ts) {
			const double tv = t.doubleValue;
			int lo = 0, hi = links - 1;
			while (lo < hi) {
				const int mid = (lo + hi) / 2;
				if ((double)gLinkTimes[mid] * tb.numer / tb.denom / 1e9 < tv)
					lo = mid + 1;
				else
					hi = mid;
			}
			double best = 1e9;
			for (int k = MAX(lo - 1, 0); k <= MIN(lo, links - 1); k++) {
				const double d = tv - (double)gLinkTimes[k] * tb.numer / tb.denom / 1e9;
				if (fabs(d) < fabs(best))
					best = d;
			}
			if (links > 0) {
				[offs addObject:@(best * 1e3)];
				[absOffs addObject:@(fabs(best) * 1e3)];
			}
		}
		result = [NSString
		    stringWithFormat:@"OK mode=%@ size=%.0fx%.0f sync=%d drawables=%d submitted_fps=%.1f presented=%lu "
		                     @"dropped=%d unanswered=%d presented_fps=%.1f gap_ms_p50=%.2f gap_ms_max=%.2f "
		                     @"latency_ms_p10=%.2f latency_ms_p50=%.2f latency_ms_p90=%.2f at_ms_p05=%.2f at_ms_p50=%.2f at_ms_p95=%.2f wait_ms_p50=%.2f wait_ms_max=%.2f nil_drawables=%d "
		                     @"link_hz=%.1f link_period_ms=%.3f vsync_ms_p05=%.3f vsync_ms_p50=%.3f vsync_ms_p95=%.3f "
		                     @"vsync_ms_absmax=%.3f "
		                     @"screen_max_fps=%ld seconds=%.0f",
		                     windowed ? @"windowed" : @"fullscreen", layer.drawableSize.width,
		                     layer.drawableSize.height, gSync, gCount, submitted / gSeconds,
		                     (unsigned long)ts.count, atomic_load(&dropped), submitted - atomic_load(&answered),
		                     span > 0 ? (double)(ts.count - 1) / span : 0, percentile(gaps, 0.5),
		                     percentile(gaps, 1), percentile(latency, 0.1), percentile(latency, 0.5),
		                     percentile(latency, 0.9), percentile(late, 0.05),
		                     percentile(late, 0.5), percentile(late, 0.95), percentile(waits, 0.5),
		                     percentile(waits, 1), nils, linkTicks / gSeconds,
		                     atomic_load(&gLinkPeriodNs) / 1e6, percentile(offs, 0.05), percentile(offs, 0.5),
		                     percentile(offs, 0.95), percentile(absOffs, 1),
		                     (long)screen.maximumFramesPerSecond, gSeconds];
		NSMutableArray *waitMax = [NSMutableArray arrayWithCapacity:(NSUInteger)buckets];
		for (int i = 0; i < buckets; i++)
			[waitMax addObject:[NSString stringWithFormat:@"%.0f", gWaitMaxB[i]]];
		NSString *seen;
		@synchronized(occlusion) {
			seen = [occlusion componentsJoinedByString:@","];
		}
		gTimeline = [NSString
		    stringWithFormat:@"occlusion=%@ bucket_s=%.1f committed=%@ presented=%@ dropped=%@ wait_ms_max=%@ "
		                     @"first_waits_ms=%@ long_waits_ms=%@ presented_first_s=%.2f presented_last_s=%.2f",
		                     seen.length ? seen : @"none", kBucket, joinInts(gCommittedB, buckets),
		                     joinInts(gPresentedB, buckets), joinInts(gDroppedB, buckets),
		                     [waitMax componentsJoinedByString:@","], [firstWaits componentsJoinedByString:@","],
		                     longWaits.count ? [longWaits componentsJoinedByString:@","] : @"none",
		                     ts.count ? [ts.firstObject doubleValue] - gT0 : 0,
		                     ts.count ? [ts.lastObject doubleValue] - gT0 : 0];
		[lock unlock];
		dispatch_async(dispatch_get_main_queue(), ^{
			[NSApp stop:nil];
			[NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
			                                    location:NSZeroPoint
			                               modifierFlags:0
			                                   timestamp:0
			                                windowNumber:0
			                                     context:nil
			                                     subtype:0
			                                       data1:0
			                                       data2:0]
			         atStart:YES];
		});
	}];
	[render start];
	[NSApp run];
	if (link) {
		CVDisplayLinkStop(link);
		CVDisplayLinkRelease(link);
	}
	[NSNotificationCenter.defaultCenter removeObserver:occlusionObserver];
	[win orderOut:nil];
	return result ?: @"FAIL the render thread gave no result";
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		int c;
		while ((c = getopt(argc, argv, "t:w:nc:f:a:")) != -1) {
			switch (c) {
			case 't':
				gSeconds = atof(optarg);
				break;
			case 'w':
				if (sscanf(optarg, "%lfx%lf", &gWidth, &gHeight) != 2) {
					fprintf(stderr, "present_probe: -w takes WIDTHxHEIGHT\n");
					return 2;
				}
				break;
			case 'n':
				gSync = NO;
				break;
			case 'c':
				gCount = atoi(optarg);
				break;
			case 'f':
				gFps = atof(optarg);
				break;
			case 'a':
				gAt = atof(optarg);
				break;
			default:
				fprintf(stderr, "usage: present_probe [-t SECONDS] [-w WxH] [-n] [-c COUNT] [-f FPS] [-a MS]\n");
				return 2;
			}
		}
		if (gSeconds <= 0 || gCount < 2 || gCount > 3 || gAt < 0) {
			fprintf(stderr, "present_probe: -t must be positive, -c 2 or 3 and -a not negative\n");
			return 2;
		}
		NSString *r = run();
		printf("present_probe: %s\n", r.UTF8String);
		if (gTimeline)
			printf("present_probe-timeline: %s\n", gTimeline.UTF8String);
		return [r hasPrefix:@"OK"] ? 0 : 1;
	}
}
