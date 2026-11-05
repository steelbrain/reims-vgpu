// display_rate_probe: the answers a Mac gives about its display's refresh rate, and the rate
// WebKit picks from them for requestAnimationFrame and its rendering updates.
//   display_rate_probe [-t SECONDS] [-k SAMPLES] [-n] [-w]
//     -t SECONDS  how long the display links run (default 3), and with -w each page's measure.
//     -k SAMPLES  fresh CVDisplayLinks to read the nominal period from, 250 ms apart (default 12).
//     -n          only the nominal samples (the nominal section), then the OK line.
//     -w          also measures requestAnimationFrame in a WKWebView in a window of its own, once
//                 with WebKit's defaults and once with "Prefer Page Rendering Updates near 60fps"
//                 off. It shows a window: never on a host's own screen, only in a guest.
// What WebKit reads on macOS (WebKit's UIProcess/mac/DisplayLinkMac.cpp, read): a UI process
// makes one CVDisplayLink for each display, the first time a page shows there, and takes
// round(timeScale / timeValue) of CVDisplayLinkGetNominalOutputVideoRefreshPeriod as the
// display's nominal frames a second, once. With PreferPageRenderingUpdatesNear60FPSEnabled
// (on by default), a page asks for nominal / (nominal / 60) frames a second, in integers
// (AnimationFrameRate.cpp, framesPerSecondNearestFullSpeed): 60 at 120 Hz, 68 at 136, 119 at
// 119. The link's callbacks then reach the page every (nominal / asked)th time when that is
// a whole number, and every time when it is not (DisplayUpdate.cpp). So a page's rate is the
// link's callback rate over that factor: 60 on a 120 Hz display, the callback rate at 119.
// Lines, each "display_rate_probe: SECTION key=value ...":
//   display     each active display: CGDisplayMode's refresh, size, IOFlags and mode ID, and the
//               refresh rates its modes list;
//   screen      each NSScreen: maximumFramesPerSecond, minimum and maximumRefreshInterval,
//               displayUpdateGranularity;
//   nominal     the main display's fresh links' nominal periods: each sample's rate as WebKit
//               rounds it, the exact rates' range, and the period's value, scale and flags;
//   link        one link run for -t seconds: callbacks a second, the callbacks' interval p50,
//               the output time stamps' videoRefreshPeriod range, rateScalar, the actual period;
//   cadisplay   NSScreen's CADisplayLink (macOS 14) run for -t seconds: ticks a second, duration;
//   webkit      the near-60fps feature's default and its value in a fresh WKPreferences, as the
//               installed WebKit gives them, and the rate WebKit's rules predict per nominal;
//   page        with -w, each WKWebView run: the nominal read just before, the rAF rate and
//               interval p50, and what WebKit's rules predict;
//   then "display_rate_probe: OK ..." with the summary, or "display_rate_probe: FAIL why".
// It needs the logged-in user's window server session (launchctl asuser in a guest).
// Build (-arch arm64 for an Apple-silicon Mac):
//   xcrun clang -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -Wall -Wextra -Werror
//       -Wno-unused-parameter -framework AppKit -framework CoreVideo -framework QuartzCore
//       -framework WebKit display_rate_probe.m -o display_rate_probe
#import <AppKit/AppKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <WebKit/WebKit.h>

#include <limits.h>
#include <mach/mach_time.h>
#include <math.h>
#include <objc/message.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

// CVDisplayLink is deprecated from macOS 15, and WebKit still paces pages by it: it is the
// link this reads.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static double gSeconds = 3;
static int gSamples = 12;
static BOOL gWebView = NO;
static BOOL gNominalOnly = NO;

static NSString *const kNear60Key = @"PreferPageRenderingUpdatesNear60FPSEnabled";

static void say(NSString *section, NSString *text)
{
	printf("display_rate_probe: %s %s\n", section.UTF8String, text.UTF8String);
	fflush(stdout);
}

static double percentile(NSArray<NSNumber *> *values, double p)
{
	if (values.count == 0)
		return 0;
	NSArray *s = [values sortedArrayUsingSelector:@selector(compare:)];
	NSUInteger i = (NSUInteger)(p * (double)(s.count - 1) + 0.5);
	return [s[i] doubleValue];
}

// WebKit's rules (read, AnimationFrameRate.cpp and DisplayUpdate.cpp), in its unsigned
// arithmetic: what a page asks for, and every how many link callbacks it is served.
static unsigned webkitNearestFullSpeed(unsigned nominal)
{
	if (nominal <= 60)
		return nominal;
	const float ratio = nominal / 60u;
	const unsigned floorSpeed = (unsigned)(nominal / floorf(ratio));
	const unsigned ceilSpeed = (unsigned)(nominal / ceilf(ratio));
	return ratio - floorf(ratio) <= 0.5f ? floorSpeed : ceilSpeed;
}

static unsigned webkitAsked(unsigned nominal, BOOL near60)
{
	if (nominal == 0 || nominal == 60)
		return 60;
	return near60 ? webkitNearestFullSpeed(nominal) : nominal;
}

static unsigned webkitFactor(unsigned nominal, unsigned asked)
{
	const unsigned updates = nominal ? nominal : 60;
	if (asked == 0 || fmodf((float)updates / (float)asked, 1.0f) != 0)
		return 1;
	const unsigned f = updates / asked;
	return f ? f : 1;
}

// round(timeScale / timeValue), WebKit's nominal frames a second (60 for no period).
static unsigned webkitNominal(CVTime period)
{
	if (!period.timeValue)
		return 60;
	const unsigned r = (unsigned)round((double)period.timeScale / (double)period.timeValue);
	return r ? r : 60;
}

// MARK: - CoreGraphics and NSScreen

static void dumpDisplays(void)
{
	CGDirectDisplayID ids[16];
	uint32_t n = 0;
	if (CGGetActiveDisplayList(16, ids, &n) != kCGErrorSuccess || n == 0) {
		say(@"display", @"active=0");
		return;
	}
	for (uint32_t i = 0; i < n; i++) {
		CGDisplayModeRef mode = CGDisplayCopyDisplayMode(ids[i]);
		NSMutableString *line = [NSMutableString
		    stringWithFormat:@"id=%u main=%d builtin=%d vendor=0x%x model=0x%x", ids[i], CGDisplayIsMain(ids[i]),
		                     CGDisplayIsBuiltin(ids[i]), CGDisplayVendorNumber(ids[i]), CGDisplayModelNumber(ids[i])];
		if (mode) {
			[line appendFormat:@" refresh=%.4f size=%zux%zu pixels=%zux%zu ioflags=0x%x mode_id=%d",
			                   CGDisplayModeGetRefreshRate(mode), CGDisplayModeGetWidth(mode),
			                   CGDisplayModeGetHeight(mode), CGDisplayModeGetPixelWidth(mode),
			                   CGDisplayModeGetPixelHeight(mode), CGDisplayModeGetIOFlags(mode),
			                   CGDisplayModeGetIODisplayModeID(mode)];
			CGDisplayModeRelease(mode);
		}
		NSDictionary *opts = @{(__bridge NSString *)kCGDisplayShowDuplicateLowResolutionModes : @YES};
		CFArrayRef modes = CGDisplayCopyAllDisplayModes(ids[i], (__bridge CFDictionaryRef)opts);
		NSMutableOrderedSet<NSString *> *rates = [NSMutableOrderedSet orderedSet];
		const CFIndex count = modes ? CFArrayGetCount(modes) : 0;
		for (CFIndex j = 0; j < count; j++) {
			CGDisplayModeRef m = (CGDisplayModeRef)CFArrayGetValueAtIndex(modes, j);
			[rates addObject:[NSString stringWithFormat:@"%.4f", CGDisplayModeGetRefreshRate(m)]];
		}
		if (modes)
			CFRelease(modes);
		[line appendFormat:@" modes=%ld mode_rates=%@", (long)count, [rates.array componentsJoinedByString:@","]];
		say(@"display", line);
	}
}

static long dumpScreens(void)
{
	long mainMax = 0;
	for (NSScreen *s in NSScreen.screens) {
		NSNumber *num = s.deviceDescription[@"NSScreenNumber"];
		if (s == NSScreen.mainScreen)
			mainMax = (long)s.maximumFramesPerSecond;
		say(@"screen", [NSString stringWithFormat:@"id=%u main=%d max_fps=%ld min_interval_ms=%.4f "
		                                          @"max_interval_ms=%.4f granularity_ms=%.4f name=\"%@\"",
		                                          num.unsignedIntValue, s == NSScreen.mainScreen,
		                                          (long)s.maximumFramesPerSecond, s.minimumRefreshInterval * 1e3,
		                                          s.maximumRefreshInterval * 1e3, s.displayUpdateGranularity * 1e3,
		                                          s.localizedName]);
	}
	return mainMax;
}

// MARK: - CVDisplayLink

typedef struct {
	unsigned min, max;
	double hzMin, hzMax;
} NominalRange;

// A fresh link's nominal rate, as WebKit reads it (made, not started).
static BOOL readNominal(CGDirectDisplayID display, CVTime *period)
{
	CVDisplayLinkRef link = NULL;
	if (CVDisplayLinkCreateWithCGDisplay(display, &link) != kCVReturnSuccess || !link)
		return NO;
	*period = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(link);
	CVDisplayLinkRelease(link);
	return YES;
}

static NominalRange sampleNominal(CGDirectDisplayID display)
{
	NominalRange r = {UINT_MAX, 0, INFINITY, 0};
	NSMutableArray<NSString *> *fps = [NSMutableArray array];
	CVTime last;
	bzero(&last, sizeof(last));
	for (int i = 0; i < gSamples; i++) {
		CVTime p;
		if (!readNominal(display, &p)) {
			[fps addObject:@"none"];
		} else {
			const unsigned w = webkitNominal(p);
			const double hz = p.timeValue ? (double)p.timeScale / (double)p.timeValue : 0;
			[fps addObject:[NSString stringWithFormat:@"%u", w]];
			r.min = MIN(r.min, w);
			r.max = MAX(r.max, w);
			r.hzMin = fmin(r.hzMin, hz);
			r.hzMax = fmax(r.hzMax, hz);
			last = p;
		}
		if (i + 1 < gSamples)
			usleep(250000);
	}
	if (r.max == 0) {
		r.min = 0;
		r.hzMin = 0;
	}
	say(@"nominal", [NSString stringWithFormat:@"display=%u samples=%d fps=%@ fps_min=%u fps_max=%u hz_min=%.4f "
	                                          @"hz_max=%.4f last_value=%lld last_scale=%d last_flags=0x%x",
	                                          display, gSamples, [fps componentsJoinedByString:@","], r.min, r.max,
	                                          r.hzMin, r.hzMax, last.timeValue, last.timeScale, last.flags]);
	return r;
}

static _Atomic int gTicks;
enum { kMaxTicks = 1 << 14 };
static uint64_t gTickHost[kMaxTicks];
static double gOutPeriodMin = INFINITY, gOutPeriodMax = 0, gRateScalar = 0;
static NSLock *gTickLock;

static CVReturn linkTick(CVDisplayLinkRef link, const CVTimeStamp *nowTs, const CVTimeStamp *out,
                         CVOptionFlags flagsIn, CVOptionFlags *flagsOut, void *context)
{
	const int n = atomic_fetch_add(&gTicks, 1);
	if (n < kMaxTicks)
		gTickHost[n] = nowTs->hostTime;
	[gTickLock lock];
	if (out->videoRefreshPeriod && out->videoTimeScale) {
		const double ms = (double)out->videoRefreshPeriod * 1e3 / (double)out->videoTimeScale;
		gOutPeriodMin = fmin(gOutPeriodMin, ms);
		gOutPeriodMax = fmax(gOutPeriodMax, ms);
	}
	if (out->flags & kCVTimeStampRateScalarValid)
		gRateScalar = out->rateScalar;
	[gTickLock unlock];
	return kCVReturnSuccess;
}

// Callbacks a second of one link run for -t seconds.
static double runLink(CGDirectDisplayID display)
{
	CVDisplayLinkRef link = NULL;
	if (CVDisplayLinkCreateWithCGDisplay(display, &link) != kCVReturnSuccess || !link) {
		say(@"link", @"made=0");
		return 0;
	}
	gTickLock = [NSLock new];
	atomic_store(&gTicks, 0);
	CVDisplayLinkSetOutputCallback(link, linkTick, NULL);
	CVDisplayLinkStart(link);
	usleep(500000);
	const int t0 = atomic_load(&gTicks);
	usleep((useconds_t)(gSeconds * 1e6));
	const int t1 = atomic_load(&gTicks);
	const double actual = CVDisplayLinkGetActualOutputVideoRefreshPeriod(link);
	const CVTime nominal = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(link);
	CVDisplayLinkStop(link);
	CVDisplayLinkRelease(link);
	mach_timebase_info_data_t tb;
	mach_timebase_info(&tb);
	NSMutableArray<NSNumber *> *gaps = [NSMutableArray array];
	for (int i = t0 + 1; i < MIN(t1, kMaxTicks); i++)
		[gaps addObject:@((double)(gTickHost[i] - gTickHost[i - 1]) * tb.numer / tb.denom / 1e6)];
	const double hz = (t1 - t0) / gSeconds;
	[gTickLock lock];
	say(@"link", [NSString stringWithFormat:@"display=%u callbacks_hz=%.2f interval_ms_p05=%.3f interval_ms_p50=%.3f "
	                                       @"interval_ms_p95=%.3f out_period_ms_min=%.4f out_period_ms_max=%.4f "
	                                       @"rate_scalar=%.6f actual_period_ms=%.4f nominal_after=%u",
	                                       display, hz, percentile(gaps, 0.05), percentile(gaps, 0.5),
	                                       percentile(gaps, 0.95), isfinite(gOutPeriodMin) ? gOutPeriodMin : 0,
	                                       gOutPeriodMax, gRateScalar, actual * 1e3, webkitNominal(nominal)]);
	[gTickLock unlock];
	return hz;
}

// MARK: - CADisplayLink (NSScreen's, macOS 14)

@interface Ticker : NSObject
@property(nonatomic) int ticks;
@property(nonatomic) double duration;
@end
@implementation Ticker
- (void)tick:(CADisplayLink *)link
{
	self.ticks++;
	self.duration = link.duration;
}
@end

static void spin(double seconds, BOOL (^done)(void))
{
	NSDate *end = [NSDate dateWithTimeIntervalSinceNow:seconds];
	while ((!done || !done()) && end.timeIntervalSinceNow > 0) {
		NSEvent *e = [NSApp nextEventMatchingMask:NSEventMaskAny
		                                untilDate:[NSDate dateWithTimeIntervalSinceNow:0.02]
		                                   inMode:NSDefaultRunLoopMode
		                                  dequeue:YES];
		if (e)
			[NSApp sendEvent:e];
	}
}

static void runCADisplayLink(NSScreen *screen)
{
	Ticker *t = [Ticker new];
	CADisplayLink *dl = [screen displayLinkWithTarget:t selector:@selector(tick:)];
	if (!dl) {
		say(@"cadisplay", @"made=0");
		return;
	}
	[dl addToRunLoop:NSRunLoop.mainRunLoop forMode:NSRunLoopCommonModes];
	spin(0.5, nil);
	const int t0 = t.ticks;
	spin(gSeconds, nil);
	const int t1 = t.ticks;
	const CAFrameRateRange range = dl.preferredFrameRateRange;
	[dl invalidate];
	say(@"cadisplay", [NSString stringWithFormat:@"ticks_hz=%.2f duration_ms=%.4f preferred_min=%.1f "
	                                            @"preferred_max=%.1f preferred=%.1f",
	                                            (t1 - t0) / gSeconds, t.duration * 1e3, range.minimum,
	                                            range.maximum, range.preferred]);
}

// MARK: - WebKit

// WebKit's feature list SPI (WKPreferencesPrivate.h, macOS 13.3): the near-60fps feature,
// or nil where this WebKit has no such SPI or feature.
static id near60Feature(void)
{
	SEL list = NSSelectorFromString(@"_features");
	if (![WKPreferences respondsToSelector:list])
		return nil;
	NSArray *features = ((NSArray * (*)(id, SEL)) objc_msgSend)([WKPreferences class], list);
	for (id f in features)
		if ([f respondsToSelector:NSSelectorFromString(@"key")] && [[f valueForKey:@"key"] isEqual:kNear60Key])
			return f;
	return nil;
}

static int near60Enabled(WKPreferences *prefs, id feature)
{
	SEL get = NSSelectorFromString(@"_isEnabledForFeature:");
	if (!feature || ![prefs respondsToSelector:get])
		return -1;
	return ((BOOL (*)(id, SEL, id))objc_msgSend)(prefs, get, feature) ? 1 : 0;
}

static BOOL setNear60(WKPreferences *prefs, id feature, BOOL on)
{
	SEL set = NSSelectorFromString(@"_setEnabled:forFeature:");
	if (!feature || ![prefs respondsToSelector:set])
		return NO;
	((void (*)(id, SEL, BOOL, id))objc_msgSend)(prefs, set, on, feature);
	return near60Enabled(prefs, feature) == (on ? 1 : 0);
}

@interface PageResult : NSObject <WKScriptMessageHandler>
@property(nonatomic, copy) NSDictionary *result;
@end
@implementation PageResult
- (void)userContentController:(WKUserContentController *)controller didReceiveScriptMessage:(WKScriptMessage *)message
{
	NSData *d = [[message.body description] dataUsingEncoding:NSUTF8StringEncoding];
	self.result = [NSJSONSerialization JSONObjectWithData:d options:0 error:nil] ?: @{};
}
@end

// One WKWebView in a window of its own: requestAnimationFrame's rate over -t seconds, after a
// second's warm-up, with a box moved each frame so every frame is a rendering update.
static NSDictionary *runPage(NSScreen *screen, id feature, BOOL near60Off, unsigned *nominalBefore)
{
	WKWebViewConfiguration *cfg = [WKWebViewConfiguration new];
	if (near60Off && !setNear60(cfg.preferences, feature, NO))
		return @{@"error" : @"near-60fps feature could not be turned off"};
	PageResult *handler = [PageResult new];
	[cfg.userContentController addScriptMessageHandler:handler name:@"r"];
	NSString *html = [NSString
	    stringWithFormat:
	        @"<!doctype html><html><body style='margin:0;background:#777'>"
	        @"<div id=b style='width:200px;height:200px;background:#c33'></div><script>"
	        @"const W=1000,S=%.0f;let t0=null,ts=[];"
	        @"function f(t){if(t0===null)t0=t;if(t-t0>=W)ts.push(t);"
	        @"if(t-t0<W+S){document.getElementById('b').style.transform='translateX('+(t%%600)+'px)';"
	        @"requestAnimationFrame(f);}else{const g=[];for(let i=1;i<ts.length;i++)g.push(ts[i]-ts[i-1]);"
	        @"g.sort((a,b)=>a-b);const span=(ts[ts.length-1]-ts[0])/1000;"
	        @"window.webkit.messageHandlers.r.postMessage(JSON.stringify({frames:ts.length,"
	        @"fps:span>0?(ts.length-1)/span:0,p50:g.length?g[Math.floor(g.length/2)]:0,"
	        @"p05:g.length?g[Math.floor(g.length*0.05)]:0,p95:g.length?g[Math.floor(g.length*0.95)]:0}));}}"
	        @"requestAnimationFrame(f);</script></body></html>",
	        gSeconds * 1e3];
	const NSRect frame = NSMakeRect(80, 80, 800, 500);
	NSWindow *win = [[NSWindow alloc] initWithContentRect:frame
	                                            styleMask:NSWindowStyleMaskTitled
	                                              backing:NSBackingStoreBuffered
	                                                defer:NO
	                                               screen:screen];
	WKWebView *view = [[WKWebView alloc] initWithFrame:NSMakeRect(0, 0, 800, 500) configuration:cfg];
	win.contentView = view;
	CVTime p;
	*nominalBefore = readNominal(CGMainDisplayID(), &p) ? webkitNominal(p) : 0;
	[win makeKeyAndOrderFront:nil];
	[win orderFrontRegardless];
	[view loadHTMLString:html baseURL:nil];
	spin(gSeconds + 15, ^BOOL {
		return handler.result != nil;
	});
	NSDictionary *r = handler.result ?: @{@"error" : @"no result from the page"};
	[cfg.userContentController removeScriptMessageHandlerForName:@"r"];
	[win orderOut:nil];
	return r;
}

static NSString *run(void)
{
	[NSApplication sharedApplication];
	[NSApp setActivationPolicy:gWebView ? NSApplicationActivationPolicyRegular
	                                    : NSApplicationActivationPolicyAccessory];
	[NSApp finishLaunching];
	NSScreen *screen = NSScreen.mainScreen;
	if (screen == nil)
		return @"FAIL no screen (run it in the logged-in user's session)";
	const CGDirectDisplayID main = CGMainDisplayID();

	if (gNominalOnly) {
		const NominalRange n = sampleNominal(main);
		return [NSString stringWithFormat:@"OK nominal_fps_min=%u nominal_fps_max=%u", n.min, n.max];
	}
	dumpDisplays();
	const long screenMax = dumpScreens();
	const NominalRange nominal = sampleNominal(main);
	const double linkHz = runLink(main);
	runCADisplayLink(screen);

	id feature = near60Feature();
	WKPreferences *fresh = [WKPreferences new];
	const int near60 = near60Enabled(fresh, feature);
	const int near60Default = feature ? (int)[[feature valueForKey:@"defaultValue"] boolValue] : -1;
	const BOOL on = near60 != 0;
	NSString *(^predict)(unsigned) = ^NSString *(unsigned n) {
		const unsigned asked = webkitAsked(n, on);
		const unsigned factor = webkitFactor(n, asked);
		return [NSString stringWithFormat:@"%u:asks_%u_every_%u=%.1f", n, asked, factor, linkHz / factor];
	};
	say(@"webkit",
	    [NSString stringWithFormat:@"feature_spi=%d near60_default=%d near60_fresh=%d predicted=%@%@%@",
	                               feature != nil, near60Default, near60, predict(nominal.min),
	                               nominal.max != nominal.min ? @"," : @"",
	                               nominal.max != nominal.min ? predict(nominal.max) : @""]);
	const unsigned asked = webkitAsked(nominal.max, on);
	const double predictedHz = linkHz / webkitFactor(nominal.max, asked);

	NSMutableString *pages = [NSMutableString string];
	if (gWebView) {
		for (int off = 0; off < 2; off++) {
			unsigned before = 0;
			NSDictionary *r = runPage(screen, feature, off, &before);
			const BOOL n60 = off ? NO : on;
			const unsigned a = webkitAsked(before, n60);
			const double pred = linkHz / webkitFactor(before, a);
			NSString *name = off ? @"near60_off" : @"default";
			if (r[@"error"]) {
				say(@"page", [NSString stringWithFormat:@"run=%@ error=\"%@\"", name, r[@"error"]]);
				[pages appendFormat:@" raf_%@=error", name];
				continue;
			}
			say(@"page", [NSString stringWithFormat:@"run=%@ nominal_before=%u asks=%u predicted_hz=%.1f raf_hz=%.2f "
			                                       @"frames=%@ interval_ms_p05=%.2f interval_ms_p50=%.2f "
			                                       @"interval_ms_p95=%.2f",
			                                       name, before, a, pred, [r[@"fps"] doubleValue], r[@"frames"],
			                                       [r[@"p05"] doubleValue], [r[@"p50"] doubleValue],
			                                       [r[@"p95"] doubleValue]]);
			[pages appendFormat:@" raf_%@=%.1f predicted_%@=%.1f", name, [r[@"fps"] doubleValue], name, pred];
		}
	}
	return [NSString stringWithFormat:@"OK nominal_fps_min=%u nominal_fps_max=%u link_hz=%.1f screen_max_fps=%ld "
	                                  @"near60=%d predicted_raf_hz=%.1f%@",
	                                  nominal.min, nominal.max, linkHz, screenMax, near60, predictedHz, pages];
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		int c;
		while ((c = getopt(argc, argv, "t:k:nw")) != -1) {
			switch (c) {
			case 't':
				gSeconds = atof(optarg);
				break;
			case 'k':
				gSamples = atoi(optarg);
				break;
			case 'n':
				gNominalOnly = YES;
				break;
			case 'w':
				gWebView = YES;
				break;
			default:
				fprintf(stderr, "usage: display_rate_probe [-t SECONDS] [-k SAMPLES] [-n] [-w]\n");
				return 2;
			}
		}
		if (gSeconds <= 0 || gSamples < 1) {
			fprintf(stderr, "display_rate_probe: -t must be positive and -k at least 1\n");
			return 2;
		}
		NSString *r = run();
		printf("display_rate_probe: %s\n", r.UTF8String);
		return [r hasPrefix:@"OK"] ? 0 : 1;
	}
}
