// winmon: what the window server says of the windows on screen, for a driver that moves them from
// outside the guest (the lane gate's window drags) and must know a window really moved.
//   winmon screen              the main display's size in points: "screen W H"
//   winmon list                the on-screen windows of the normal layer, front to back, one a
//                              line: "window ID PID OWNER X Y W H" (points, from the top left)
//   winmon all                 every on-screen window but the desktop's, front to back, one a
//                              line: "window ID PID OWNER LAYER X Y W H"
//   winmon track ID SECONDS [MS] [size]
//                              the window's origin (or with size, its size) every MS milliseconds
//                              (default 10) for SECONDS: "at EPOCH X Y" (or "at EPOCH W H") each
//                              time it changes, then "track samples=N moves=M" (M the samples
//                              whose value differed from the one before)
// The window server answers bounds, owners and layers without a screen-recording grant; titles
// it would not, and winmon reads none.
// Build (-arch arm64 for an Apple-silicon Mac):
//   xcrun clang -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -O2 -Wall -Wextra -Werror
//       -Wno-unused-parameter -framework Foundation -framework CoreGraphics winmon.m -o winmon
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

static double epoch(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1e6;
}

static BOOL boundsOf(NSDictionary *w, CGRect *r)
{
	CFDictionaryRef b = (__bridge CFDictionaryRef)w[(__bridge NSString *)kCGWindowBounds];
	return b && CGRectMakeWithDictionaryRepresentation(b, r);
}

static int list(BOOL all)
{
	NSArray *ws = CFBridgingRelease(CGWindowListCopyWindowInfo(
	    kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID));
	for (NSDictionary *w in ws) {
		int layer = [w[(__bridge NSString *)kCGWindowLayer] intValue];
		if (!all && layer != 0)
			continue;
		CGRect r;
		if (!boundsOf(w, &r))
			continue;
		NSString *owner = w[(__bridge NSString *)kCGWindowOwnerName] ?: @"?";
		const char *name = [[owner stringByReplacingOccurrencesOfString:@" " withString:@"_"] UTF8String];
		unsigned wid = [w[(__bridge NSString *)kCGWindowNumber] unsignedIntValue];
		int pid = [w[(__bridge NSString *)kCGWindowOwnerPID] intValue];
		if (all)
			printf("window %u %d %s %d %.0f %.0f %.0f %.0f\n", wid, pid, name, layer, r.origin.x, r.origin.y,
			       r.size.width, r.size.height);
		else
			printf("window %u %d %s %.0f %.0f %.0f %.0f\n", wid, pid, name, r.origin.x, r.origin.y, r.size.width,
			       r.size.height);
	}
	return 0;
}

static int track(CGWindowID wid, double seconds, int ms, BOOL size)
{
	CFArrayRef ids = CFArrayCreate(NULL, (const void *[]){(const void *)(uintptr_t)wid}, 1, NULL);
	double end = epoch() + seconds;
	CGPoint last = CGPointMake(-1e9, -1e9);
	long samples = 0, moves = 0;
	while (epoch() < end) {
		NSArray *d = CFBridgingRelease(CGWindowListCreateDescriptionFromArray(ids));
		CGRect r;
		if (d.count == 1 && boundsOf(d[0], &r)) {
			samples++;
			CGPoint v = size ? CGPointMake(r.size.width, r.size.height) : r.origin;
			if (v.x != last.x || v.y != last.y) {
				if (samples > 1)
					moves++;
				printf("at %.3f %.0f %.0f\n", epoch(), v.x, v.y);
				fflush(stdout);
				last = v;
			}
		}
		usleep(ms * 1000);
	}
	CFRelease(ids);
	printf("track samples=%ld moves=%ld\n", samples, moves);
	return samples ? 0 : 1;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc == 2 && !strcmp(argv[1], "screen")) {
			CGRect b = CGDisplayBounds(CGMainDisplayID());
			printf("screen %.0f %.0f\n", b.size.width, b.size.height);
			return 0;
		}
		if (argc == 2 && (!strcmp(argv[1], "list") || !strcmp(argv[1], "all")))
			return list(!strcmp(argv[1], "all"));
		BOOL size = argc == 6 && !strcmp(argv[5], "size");
		if ((argc == 4 || argc == 5 || size) && !strcmp(argv[1], "track"))
			return track((CGWindowID)strtoul(argv[2], NULL, 10), atof(argv[3]), argc >= 5 ? atoi(argv[4]) : 10, size);
		fprintf(stderr, "usage: winmon screen | list | all | track ID SECONDS [MS] [size]\n");
		return 2;
	}
}
