// oracle_common.h: helpers the oracle probes share. Each probe is one x86-64 (or arm64) ObjC
// program that picks a Metal device by a substring of its name from MTLCopyAllDevices (never
// MTLCreateSystemDefaultDevice, which switches a dual-GPU MacBook's display to its discrete
// GPU), runs its steps and prints one table line per step.
#ifndef ORACLE_COMMON_H
#define ORACLE_COMMON_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static inline double oracle_now_ms(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// The device whose name holds `want` (case-insensitive); exits 2 naming the devices there are.
static inline id<MTLDevice> oracle_device(const char *want)
{
	NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
	for (id<MTLDevice> d in devs) {
		if (strcasestr(d.name.UTF8String, want))
			return d;
	}
	fprintf(stderr, "no Metal device matching '%s'; there are:", want);
	for (id<MTLDevice> d in devs)
		fprintf(stderr, " [%s]", d.name.UTF8String);
	fprintf(stderr, "\n");
	exit(2);
}

// One header line: the device, its memory model, the machine and the OS.
static inline void oracle_header(const char *probe, id<MTLDevice> d)
{
	char model[64] = "?";
	size_t len = sizeof(model);
	sysctlbyname("hw.model", model, &len, NULL, 0);
	char osb[32] = "?";
	len = sizeof(osb);
	sysctlbyname("kern.osversion", osb, &len, NULL, 0);
	NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
	printf("# %s on \"%s\" (unified memory %d, low power %d), %s, macOS %ld.%ld.%ld (%s), %s\n",
	       probe, d.name.UTF8String, (int)d.hasUnifiedMemory, (int)d.isLowPower, model,
	       (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, osb,
#if defined(__x86_64__)
	       "x86_64"
#else
	       "arm64"
#endif
	);
}

// A command buffer's commit-to-completion wall time and GPU time, in ms.
static inline void oracle_commit_wait(id<MTLCommandBuffer> cb, double *wall_ms, double *gpu_ms)
{
	double t0 = oracle_now_ms();
	[cb commit];
	[cb waitUntilCompleted];
	if (wall_ms)
		*wall_ms = oracle_now_ms() - t0;
	if (gpu_ms)
		*gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
	if (cb.status != MTLCommandBufferStatusCompleted)
		printf("  ! command buffer status %ld error %s\n", (long)cb.status,
		       cb.error ? cb.error.description.UTF8String : "(none)");
}

// The metallib next to the executable (NAME.metallib), or nil with a line saying why.
static inline id<MTLLibrary> oracle_library(id<MTLDevice> d, const char *argv0, const char *name)
{
	NSString *exe = [NSString stringWithUTF8String:argv0];
	if (![exe containsString:@"/"])
		exe = [@"./" stringByAppendingString:exe];
	NSString *dir = [[exe stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent];
	NSString *path = [dir stringByAppendingPathComponent:
	                          [NSString stringWithFormat:@"%s.metallib", name]];
	NSError *err = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
	if (!lib)
		printf("  ! library %s: %s\n", path.UTF8String,
		       err ? err.description.UTF8String : "nil");
	return lib;
}

#endif
