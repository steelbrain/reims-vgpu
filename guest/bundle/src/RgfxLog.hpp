// RgfxLog: the os_log object the bundle's os_log lines go to. Its subsystem is the bundle's
// identifier, so `log config --subsystem ai.aneesiqbal.reims-graphics` sets the level those lines
// are kept at and persisted to, on every release: a line on OS_LOG_DEFAULT has no subsystem, and
// no mode persists its debug lines (docs/lessons.md). Its Objective-C sources log through
// rgfxLogLine, at the default level, which the log store keeps, with their text public.
// tests/bundle_log_check.sh keeps the subsystem equal to Info.plist's CFBundleIdentifier and the
// bundle off OS_LOG_DEFAULT and NSLog.

#pragma once

#include <os/log.h>

// The subsystem of the bundle's lines: its CFBundleIdentifier.
#define RGFX_LOG_SUBSYSTEM "ai.aneesiqbal.reims-graphics"
// Their category: one for the whole bundle, Metal device and video driver alike.
#define RGFX_LOG_CATEGORY "bundle"

// The bundle's log object, made at the first call in a process and never released.
os_log_t rgfxLog();

// Whether a line also goes to the process's stderr, as NSLog's does: where it is a file, a
// socket, a pipe or a terminal, not /dev/null or nothing.
bool rgfxLogToStderr();

// A nil a creator of the device's or of its heaps' (`creator`) answers for `cause`, asked for
// `asked`: logged at the info level (`log show --info`) once per creator and cause in the process
// (docs/conventions.md), what was asked the first time named, so a process that ends at a nil it
// was given (14.6.1's WindowServer ends at a texture's, in SkyLight's
// MetalBacking::AbortWithTextureInfo) has the device's answer named beside it. A creator asked
// every frame, the same or not, logs once.
void rgfxNoteNil(const char *creator, const char *cause, const char *asked);

#ifdef __OBJC__
#import <Foundation/Foundation.h>

#include <stdio.h>

// A line of the bundle's at the default level, on its subsystem, with all of its text public:
// it is formatted here, as NSLog formats, and logged as one public string, and written to stderr
// where NSLog writes there too (a death's words among them, which tests/shell_tests.mm's death
// checks read). An NSLog argument the unified log takes for private (an object, or a string made
// at run time) reads `<private>` on macOS 26, so a line naming what a guest's framework asked for
// named nothing there.
NS_FORMAT_FUNCTION(1, 2) static inline void rgfxLogLine(NSString *format, ...)
{
	va_list args;
	va_start(args, format);
	NSString *line = [[NSString alloc] initWithFormat:format arguments:args];
	va_end(args);
	os_log(rgfxLog(), "%{public}@", line);
	if (rgfxLogToStderr())
		fprintf(stderr, "%s\n", line.UTF8String);
}
#endif
