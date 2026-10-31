/*
 * RgfxUnsupported: how the bundle answers a member of Metal's public protocols whose feature
 * the device does not support: one the wire does not carry yet (argument encoders,
 * automatic heap placement), or one the device answers NO for (ray tracing, function pointers, Metal 4, mesh
 * shaders, counters, residency sets, indirect command buffers, ...). The device must not
 * promise such a feature, so such a member fails the way Metal fails an unsupported use:
 *   - where the member has an error out, it answers nil and an NSError (RGFX_UNSUPPORTED_ERROR);
 *   - where it returns an object, it answers nil;
 *   - otherwise it does nothing, or answers the value a device without the feature answers.
 * Every such member logs, once per class and selector, that it was asked
 * ("RGFX: unsupported: -[RgfxHeap newAccelerationStructureWithSize:]: ray tracing"), so the
 * work that relied on it is seen, and is marked RGFX_UNSUPPORTED in the source, so the count of
 * knowingly stubbed members shows (the bundle's Makefile prints it at build time; each one, with
 * what it needs on the wire, is listed in the effort's protocol-gaps-wire.md).
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_UNSUPPORTED_H_
#define RGFX_UNSUPPORTED_H_

#import <Foundation/Foundation.h>

#import "RgfxMetalError.h"

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/// Logs, the first time `cls` is asked `cmd`, that the member is not supported: `what` names
/// the missing feature. Thread-safe.
void RgfxLogUnsupported(Class cls, SEL cmd, const char *what);

/// Whether `cls` has been asked `cmd` and logged it (for the tests).
BOOL RgfxUnsupportedWasLogged(Class cls, SEL cmd);

/// The error an unsupported member answers through its error out: `domain` and `code` where
/// Metal names an unsupported use in its own domain (MTLLibraryErrorUnsupported,
/// MTLDynamicLibraryErrorUnsupported), else RgfxMetalErrorDomain / RgfxMetalErrorUnsupported.
NSError *RgfxUnsupportedError(Class cls, SEL cmd, const char *what, NSErrorDomain domain,
                              NSInteger code);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END

/// Marks a member stubbed knowingly and logs its first use. `what` is a C string naming the
/// feature the device does not support.
#define RGFX_UNSUPPORTED(what) RgfxLogUnsupported([self class], _cmd, what)

/// RGFX_UNSUPPORTED, and sets `*(errp)` (when `errp` is not null) to the bundle's
/// unsupported error.
#define RGFX_UNSUPPORTED_ERROR(errp, what)                                                         \
	do {                                                                                           \
		RgfxLogUnsupported([self class], _cmd, what);                                              \
		if ((errp) != nullptr)                                                                     \
			*(errp) = RgfxUnsupportedError([self class], _cmd, what, RgfxMetalErrorDomain,         \
			                               RgfxMetalErrorUnsupported);                             \
	} while (0)

/// RGFX_UNSUPPORTED_ERROR in Metal's own `domain`, with its `code`.
#define RGFX_UNSUPPORTED_ERROR_IN(errp, what, domain, code)                                        \
	do {                                                                                           \
		RgfxLogUnsupported([self class], _cmd, what);                                              \
		if ((errp) != nullptr)                                                                     \
			*(errp) = RgfxUnsupportedError([self class], _cmd, what, domain, code);                \
	} while (0)

#endif // RGFX_UNSUPPORTED_H_
