/*
 * RgfxMetalError: the reims-graphics bundle's NSError domain and a small helper for
 * the shell's refusals. The pipeline/library factories and the device category hand
 * these back where a Metal call cannot be satisfied — a missing serialiser route, a
 * create that did not fit the ring, or a path the shell does not implement (source
 * compilation, a default library). The shell refuses such a call rather than
 * fabricating a result, the way the host device-info answers an unsupported query.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_METAL_ERROR_H_
#define RGFX_METAL_ERROR_H_

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// The bundle's NSError domain.
static NSErrorDomain const RgfxMetalErrorDomain = @"RgfxMetalErrorDomain";

typedef NS_ENUM(NSInteger, RgfxMetalErrorCode) {
	/// No serialiser route: the device's ring transport is not up.
	RgfxMetalErrorNoRoute = 1,
	/// The create's wire message did not fit the control ring.
	RgfxMetalErrorRingFull = 2,
	/// The descriptor was missing a required part (e.g. a pipeline's function).
	RgfxMetalErrorInvalidDescriptor = 3,
	/// A path the shell does not implement off the host compiler (source→AIR
	/// compilation, a default library baked into the bundle).
	RgfxMetalErrorUnsupported = 4,
	/// The device's GPU refused the object (a library whose bytes it could not build).
	RgfxMetalErrorRefused = 5,
};

/// Builds an NSError in RgfxMetalErrorDomain with `code` and `message`.
static inline NSError *RgfxMakeError(RgfxMetalErrorCode code, NSString *message)
{
	return [NSError errorWithDomain:RgfxMetalErrorDomain
	                           code:code
	                       userInfo:@{NSLocalizedDescriptionKey : message}];
}

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_ERROR_H_
