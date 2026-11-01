/*
 * RgfxMetalDevice (IO): see the header. Our own ObjC++.
 */

#import "RgfxMetalDevice+IO.h"

#import "RgfxMetalPrivate.h"
#import "RgfxUnsupported.h"

#import <objc/runtime.h>

namespace {
constexpr const char *kIO = "Metal IO command queues";

/// The class named `name` where it answers `init`, else Nil.
Class rgfxMetalIOClass(const char *name, SEL init)
{
	Class c = objc_getClass(name);
	return c != Nil && [c instancesRespondToSelector:init] ? c : Nil;
}

/// What Metal's device checks of a handle's URL before it opens the file (measured on the M3 Max):
/// a URL that is not a file's is MTLIOErrorDomain's 1, "URL is not a file"; a file that is not
/// there is the error checkResourceIsReachableAndReturnError: gives (NSCocoaErrorDomain 260, with
/// the URL, the path and the POSIX error). Every other answer is the handle's own.
BOOL rgfxMetalIOFileIsThere(NSURL *url, NSError **error)
{
	if (url == nil || !url.isFileURL) {
		if (error)
			*error = [NSError errorWithDomain:MTLIOErrorDomain
			                             code:MTLIOErrorURLInvalid
			                         userInfo:@{NSLocalizedDescriptionKey : @"URL is not a file"}];
		return NO;
	}
	return [url checkResourceIsReachableAndReturnError:error];
}
} // namespace

@implementation RgfxMetalDevice (IO)

- (id<MTLIOCommandQueue>)newIOCommandQueueWithDescriptor:(MTLIOCommandQueueDescriptor *)descriptor
                                                   error:(NSError **)error
{
	Class queue = rgfxMetalIOClass("_MTLIOCommandQueue", @selector(initWithDevice:descriptor:));
	if (queue == Nil) {
		RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
		return nil;
	}
	return [[queue alloc] initWithDevice:self descriptor:descriptor];
}

- (id<MTLIOFileHandle>)newIOFileHandleWithURL:(NSURL *)url error:(NSError **)error
{
	Class handle =
	    rgfxMetalIOClass("_MTLIOHandleRaw", @selector(initWithDevice:path:error:uncached:));
	if (handle == Nil) {
		RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
		return nil;
	}
	if (!rgfxMetalIOFileIsThere(url, error))
		return nil;
	return [[handle alloc] initWithDevice:self
	                                 path:url.fileSystemRepresentation
	                                error:error
	                             uncached:NO];
}

- (id<MTLIOFileHandle>)newIOFileHandleWithURL:(NSURL *)url
                            compressionMethod:(MTLIOCompressionMethod)compressionMethod
                                        error:(NSError **)error
{
	Class handle = rgfxMetalIOClass("_MTLIOHandleCompressed",
	                                @selector(initWithDevice:path:compressionType:error:uncached:));
	if (handle == Nil) {
		RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
		return nil;
	}
	if (!rgfxMetalIOFileIsThere(url, error))
		return nil;
	return [[handle alloc] initWithDevice:self
	                                 path:url.fileSystemRepresentation
	                      compressionType:compressionMethod
	                                error:error
	                             uncached:NO];
}

// The names Metal deprecated for the two above.
- (id<MTLIOFileHandle>)newIOHandleWithURL:(NSURL *)url error:(NSError **)error
{
	return [self newIOFileHandleWithURL:url error:error];
}

- (id<MTLIOFileHandle>)newIOHandleWithURL:(NSURL *)url
                        compressionMethod:(MTLIOCompressionMethod)compressionMethod
                                    error:(NSError **)error
{
	return [self newIOFileHandleWithURL:url compressionMethod:compressionMethod error:error];
}

@end
