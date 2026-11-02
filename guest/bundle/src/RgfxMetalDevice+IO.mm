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

// A raw or a compressed handle over the file at `url`, reading through the page cache unless
// `uncached`.
- (nullable id<MTLIOFileHandle>)rgfxIOHandleWithURL:(NSURL *)url
                                         compressed:(BOOL)compressed
                                    compressionType:(NSInteger)compressionType
                                           uncached:(BOOL)uncached
                                              error:(NSError **)error
{
	Class handle =
	    compressed
	        ? rgfxMetalIOClass("_MTLIOHandleCompressed",
	                           @selector(initWithDevice:path:compressionType:error:uncached:))
	        : rgfxMetalIOClass("_MTLIOHandleRaw", @selector(initWithDevice:path:error:uncached:));
	if (handle == Nil) {
		RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
		return nil;
	}
	if (!rgfxMetalIOFileIsThere(url, error))
		return nil;
	if (!compressed)
		return [[handle alloc] initWithDevice:self
		                                 path:url.fileSystemRepresentation
		                                error:error
		                             uncached:uncached];
	return [[handle alloc] initWithDevice:self
	                                 path:url.fileSystemRepresentation
	                      compressionType:compressionType
	                                error:error
	                             uncached:uncached];
}

- (id<MTLIOFileHandle>)newIOFileHandleWithURL:(NSURL *)url error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url compressed:NO compressionType:0 uncached:NO error:error];
}

- (id<MTLIOFileHandle>)newIOFileHandleWithURL:(NSURL *)url
                            compressionMethod:(MTLIOCompressionMethod)compressionMethod
                                        error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url
	                      compressed:YES
	                 compressionType:compressionMethod
	                        uncached:NO
	                           error:error];
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

// The forms no header declares (measured: IOGPUMetalDevice's on macOS 27's x86-64 slice,
// `@40@0:8@16q24^@32` with a compression type, `@32@0:8@16^@24` without): the same handles,
// uncached for the Uncached ones. The compression type is the compressed handle's own
// (`initWithDevice:path:compressionType:error:uncached:`), which the public forms pass their
// compression method as.
- (id<MTLIOFileHandle>)newIOHandleWithURL:(NSURL *)url
                          compressionType:(NSInteger)compressionType
                                    error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url
	                      compressed:YES
	                 compressionType:compressionType
	                        uncached:NO
	                           error:error];
}

- (id<MTLIOFileHandle>)newUncachedIOHandleWithURL:(NSURL *)url error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url compressed:NO compressionType:0 uncached:YES error:error];
}

- (id<MTLIOFileHandle>)newUncachedIOFileHandleWithURL:(NSURL *)url error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url compressed:NO compressionType:0 uncached:YES error:error];
}

- (id<MTLIOFileHandle>)newUncachedIOHandleWithURL:(NSURL *)url
                                  compressionType:(NSInteger)compressionType
                                            error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url
	                      compressed:YES
	                 compressionType:compressionType
	                        uncached:YES
	                           error:error];
}

- (id<MTLIOFileHandle>)newUncachedIOFileHandleWithURL:(NSURL *)url
                                    compressionMethod:(NSInteger)compressionMethod
                                                error:(NSError **)error
{
	return [self rgfxIOHandleWithURL:url
	                      compressed:YES
	                 compressionType:compressionMethod
	                        uncached:YES
	                           error:error];
}

@end
