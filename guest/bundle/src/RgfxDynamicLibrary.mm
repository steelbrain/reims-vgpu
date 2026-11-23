/*
 * RgfxDynamicLibrary: see RgfxDynamicLibrary.h. Our own ObjC++.
 */

#import "RgfxDynamicLibrary.h"

#import "RgfxLibrary.h"
#include "RgfxFiles.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"

#include <cerrno>

namespace {

// Foundation's error for a write that failed with `posix`: NSCocoaErrorDomain, its write code for
// the errno, the file's path and the POSIX error under it (read from code: swift-corelibs-
// foundation's _NSErrorWithErrno, for a write).
NSError *writeError(int posix, NSString *path)
{
	NSInteger code;
	switch (posix) {
	case ENOENT:
		code = NSFileNoSuchFileError;
		break;
	case EPERM:
	case EACCES:
		code = NSFileWriteNoPermissionError;
		break;
	case ENAMETOOLONG:
		code = NSFileWriteInvalidFileNameError;
		break;
	case EDQUOT:
	case ENOSPC:
		code = NSFileWriteOutOfSpaceError;
		break;
	case EROFS:
		code = NSFileWriteVolumeReadOnlyError;
		break;
	case EEXIST:
		code = NSFileWriteFileExistsError;
		break;
	default:
		code = NSFileWriteUnknownError;
		break;
	}
	return [NSError errorWithDomain:NSCocoaErrorDomain
	                           code:code
	                       userInfo:@{
		                       NSFilePathErrorKey : path ?: @"",
		                       NSUnderlyingErrorKey : [NSError errorWithDomain:NSPOSIXErrorDomain
		                                                                  code:posix
		                                                              userInfo:nil]
	                       }];
}

} // namespace

@implementation RgfxDynamicLibrary {
	RgfxLibrary *_library;
}

// The host drops the dynamic library as the app lets it go; a pipeline made with it keeps what it
// linked.
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)dynamicLibraryOfLibrary:(RgfxLibrary *)library
                                 device:(id<MTLDevice>)device
                             serializer:(RgfxSerializer *)serializer
                                 status:(uint32_t *)status
{
	*status = kRgfxNoAnswer;
	if (serializer == nullptr || library == nil)
		return nil;
	RgfxCreated created = serializer->createDynamicLibrary(library.rgfxObjId);
	if (!created.posted) {
		*status = kRgfxNoAnswer;
		return nil;
	}
	*status = created.status;
	if (created.status != RGFX_STATUS_OK)
		return nil;
	RgfxDynamicLibrary *made = [[RgfxDynamicLibrary alloc] initWithObjId:created.obj_id
	                                                               device:device
	                                                           serializer:serializer];
	if (made)
		made->_library = library;
	return made;
}

- (RgfxLibrary *)rgfxLibrary
{
	return _library;
}

- (dispatch_data_t)airData
{
	return [_library rgfxSource];
}

// --- MTLDynamicLibrary -----------------------------------------------------------------

- (NSString *)installName
{
	return _library.installName;
}

// The library's .metallib, written through a file beside `url` and a rename (RgfxWriteFileWhole,
// with the bundle's own file calls, which a sandboxed app's profile takes: RgfxFiles.hpp). A URL
// that is no file URL, or in a directory that is not there, is Metal's "Invalid URL"
// (MTLDynamicLibraryDomain 1, measured: the AMD Radeon Pro 5300M and the Intel UHD Graphics 630,
// macOS 26.7.1, oracle_fnptr), and writes nothing; another failure to write is Foundation's error
// for the write's errno (writeError).
- (BOOL)serializeToURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	auto invalid = ^BOOL(NSString *what) {
		if (error)
			*error = [NSError errorWithDomain:MTLDynamicLibraryDomain
			                             code:MTLDynamicLibraryErrorInvalidFile
			                         userInfo:@{NSLocalizedDescriptionKey : what}];
		return NO;
	};
	const char *directory = url.fileURL ? url.URLByDeletingLastPathComponent.fileSystemRepresentation : nullptr;
	if (directory == nullptr || !RgfxIsDirectory(directory))
		return invalid(@"Invalid URL");
	dispatch_data_t source = [_library rgfxSource];
	if (source == nil)
		return invalid(@"the library kept no bytes to write");
	const void *bytes = nullptr;
	size_t size = 0;
	dispatch_data_t map = dispatch_data_create_map(source, &bytes, &size);
	const char *path = url.fileSystemRepresentation;
	const int failure = path != nullptr ? RgfxWriteFileWhole(path, bytes, size) : ENOENT;
	(void)map;
	if (failure != 0) {
		if (error)
			*error = writeError(failure, url.path);
		return NO;
	}
	return YES;
}

@end
