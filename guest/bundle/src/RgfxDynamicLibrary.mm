/*
 * RgfxDynamicLibrary: see RgfxDynamicLibrary.h. Our own ObjC++.
 */

#import "RgfxDynamicLibrary.h"

#import "RgfxLibrary.h"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"

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

// The library's .metallib, written through a file beside `url` and a rename. A URL that is no file
// URL, or in a directory that is not there, is Metal's "Invalid URL" (MTLDynamicLibraryDomain 1,
// measured: the AMD Radeon Pro 5300M and the Intel UHD Graphics 630, macOS 26.7.1, oracle_fnptr),
// and writes nothing; another failure to write is the write's own error.
- (BOOL)serializeToURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	auto invalid = ^BOOL(NSString *what) {
		if (error)
			*error = [NSError errorWithDomain:MTLDynamicLibraryDomain
			                             code:MTLDynamicLibraryErrorInvalidFile
			                         userInfo:@{NSLocalizedDescriptionKey : what}];
		return NO;
	};
	BOOL isDirectory = NO;
	NSString *directory = url.fileURL ? url.URLByDeletingLastPathComponent.path : nil;
	if (directory == nil ||
	    ![[NSFileManager defaultManager] fileExistsAtPath:directory isDirectory:&isDirectory] ||
	    !isDirectory)
		return invalid(@"Invalid URL");
	dispatch_data_t source = [_library rgfxSource];
	if (source == nil)
		return invalid(@"the library kept no bytes to write");
	const void *bytes = nullptr;
	size_t size = 0;
	dispatch_data_t map = dispatch_data_create_map(source, &bytes, &size);
	NSData *data = [NSData dataWithBytes:bytes length:size];
	(void)map;
	return [data writeToURL:url options:NSDataWritingAtomic error:error];
}

@end
