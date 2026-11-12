/*
 * RgfxLibraryFile: see the header. Our own ObjC++.
 */

#import "RgfxLibraryFile.h"

#import <Metal/Metal.h>

#include <errno.h>
#include <string.h>
#include <sys/stat.h>

namespace {

NSError *libraryError(MTLLibraryError code, NSString *description)
{
	return [NSError errorWithDomain:MTLLibraryErrorDomain
	                           code:code
	                       userInfo:@{NSLocalizedDescriptionKey : description}];
}

NSError *notFound()
{
	return libraryError(MTLLibraryErrorFileNotFound, @"library not found");
}

NSError *endOfFile()
{
	return libraryError(MTLLibraryErrorUnsupported, @"Invalid library file (unexpected end of file)");
}

// Metal's error for a file it could not open, `reason` strerror's words for the open's error.
NSError *notOpened(NSString *path, NSString *reason)
{
	return libraryError(MTLLibraryErrorUnsupported,
	                    [NSString stringWithFormat:@"metal library file %@ failed to open with error: %@", path,
	                                               reason]);
}

NSString *reasonOf(int code)
{
	return @(strerror(code));
}

// The POSIX error under Foundation's error of a read, 0 where it names none.
int posixCodeOf(NSError *error)
{
	for (NSError *e = error; e != nil; e = e.userInfo[NSUnderlyingErrorKey]) {
		if ([e.domain isEqualToString:NSPOSIXErrorDomain])
			return static_cast<int>(e.code);
	}
	return 0;
}

} // namespace

NSString *RgfxLibraryFilePath(NSURL *url)
{
	NSString *path = url.path;
	return path.length != 0 ? path : nil;
}

NSData *RgfxLibraryFileData(NSString *path, NSError *__autoreleasing *error)
{
	NSError *answer = nil;
	NSData *data = nil;
	struct stat st = {};
	if (path == nil) {
		answer = libraryError(MTLLibraryErrorFileNotFound, @"Invalid file url");
	} else if (stat(path.fileSystemRepresentation, &st) != 0) {
		const int code = errno;
		answer = code == ENOENT ? notFound() : notOpened(path, reasonOf(code));
	} else if (S_ISDIR(st.st_mode)) {
		answer = endOfFile();
	} else {
		NSError *readError = nil;
		data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:&readError];
		if (data == nil) {
			// Foundation's error names the read's POSIX error; where it names none, its own words
			// stand for strerror's.
			const int code = posixCodeOf(readError);
			answer = code == ENOENT ? notFound()
			                        : notOpened(path, code != 0 ? reasonOf(code) : readError.localizedDescription);
		} else if (data.length == 0) {
			data = nil;
			answer = endOfFile();
		}
	}
	if (data == nil && error != nullptr)
		*error = answer;
	return data;
}
