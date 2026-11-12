/*
 * RgfxLibraryFile: the bytes of the library file a URL names, read as Metal's
 * newLibraryWithURL:error: reads them, for the device's newLibraryWithURL:error: (and
 * newLibraryWithFile:error:, which makes a file URL of its path).
 *
 * Metal opens the file at the URL's path, whatever the URL's scheme: a URL of no scheme is a path,
 * relative to the current directory where it is relative, its percent escapes decoded; a URL of
 * another scheme (http, one of no meaning) is the file at its path too. It answers nil and an
 * error of MTLLibraryErrorDomain for a file it cannot read (measured: macOS 26.7.1, the AMD Radeon
 * Pro 5300M and the Intel UHD 630, and macOS 27.0.1, the M3 Max, all alike, a .metallib of our own
 * at each URL form):
 *   - a URL of no path (data:): MTLLibraryErrorFileNotFound (6), "Invalid file url";
 *   - no file at the path: MTLLibraryErrorFileNotFound (6), "library not found";
 *   - a file it may not open: MTLLibraryErrorUnsupported (1), "metal library file PATH failed to
 *     open with error: Permission denied", strerror's words for EACCES, the one error of an open
 *     measured besides ENOENT; another error is answered in the same form, with its own words;
 *   - a directory, or an empty file: MTLLibraryErrorUnsupported (1), "Invalid library file
 *     (unexpected end of file)".
 * Each error's userInfo holds its description alone. Bytes that are not a library are the
 * library's to refuse, as newLibraryWithData:error: refuses them.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_LIBRARY_FILE_H_
#define RGFX_LIBRARY_FILE_H_

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// The path Metal opens for `url`: its path, whatever its scheme; nil where it has none.
NSString *_Nullable RgfxLibraryFilePath(NSURL *url);

/// The bytes of the file at `path` (RgfxLibraryFilePath's), mapped where that is safe; nil, with
/// Metal's error in `*error`, where it cannot be read.
NSData *_Nullable RgfxLibraryFileData(NSString *_Nullable path, NSError *_Nullable __autoreleasing *_Nullable error);

NS_ASSUME_NONNULL_END

#endif // RGFX_LIBRARY_FILE_H_
