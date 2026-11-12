/*
 * RgfxDynamicLibrary: the reims-graphics bundle's MTLDynamicLibrary (the wrapper convention is
 * RgfxResource.h): a dynamic library of one of the device's libraries of type dynamic, made by
 * `+dynamicLibraryOfLibrary:device:serializer:status:`, which carries a DYNAMIC_LIBRARY_CREATE of
 * the library and waits for the host's answer. It keeps its library, whose bytes it serializes
 * and whose install name it answers.
 *
 * A compute pipeline's kernel links dynamic libraries (RgfxComputePipelineState): those of its
 * descriptor's preloadedLibraries, and the libraries the install names of its libraries load,
 * which Metal reads from the files they name where the library naming them was made from a file:
 * the bundle reads each in the guest's process (pipelineLibraries, RgfxComputePipelineState.mm),
 * so the host loads none from a file of its own.
 *
 * Metal's compile of source reads each dynamic library of its options' libraries through the
 * private member airData, which this class answers.
 *
 * What a dynamic library answers, measured on the AMD Radeon Pro 5300M and the Intel UHD
 * Graphics 630 (macOS 26.7.1, guest/bundle/tools/oracle/oracle_fnptr), and where ours differ:
 *  - Its install name is its library's; its label nil until set, for one made of a library and
 *    for one read from a file alike.
 *  - serializeToURL: writes a file both read back as the same library, of 10080 and 24480 bytes:
 *    Metal's universal file, its GPU's code beside the AIR. Ours writes the library's .metallib,
 *    which Metal reads back as it reads the offline `metal -dynamiclib` output. A URL in a
 *    directory that is not there is NO with MTLDynamicLibraryDomain 1, "Invalid URL", and no file.
 *
 * Our own ObjC++: conforms to the Metal protocol but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_DYNAMIC_LIBRARY_H_
#define RGFX_DYNAMIC_LIBRARY_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

@class RgfxLibrary;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxDynamicLibrary : RgfxResource <MTLDynamicLibrary>

/// The dynamic library of `library`, a library of type dynamic, through `serializer` (which picks
/// the objId); `*status` is the host's answer (RGFX_STATUS_OK where it made it; kRgfxNoAnswer
/// where none came back or the host makes no dynamic libraries). Nil but where the host made it.
/// `device` is the owning shell (nil in a unit test).
+ (nullable instancetype)dynamicLibraryOfLibrary:(RgfxLibrary *)library
                                          device:(nullable id<MTLDevice>)device
                                      serializer:(nullable RgfxSerializer *)serializer
                                          status:(uint32_t *)status;

/// The library it was made of.
@property(nonatomic, readonly) RgfxLibrary *rgfxLibrary;

/// The bytes of the .metallib it was made of: what Metal's compile of source reads of each
/// dynamic library of MTLCompileOptions' libraries, and the one message it sends it (measured:
/// macOS 26.7.1, the AMD Radeon Pro 5300M and the Intel UHD Graphics 630, oracle_fnptr's "dylib
/// airData" and "dylib source compile sent"; `@16@0:8`, _MTLDynamicLibrary's, a dispatch_data_t
/// of the file's bytes for one read from a file). The caller owns it: Metal's answers a new
/// object at each call, retained and not autoreleased, and its compile releases what it is
/// answered, so an object answered unowned is released once too often (measured: the 5300M; the
/// guest's process ended with SIGSEGV after the compile where ours was unowned). Nil where its
/// library kept none, which a library of type dynamic does not: it is made of bytes.
- (nullable dispatch_data_t)airData NS_RETURNS_RETAINED;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_DYNAMIC_LIBRARY_H_
