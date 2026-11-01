/*
 * RgfxBinaryArchive: the device's MTLBinaryArchive, kept in the guest.
 *
 * The device compiles every pipeline on the host from its functions and state, so the archive
 * holds no compiled code. It holds, for each function added through it, the key Metal looks the
 * function's compile up by, so that a pipeline create with
 * MTLPipelineOptionFailOnBinaryArchiveMiss hits or misses as Metal's does, and it writes and reads
 * those keys as a file of its own (RgfxBinaryArchiveFile.hpp), per architecture. A create without
 * the option compiles as it would with no archive, as Metal's compiles a missing pipeline.
 *
 * Every answer and error is Metal's, measured on the M3 Max (macOS 27) with
 * guest/bundle/tools/archive_probe.m. Where Metal asserts on a descriptor (no compute or vertex
 * function), the archive answers NO with Metal's words in an NSError: the lane ends no process on
 * its input.
 *
 * Our own ObjC++: it subclasses only NSObject, so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_BINARY_ARCHIVE_H_
#define RGFX_BINARY_ARCHIVE_H_

#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// The domain of a pipeline create's miss: the driver's name, as Metal's driver names its own
/// (the M3 Max answers AGXMetalG15X_M1, its driver bundle's).
extern NSErrorDomain const RgfxBinaryArchiveMissDomain;

@interface RgfxBinaryArchive : NSObject <MTLBinaryArchive>

/// The archive `descriptor` names on `device`, whose `architecture.name` is `architecture`: empty
/// where the descriptor has no URL, else the file at its URL. Nil and Metal's error
/// (MTLBinaryArchiveDomain, MTLBinaryArchiveErrorInvalidFile) for a URL that names no readable
/// file ("Invalid URL") or a file that is no archive ("The file … has an invalid format.").
/// A file of this device's format keeps every architecture's entries and looks up its own; a
/// Metal archive, whose slices are Apple GPUs', loads with none of this device's, as Metal loads
/// an archive of another GPU and misses every lookup in it.
+ (nullable instancetype)archiveWithDescriptor:(MTLBinaryArchiveDescriptor *)descriptor
                                        device:(nullable id<MTLDevice>)device
                                  architecture:(NSString *)architecture
                                         error:(NSError *_Nullable __autoreleasing *_Nullable)error;

- (instancetype)init NS_UNAVAILABLE;

/// The private members SkyLight, MPS and QuartzCore send an archive (measured: macOS 27, M3 Max):
/// a key for each entry this device's architecture holds, equal to the key of the same entry and
/// to its copy; and the options of the descriptor it was made of, with 8 added where it has any,
/// as Metal's archive answers them.
- (NSArray<id<NSCopying>> *)keys;
- (NSUInteger)options;

/// Metal's miss of a render pipeline of `descriptor` created with
/// MTLPipelineOptionFailOnBinaryArchiveMiss: nil where its `binaryArchives` hold the key of each
/// of its functions' compiles, or where it names no archive, as Metal then compiles it.
+ (nullable NSError *)rgfxMissOfRenderPipeline:(MTLRenderPipelineDescriptor *)descriptor;

/// The same, of a compute pipeline of `descriptor`.
+ (nullable NSError *)rgfxMissOfComputePipeline:(MTLComputePipelineDescriptor *)descriptor;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_BINARY_ARCHIVE_H_
