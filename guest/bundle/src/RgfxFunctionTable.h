/*
 * RgfxFunctionTable: the reims-graphics bundle's function pointers, a compute pipeline's
 * MTLFunctionHandle and MTLVisibleFunctionTable (the wrapper convention is RgfxResource.h).
 *
 * A handle is made by `+handleOfFunction:pipeline:device:serializer:`, which carries the function
 * (its library, name, constants, options and specializedName, as a pipeline's linked function) in
 * a FUNCTION_HANDLE_CREATE and waits for the host's answer: the host's platform makes the handle
 * of that function among the functions the pipeline's kernel links, and answers not found for a
 * function the kernel does not link, which is nil here, as Metal answers it. A table is made by
 * `+tableOfPipeline:count:device:serializer:` (FUNCTION_TABLE_CREATE): the host's platform makes
 * the table, and its answer carries what the table answers of it: its gpuResourceID, which an
 * argument buffer holds, its allocatedSize and its resourceOptions. Its entries are set on the
 * host's table (FUNCTION_TABLE_SET), which the device carries out before any command buffer the
 * process posts after the set, so a dispatch encoded before a set and committed after it calls
 * the entry set, as on Metal (measured: AMD Radeon Pro 5300M, macOS 26.7.1, oracle_fnptr).
 *
 * What a handle and a table answer, measured on the AMD Radeon Pro 5300M (macOS 26.7.1,
 * guest/bundle/tools/oracle/oracle_fnptr), and where ours differ:
 *  - A handle answers its function's name and type (MTLFunctionTypeVisible for a visible
 *    function), and raises an unrecognized selector at gpuResourceID; the pipeline answers one
 *    handle object for a function each time it is asked.
 *  - A table answers no heap, no label until set, not aliasable, purgeable state as every
 *    resource; its options and allocatedSize are the host's table's (the 5300M's answer
 *    Managed and tracked, 528, and 4096 bytes for up to 2016 entries).
 *  - Its gpuResourceID is the host's (the 5300M numbers its tables from 1, the lowest free).
 *
 * Every create is nil where the host makes no function pointers
 * (RgfxSerializer::carriesFunctionPointers).
 *
 * Our own ObjC++: conforms to the Metal protocols but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_FUNCTION_TABLE_H_
#define RGFX_FUNCTION_TABLE_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

NS_ASSUME_NONNULL_BEGIN

@interface RgfxFunctionHandle : RgfxResource <MTLFunctionHandle>

/// The handle of `function` among the functions compute pipeline `pipeline` (its objId) links,
/// through `serializer` (which picks the objId). Nil where `serializer` is null, `function` is
/// not one of ours, the host makes no function pointers, or the pipeline links no such function.
/// `device` is the owning shell (nil in a unit test).
+ (nullable instancetype)handleOfFunction:(id<MTLFunction>)function
                                 pipeline:(uint64_t)pipeline
                                   device:(nullable id<MTLDevice>)device
                               serializer:(nullable RgfxSerializer *)serializer;

@end

@interface RgfxVisibleFunctionTable : RgfxResource <MTLVisibleFunctionTable>

/// A table of `count` entries, each nil, of compute pipeline `pipeline` (its objId), through
/// `serializer`. Nil where `serializer` is null, the host makes no function pointers, or the
/// host made none.
+ (nullable instancetype)tableOfPipeline:(uint64_t)pipeline
                                   count:(NSUInteger)count
                                  device:(nullable id<MTLDevice>)device
                              serializer:(nullable RgfxSerializer *)serializer;

/// The number of entries it was made with (MTLVisibleFunctionTableDescriptor's functionCount).
@property(nonatomic, readonly) NSUInteger rgfxFunctionCount;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_FUNCTION_TABLE_H_
