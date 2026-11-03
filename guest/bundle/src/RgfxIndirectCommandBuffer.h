/*
 * RgfxIndirectCommandBuffer: the reims-graphics bundle's MTLIndirectCommandBuffer, and the
 * render and compute commands it hands out (resource slice; the wrapper convention is
 * RgfxResource.h).
 *
 * Made by `+icbWithDescriptor:...`, which carries the descriptor in an ICB_CREATE through the
 * serialiser and waits for the host's ICB_STATE: the host makes the one copy of the buffer, on
 * its own GPU, and numbers it (`gpuResourceID`, which an argument buffer holds, so a kernel
 * encoding through one reaches it). The guest keeps no commands of its own: each CPU encode, a
 * setter or a draw on a command, reaches the host's buffer before the next command buffer the
 * process posts, so a command buffer sees every encode made before its commit, as on every Mac
 * (measured: M3 Max, AMD 5300M, Intel 630; docs in crates/reims-graphics/src/icb.rs). Where the
 * device took ICB_SLOTS, an encode is written into the command's slot in a block of the
 * process's region, as a GPU's driver writes a command's record, and the flush before a commit
 * names the commands written (RgfxIcbSlotTable.hpp); else, and for a buffer whose shape a slot does
 * not hold, it is a record the serialiser sends (ICB_WRITE). GPU encodes, executes, copies,
 * resets and optimizes in a command buffer reach the host's buffer through the command buffer's
 * ops.
 *
 * What the buffer and its commands answer, measured on the AMD 5300M and the Intel 630 (macOS
 * 26.7.1) and the M3 Max (macOS 27.0.1), and where they differ, the lane's choice:
 *  - `size` is maxCommandCount; the options' storage, cache and hazard modes (a Default hazard
 *    mode answers Tracked); no heap, no label, not aliasable; purgeable state as every resource.
 *  - `allocatedSize`: 64 bytes a command, in whole pages, at least one (the AMD's for a draw;
 *    the M3 answers 672 a command, the Intel 3428).
 *  - A command past the size, or of a kind the buffer's types lack, is nil (the Intel's; the
 *    AMD raises, the M3 answers an object). Each lookup makes a new object (the AMD's and M3's).
 *  - The macOS 26 render-state setters (setCullMode: and the rest) raise unrecognized selector,
 *    as on the AMD and the Intel, which answer respondsToSelector: YES for them too.
 *  - Mesh setters are logged no-ops: the device makes no mesh pipeline, so no mesh command runs.
 *  - Render barriers are sent on (the M3's; the AMD faults, the Intel raises).
 *  - A Memoryless buffer ends the process, as on all three.
 *
 * Our own ObjC++: conforms to the Metal protocols but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_INDIRECT_COMMAND_BUFFER_H_
#define RGFX_INDIRECT_COMMAND_BUFFER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include "reims_graphics_wire.h" // RgfxIcbCreate
class RgfxRegion;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxIndirectCommandBuffer : RgfxResource <MTLIndirectCommandBuffer>

/// Makes a buffer of `maxCount` commands of shape `descriptor`, with `options`, through
/// `serializer` (which picks the objId and asks the host). Nil where `serializer` is null,
/// `descriptor` is nil, or the host made none. A Memoryless one ends the process. `device` is
/// the owning shell (nil in a unit test).
+ (nullable instancetype)icbWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                           maxCommandCount:(NSUInteger)maxCount
                                   options:(MTLResourceOptions)options
                                    device:(nullable id<MTLDevice>)device
                                serializer:(nullable RgfxSerializer *)serializer;

/// As icbWithDescriptor:maxCommandCount:options:device:serializer:, the CPU's encodes written as
/// slots in a block of `region` where it is given (the device took ICB_SLOTS) and the buffer's
/// shape takes them (RgfxIcbSlotTable.hpp), else as records.
+ (nullable instancetype)icbWithDescriptor:(MTLIndirectCommandBufferDescriptor *)descriptor
                           maxCommandCount:(NSUInteger)maxCount
                                   options:(MTLResourceOptions)options
                                    device:(nullable id<MTLDevice>)device
                                serializer:(nullable RgfxSerializer *)serializer
                               slotsRegion:(nullable RgfxRegion *)region;

/// `range` within the buffer's commands: its location at most `size`, its length at most what
/// is left from there. Executes, resets, copies and optimizes name only commands it has.
- (NSRange)rgfxClampRange:(NSRange)range;

@end

/// A render command of an RgfxIndirectCommandBuffer: each setter and draw written into its slot,
/// or a record to the host's buffer.
@interface RgfxIndirectRenderCommand : NSObject <MTLIndirectRenderCommand>
@end

/// A compute command of an RgfxIndirectCommandBuffer, as RgfxIndirectRenderCommand.
@interface RgfxIndirectComputeCommand : NSObject <MTLIndirectComputeCommand>
@end

#ifdef __cplusplus
/// `descriptor`'s state as ICB_CREATE carries it, for `maxCount` commands with `options`: each
/// count and type its Metal value, the inherits in `flags`, and the SPI members where the
/// descriptor answers them, else a fresh descriptor's (1 scissor rect, 1 viewport, the SPI
/// states inherited, no override: measured, AMD 5300M and Intel 630).
RgfxIcbCreate RgfxIcbCreateOf(MTLIndirectCommandBufferDescriptor *descriptor, NSUInteger maxCount,
                              MTLResourceOptions options);

/// `range` clamped to `icb`'s commands where `icb` is one of ours (rgfxClampRange:), else as
/// given.
NSRange RgfxIcbClamp(id _Nullable icb, NSRange range);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_INDIRECT_COMMAND_BUFFER_H_
