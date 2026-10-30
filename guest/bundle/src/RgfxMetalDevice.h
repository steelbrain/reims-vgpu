/*
 * RgfxMetalDevice: the reims-graphics guest bundle's MTLDevice — the class Metal
 * loads from the bundle (via MetalPluginClassName) and instantiates with
 * `-initWithAcceleratorPort:` as the guest's GPU device. It subclasses Apple's
 * private IOGPU-backed Metal device base (IOGPUMetalDevice, forward-declared in
 * RgfxMetalPrivate.h) and conforms to <MTLDevice, MTLDeviceSPI>.
 *
 * Because it subclasses an Apple-private class whose symbols resolve only at load
 * time in the guest, this class is COMPILED (x86_64, into the bundle) but NOT
 * linked into the user-space unit test — it cannot be instantiated off the guest.
 * The NSObject-based wrappers it vends (RgfxBuffer and the rest) are what the unit
 * test exercises.
 *
 * ===========================================================================
 * THE CATEGORY SPLIT (established by slice 0; the four later slices each own one
 * category file so the slices never collide in one .mm):
 *
 *   RgfxMetalDevice.mm            slice 0 — THIS FILE. init + wire handshake, the
 *                                 device identity (name/registryID/location/peer…),
 *                                 supportsFamily:/supportsFeatureSet:/sample counts,
 *                                 and the limit & feature getters, answered from
 *                                 RgfxCaps. Holds the RgfxSerializer / RgfxIdSpace /
 *                                 RgfxCaps the categories reach through the
 *                                 accessors below.
 *   RgfxMetalDevice+Resources.mm  slice 1 — newBufferWith…, newTextureWith…,
 *                                 newSamplerStateWith…, newHeapWith…,
 *                                 newDepthStencilStateWith…; each serialises a
 *                                 create and wraps the result (RgfxBuffer-style).
 *   RgfxMetalDevice+Pipelines.mm  slice 2 — newLibraryWith…, newRenderPipelineState…,
 *                                 newComputePipelineState…, argument encoders,
 *                                 reflection.
 *   RgfxMetalDevice+CommandBuffer.mm  slice 3 — newCommandQueue* (RgfxEncoder path),
 *                                 the command-buffer and encoder object graph.
 *   RgfxMetalDevice+Sync.mm       slice 4 — newEvent/newFence/newSharedEvent and
 *                                 event wrappers.
 *
 * Each category implements its MTLDevice/MTLDeviceSPI selectors using the three
 * accessors below. Each slice adds any private selectors it overrides to the
 * forward-declared protocols in RgfxMetalPrivate.h, tagged measured.
 * ===========================================================================
 *
 * Our own ObjC++; the only Apple-private names it touches are the minimal
 * forward-declarations in RgfxMetalPrivate.h.
 */

#ifndef RGFX_METAL_DEVICE_H_
#define RGFX_METAL_DEVICE_H_

#import "RgfxMetalPrivate.h"

#include "RgfxCaps.hpp"
#include "RgfxObjectModel.hpp"

#ifdef __cplusplus
class RgfxSerializer;
class RgfxRegion;
class RgfxReplyReader;
class RgfxRingWriter;
class RgfxCompletionThread;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice : IOGPUMetalDevice <MTLDevice, MTLDeviceSPI>

/// Metal's entry point: `[super initWithAcceleratorPort:]` identifies the GPU, then
/// (stub, until the region/ring transport lands) we would map the device region,
/// read the caps blob, and stand up the submission rings. May return nil to
/// decline a process, as Apple's base allows.
- (nullable instancetype)initWithAcceleratorPort:(mach_port_t)port;

// --- accessors the category slices route through -----------------------------

/// The resource/create serialiser (null until the ring transport is wired).
- (nullable RgfxSerializer *)serializer;

/// The per-device id space (object ids, GPU addresses, resource ids).
- (RgfxIdSpace *)ids;

/// The host device's caps (identity, limits, features). Invalid until the device
/// region's caps blob is read.
- (const RgfxCaps *)caps;

/// The BAR-region allocator (RgfxRegion), from which queue rings and Shared/no-copy
/// resource blocks are placed. Null until the wire handshake maps the region (a later
/// piece); the resource and command-queue slices fall back to their non-region path
/// (or nil) while it is null.
- (nullable RgfxRegion *)region;

/// The reader of the control completion ring (RgfxReplyReader): command-buffer
/// completions, pipeline reflections and shared-event signals. Null until the
/// handshake stands it up over the region.
- (nullable RgfxReplyReader *)replyReader;

/// The process's completion thread (RgfxCompletionThread), which takes each queue's
/// completions off its ring as the kext wakes it. Null until the handshake opens the
/// context, or where the thread could not start (the waiters then take the rings).
- (nullable RgfxCompletionThread *)completionThread;

/// Places a queue submission ring at `regionOffset` in the device region with a byte
/// ring of `dataSize` bytes, owns the RgfxRingWriter over it for the device's life, and
/// returns it (null when there is no region). The command-queue slice calls this.
- (nullable RgfxRingWriter *)addQueueSubmitRingAtOffset:(uint64_t)regionOffset
                                              dataSize:(uint32_t)dataSize;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_H_
