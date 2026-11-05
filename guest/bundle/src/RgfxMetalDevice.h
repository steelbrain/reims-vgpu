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
 *   RgfxMetalDevice+Video.mm      the video driver's way to the host (RgfxVideoChannel.hpp):
 *                                 its sessions, bitstream buffers and decodes.
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

#import "RgfxArgumentEncoder.h" // RgfxRegionHolder
#import "RgfxMetalPrivate.h"
#import "RgfxStateCache.h"
#import "RgfxTexture.h" // RgfxLinearAlignmentSource

#include "RgfxCaps.hpp"
#include "RgfxObjectModel.hpp"

#ifdef __cplusplus
#include <memory>

class RgfxCommitOrder;
class RgfxKnownPipelines;
class RgfxSerializer;
class RgfxRegion;
class RgfxReplyReader;
class RgfxRingWriter;
class RgfxCompletionThread;
class RgfxSlabs;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice
    : IOGPUMetalDevice <MTLDevice, MTLDeviceSPI, RgfxRegionHolder, RgfxLinearAlignmentSource>

/// Metal's entry point: `[super initWithAcceleratorPort:]` identifies the GPU, then the
/// wire handshake maps the process's window of the device region, reads the caps blob, and
/// stands up the device's rings (the .mm). May return nil to decline a process, as Apple's
/// base allows.
- (nullable instancetype)initWithAcceleratorPort:(mach_port_t)port;

// --- accessors the category slices route through -----------------------------

/// The resource/create serialiser (null where the wire handshake failed).
- (nullable RgfxSerializer *)serializer;

/// The pipeline creates the device accepted for this process (RgfxKnownPipelines), against which
/// an equal create is posted without waiting: null where the kext keeps no answers for later
/// (its feature word lacks kRgfxKextFeatureDeferredAnswers), so every create waits.
- (std::shared_ptr<RgfxKnownPipelines>)knownPipelines;

/// The per-device id space (object ids, GPU addresses, resource ids).
- (RgfxIdSpace *)ids;

/// The order of the process's command buffers across its queues, which each queue takes part in
/// where the device's word holds RGFX_FEATURE_COMMIT_ORDER (RgfxCommitOrder.hpp).
- (std::shared_ptr<RgfxCommitOrder>)commitOrder;

/// The device's samplers and depth/stencil states, one for equal descriptors (RgfxStateCache.h).
/// Nil where the wire handshake failed.
- (nullable RgfxStateCache *)stateCache;

/// The host device's caps (identity, limits, features). Invalid until the device
/// region's caps blob is read.
- (const RgfxCaps *)caps;

/// The host's video codecs, as the video caps the host wrote say (RgfxVideoCaps): no codec
/// where the host wrote none or the kext does not read them.
- (const RgfxVideoCaps *)videoCaps;

/// The BAR-region allocator (RgfxRegion), from which queue rings and Shared/no-copy
/// resource blocks are placed. Null until the wire handshake maps the region (a later
/// piece); the resource and command-queue slices fall back to their non-region path
/// (or nil) while it is null.
- (nullable RgfxRegion *)region;

/// The slabs of the region the device's Shared and Managed buffers lie in (RgfxSlabs). Null
/// until the wire handshake maps the region.
- (nullable RgfxSlabs *)slabs;

/// The reader of the control completion ring (RgfxReplyReader): command-buffer
/// completions, pipeline reflections and shared-event signals. Null until the
/// handshake stands it up over the region.
- (nullable RgfxReplyReader *)replyReader;

/// The process's completion thread (RgfxCompletionThread), which takes each queue's
/// completions off its ring as the kext wakes it. Null until the handshake opens the
/// context, or where the thread could not start (the waiters then take the rings).
- (nullable RgfxCompletionThread *)completionThread;

/// Places a queue submission ring at `regionOffset` in the device region with a byte
/// ring of `dataSize` bytes, owns the RgfxRingWriter over it until the queue's end
/// (dropQueueSubmitRing:), and returns it (null when there is no region). The
/// command-queue slice calls this.
- (nullable RgfxRingWriter *)addQueueSubmitRingAtOffset:(uint64_t)regionOffset
                                              dataSize:(uint32_t)dataSize;

/// Frees the writer addQueueSubmitRingAtOffset: returned, at its queue's end.
- (void)dropQueueSubmitRing:(RgfxRingWriter *)ring;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_H_
