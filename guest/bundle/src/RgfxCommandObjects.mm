/*
 * RgfxCommandObjects: see RgfxCommandObjects.h for the object graph and conventions.
 * Each encoder selector is a thin forwarder to the matching RgfxEncoder op; the op
 * stream, its framing and the command-buffer header are RgfxEncoder's (RgfxEncoder.hpp),
 * which mirrors the host decoder crates/reims-vgpu/src/graphics/decode/{cmdbuf,render,
 * compute,blit}.rs. Our own ObjC++; no Apple private SPI.
 *
 * ARGUMENT HANDLES. Every resource/pipeline/state/event argument Metal hands an
 * encoder is one of our RgfxResource wrappers; `rgfxId` reads its host handle (0 for
 * nil), as the convention requires.
 *
 * SINGLE vs ARRAY ops. Where RgfxEncoder has only the array form of a binding op
 * (e.g. renderSetVertexBuffers but no single vertex-buffer op), the single Metal
 * selector (setVertexBuffer:offset:atIndex:) forwards a one-element array at the given
 * index, exactly as the host reads it.
 *
 * INLINE BYTES. The set*Bytes:length:atIndex: setters have no op of their own: the bytes go
 * to an upload chunk of the command buffer's in the device region (RgfxUploads.h), bound as a
 * buffer at their offset there.
 *
 * QUEUE BARRIERS. barrierAfterQueueStages:beforeStages: has no op of its own either: it waits
 * for every encoder made before it on the queue, through the queue's eight barrier fences
 * (RgfxBarrierSlot). Every encoder takes the next fence in turn, waits at its start for the
 * encoder that last took it, and updates it at its end, so the fence each encoder updated last
 * is done only once every earlier encoder of that fence is. A barrier then waits for the other
 * seven, which covers every encoder before it: over-synchronising, as the barrier may (all
 * stages, all resources). The order is the order the encoders were made in, which is the
 * queue's where its command buffers are committed in the order they were encoded.
 *
 * NOT MAPPED HERE (no RgfxEncoder op yet — follow-ups, no wire invented): the render SPI-only
 * ops that have no public selector (provoking-vertex, alpha-test ref, line width, point size,
 * viewport-transform, primitive-restart, depth/stencil "cleared", resolve textures,
 * tile-dispatch in-region, vertex-amplification mode); the compute indirect-command-encoding
 * ops (encodeStartIf/While/Else/DoWhile, compressed-texture flush) and
 * computeDispatchThreadsIndirect; the blit fill-texture, fill-buffer-pattern4 and
 * invalidate-compressed ops. The public members of features the device does not support
 * (function tables, acceleration structures, mesh and object stages, counters, tensors,
 * depth bounds, colour attachment maps, depth store-action options) do nothing on the wire
 * and are logged (RgfxUnsupported.h).
 */

#import "RgfxCommandObjects.h"
#include "RgfxLogOnce.hpp"
#import "RgfxBuffer.h"
#import "RgfxComputePipelineState.h"
#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxSharedEvent.h"
#import "RgfxIOSurface.h"
#import "RgfxIndirectCommandBuffer.h"
#import "RgfxInterop.h"
#import "RgfxMetalError.h"
#import "RgfxMetalPrivate.h"
#import "RgfxRenderPipelineState.h"
#import "RgfxUnsupported.h"
#import "RgfxUploads.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <mach/mach_time.h>
#include <mutex>
#include <os/lock.h>
#include <vector>

#include "RgfxCommitOrder.hpp"
#include "RgfxCompletionThread.hpp"
#include "RgfxEncoder.hpp"
#include "RgfxMemoryShadow.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"

namespace {

// The guest's host time in seconds, the clock CACurrentMediaTime reads (mach_absolute_time).
CFTimeInterval rgfxHostTime()
{
	static mach_timebase_info_data_t timebase = [] {
		mach_timebase_info_data_t t = {};
		mach_timebase_info(&t);
		return t;
	}();
	return static_cast<CFTimeInterval>(mach_absolute_time()) * timebase.numer / timebase.denom /
	       1e9;
}

} // namespace

// The logs of a command buffer: none, as the device runs no shader that logs.
@interface RgfxLogContainer : NSObject <MTLLogContainer>
@end

@implementation RgfxLogContainer
RGFX_LOG_UNRECOGNIZED

- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state
                                  objects:(id __unsafe_unretained _Nullable[_Nonnull])buffer
                                    count:(NSUInteger)len
{
	return 0;
}
@end

namespace {

// What a mesh command answers where the host makes no mesh pipeline (RGFX_FEATURE_MESH_PIPELINES):
// no mesh pipeline can be made, so it draws nothing.
constexpr const char *kNoMeshes = "mesh shading (the host makes no mesh pipeline)";

/// Whether the `n`th (from 0) occurrence of a repeated log line is logged: the first alone, as
/// for every degradation (docs/conventions.md); `count` counts them all.
bool logEventOp(std::atomic<uint64_t> &count, uint64_t *n)
{
	*n = count.fetch_add(1, std::memory_order_relaxed);
	return *n == 0;
}

std::atomic<uint64_t> g_foreignEvents{0};
std::atomic<uint64_t> g_bytesDropped{0};

/// Puts an encoder's inline bytes (set*Bytes:length:atIndex:) in its command buffer's upload
/// chunks (RgfxUploads.h), answering the buffer and offset they lie at to bind; false, said in
/// the log, where no chunk could be had (the region has no room): the binding is not made, and
/// the command buffer fails with MTLCommandBufferErrorOutOfMemory when it completes.
bool rgfxPutBytes(RgfxUploadArena *uploads, const void *bytes, NSUInteger length,
                  __unsafe_unretained id<MTLBuffer> *buffer, uint64_t *offset)
{
	if (uploads != nullptr && uploads->put(bytes, length, buffer, offset)) {
		return true;
	}
	uint64_t n = 0;
	if (logEventOp(g_bytesDropped, &n)) {
		NSLog(@"RGFX: %lu inline bytes have no upload chunk (no region room): their binding is "
		      @"not made, and the command buffer fails",
		      (unsigned long)length);
	}
	return false;
}

/// The host handle naming `obj` (0 = nil). Every Metal object the guest sees is one
/// of our RgfxResource wrappers. These helpers take their object unretained: their callers hold
/// it for the call.
uint64_t rgfxId(__unsafe_unretained id obj)
{
	return RgfxResourceId(obj);
}

/// Records on `enc` that its command buffer uses the copy of app memory `obj` reaches, if any
/// (rgfxShadowOf: a no-copy buffer's, or a texture's over one), which its commit syncs in and,
/// where it `writes`, its completion writes back.
void rgfxNoteShadowUse(RgfxEncoder *enc, __unsafe_unretained id obj, bool writes)
{
	if (enc == nullptr || obj == nil || !RgfxMemoryShadow::anyLive())
		return;
	if (std::shared_ptr<RgfxMemoryShadow> shadow = rgfxShadowOf(obj))
		enc->noteShadowUse(shadow, writes);
}

/// The objects a command buffer keeps from its encode until it completes or goes
/// (RgfxEncoder::setKeeper): locked, as a parallel pass's sub-encoders keep on their own threads.
/// Each object comes once a command buffer (its keep stamp), so the lock is taken once an object.
/// The resources among them are in flight from the buffer's submission (begin) until its
/// results land where the CPU sees them (land): the resource SPI's isComplete answers NO and its
/// waitUntilComplete waits meanwhile (RGFX_RESOURCE_SPI).
struct RgfxKept {
	os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
	std::vector<id> objects;
	// The objects counted in flight at begin, the first of `objects`; whether begin and land ran.
	size_t inFlight = 0;
	bool begun = false;
	bool landed = false;

	/// Counts the resources kept so far in flight (RgfxInFlightBegin), once, and only before
	/// land: the buffer is submitted.
	void begin()
	{
		os_unfair_lock_lock(&lock);
		if (!begun && !landed) {
			for (__unsafe_unretained id object : objects) {
				if (RgfxIsResource(object))
					RgfxInFlightBegin((RgfxResource *)object);
			}
			inFlight = objects.size();
			begun = true;
		}
		os_unfair_lock_unlock(&lock);
	}

	/// Counts the resources begin counted out of flight (RgfxInFlightEnd), once: the buffer's
	/// results are where the CPU sees them, or it goes without them.
	void land()
	{
		os_unfair_lock_lock(&lock);
		if (begun && !landed) {
			for (size_t i = 0; i < inFlight; i++) {
				if (RgfxIsResource(objects[i]))
					RgfxInFlightEnd((RgfxResource *)objects[i]);
			}
		}
		landed = true;
		os_unfair_lock_unlock(&lock);
	}

	/// Lets every object go, outside the lock (a dealloc may run), landed first.
	void clear()
	{
		land();
		std::vector<id> gone;
		os_unfair_lock_lock(&lock);
		gone.swap(objects);
		os_unfair_lock_unlock(&lock);
	}
};

/// The keeper of a command buffer with retained references (RgfxEncoder::setKeeper): `context`
/// is the buffer's RgfxKept.
void rgfxKeep(void *context, const void *object)
{
	RgfxKept *kept = static_cast<RgfxKept *>(context);
	os_unfair_lock_lock(&kept->lock);
	kept->objects.push_back((__bridge id)object);
	os_unfair_lock_unlock(&kept->lock);
}

/// How an encoder names a resource (rgfxNote): only by its id (a pipeline, a state, a fence);
/// as one the command buffer reads, writes, or binds to a function, which may write it.
enum RgfxNaming { kRgfxNamed, kRgfxReads, kRgfxWrites, kRgfxBinds };

/// Records on `enc` the use of the copy of app memory our `obj` reaches (rgfxNoteShadowUse), as
/// the naming `how` uses it: a bound texture writes it where its usage has ShaderWrite.
void rgfxNoteShadowAs(RgfxEncoder *enc, __unsafe_unretained id obj, RgfxNaming how)
{
	if (how != kRgfxBinds) {
		rgfxNoteShadowUse(enc, obj, how == kRgfxWrites);
		return;
	}
	if (std::shared_ptr<RgfxMemoryShadow> shadow = rgfxShadowOf(obj)) {
		const bool writes = ![obj isKindOfClass:[RgfxTexture class]] ||
		                    ([(RgfxTexture *)obj usage] & MTLTextureUsageShaderWrite) != 0;
		enc->noteShadowUse(shadow, writes);
	}
}

/// rgfxId of `obj`, named by an encoder of `enc`'s command buffer as `how` says, which the buffer
/// keeps where it keeps what it encodes (RgfxEncoder::setKeeper): one of ours once a command
/// buffer, by the stamp it records (RgfxResource's _rgfxKeptBy), any other each time it is not
/// the last named. A resource read, written or bound also has its copies noted: of app memory
/// (rgfxNoteShadowAs) and of an IOSurface (rgfxIOSurfaceNoteUse); `*overSurface` (when not null)
/// says whether it is a texture over an IOSurface or a view of one. One class lookup in all.
template <RgfxNaming how>
__attribute__((always_inline)) inline uint64_t rgfxNote(RgfxEncoder *enc, __unsafe_unretained id obj,
                                                        bool *overSurface = nullptr)
{
	uint64_t id = 0;
	if (enc != nullptr && enc->namedBefore((__bridge const void *)obj, &id))
		return id; // kept, and over no copy (RgfxEncoder::namedNote)
	if (!RgfxIsResource(obj)) {
		if (enc != nullptr && obj != nil)
			enc->keep((__bridge const void *)obj);
		return 0;
	}
	__unsafe_unretained RgfxResource *r = (RgfxResource *)obj;
	const bool surface = r->_rgfxSurfaceRoot != nil;
	const bool shadows = RgfxMemoryShadow::anyLive();
	if (how != kRgfxNamed) {
		if (overSurface != nullptr)
			*overSurface = surface;
		if (enc != nullptr && shadows)
			rgfxNoteShadowAs(enc, obj, how);
		if (surface)
			rgfxIOSurfaceNoteUse(enc, obj);
	}
	id = r->_objId;
	if (enc != nullptr) {
		const uint64_t stamp = enc->keepStamp();
		if (__atomic_load_n(&r->_rgfxKeptBy, __ATOMIC_RELAXED) != stamp) {
			__atomic_store_n(&r->_rgfxKeptBy, stamp, __ATOMIC_RELAXED);
			enc->keep((__bridge const void *)obj);
		}
		// With no copy of app memory alive, this one has none, and never will: a resource's
		// copy is made with it.
		if (!surface && !shadows)
			enc->namedNote((__bridge const void *)obj, id);
	}
	return id;
}

/// rgfxId of an object an encoder of the command buffer names, which the buffer keeps where it
/// keeps what it encodes.
uint64_t rgfxNamed(RgfxEncoder *enc, __unsafe_unretained id obj)
{
	return rgfxNote<kRgfxNamed>(enc, obj);
}

/// rgfxId of a resource the command buffer only reads (a blit's source, an index or indirect
/// buffer).
uint64_t rgfxReads(RgfxEncoder *enc, __unsafe_unretained id obj)
{
	return rgfxNote<kRgfxReads>(enc, obj);
}

/// rgfxId of a resource the command buffer writes (a blit's destination, a fill, a synchronize,
/// a render attachment).
uint64_t rgfxWrites(RgfxEncoder *enc, __unsafe_unretained id obj)
{
	return rgfxNote<kRgfxWrites>(enc, obj);
}

/// rgfxId of a resource bound to a shader, which may write a buffer, and a texture whose usage
/// has ShaderWrite; `*overSurface` as rgfxNote's.
uint64_t rgfxBinds(RgfxEncoder *enc, __unsafe_unretained id obj, bool *overSurface = nullptr)
{
	return rgfxNote<kRgfxBinds>(enc, obj, overSurface);
}

/// rgfxId of a resource used with `usage` (useResource:usage:), which writes it where the usage
/// has Write: a texture over an IOSurface, or a view of one, among them.
uint64_t rgfxUses(RgfxEncoder *enc, __unsafe_unretained id obj, MTLResourceUsage usage)
{
	const bool writes = (usage & MTLResourceUsageWrite) != 0;
	if (writes)
		rgfxIOSurfaceNoteWrite(enc, obj);
	rgfxNoteShadowUse(enc, obj, writes);
	rgfxIOSurfaceNoteUse(enc, obj);
	return rgfxNamed(enc, obj);
}

/// `resource`, used with `usage`, resident in `bindings` where the usage has no Write: a
/// function may still write it through an argument buffer (RgfxSurfaceBindings::resident).
void rgfxNoteResident(RgfxSurfaceBindings &bindings, __unsafe_unretained id resource,
                      MTLResourceUsage usage)
{
	if ((usage & MTLResourceUsageWrite) == 0)
		bindings.resident(resource);
}

/// Logs a command buffer's flush of ICB records (RgfxSerializer::flushIcbWrites): the process's
/// first 16, then each one longer than any before, with its messages, bytes and time.
void rgfxNoteIcbFlush(const RgfxIcbFlush &flush)
{
	static std::atomic<uint32_t> logged{0};
	static std::atomic<uint64_t> longest{0};
	uint64_t was = longest.load(std::memory_order_relaxed);
	bool longer = false;
	while (flush.nanoseconds > was &&
	       !(longer = longest.compare_exchange_weak(was, flush.nanoseconds,
	                                                std::memory_order_relaxed))) {
	}
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16 || longer) {
		NSLog(@"RGFX: ICB_WRITE flush: %u messages, %llu bytes, %llu us", flush.messages,
		      (unsigned long long)flush.bytes, (unsigned long long)(flush.nanoseconds / 1000));
	}
}

/// The RGFX_RESOURCE_KIND_* of a Metal resource, for the resource-reference ops.
uint32_t rgfxResourceKind(id<MTLResource> res)
{
	if ([res conformsToProtocol:@protocol(MTLBuffer)])
		return RGFX_RESOURCE_KIND_BUFFER;
	if ([res conformsToProtocol:@protocol(MTLTexture)])
		return RGFX_RESOURCE_KIND_TEXTURE;
	if ([res conformsToProtocol:@protocol(MTLIndirectCommandBuffer)])
		return RGFX_RESOURCE_KIND_ICB;
	return RGFX_RESOURCE_KIND_NONE;
}

/// The level of `t` whose size is `other`'s first level's, in `*level`; false where none is.
bool rgfxLevelMatching(id<MTLTexture> t, id<MTLTexture> other, NSUInteger *level)
{
	for (NSUInteger m = 0; m < t.mipmapLevelCount; m++) {
		if (std::max<NSUInteger>(1, t.width >> m) == other.width &&
		    std::max<NSUInteger>(1, t.height >> m) == other.height &&
		    std::max<NSUInteger>(1, t.depth >> m) == other.depth) {
			*level = m;
			return true;
		}
	}
	return false;
}

/// The slices a blit's whole-surface copy counts in `t`: its array length, six faces a cube.
NSUInteger rgfxSurfaceSlices(id<MTLTexture> t)
{
	NSUInteger slices = t.arrayLength != 0 ? t.arrayLength : 1;
	if (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray)
		slices *= 6;
	return slices;
}

/// The f64 bits of `v`, as a clear-colour/viewport field travels on the wire.
uint64_t f64bits(double v)
{
	uint64_t bits;
	std::memcpy(&bits, &v, sizeof(bits));
	return bits;
}

/// Fills a colour attachment record from a Metal descriptor.
void fillColor(RgfxRenderPassColorAttachment *c, MTLRenderPassColorAttachmentDescriptor *d)
{
	*c = {};
	c->texture_obj_id = rgfxId(d.texture);
	c->resolve_texture_obj_id = rgfxId(d.resolveTexture);
	MTLClearColor cc = d.clearColor;
	c->clear_color_bits[0] = f64bits(cc.red);
	c->clear_color_bits[1] = f64bits(cc.green);
	c->clear_color_bits[2] = f64bits(cc.blue);
	c->clear_color_bits[3] = f64bits(cc.alpha);
	c->load_action = static_cast<uint32_t>(d.loadAction);
	c->store_action = static_cast<uint32_t>(d.storeAction);
	c->store_action_options = static_cast<uint32_t>(d.storeActionOptions);
	c->level = static_cast<uint32_t>(d.level);
	c->slice = static_cast<uint32_t>(d.slice);
	c->depth_plane = static_cast<uint32_t>(d.depthPlane);
	c->resolve_level = static_cast<uint32_t>(d.resolveLevel);
	c->resolve_slice = static_cast<uint32_t>(d.resolveSlice);
	c->resolve_depth_plane = static_cast<uint32_t>(d.resolveDepthPlane);
}

/// Translates an MTLRenderPassDescriptor into the wire pass descriptor and its colour
/// attachments, returning the colour count (highest set attachment index + 1, so
/// gaps stay as absent attachments). `colors` is filled with `count` records.
/// Records on `enc` the textures the pass `d` writes, its attachments and their resolve
/// targets: those over an IOSurface or over app memory (a no-copy buffer's), and every one as
/// the command buffer's to keep.
void noteRenderPassWrites(RgfxEncoder *enc, MTLRenderPassDescriptor *d)
{
	auto note = [enc](id<MTLTexture> t) {
		rgfxIOSurfaceNoteWrite(enc, t);
		(void)rgfxWrites(enc, t);
	};
	for (uint32_t i = 0; i < 8; i++) {
		note(d.colorAttachments[i].texture);
		note(d.colorAttachments[i].resolveTexture);
	}
	note(d.depthAttachment.texture);
	note(d.depthAttachment.resolveTexture);
	note(d.stencilAttachment.texture);
	note(d.stencilAttachment.resolveTexture);
}

/// The sample positions `d` holds (setSamplePositions:count:), as the pass's record carries them;
/// false where it holds none, so the pass keeps the device's default positions. Metal's descriptor
/// takes only counts of 2, 4 or 8 (measured on the M3 Max, macOS 27: any other count asserts in
/// setSamplePositions:count:), all within the record's sixteen.
bool fillSamplePositions(MTLRenderPassDescriptor *d, RgfxRenderPassSamplePositions *out)
{
	*out = {};
	const NSUInteger n = [d getSamplePositions:NULL count:0];
	if (n == 0 || n > RGFX_PASS_MAX_SAMPLE_POSITIONS)
		return false;
	MTLSamplePosition positions[RGFX_PASS_MAX_SAMPLE_POSITIONS];
	[d getSamplePositions:positions count:n];
	out->count = static_cast<uint32_t>(n);
	for (NSUInteger i = 0; i < n; i++) {
		std::memcpy(&out->position_bits[i][0], &positions[i].x, sizeof(uint32_t));
		std::memcpy(&out->position_bits[i][1], &positions[i].y, sizeof(uint32_t));
	}
	return true;
}

/// Whether `device` takes the sample positions `p` of a pass. Metal's render pass validation fails
/// an assertion at a count the device makes no texture of ("Custom sample positions were specified,
/// but the device does not support the number of samples supplied": measured on the M3 Max, macOS
/// 27, at 8), and takes any count it makes, whatever the attachments' sample count (4 positions on
/// a 2-sample pass, 2 on a 4-sample or a 1-sample one). The bundle answers no encoder for the pass
/// it refuses, sending nothing, and logs the first.
bool rgfxTakesSamplePositions(id<MTLDevice> device, const RgfxRenderPassSamplePositions &p)
{
	if ([device supportsTextureSampleCount:p.count])
		return true;
	static std::atomic<bool> logged{false};
	if (!logged.exchange(true))
		NSLog(@"RGFX: a render pass of %u sample positions, a sample count the device does not "
		      @"make: answered no encoder",
		      p.count);
	return false;
}

/// The visibility result buffer of pass `d` (visibilityResultBuffer) and its type
/// (visibilityResultType, macOS 26), as the pass's record carries them, the buffer's write noted
/// on `enc` as an attachment's is (noteRenderPassWrites); false where the pass has none of ours,
/// so it carries none. A pass of type Accumulate adds each Counting result to the word at its
/// offset and ORs each Boolean one into it, as Metal's docs promise and the M3 Max under macOS
/// 27 does; the AMD Radeon Pro 5300M and the Intel UHD 630 under macOS 26.7.1 reset it
/// (measured).
bool fillVisibility(RgfxEncoder *enc, MTLRenderPassDescriptor *d, RgfxRenderPassVisibility *out)
{
	*out = {};
	id<MTLBuffer> buffer = d.visibilityResultBuffer;
	if (buffer == nil)
		return false;
	rgfxIOSurfaceNoteWrite(enc, buffer);
	out->buffer_obj_id = rgfxWrites(enc, buffer);
	if (out->buffer_obj_id == 0)
		return false;
	static const bool typed =
	    [MTLRenderPassDescriptor instancesRespondToSelector:@selector(visibilityResultType)];
	if (typed)
		out->result_type = static_cast<uint32_t>(d.visibilityResultType);
	return true;
}

/// The size of the target of pass `d`, which Metal's default viewport and scissor rectangle
/// cover: the smallest of its attachments (ours, RgfxTexture) at their levels, else its
/// renderTargetWidth and renderTargetHeight where it has none.
RgfxRenderTargetSize rgfxTargetSize(MTLRenderPassDescriptor *d)
{
	RgfxRenderTargetSize size = {UINT64_MAX, UINT64_MAX};
	auto meet = [&size](MTLRenderPassAttachmentDescriptor *a) {
		id t = a.texture;
		if (![t isKindOfClass:[RgfxTexture class]])
			return;
		RgfxTexture *texture = t;
		size.width = std::min<uint64_t>(size.width, std::max<NSUInteger>(1, texture.width >> a.level));
		size.height =
		    std::min<uint64_t>(size.height, std::max<NSUInteger>(1, texture.height >> a.level));
	};
	for (uint32_t i = 0; i < 8; i++)
		meet(d.colorAttachments[i]);
	meet(d.depthAttachment);
	meet(d.stencilAttachment);
	if (size.width == UINT64_MAX)
		size = {d.renderTargetWidth, d.renderTargetHeight};
	return size;
}

/// The record of pass `d`: its colour attachments (`colors`, as many as up to its last of ours),
/// its size, depth and stencil. Its supportColorAttachmentMapping (macOS 26) is not carried, so
/// each fragment output lands in the attachment of its index: the device answers no Apple family,
/// and a colour attachment map the encoder is given is logged and dropped
/// (setColorAttachmentMap:). With it and no map, the M3 Max under macOS 27 draws so too, where
/// the AMD Radeon Pro 5300M and the Intel UHD 630 under macOS 26.7.1 draw no colour at all
/// (measured), and Metal's docs say neither: the lane takes the M3's.
uint32_t fillRenderPass(MTLRenderPassDescriptor *d, RgfxRenderPassDesc *desc,
                        std::vector<RgfxRenderPassColorAttachment> *colors)
{
	*desc = {};

	uint32_t count = 0;
	for (uint32_t i = 0; i < 8; i++) {
		if (rgfxId(d.colorAttachments[i].texture) != 0) {
			count = i + 1;
		}
	}
	colors->resize(count);
	for (uint32_t i = 0; i < count; i++) {
		fillColor(&(*colors)[i], d.colorAttachments[i]);
	}

	uint32_t flags = 0;
	if (d.renderTargetWidth != 0 || d.renderTargetHeight != 0) {
		flags |= RGFX_PASS_FLAG_RT_SIZE;
		desc->render_target_width = static_cast<uint32_t>(d.renderTargetWidth);
		desc->render_target_height = static_cast<uint32_t>(d.renderTargetHeight);
	}
	if (d.renderTargetArrayLength != 0) {
		flags |= RGFX_PASS_FLAG_RT_ARRAY_LENGTH;
		desc->render_target_array_length = static_cast<uint32_t>(d.renderTargetArrayLength);
	}
	if (d.defaultRasterSampleCount != 0) {
		flags |= RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT;
		desc->default_raster_sample_count = static_cast<uint32_t>(d.defaultRasterSampleCount);
	}
	desc->flags = flags;

	MTLRenderPassDepthAttachmentDescriptor *dep = d.depthAttachment;
	if (rgfxId(dep.texture) != 0) {
		desc->depth.texture_obj_id = rgfxId(dep.texture);
		desc->depth.resolve_texture_obj_id = rgfxId(dep.resolveTexture);
		desc->depth.clear_depth_bits = f64bits(dep.clearDepth);
		desc->depth.load_action = static_cast<uint32_t>(dep.loadAction);
		desc->depth.store_action = static_cast<uint32_t>(dep.storeAction);
		desc->depth.store_action_options = static_cast<uint32_t>(dep.storeActionOptions);
		desc->depth.level = static_cast<uint32_t>(dep.level);
		desc->depth.slice = static_cast<uint32_t>(dep.slice);
		desc->depth.depth_plane = static_cast<uint32_t>(dep.depthPlane);
		desc->depth.resolve_level = static_cast<uint32_t>(dep.resolveLevel);
		desc->depth.resolve_slice = static_cast<uint32_t>(dep.resolveSlice);
		desc->depth.resolve_depth_plane = static_cast<uint32_t>(dep.resolveDepthPlane);
		desc->depth.depth_resolve_filter = static_cast<uint32_t>(dep.depthResolveFilter);
	}

	MTLRenderPassStencilAttachmentDescriptor *st = d.stencilAttachment;
	if (rgfxId(st.texture) != 0) {
		desc->stencil.texture_obj_id = rgfxId(st.texture);
		desc->stencil.resolve_texture_obj_id = rgfxId(st.resolveTexture);
		desc->stencil.load_action = static_cast<uint32_t>(st.loadAction);
		desc->stencil.store_action = static_cast<uint32_t>(st.storeAction);
		desc->stencil.store_action_options = static_cast<uint32_t>(st.storeActionOptions);
		desc->stencil.level = static_cast<uint32_t>(st.level);
		desc->stencil.slice = static_cast<uint32_t>(st.slice);
		desc->stencil.depth_plane = static_cast<uint32_t>(st.depthPlane);
		desc->stencil.resolve_level = static_cast<uint32_t>(st.resolveLevel);
		desc->stencil.resolve_slice = static_cast<uint32_t>(st.resolveSlice);
		desc->stencil.resolve_depth_plane = static_cast<uint32_t>(st.resolveDepthPlane);
		desc->stencil.stencil_resolve_filter = static_cast<uint32_t>(st.stencilResolveFilter);
		desc->stencil.clear_stencil = static_cast<uint32_t>(st.clearStencil);
	}

	return count;
}

} // namespace

// =============================================================================
// RgfxCommandQueue
// =============================================================================

@interface RgfxCmdBuffer ()
/// The buffer holds one of its queue's `slots` (maxCommandBufferCount), given back once it
/// completes, or at its dealloc where it never does.
- (void)rgfxHoldSlot:(dispatch_semaphore_t)slots;
/// Under its queue's order lock: whether it was not yet enqueued; it is from now on.
- (BOOL)rgfxTakeEnqueue;
/// Whether it is committed and ready to post (rgfxPost), once every buffer before it is.
- (BOOL)rgfxReadyToPost;
/// Posts its message on the queue's submission ring: its queue's poster calls it in enqueue
/// order (rgfxPostReady).
- (void)rgfxPost;
@end

/// How many barrier fences a queue has (QUEUE BARRIERS, above).
constexpr uint32_t kRgfxBarrierFences = 8;

/// The quality of service of a queue no descriptor names (qosLevel).
constexpr NSUInteger kRgfxDefaultQosLevel = 2;

/// An encoder's place among its queue's barrier fences: the fences, the one it takes, and how
/// many of them encoders before it took. All fences 0 where the queue has none (no route).
struct RgfxBarrierSlot {
	RgfxObjId fences[kRgfxBarrierFences] = {};
	uint32_t slot = 0;    // the fence the encoder updates at its end
	uint32_t earlier = 0; // the fences earlier encoders took: 0..earlier-1, all once past eight
	bool waitOwn = false; // an earlier encoder took the encoder's fence: wait for it first
};

@interface RgfxCommandQueue ()
- (std::shared_ptr<RgfxUploadPool>)rgfxUploadPool;
- (std::shared_ptr<RgfxBodyPool>)rgfxBodyPool;
/// Takes a committed message's `bytes` into rgfxMessageReserve.
- (void)rgfxNoteMessageBytes:(uint64_t)bytes;
/// At the take of the completion of a buffer whose writes are named to the kext: whether its
/// writes retire there, `now` (all in place and named) and no buffer before it retiring later;
/// else it is one of those until rgfxRetiredLate.
- (BOOL)rgfxRetiresAtTake:(BOOL)now;
/// A buffer rgfxRetiresAtTake: left to retire later has retired.
- (void)rgfxRetiredLate;
/// How many of its posted buffers are held: posted with their surface accesses not yet named,
/// until they are scheduled (RgfxCmdBuffer's rgfxPost). Read and raised by its posts, one at a
/// time; lowered as each held buffer is scheduled, or goes unposted.
- (uint32_t)rgfxHeldBuffers;
- (void)rgfxNoteHeld;
- (void)rgfxNoteHeldGone;
/// Where the queue's buffers whose writes do not retire at the take write their copies back and
/// retire, one at a time in the ring's order (rgfxHostDone:): a serial queue of the queue's own,
/// made with its first such buffer, which runs no handler.
- (dispatch_queue_t)rgfxWriteBackQueue;
/// The next encoder's barrier fence, of the fences the queue made with itself.
- (RgfxBarrierSlot)rgfxTakeBarrierSlot;
/// Runs `record` on the queue's completion queue, behind the records before it, or here.
- (void)rgfxDeliver:(dispatch_block_t)record here:(BOOL)here;
/// Puts `cb` last in the enqueue order, where it is not in it yet: whether it was not.
- (BOOL)rgfxEnqueue:(RgfxCmdBuffer *)cb;
/// Posts the buffers at the front of the enqueue order that are ready, in that order.
- (void)rgfxPostReady;
/// Before a buffer of the queue goes on its ring: the buffers of the process's other queues it
/// is to be committed after (RgfxCommitOrder::name), none where the queue takes no part.
- (std::vector<RgfxSurfaceOrderWait>)rgfxNameCommitOrder;
/// A buffer of the queue went on its ring at `seqno` (RgfxCommitOrder::posted).
- (void)rgfxNotePosted:(uint64_t)seqno;
/// The group finish waits on: a buffer committed with a reader enters it, and leaves it once it
/// has completed and run its handlers.
- (dispatch_group_t)rgfxCommittedGroup;
/// Whether its command buffers profile from their start (setProfilingEnabled:, MTLCommandQueueSPI).
- (BOOL)isProfilingEnabled;
@end

@implementation RgfxCommandQueue {
	RgfxRingWriter *_submitRing;   // borrowed: the queue's submission ring
	RgfxIdSpace *_ids;             // borrowed: the device id space, for cb ids
	NSUInteger _maxCount;
	// maxCommandBufferCount slots, one taken by each command buffer made while the queue has a
	// reader, given back as it completes.
	dispatch_semaphore_t _slots;
	NSString *_queueLabel;
	std::shared_ptr<RgfxReplyReader> _ownedReader; // adoptReplyReader:'s per-queue completion reader
	std::function<void()> _release;                // rgfxSetRelease:'s, run at the queue's end
	dispatch_queue_t _submissionQueue; // kept for -submissionQueue (setSubmissionQueue: SPI)
	// Where its command buffers' records run (rgfxDeliver:): the queue setCompletionQueue: gave,
	// under `_completionQueueLock`, else the serial queue of its own it is made with.
	std::mutex _completionQueueLock;
	dispatch_queue_t _completionQueue;
	dispatch_queue_t _ownCompletionQueue;
	// The records handed to a dispatch queue and not yet run: while there are any, a record the
	// taker of the ring could run itself goes behind them instead.
	std::atomic<uint32_t> _deliveriesInFlight;
	// Its buffers taken whose writes retire on the write-back queue, not yet retired: while there
	// are any, a later buffer's retire goes behind them too, as the kext retires a queue's writes
	// up to a seqno (rgfxRetiresAtTake:).
	std::atomic<uint32_t> _lateRetires;
	// rgfxHeldBuffers.
	std::atomic<uint32_t> _heldBuffers;
	// rgfxWriteBackQueue's queue, made once.
	std::once_flag _writeBackOnce;
	dispatch_queue_t _writeBackQueue;
	NSUInteger _gpuPriority;           // kept for -GPUPriority (setGPUPriority: SPI)
	unsigned short _gpuPriorityOffset; // and the offsets and background priority with it
	NSUInteger _backgroundGpuPriority;
	unsigned short _backgroundGpuPriorityOffset;
	// The chunks its command buffers' inline bytes go to (RgfxUploads.h), made with the first
	// command buffer, over the queue's region.
	std::shared_ptr<RgfxUploadPool> _uploadPool;
	// The blocks of its command buffers too large for its ring (RgfxUploads.h), made with the
	// first such buffer, over the queue's region.
	std::shared_ptr<RgfxBodyPool> _bodyPool;
	// The barrier fences (QUEUE BARRIERS), made with the queue, and the encoders made.
	NSArray<RgfxFence *> *_barrierFences;
	uint64_t _encodersMade;
	uint64_t _traceId; // globalTraceObjectID
	// rgfxMessageReserve's bytes. Command buffers commit on any thread; a lost update only
	// sizes one message's first allocation.
	std::atomic<uint32_t> _messageReserve;
	// The enqueue order (rgfxEnqueue:): the buffers enqueued and not yet posted, oldest first,
	// under `_orderLock`; `_postingFront` while the oldest is being posted. One thread at a time
	// posts the ready ones (`_postLock`, which the poster's own handlers may take again where it
	// runs them, with no thread driving the ring).
	std::mutex _orderLock;
	std::deque<RgfxCmdBuffer *> _order;
	bool _postingFront;
	std::recursive_mutex _postLock;
	// Its process's commit order (rgfxJoinCommitOrder:), its slot there, and what its posts named
	// last, which only its posts read and write, one at a time (`_postLock`).
	std::shared_ptr<RgfxCommitOrder> _commitOrder;
	size_t _commitSlot;
	RgfxCommitNamed _commitNamed;
	// What Metal's base queue keeps for its SPI (MTLCommandQueueSPI, below): what its descriptor
	// named (rgfxTakeDescriptor:), its switches, and the statistics' and counters' settings.
	NSUInteger _qosLevel;
	dispatch_queue_t _commitQueue;
	BOOL _commitSynchronously;
	BOOL _commitsWithQoS;
	BOOL _disableCrossQueueHazardTracking;
	BOOL _openGLQueue;
	int _backgroundTrackingPID;
	BOOL _executionEnabled;
	BOOL _skipRender;
	BOOL _profilingEnabled;
	BOOL _statEnabled;
	NSUInteger _statOptions;
	NSUInteger _statLocations;
	NSUInteger _numRequestedCounters;
	NSUInteger _numInternalSampleCounters;
	std::mutex _counterInfoLock;
	NSMutableDictionary *_counterInfo; // under `_counterInfoLock`, as Metal's property is atomic
	// finish's: each command buffer committed to it with a reader is in it until it has
	// completed (RgfxCmdBuffer's commit and rgfxDeliverCompletion:).
	dispatch_group_t _committedGroup;
}

+ (instancetype)queueWithSubmitRing:(RgfxRingWriter *)submitRing
                       submitOffset:(uint64_t)submitOffset
                     completeOffset:(uint64_t)completeOffset
                   completeDataSize:(uint32_t)completeDataSize
                                ids:(RgfxIdSpace *)ids
                           maxCount:(NSUInteger)maxCount
                             device:(id<MTLDevice>)device
                         serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || ids == nullptr || submitRing == nullptr) {
		return nil; // a queue must have a route, an id space and a ring
	}
	// The ring offsets and the completion-ring size are the region allocator's
	// (RgfxRegion::allocateRingPair); the submission ring's size is the ring we hold,
	// and the queue's own objId is picked by the create. The priority/vector are the
	// host's scheduling knobs, 0 (normal) here.
	RgfxCreated created = serializer->createQueue(submitOffset, completeOffset,
	                                              submitRing->dataSize(), completeDataSize,
	                                              /*priority*/ 0, /*vector*/ 0);
	// One the device never opened (the kext refuses a context's queues past its limit) is no
	// queue: nothing would take its ring, and a wait for a buffer committed to it would never
	// return. nil, as Metal's newCommandQueue answers where it can make none.
	if (!created.posted)
		return nil;
	RgfxCommandQueue *q = [[RgfxCommandQueue alloc] initWithObjId:created.obj_id
	                                                       device:device
	                                                   serializer:serializer];
	if (q != nil) {
		q->_traceId = RgfxNextTraceObjectId();
		q->_submitRing = submitRing;
		q->_ids = ids;
		q->_maxCount = maxCount != 0 ? maxCount : 64;
		q->_slots = dispatch_semaphore_create(static_cast<long>(q->_maxCount));
		// Its command buffers' handlers run on a serial queue of its own, as Metal's do where no
		// completion queue is set: Metal's has this label and user-interactive QoS, one queue's
		// handlers run one at a time and two queues' at once (measured on the M3 Max, an AMD
		// 5300M and an Intel 630).
		q->_ownCompletionQueue = dispatch_queue_create(
		    "com.Metal.CompletionQueueDispatch",
		    dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE,
		                                            0));
		// Its barrier fences (QUEUE BARRIERS), made with it, so every encoder takes one.
		NSMutableArray<RgfxFence *> *fences = [NSMutableArray array];
		for (uint32_t i = 0; i < kRgfxBarrierFences; i++) {
			RgfxFence *f = [RgfxFence fenceWithDevice:device serializer:serializer];
			if (f == nil)
				break;
			[fences addObject:f];
		}
		q->_barrierFences = fences.count == kRgfxBarrierFences ? fences : @[];
		// As Metal's base queue starts (read from code, _MTLCommandQueue, macOS 27.0.1): its
		// process tracked for background work, its buffers run, and the quality of service of
		// a queue no descriptor names, 2 (measured: the AMD 5300M's and the Intel 630's queues
		// of newCommandQueue, macOS 26.7.1).
		q->_qosLevel = kRgfxDefaultQosLevel;
		q->_backgroundTrackingPID = getpid();
		q->_executionEnabled = YES;
		q->_committedGroup = dispatch_group_create();
	}
	return q;
}

// The device ends the queue as it goes, once its command buffers have (each keeps the queue
// alive), and only then do its rings go back: the completion ring is read no more first, as the
// completion thread may still hold its reader.
- (void)dealloc
{
	if (_ownedReader != nullptr)
		_ownedReader->close();
	if (_commitOrder != nullptr)
		_commitOrder->leave(_commitSlot);
	if (_serializer != nullptr)
		_serializer->destroyQueue(_objId);
	if (_release)
		_release();
}

- (void)rgfxSetRelease:(std::function<void()>)release
{
	_release = std::move(release);
}

- (void)rgfxJoinCommitOrder:(std::shared_ptr<RgfxCommitOrder>)order
{
	if (order == nullptr || _commitOrder != nullptr)
		return;
	_commitSlot = order->join(_objId);
	_commitOrder = std::move(order);
}

- (std::vector<RgfxSurfaceOrderWait>)rgfxNameCommitOrder
{
	std::vector<RgfxSurfaceOrderWait> waits;
	if (_commitOrder != nullptr)
		_commitOrder->name(_commitSlot, _commitNamed, waits);
	return waits;
}

- (void)rgfxNotePosted:(uint64_t)seqno
{
	if (_commitOrder != nullptr)
		_commitOrder->posted(_commitSlot, seqno);
}

- (NSUInteger)maxCommandBufferCount
{
	return _maxCount;
}

- (uint32_t)rgfxMessageReserve
{
	return _messageReserve.load(std::memory_order_relaxed);
}

- (void)rgfxNoteMessageBytes:(uint64_t)bytes
{
	const uint32_t was = _messageReserve.load(std::memory_order_relaxed);
	const uint64_t next = std::max<uint64_t>(bytes, was - was / 8);
	_messageReserve.store(static_cast<uint32_t>(std::min<uint64_t>(next, UINT32_MAX)),
	                      std::memory_order_relaxed);
}

// The takes come in the ring's order, one at a time; a write-back lets one go only after its
// retire, so a take that finds none pending finds every earlier buffer's copies written back.
- (BOOL)rgfxRetiresAtTake:(BOOL)now
{
	if (now && _lateRetires.load(std::memory_order_acquire) == 0)
		return YES;
	_lateRetires.fetch_add(1, std::memory_order_acq_rel);
	return NO;
}

- (void)rgfxRetiredLate
{
	_lateRetires.fetch_sub(1, std::memory_order_acq_rel);
}

- (uint32_t)rgfxHeldBuffers
{
	return _heldBuffers.load(std::memory_order_acquire);
}

- (void)rgfxNoteHeld
{
	_heldBuffers.fetch_add(1, std::memory_order_acq_rel);
}

- (void)rgfxNoteHeldGone
{
	_heldBuffers.fetch_sub(1, std::memory_order_acq_rel);
}

- (dispatch_queue_t)rgfxWriteBackQueue
{
	std::call_once(_writeBackOnce, [self] {
		self->_writeBackQueue = dispatch_queue_create(
		    "com.reims.rgfx.surface-write-back",
		    dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE,
		                                            0));
	});
	return _writeBackQueue;
}

// Takes one of the queue's slots, blocking while every one is held, as Metal's -commandBuffer
// does. The completion thread gives each back as it takes the buffer's completion off the ring,
// before its handlers run, so a handler that makes a buffer never waits for handlers behind it;
// where no thread drives the ring (a unit test, or its thread has ended), the ring is taken here
// meanwhile.
- (void)rgfxTakeSlot
{
	if (dispatch_semaphore_wait(_slots, DISPATCH_TIME_NOW) == 0)
		return;
	RgfxReplyReader *reader = self.replyReader;
	for (;;) {
		if (reader->driven()) {
			if (dispatch_semaphore_wait(_slots, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC)) == 0)
				return;
			continue;
		}
		reader->pump();
		if (dispatch_semaphore_wait(_slots, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_MSEC)) == 0)
			return;
	}
}

// --- MTLCommandQueue ---------------------------------------------------------

- (nullable id<MTLCommandBuffer>)commandBuffer
{
	return [self rgfxCommandBufferRetained:YES];
}

- (nullable id<MTLCommandBuffer>)commandBufferWithUnretainedReferences
{
	return [self rgfxCommandBufferRetained:NO];
}

- (nullable id<MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)descriptor
{
	BOOL retained = descriptor == nil || descriptor.retainedReferences;
	RgfxCmdBuffer *cb = [self rgfxCommandBufferRetained:retained];
	// The error options are diagnostics: kept for the buffer to answer, they change nothing.
	if (descriptor != nil)
		[cb rgfxSetErrorOptions:descriptor.errorOptions];
	return cb;
}

// The queue's pool of message blocks, made over its region on first use.
- (std::shared_ptr<RgfxBodyPool>)rgfxBodyPool
{
	@synchronized(self) {
		if (_bodyPool == nullptr && self.region != nullptr) {
			_bodyPool = std::make_shared<RgfxBodyPool>(self.region);
		}
		return _bodyPool;
	}
}

// The queue's upload pool, made over its region on first use.
- (std::shared_ptr<RgfxUploadPool>)rgfxUploadPool
{
	@synchronized(self) {
		if (_uploadPool == nullptr) {
			_uploadPool = std::make_shared<RgfxUploadPool>(self.region, [self serializer],
			                                               self.device, self.slabs);
		}
		return _uploadPool;
	}
}

- (RgfxBarrierSlot)rgfxTakeBarrierSlot
{
	RgfxBarrierSlot b;
	@synchronized(self) {
		if (_barrierFences.count != kRgfxBarrierFences)
			return b;
		for (uint32_t i = 0; i < kRgfxBarrierFences; i++)
			b.fences[i] = [_barrierFences[i] rgfxObjId];
		const uint64_t n = _encodersMade++;
		b.slot = static_cast<uint32_t>(n % kRgfxBarrierFences);
		b.earlier = static_cast<uint32_t>(n < kRgfxBarrierFences ? n : kRgfxBarrierFences);
		b.waitOwn = n >= kRgfxBarrierFences;
	}
	return b;
}

// ENQUEUE ORDER. The queue's command buffers run in the order they were enqueued, whatever
// order they are committed in, as Metal's do on the M3 Max, an AMD 5300M and an Intel 630
// (measured): one committed before a buffer enqueued ahead of it waits, Committed, until that
// one is committed too, and an enqueued buffer let go uncommitted stays, holding back every
// buffer after it. A commit with no enqueue before it enqueues the buffer there. So the queue
// keeps the enqueued buffers not yet posted, oldest first, and each commit posts the ready ones
// at the front, in order, on the one ring: its own, and those committed behind one it waited for.
- (BOOL)rgfxEnqueue:(RgfxCmdBuffer *)cb
{
	std::lock_guard<std::mutex> hold(_orderLock);
	if (![cb rgfxTakeEnqueue])
		return NO;
	_order.push_back(cb);
	return YES;
}

- (void)rgfxPostReady
{
	RgfxCommandQueue *keep = self; // a buffer posted here may hold the queue's last reference
	std::lock_guard<std::recursive_mutex> posting(keep->_postLock);
	for (;;) {
		RgfxCmdBuffer *next = nil;
		{
			std::lock_guard<std::mutex> hold(_orderLock);
			// Where this thread is already posting the oldest (a handler it runs commits), the
			// outer loop posts the rest after it.
			if (_postingFront || _order.empty() || ![_order.front() rgfxReadyToPost])
				return;
			next = _order.front();
			_postingFront = true;
		}
		[next rgfxPost];
		{
			std::lock_guard<std::mutex> hold(_orderLock);
			_order.pop_front();
			_postingFront = false;
		}
	}
}

- (RgfxCmdBuffer *)rgfxCommandBufferRetained:(BOOL)retained
{
	// Only completions give a slot back: a queue without a reader (a unit test) takes none.
	bool slotted = self.replyReader != nullptr;
	if (slotted)
		[self rgfxTakeSlot];
	uint64_t cbObjId = _ids->allocObjId();
	RgfxCmdBuffer *cb = [RgfxCmdBuffer commandBufferForQueue:self
	                                                    ring:_submitRing
	                                                 cbObjId:cbObjId
	                                                retained:retained
	                                                  device:self.device
	                                              serializer:[self serializer]];
	if (slotted) {
		if (cb != nil)
			[cb rgfxHoldSlot:_slots];
		else
			dispatch_semaphore_signal(_slots);
	}
	return cb;
}

// Its trace id (RgfxInterop.h), taken as it was made.
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

- (NSString *)label
{
	return _queueLabel;
}
- (void)setLabel:(NSString *)label
{
	_queueLabel = [label copy];
}

- (void)adoptReplyReader:(RgfxReplyReader *)reader
{
	if (_ownedReader.get() != reader)
		_ownedReader.reset(reader);
	self.replyReader = reader;
}

- (std::shared_ptr<RgfxReplyReader>)sharedReplyReader
{
	return _ownedReader;
}

// A capture tool's frame boundary: no capture tool sees this device's work, so nothing
// marks one, as Metal's does with no capture under way.
- (void)insertDebugCaptureBoundary
{
}

// --- unsupported: the device makes no residency set (newResidencySetWithDescriptor:error:) ---

- (void)addResidencySet:(id<MTLResidencySet>)residencySet
{
	if (residencySet != nil)
		RGFX_UNSUPPORTED("residency sets");
}
- (void)addResidencySets:(const id<MTLResidencySet> _Nonnull[_Nonnull])residencySets
                   count:(NSUInteger)count
{
	if (count != 0)
		RGFX_UNSUPPORTED("residency sets");
}
- (void)removeResidencySet:(id<MTLResidencySet>)residencySet
{
	if (residencySet != nil)
		RGFX_UNSUPPORTED("residency sets");
}
- (void)removeResidencySets:(const id<MTLResidencySet> _Nonnull[_Nonnull])residencySets
                      count:(NSUInteger)count
{
	if (count != 0)
		RGFX_UNSUPPORTED("residency sets");
}

// --- dispatch queues and priority (MTLCommandQueue SPI WindowServer sets) -----------
// WindowServer hands its queues the dispatch queues Metal is to run submission and
// completion work on, and a GPU priority. The completion queue is where the queue's command
// buffers' scheduled and completed handlers run from then on, as Metal runs them there (measured
// on the M3 Max, an AMD 5300M and an Intel 630). Our queue submits on the caller's thread, and
// the host schedules every queue alike, so the submission queue and the priorities are kept and
// answered back, and change nothing the queue does.

- (void)setSubmissionQueue:(dispatch_queue_t)queue
{
	_submissionQueue = queue;
}

- (dispatch_queue_t)submissionQueue
{
	return _submissionQueue;
}

- (void)setCompletionQueue:(dispatch_queue_t)queue
{
	std::lock_guard<std::mutex> hold(_completionQueueLock);
	_completionQueue = queue;
}

- (dispatch_queue_t)completionQueue
{
	std::lock_guard<std::mutex> hold(_completionQueueLock);
	return _completionQueue;
}

// Runs one of its command buffers' records (the buffer's scheduling or completion, with its
// handlers) behind every record handed over before it, on the queue's completion queue: the one
// setCompletionQueue: gave, else its own serial queue. Where `here` and nothing waits on a
// dispatch queue, it runs at once on this thread instead: where no completion thread drives the
// ring, so the thread that took it off delivers, and for a buffer that runs nothing of the app's
// and writes nothing back (_deliversWhereSeen), whose waiters the thread that saw its completion
// then wakes itself, as Metal's does for a buffer with no completed handler (measured on the M3
// Max: the wait returned 3 to 10 us after the status read Completed), one wake where the queue's
// would be two; and for the scheduling of a buffer with no scheduled handler
// (_schedulesWhereSeen), which runs nothing of the app's either, so a buffer with only a completed
// handler pays the queue's wake for its completion alone.
- (void)rgfxDeliver:(dispatch_block_t)record here:(BOOL)here
{
	if (here && _deliveriesInFlight.load(std::memory_order_acquire) == 0) {
		record();
		return;
	}
	dispatch_queue_t target = nil;
	{
		std::lock_guard<std::mutex> hold(_completionQueueLock);
		target = _completionQueue != nil ? _completionQueue : _ownCompletionQueue;
	}
	_deliveriesInFlight.fetch_add(1, std::memory_order_acq_rel);
	RgfxCommandQueue *queue = self; // kept until the record has run
	dispatch_async(target, ^{
		record();
		queue->_deliveriesInFlight.fetch_sub(1, std::memory_order_acq_rel);
	});
}

// The runtime's setter answers whether the priority was taken (`c24@0:8Q16`): it always is.
- (BOOL)setGPUPriority:(NSUInteger)priority
{
	_gpuPriority = priority;
	return YES;
}

- (NSUInteger)GPUPriority
{
	return _gpuPriority;
}

// RenderBox reads the priority back as it sets up its queue (`Q16@0:8`), and the background
// priority with it; the rest of the family is the runtime's too (MTLCommandQueueSPI).
- (NSUInteger)getGPUPriority
{
	return _gpuPriority;
}

- (BOOL)setGPUPriority:(NSUInteger)priority offset:(unsigned short)offset
{
	_gpuPriority = priority;
	_gpuPriorityOffset = offset;
	return YES;
}

- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority
{
	_backgroundGpuPriority = priority;
	return YES;
}

- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(unsigned short)offset
{
	_backgroundGpuPriority = priority;
	_backgroundGpuPriorityOffset = offset;
	return YES;
}

- (NSUInteger)getBackgroundGPUPriority
{
	return _backgroundGpuPriority;
}

// --- what Core Image sends a queue beyond MTLCommandQueue (RgfxInterop.h) -------------
// Each as Metal's queue answers it, measured on the M3 Max (macOS 27).

// Metal's never called the handler after a command buffer completed; nor does this.
- (void)addPerfSampleHandler:(void (^)(id))handler
{
	(void)handler;
}

// -1 for every list tried (nil, empty, one counter).
- (int)requestCounters:(NSArray *)counters withIndex:(NSUInteger)index
{
	(void)counters;
	(void)index;
	return -1;
}

// Statistics: Metal keeps the settings, which its getters answer back (MTLCommandQueueSPI,
// below); on is logged here, as its statistics, whatever they are, are not gathered.
- (void)setStatEnabled:(BOOL)enabled
{
	if (enabled)
		RGFX_UNSUPPORTED("command statistics");
	_statEnabled = enabled;
}
- (void)setStatLocations:(NSUInteger)locations
{
	_statLocations = locations;
}
- (void)setStatOptions:(NSUInteger)options
{
	_statOptions = options;
}

// --- MTLCommandQueueSPI ----------------------------------------------------------------------
// The members of Metal's command-queue SPI that the AMD 5300M's and the Intel 630's queues both
// answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m),
// answered as Metal's base queue, _MTLCommandQueue, answers them (read from code, macOS 27.0.1);
// the getters of a fresh queue answer what those two GPUs' queues answer (measured:
// oracle_spi_members answers).

// What the descriptor the queue was made with names, as Metal's base queue takes it at its start:
// its quality of service, commit and completion queues, and switches. A guest's descriptor that
// lacks one of them leaves the queue's default.
- (void)rgfxTakeDescriptor:(MTLCommandQueueDescriptor *)descriptor
{
	id<RgfxMetalQueueDescriptorInternals> spi = (id<RgfxMetalQueueDescriptorInternals>)descriptor;
	if ([spi respondsToSelector:@selector(qosLevel)])
		_qosLevel = [spi qosLevel];
	if ([spi respondsToSelector:@selector(commitQueue)])
		_commitQueue = [spi commitQueue];
	if ([spi respondsToSelector:@selector(commitSynchronously)])
		_commitSynchronously = [spi commitSynchronously];
	if ([spi respondsToSelector:@selector(commitsWithQoS)])
		_commitsWithQoS = [spi commitsWithQoS];
	if ([spi respondsToSelector:@selector(disableCrossQueueHazardTracking)])
		_disableCrossQueueHazardTracking = [spi disableCrossQueueHazardTracking];
	if ([spi respondsToSelector:@selector(isOpenGLQueue)])
		_openGLQueue = [spi isOpenGLQueue];
	if ([spi respondsToSelector:@selector(completionQueue)]) {
		dispatch_queue_t completion = [spi completionQueue];
		if (completion != nil)
			[self setCompletionQueue:completion];
	}
}

- (NSUInteger)qosLevel
{
	return _qosLevel;
}

// The queue the descriptor named for Metal's commits, nil where none: the device's buffers post
// on the thread that commits them, which the guest's results do not see.
- (dispatch_queue_t)commitQueue
{
	return _commitQueue;
}

- (BOOL)commitSynchronously
{
	return _commitSynchronously;
}

- (BOOL)commitsWithQoS
{
	return _commitsWithQoS;
}

// Whether the descriptor asked for no tracking of hazards between queues: the device orders its
// queues' buffers by their commits either way (RgfxCommitOrder.hpp), which Metal's tracking does
// at most.
- (BOOL)disableCrossQueueHazardTracking
{
	return _disableCrossQueueHazardTracking;
}

- (BOOL)isOpenGLQueue
{
	return _openGLQueue;
}

- (int)backgroundTrackingPID
{
	return _backgroundTrackingPID;
}

- (void)setBackgroundTrackingPID:(int)pid
{
	_backgroundTrackingPID = pid;
}

- (BOOL)executionEnabled
{
	return _executionEnabled;
}

// Metal's queue completes the buffers committed while execution is off without running them;
// the device runs them.
- (void)setExecutionEnabled:(BOOL)enabled
{
	if (!enabled)
		RGFX_UNSUPPORTED("completing a queue's command buffers without running them");
	_executionEnabled = enabled;
}

- (BOOL)skipRender
{
	return _skipRender;
}

// Metal's buffers take the switch from their queue as they are made; the device's render
// passes run.
- (void)setSkipRender:(BOOL)skip
{
	if (skip)
		RGFX_UNSUPPORTED("skipping a queue's render work (setSkipRender:)");
	_skipRender = skip;
}

- (BOOL)isProfilingEnabled
{
	return _profilingEnabled;
}

// Its command buffers made from now on profile (RgfxCmdBuffer's profilingResults), as Metal's
// take the switch from their queue as they are made.
- (void)setProfilingEnabled:(BOOL)enabled
{
	_profilingEnabled = enabled;
}

- (BOOL)isStatEnabled
{
	return _statEnabled;
}

- (NSUInteger)StatOptions
{
	return _statOptions;
}

- (NSUInteger)getStatOptions
{
	return _statOptions;
}

- (NSUInteger)StatLocations
{
	return _statLocations;
}

- (NSUInteger)getStatLocations
{
	return _statLocations;
}

- (NSUInteger)numRequestedCounters
{
	return _numRequestedCounters;
}

- (void)setNumRequestedCounters:(NSUInteger)count
{
	_numRequestedCounters = count;
}

- (NSUInteger)numInternalSampleCounters
{
	return _numInternalSampleCounters;
}

- (void)setNumInternalSampleCounters:(NSUInteger)count
{
	_numInternalSampleCounters = count;
}

- (NSMutableDictionary *)counterInfo
{
	std::lock_guard<std::mutex> hold(_counterInfoLock);
	return _counterInfo;
}

- (void)setCounterInfo:(NSMutableDictionary *)info
{
	std::lock_guard<std::mutex> hold(_counterInfoLock);
	_counterInfo = info;
}

// The counters the queue's statistics can gather: the device gathers none (setStatEnabled: YES
// is unsupported) and samples none, where the AMD 5300M's and the Intel 630's queues list their
// GPUs' own (measured).
- (NSArray *)availableCounters
{
	return @[];
}

// The counters requested, none where none were: nil, as both GPUs' queues answer (measured).
- (NSArray *)getRequestedCounters
{
	return nil;
}

// -1, as requestCounters:withIndex: answers for every list tried (measured: M3 Max, macOS 27):
// the device samples no counters.
- (int)requestCounters:(NSArray *)counters
{
	(void)counters;
	return -1;
}

// The list in a list of its own, as Metal's base queue answers.
- (NSArray *)subdivideCounterList:(NSArray *)list
{
	NSMutableArray *lists = [[NSMutableArray alloc] init];
	[lists addObject:list];
	return lists;
}

// Metal's base queue delivers its notifications as they come, and does nothing here; so does
// this queue (rgfxDeliver:here:).
- (void)dispatchAvailableCompletionNotifications
{
}

// Returns once every command buffer committed to the queue before the call has completed and
// run its completed handlers, as Metal's waits on its buffers' group.
- (void)finish
{
	dispatch_group_wait(_committedGroup, DISPATCH_TIME_FOREVER);
}

- (dispatch_group_t)rgfxCommittedGroup
{
	return _committedGroup;
}

@end

// =============================================================================
// RgfxCmdBuffer
// =============================================================================

// The delivery entries the reader's handlers call (defined below); declared here so the
// lambdas in `commit` see their signatures.
@interface RgfxCmdBuffer ()
- (void)rgfxHostDone:(const RgfxCbComplete *)msg;
- (void)rgfxDeliverCompletion:(const RgfxCbComplete *)msg;
- (void)rgfxWriteBackAndRetire:(uint64_t)seqno;
- (void)rgfxWriteBackCopies;
- (void)rgfxReachScheduled;
- (void)rgfxReachSubmitted;
- (void)rgfxAskScheduled;
- (void)rgfxPostScheduledAsk;
- (void)rgfxWaitFor:(dispatch_group_t)group reached:(const std::atomic<bool> *)reached;
@end

// The encoders' inline bytes go to their command buffer's upload arena, handed over as the
// command buffer vends them.
@interface RgfxRenderCommandEncoder ()
// At a draw: each stage's bound textures over IOSurfaces its function writes, as written. Direct:
// a draw calls it with no message.
- (void)rgfxNoteDrawWrites __attribute__((objc_direct));
// The command buffer that made it (commandBuffer).
- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads;
// The tile size its pass descriptor asked for (0: the device's choice).
- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height;
// Its barrier fence (QUEUE BARRIERS): `begin` for an encoder of its own pass, which waits for
// its fence's last encoder now and updates the fence at its end; not for a parallel pass's
// sub-encoder, whose pass does both.
- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot begin:(BOOL)begin;
// A parallel pass's sub-encoder: `segment` is the stream it writes, its own (SegmentOf), which
// the pass joins at its end (rgfxSegment).
- (void)rgfxAdoptSegment:(std::unique_ptr<RgfxEncoder>)segment;
- (RgfxEncoder *)rgfxSegment;
@end
@interface RgfxComputeCommandEncoder ()
// The command buffer that made it (commandBuffer).
- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads;
// The dispatch type its pass was opened with.
- (void)rgfxSetDispatchType:(MTLDispatchType)dispatchType;
- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot;
@end
@interface RgfxBlitCommandEncoder ()
// The command buffer that made it (commandBuffer).
- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot;
@end
@interface RgfxParallelRenderCommandEncoder ()
// The command buffer that made it (commandBuffer).
- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads;
- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height;
// The pass's descriptor, a copy of which gives the size of its target, the default viewport and
// scissor rectangle each sub-encoder starts from, where one is to be set back.
- (void)rgfxSetPass:(MTLRenderPassDescriptor *)pass;
- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot;
@end

namespace {

// The render stages a queue barrier's fences wait before and update after: every stage the
// device's pipelines have.
constexpr uint64_t kRgfxBarrierRenderStages = MTLRenderStageVertex | MTLRenderStageFragment;

// A command buffer's surface accesses (RgfxCmdBuffer's _surfaceSlots): named in its surfaces'
// slots and to the kext; the host has finished it; posted held, its accesses left to its
// scheduling; and scheduled.
constexpr uint32_t kRgfxSlotsNamed = 1, kRgfxSlotsFinished = 2, kRgfxSlotsHeld = 4,
                   kRgfxSlotsScheduled = 8;

// A waiting thread's ask of a command buffer's scheduling (RgfxCmdBuffer's _scheduledAsk): a
// thread waits for it to be scheduled; its message is on its ring, held by nothing.
constexpr uint32_t kRgfxAskWanted = 1, kRgfxAskPosted = 2;

// The fences a barrier in the encoder of `b` waits for: every one but its own that an earlier
// encoder took, calling `wait` with each.
template <typename Wait> void rgfxWaitForEarlierEncoders(const RgfxBarrierSlot &b, Wait wait)
{
	for (uint32_t t = 0; t < b.earlier; t++) {
		if (t != b.slot && b.fences[t] != 0)
			wait(b.fences[t]);
	}
}

} // namespace

// The stages profilingResults times, in the order of its keys (MTLCommandBufferSPI, below).
enum RgfxStage : unsigned {
	kRgfxStageCreation,
	kRgfxStageEnqueue,
	kRgfxStageCommit,
	kRgfxStageSubmitToKernel,
	kRgfxStageKernelScheduled,
	kRgfxStageSubmitToHW,
	kRgfxStageCompletionInterrupt,
	kRgfxStageKernelComplete,
	kRgfxStageCompletionHandlerEnqueue,
	kRgfxStageCompletionHandlerExecution,
	kRgfxStageCount,
};

@implementation RgfxCmdBuffer {
	RgfxCommandQueue *_queue; // kept alive, as Metal's command buffers keep their queue
	RgfxEncoder *_encoder; // owned: built incrementally, posted on commit
	BOOL _retained;
	BOOL _committed;
	// Committed by commitAndWaitUntilSubmitted, so it asks for its submission where the host
	// answers it.
	BOOL _waitsSubmitted;
	// It asks the host for its submission (RGFX_OP_SUBMITTED), so its post expects CB_SUBMITTED.
	BOOL _asksSubmitted;
	// Its queue's submission ring, which its message and a waiting thread's ask of its scheduling
	// go on; and where that ask is (kRgfxAsk*: a thread waits, the message is on the ring), so the
	// one of the two that comes second posts it, once (rgfxAskScheduled).
	RgfxRingWriter *_ring;
	std::atomic<uint32_t> _scheduledAsk;
	// In its queue's enqueue order (under the queue's order lock), and committed and ready to
	// post once every buffer before it is.
	BOOL _enqueued;
	std::atomic<bool> _readyToPost;
	uint64_t _seqno;
	// Written by the thread that delivers the completion, read by any: atomic.
	std::atomic<MTLCommandBufferStatus> _status;
	NSString *_cbLabel;
	NSMutableDictionary *_userDictionary; // userDictionary, made when first asked
	// What the client asked of the buffer's content protection (setProtectionOptions:).
	uint64_t _protectionOptions;
	// The error options its descriptor asked for (commandBufferWithDescriptor:).
	MTLCommandBufferErrorOption _errorOptions;
	// Its times, in the guest's host time (rgfxHostTime), 0 until known: the CPU's
	// scheduling from commit until the buffer is posted to the device, and the GPU's run
	// (rgfxDeliverCompletion:). Each is written once, before the status that makes it known.
	std::atomic<CFTimeInterval> _kernelStart;
	std::atomic<CFTimeInterval> _kernelEnd;
	std::atomic<CFTimeInterval> _gpuStart;
	std::atomic<CFTimeInterval> _gpuEnd;
	NSError *_cbError;
	// The handlers, added until the commit, which `_handlersLock` orders them against: once
	// committed, the lists are only read, by the thread that delivers.
	std::mutex _handlersLock;
	NSMutableArray *_completedHandlers;
	NSMutableArray *_scheduledHandlers;
	// The queue's completion-ring reader, taken at commit; null when the queue has none
	// (a unit test driving the queue without one), in which case the buffer holds its
	// seqno. `_readerHold` keeps a reader the queue owns alive while the buffer is in
	// flight, though the queue may go first.
	RgfxReplyReader *_reader;
	std::shared_ptr<RgfxReplyReader> _readerHold;
	// Entered at commit with a reader, left once the completion's handlers have run:
	// what waitUntilCompleted waits on while the completion thread delivers. `_left`
	// says it was left, so a buffer whose completion never came leaves it at dealloc.
	dispatch_group_t _done;
	std::atomic<bool> _left;
	// Its queue's group finish waits on (rgfxCommittedGroup), entered and left with `_done`.
	dispatch_group_t _queueCommitted;
	// Entered at commit with a reader, as `_done` is: `_submitted` is left once the buffer is
	// submitted, `_scheduled` once its scheduled handlers have run, which is what
	// commitAndWaitUntilSubmitted and waitUntilScheduled wait on. The buffer is submitted at its
	// CB_SUBMITTED where it asked for one (rgfxReachSubmitted), and submitted and scheduled at
	// its CB_SCHEDULED, or at its completion where none came first (rgfxReachScheduled).
	// `_submittedReached` and `_scheduledReached` say it got there; each `_*Left` that its group
	// was left, so one never left is left at dealloc.
	dispatch_group_t _submitted;
	dispatch_group_t _scheduled;
	std::atomic<bool> _submittedReached;
	std::atomic<bool> _scheduledReached;
	std::atomic<bool> _scheduledRan;
	std::atomic<bool> _submittedLeft;
	std::atomic<bool> _scheduledLeft;
	// The queue's slot this buffer holds (rgfxHoldSlot:), and whether it still does.
	dispatch_semaphore_t _slots;
	std::atomic<bool> _slotHeld;
	// A buffer too large for its ring: the block of its queue's pool its message lies in until
	// it completes.
	std::shared_ptr<RgfxBodyPool> _bodyPool;
	RgfxBodyBlock _body;
	// What its encoders named, kept from the encode until it completes or goes, so nothing it
	// names leaves the host before it runs, as Metal's buffer with retained references keeps what
	// it encodes.
	RgfxKept _kept;
	// The IOSurface copies this buffer writes, written back to their surfaces' pages
	// when it completes (RgfxIOSurface.h).
	std::vector<std::shared_ptr<void>> _surfaceWrites;
	// The IOSurface views it uses, taken from its encoder at its commit, for its post to name its
	// order by (rgfxIOSurfaceNameOrder); let go of there.
	std::vector<std::shared_ptr<void>> _surfaceUses;
	// The IOSurface views it only reads, where the process names its buffers' order: named among
	// their readers once it is posted, and held until it completes, so the slots stay named while
	// it runs (rgfxIOSurfaceNameOrder).
	std::vector<std::shared_ptr<void>> _surfaceReads;
	// Where this buffer's writes of IOSurfaces' shared blocks are named to the kext, null where
	// it has none: its completion retires them there, so a CPU lock of one of the surfaces waits
	// until then (RgfxIOSurface.h, rgfxIOSurfaceWritesPosted).
	RgfxRegion *_surfaceWritesRegion;
	// Whether every surface it writes is in place, and whether its writes are named to the kext
	// yet: at its completion's take, a buffer of both retires its writes there (rgfxHostDone:).
	bool _writesInPlace;
	std::atomic<bool> _writesNamed;
	// Whether its surface accesses are named, in its surfaces' writer and reader slots and to the
	// kext, and whether the host has finished it (kRgfxSlotsNamed, kRgfxSlotsFinished): the second
	// of the two empties the slots that still name it (rgfxIOSurfaceNoteFinished), as the host may
	// finish a buffer before they are named. A buffer posted held has them named once it is
	// both posted and scheduled (kRgfxSlotsHeld, kRgfxSlotsScheduled: rgfxNameAccesses).
	std::atomic<uint32_t> _surfaceSlots;
	// What its accesses name (rgfxNameAccesses): the views it writes and those it only reads,
	// and where its writes go to the kext; set by its post, emptied by its write-back or its
	// delivery (`_namedGone`), all under `_namedLock`, so one that comes first leaves nothing to
	// name: no write is named to the kext after its retire.
	std::mutex _namedLock;
	std::vector<std::shared_ptr<void>> _namedWrites;
	std::vector<std::shared_ptr<void>> _namedReads;
	// The buffers of other queues its post's order named (rgfxIOSurfaceNameOrder), which it runs
	// after: where a record has no slot left for it, one of this process's naming one of them is.
	std::vector<RgfxSurfaceOrderWait> _namedOrder;
	RgfxRegion *_namedRegion;
	bool _namedGone;
	// The events it waits on, each with its value, and the plain events it signals: whether a
	// wait is met at its post decides whether the post names its accesses (rgfxPost); the host's
	// finish of the buffer raises each plain event's value as this process knows it.
	std::vector<std::pair<id, uint64_t>> _eventWaits;
	std::vector<std::pair<RgfxEvent *, uint64_t>> _plainSignals;
	// Signalled once the queue's write-back queue has written its copies back and retired its
	// writes, where its take left them to it (rgfxHostDone:); nil otherwise.
	dispatch_semaphore_t _writtenBack;
	// The copies of app memory (no-copy buffers') this buffer may write, written back to the
	// memory when it completes (RgfxMemoryShadow.hpp).
	std::vector<std::shared_ptr<RgfxMemoryShadow>> _shadowWrites;
	// Whether its records run where they are seen (rgfxDeliver:here:), fixed at its commit: it
	// has no scheduled or completed handler, and nothing to write back at its completion (every
	// surface it writes is in place, and it writes no copy of app memory).
	bool _deliversWhereSeen;
	// Whether its scheduling runs where it is seen, fixed at its commit: it has no scheduled
	// handler, so its scheduling runs nothing of the app's (rgfxDeliver:here:).
	bool _schedulesWhereSeen;
	// Where its encoders' inline bytes go: chunks of the queue's upload pool, given back once
	// the host has completed the buffer (RgfxUploads.h). The encoders share it. `_uploadsFailed`:
	// some found no chunk, noted as they go back (rgfxHostDone:).
	std::shared_ptr<RgfxUploadArena> _uploads;
	bool _uploadsFailed;
	uint64_t _traceId; // globalTraceObjectID
	// What Metal's base command buffer keeps for its SPI (MTLCommandBufferSPI, below): the
	// private data and its offset, a logging buffer it does not retain, the profiling and
	// shared-indirection-table switches, and the logs it was given.
	id _privateData;
	NSUInteger _privateDataOffset;
	__unsafe_unretained id _privateLoggingBuffer;
	std::atomic<bool> _profilingEnabled;
	BOOL _sharedIndirectionTable;
	id _logs;
	// The blocks __waitUntilScheduledAsync: and __waitUntilCompletedAsync: were given, run once
	// the scheduled or the completed handlers have; each `*Done` says they have, so a block
	// given later runs at once. Under `_handlersLock`.
	NSMutableArray *_scheduledWaiters;
	NSMutableArray *_completedWaiters;
	bool _scheduledWaitersDone;
	bool _completedWaitersDone;
	// When it reached each stage, in mach_absolute_time's units, for profilingResults: its
	// creation always, the rest where profiling is enabled then (rgfxNoteStage:).
	std::atomic<uint64_t> _stageTimes[kRgfxStageCount];
}

+ (instancetype)commandBufferForQueue:(RgfxCommandQueue *)queue
                                 ring:(RgfxRingWriter *)ring
                              cbObjId:(uint64_t)cbObjId
                             retained:(BOOL)retained
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	RgfxCmdBuffer *cb = [[RgfxCmdBuffer alloc] initWithObjId:cbObjId
	                                                          device:device
	                                                      serializer:serializer];
	if (cb != nil) {
		cb->_traceId = RgfxNextTraceObjectId();
		cb->_queue = queue;
		cb->_retained = retained;
		cb->_ring = ring;
		cb->_status.store(MTLCommandBufferStatusNotEnqueued);
		cb->_completedHandlers = [NSMutableArray array];
		cb->_scheduledHandlers = [NSMutableArray array];
		cb->_completedWaiters = [NSMutableArray array];
		cb->_scheduledWaiters = [NSMutableArray array];
		cb->_stageTimes[kRgfxStageCreation].store(mach_absolute_time());
		// It profiles from its start where its queue does, as Metal's buffer takes the switch.
		cb->_profilingEnabled.store([queue isProfilingEnabled]);
		const uint32_t reserve = queue.rgfxMessageReserve;
		cb->_encoder = new RgfxEncoder(*ring, queue.rgfxObjId, cbObjId, reserve);
		// Past its ring, the message is built in a block of the queue's pool, the size of the
		// queue's recent largest first.
		cb->_encoder->setMessageBlocks([queue rgfxBodyPool], reserve);
		// One with unretained references keeps them too, though it answers retainedReferences
		// NO: the host must not drop an object before a buffer that names it has run.
		cb->_encoder->setKeeper(&rgfxKeep, &cb->_kept);
		cb->_uploads = std::make_shared<RgfxUploadArena>([queue rgfxUploadPool]);
	}
	return cb;
}

- (void)dealloc
{
	// A group released while entered ends the process: one whose completion never came
	// (its reader went first) is left here.
	if (_done != nil && !_left.exchange(true)) {
		dispatch_group_leave(_done);
		dispatch_group_leave(_queueCommitted);
	}
	if (_submitted != nil && !_submittedLeft.exchange(true))
		dispatch_group_leave(_submitted);
	if (_scheduled != nil && !_scheduledLeft.exchange(true))
		dispatch_group_leave(_scheduled);
	// A buffer never committed, or whose completion never came, gives its slot back, and lets
	// what it kept go with it, out of flight.
	_kept.land();
	[self rgfxGiveSlotBack];
	// And the block its message lay in, as its completion would have: a buffer posted with a
	// completion to come is kept until it comes (the reader holds it), so the host has read the
	// message from any block given back here, or will never read it.
	if (_bodyPool != nullptr) {
		_bodyPool->giveBack(_body);
		_bodyPool = nullptr;
	}
	delete _encoder;
}

- (void)rgfxHoldSlot:(dispatch_semaphore_t)slots
{
	_slots = slots;
	_slotHeld.store(true);
}

- (void)rgfxGiveSlotBack
{
	if (_slots != nil && _slotHeld.exchange(false))
		dispatch_semaphore_signal(_slots);
}

// --- vending the encoders ----------------------------------------------------

- (nullable id<MTLRenderCommandEncoder>)renderCommandEncoderWithDescriptor:
    (MTLRenderPassDescriptor *)renderPassDescriptor
{
	RgfxRenderPassDesc desc = {};
	std::vector<RgfxRenderPassColorAttachment> colors;
	uint32_t count = fillRenderPass(renderPassDescriptor, &desc, &colors);
	RgfxRenderPassSamplePositions positions;
	const bool programmed = fillSamplePositions(renderPassDescriptor, &positions);
	if (programmed && !rgfxTakesSamplePositions(self.device, positions))
		return nil;
	noteRenderPassWrites(_encoder, renderPassDescriptor);
	RgfxRenderPassVisibility visibility;
	const bool visible = fillVisibility(_encoder, renderPassDescriptor, &visibility);
	_encoder->beginRenderPass(desc, colors.empty() ? nullptr : colors.data(), count,
	                          programmed ? &positions : nullptr, visible ? &visibility : nullptr);
	RgfxRenderCommandEncoder *enc = [[RgfxRenderCommandEncoder alloc] initWithEncoder:_encoder
	                                                                          emitEnd:YES
	                                                                           device:self.device];
	[enc rgfxSetCommandBuffer:self];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetTileWidth:renderPassDescriptor.tileWidth height:renderPassDescriptor.tileHeight];
	[enc rgfxSetBarrierSlot:[_queue rgfxTakeBarrierSlot] begin:YES];
	return enc;
}

- (nullable id<MTLComputeCommandEncoder>)computeCommandEncoder
{
	_encoder->beginComputePass();
	RgfxComputeCommandEncoder *enc = [[RgfxComputeCommandEncoder alloc] initWithEncoder:_encoder
	                                                                             device:self.device];
	[enc rgfxSetCommandBuffer:self];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetBarrierSlot:[_queue rgfxTakeBarrierSlot]];
	return enc;
}

- (nullable id<MTLComputeCommandEncoder>)computeCommandEncoderWithDescriptor:
    (MTLComputePassDescriptor *)computePassDescriptor
{
	return [self computeCommandEncoderWithDispatchType:computePassDescriptor.dispatchType];
}

// The pass carries the dispatch type, which the encoder answers (dispatchType).
- (nullable id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:
    (MTLDispatchType)dispatchType
{
	_encoder->beginComputePass(static_cast<uint32_t>(dispatchType));
	RgfxComputeCommandEncoder *enc = [[RgfxComputeCommandEncoder alloc] initWithEncoder:_encoder
	                                                                             device:self.device];
	[enc rgfxSetCommandBuffer:self];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetDispatchType:dispatchType];
	[enc rgfxSetBarrierSlot:[_queue rgfxTakeBarrierSlot]];
	return enc;
}

- (void)rgfxEncodeVideoSession:(uint64_t)session
                         codec:(uint32_t)codec
                 parameterSets:(const uint8_t *)parameterSets
                          size:(uint32_t)size
                     bitstream:(id<MTLBuffer>)bitstream
                       ofBytes:(uint64_t)bytes
                          luma:(id<MTLTexture>)luma
                        chroma:(id<MTLTexture>)chroma
                    outputRows:(uint32_t)outputRows
{
	_encoder->beginVideoPass(session);
	const uint64_t bitstreamId = rgfxReads(_encoder, bitstream);
	rgfxIOSurfaceNoteWrite(_encoder, luma);
	rgfxIOSurfaceNoteWrite(_encoder, chroma);
	const uint64_t lumaId = rgfxWrites(_encoder, luma);
	const uint64_t chromaId = rgfxWrites(_encoder, chroma);
	if (codec == RGFX_VIDEO_CODEC_HEVC) {
		RgfxVideoOpDecodeH265 b = {};
		b.bitstream_buffer_obj_id = bitstreamId;
		b.bitstream_size = bytes;
		b.luma_texture_obj_id = lumaId;
		b.chroma_texture_obj_id = chromaId;
		b.output_rows = outputRows;
		_encoder->videoDecodeH265(b, parameterSets, size);
	} else {
		RgfxVideoOpDecodeH264 b = {};
		b.bitstream_buffer_obj_id = bitstreamId;
		b.bitstream_size = bytes;
		b.luma_texture_obj_id = lumaId;
		b.chroma_texture_obj_id = chromaId;
		b.output_rows = outputRows;
		_encoder->videoDecodeH264(b, parameterSets, size);
	}
	_encoder->endEncoder();
}

- (nullable id<MTLBlitCommandEncoder>)blitCommandEncoder
{
	_encoder->beginBlitPass();
	RgfxBlitCommandEncoder *enc = [[RgfxBlitCommandEncoder alloc] initWithEncoder:_encoder
	                                                                       device:self.device];
	[enc rgfxSetCommandBuffer:self];
	[enc rgfxSetBarrierSlot:[_queue rgfxTakeBarrierSlot]];
	return enc;
}

- (nullable id<MTLBlitCommandEncoder>)blitCommandEncoderWithDescriptor:
    (MTLBlitPassDescriptor *)blitPassDescriptor
{
	// The blit pass descriptor carries only sample-buffer attachments (diagnostics);
	// the op stream's blit pass has no body.
	return [self blitCommandEncoder];
}

- (nullable id<MTLParallelRenderCommandEncoder>)parallelRenderCommandEncoderWithDescriptor:
    (MTLRenderPassDescriptor *)renderPassDescriptor
{
	RgfxRenderPassDesc desc = {};
	std::vector<RgfxRenderPassColorAttachment> colors;
	uint32_t count = fillRenderPass(renderPassDescriptor, &desc, &colors);
	RgfxRenderPassSamplePositions positions;
	const bool programmed = fillSamplePositions(renderPassDescriptor, &positions);
	if (programmed && !rgfxTakesSamplePositions(self.device, positions))
		return nil;
	noteRenderPassWrites(_encoder, renderPassDescriptor);
	RgfxRenderPassVisibility visibility;
	const bool visible = fillVisibility(_encoder, renderPassDescriptor, &visibility);
	_encoder->beginRenderPass(desc, colors.empty() ? nullptr : colors.data(), count,
	                          programmed ? &positions : nullptr, visible ? &visibility : nullptr);
	RgfxParallelRenderCommandEncoder *enc =
	    [[RgfxParallelRenderCommandEncoder alloc] initWithEncoder:_encoder device:self.device];
	[enc rgfxSetCommandBuffer:self];
	[enc rgfxSetUploads:_uploads];
	[enc rgfxSetTileWidth:renderPassDescriptor.tileWidth height:renderPassDescriptor.tileHeight];
	[enc rgfxSetPass:renderPassDescriptor];
	[enc rgfxSetBarrierSlot:[_queue rgfxTakeBarrierSlot]];
	return enc;
}

// --- submission --------------------------------------------------------------

// Its place in its queue's order, which it runs in (ENQUEUE ORDER, above), once: a second
// enqueue, or one after the commit (which enqueues), ends the process with Metal's words, as
// Metal does on the M3 Max, an AMD 5300M and an Intel 630 (measured).
- (void)enqueue
{
	if (![_queue rgfxEnqueue:self])
		RgfxMetalAssertionFailed("-[_MTLCommandBuffer enqueue]",
		                         @"enqueue an already enqueued command buffer");
}

- (BOOL)rgfxTakeEnqueue
{
	if (_enqueued)
		return NO;
	_enqueued = YES;
	MTLCommandBufferStatus expected = MTLCommandBufferStatusNotEnqueued;
	_status.compare_exchange_strong(expected, MTLCommandBufferStatusEnqueued);
	[self rgfxNoteStage:kRgfxStageEnqueue];
	return YES;
}

- (BOOL)rgfxReadyToPost
{
	return _readyToPost.load(std::memory_order_acquire);
}

- (void)commitAndHold
{
	[self commit];
}

- (void)commit
{
	// Commit is once: a second, or a commitAndWaitUntilSubmitted after it, ends the process with
	// Metal's words, as Metal does on the M3 Max, an AMD 5300M and an Intel 630 (measured; the
	// Intel's assertion is its completed handler's).
	{
		std::lock_guard<std::mutex> hold(_handlersLock);
		if (_committed)
			RgfxMetalAssertionFailed("-[IOGPUMetalCommandBuffer validate]",
			                         @"commit an already committed command buffer");
		_committed = YES;
	}
	// Not enqueued before, it is now, behind every buffer enqueued before it.
	[_queue rgfxEnqueue:self];
	_kernelStart.store(rgfxHostTime());
	[self rgfxNoteStage:kRgfxStageCommit];
	// The IOSurface copies the CPU changed since they last synced take the surfaces'
	// pages before the GPU reads them; the copies this buffer writes go back at its
	// completion.
	rgfxIOSurfaceSyncBeforeCommit(_encoder);
	_surfaceWrites = _encoder->takeSurfaceWrites();
	// The cached surfaces it uses are named before its first encoder, each with the CPU's
	// generation of it as of this commit, so the host fills each one's device cache first where
	// the CPU wrote it since. The buffers of other queues it runs after are named at its post.
	_surfaceUses = rgfxIOSurfaceNameUses(_encoder, _surfaceWrites);
	// The surfaces it writes are named to the kext once it is posted, and retired at its
	// completion, which may come before the naming: the kext then keeps nothing of them.
	RgfxRegion *writesRegion = rgfxIOSurfaceWritesRegion(_surfaceWrites);
	_surfaceWritesRegion = writesRegion;
	_writesInPlace = writesRegion != nullptr && rgfxIOSurfaceWritesInPlace(_surfaceWrites);
	// So do the copies of app memory it uses, which take the bytes the CPU wrote since.
	_shadowWrites = rgfxShadowsSyncIn(_encoder->takeShadowUses());
	// With no handler to run and nothing to write back, its records run on the thread that takes
	// them off the ring, where nothing of the queue's waits on the completion queue: no handler
	// can be added after the commit, so the lists are final here.
	{
		std::lock_guard<std::mutex> hold(_handlersLock);
		_deliversWhereSeen = _completedHandlers.count == 0 && _scheduledHandlers.count == 0 &&
		                     (_surfaceWrites.empty() || rgfxIOSurfaceWritesInPlace(_surfaceWrites)) &&
		                     _shadowWrites.empty();
		// With no scheduled handler, its scheduling flips its status and lets its waiters go,
		// and runs nothing of the app's: it runs where it is seen too, whatever its completion
		// runs, so a buffer with only a completed handler pays the completion queue's wake once.
		_schedulesWhereSeen = _scheduledHandlers.count == 0;
	}
	// Committed before the host can see it, so a completion never finds an older status.
	_status.store(MTLCommandBufferStatusCommitted);

	// With a reader, the RGFX_EVT_CB_COMPLETE for the buffer's seqno fires its handlers,
	// flips its status and wakes a waiter. It is expected under the ring's lock, before the
	// host can see the buffer, so the completion thread never takes the completion first.
	// The handler captures `self` strongly, so a committed buffer lives until its completion
	// lands (as Metal retains an in-flight command buffer); the reader removes the one-shot
	// handler when it fires. Without a reader the status stays Committed and the handlers
	// are held.
	//
	// What the host is done with goes back on the thread that takes a record off the ring:
	// the buffer is submitted (commitAndWaitUntilSubmitted returns), and at its completion its
	// queue's slot, its message's block and its inline bytes' chunks go back and its times are
	// known, whatever handlers of the queue still run. What the app sees, its write-backs, its
	// status, its handlers and its waiters, goes in ring order to its queue's completion queue
	// (rgfxDeliver:here:), where Metal runs its handlers. Where no thread drives the ring, the
	// thread that took it off runs that too.
	RgfxCommandQueue *queue = _queue;
	_reader = queue.replyReader;
	if (_reader != nullptr) {
		_readerHold = [queue sharedReplyReader];
		_done = dispatch_group_create();
		dispatch_group_enter(_done);
		// In its queue's group, which finish waits on, until it has completed (`_left`).
		_queueCommitted = [queue rgfxCommittedGroup];
		dispatch_group_enter(_queueCommitted);
		_submitted = dispatch_group_create();
		dispatch_group_enter(_submitted);
		_scheduled = dispatch_group_create();
		dispatch_group_enter(_scheduled);
		// One commitAndWaitUntilSubmitted commits asks for its submission where the host
		// answers it: the call then returns as the host takes it, or at the signal it or a
		// buffer before it on its queue waits for, as Metal's does, not at its scheduling.
		// Nothing else waits for a submission, so no other buffer asks.
		_asksSubmitted = _waitsSubmitted && queue.asksSubmitted;
		if (_asksSubmitted)
			_encoder->appendSubmitted();
	}
	// Posted once every buffer enqueued before it is: here, or by the commit that makes the
	// last of them ready. Either way its message goes on the ring before the commit returns,
	// unless a buffer before it is not yet committed.
	_readyToPost.store(true, std::memory_order_release);
	[queue rgfxPostReady];
}

// Posts the committed buffer's message on its queue's submission ring, on the thread of the
// commit that made it the oldest ready buffer of the queue (rgfxPostReady).
- (void)rgfxPost
{
	[self rgfxNoteStage:kRgfxStageSubmitToKernel];
	RgfxCommandQueue *queue = _queue;
	// Where the process names its buffers' order, the buffers of other queues it runs after, as
	// the surfaces' blocks name them now, as it goes on its ring: a Mac orders a buffer's
	// accesses at its submission, which waits for the buffers enqueued before it on its queue to
	// be committed (rgfxIOSurfaceNameOrder).
	std::vector<RgfxSurfaceOrderWait> order;
	_surfaceReads =
	    rgfxIOSurfaceNameOrder(_encoder, _surfaceWrites, _surfaceUses, queue.rgfxObjId, &order);
	_surfaceUses.clear();
	// The copies it writes, named among their writers once it is posted: its completion may
	// clear _surfaceWrites by then.
	std::vector<std::shared_ptr<void>> written = _surfaceWrites;
	std::vector<std::shared_ptr<void>> read = _surfaceReads;
	RgfxRegion *writesRegion = _surfaceWritesRegion;
	std::function<void(uint64_t)> expect;
	std::function<void()> whileWaiting;
	if (_reader != nullptr) {
		RgfxCmdBuffer *strongSelf = self;
		RgfxReplyReader *reader = _reader;
		const bool asks = _asksSubmitted;
		const bool whereSeen = _deliversWhereSeen;
		const bool schedulesWhereSeen = _schedulesWhereSeen;
		expect = [reader, strongSelf, queue, asks, whereSeen, schedulesWhereSeen](uint64_t seqno) {
			if (asks) {
				reader->expectSubmitted(seqno, [strongSelf](const RgfxCbSubmitted &) {
					[strongSelf rgfxReachSubmitted];
				});
			}
			reader->expectScheduled(seqno, [strongSelf, queue, reader, schedulesWhereSeen](
			                                   const RgfxCbScheduled &) {
				[strongSelf rgfxReachSubmitted];
				[strongSelf rgfxScheduledNamesAccesses];
				[queue rgfxDeliver:^{
					@autoreleasepool {
						[strongSelf rgfxReachScheduled];
					}
				}
				              here:schedulesWhereSeen || !reader->driven()];
			});
			reader->expectCompletion(seqno, [strongSelf, queue, reader, whereSeen](const RgfxCbComplete &m) {
				[strongSelf rgfxHostDone:&m];
				const RgfxCbComplete msg = m;
				[queue rgfxDeliver:^{
					@autoreleasepool {
						[strongSelf rgfxDeliverCompletion:&msg];
					}
				}
				              here:whereSeen || !reader->driven()];
			});
		};
		// While the ring is full: where no thread drives the ring, take the queue's
		// completions here, as the host may be waiting for room on the completion ring before
		// it takes more. A driving thread takes them whatever handlers run.
		if (!reader->driven())
			whileWaiting = [reader] { reader->pump(); };
	}

	// The buffers of the process's other queues put on their rings since this queue last named
	// them, which the host commits this one after, as Metal orders a tracked resource across
	// queues by commit (RgfxCommitOrder.hpp); read before this one goes on its ring.
	const std::vector<RgfxSurfaceOrderWait> commitOrder = [queue rgfxNameCommitOrder];
	_encoder->prependCommitOrder(commitOrder.data(), commitOrder.size());

	// Post the command buffer on the queue's submission ring, waiting for room; one too
	// large for the ring lies in a block of the queue's pool, which the ring names, held until
	// it completes. Its size sizes the queue's next messages first.
	// The block is the buffer's before the host can see it: its completion may come before the
	// post returns.
	const uint64_t messageBytes = _encoder->messageSize();
	[queue rgfxNoteMessageBytes:messageBytes];
	const bool outOfLine = !_encoder->fitsRing();
	if (outOfLine)
		_bodyPool = [queue rgfxBodyPool];
	// The CPU's encodes into indirect command buffers go first, on the control ring, which the
	// host carries out before any later command buffer of the process: the buffer sees every
	// encode made before its commit, as on every Mac.
	// So do the destroys the kext refused before (RgfxSerializer::destroy), which free what their
	// objects lay over once posted.
	if (RgfxSerializer *serializer = [queue serializer]) {
		const RgfxIcbFlush flush = serializer->flushIcbWrites();
		if (flush.messages != 0)
			rgfxNoteIcbFlush(flush);
		serializer->postWaitingDestroys();
	}
	// Held: it waits on an event not at its value as far as this process knows, or is posted
	// behind a held buffer of its queue. Metal submits such a buffer only once it is scheduled,
	// and orders no access of another queue or process after it, nor a CPU lock of a surface it
	// writes, until then (measured on the 5300M and the 630, macOS 26.7.1: another process's
	// blit of the surface, and its CPU lock, went on at once while the write waited on its
	// event); so its accesses are named at its CB_SCHEDULED, not here.
	const bool held = [queue rgfxHeldBuffers] > 0 || ![self rgfxWaitsMet];
	if (held)
		[queue rgfxNoteHeld];
	const bool posted = _encoder->finish(&_seqno, whileWaiting, expect, outOfLine ? &_body : nullptr);
	// On its ring: the posts of other queues from here name it (RgfxCommitOrder::posted).
	if (posted)
		[queue rgfxNotePosted:_seqno];
	if (posted) {
		{
			std::lock_guard<std::mutex> g(_namedLock);
			if (!_namedGone) {
				_namedWrites = std::move(written);
				_namedReads = std::move(read);
				_namedOrder = std::move(order);
				_namedRegion = writesRegion;
			}
		}
		if (!held)
			[self rgfxNameAccesses];
		else if ((_surfaceSlots.fetch_or(kRgfxSlotsHeld, std::memory_order_acq_rel) &
		          kRgfxSlotsScheduled) != 0)
			[self rgfxNameAccesses];
	} else if (held) {
		[queue rgfxNoteHeldGone];
	}
	if (!posted)
		_surfaceWritesRegion = nullptr;
	// The CPU's part ends here, posted or not; the completion may already have come.
	CFTimeInterval unset = 0;
	_kernelEnd.compare_exchange_strong(unset, rgfxHostTime());
	// A host that answers the buffer on its doorbell's thread posted its CB_SUBMITTED before
	// the doorbell write in the post returned. Where that record is on the ring, the buffer is
	// submitted from here, after everything above that a submitted buffer is held to, so
	// commitAndWaitUntilSubmitted returns with no wake of the completion thread, whose take of
	// the record then finds the buffer submitted already.
	if (posted && _asksSubmitted && _reader != nullptr && _reader->holdsSubmitted(_seqno))
		[self rgfxReachSubmitted];
	// On its ring, and held by nothing as far as this process knows: a thread that waits for it
	// to be scheduled may ask for its record now, and one that already waits is asked for here
	// (rgfxAskScheduled). A held buffer is scheduled past its waits, so no ask is posted for it.
	if (posted && !held && _reader != nullptr &&
	    (_scheduledAsk.fetch_or(kRgfxAskPosted, std::memory_order_acq_rel) & kRgfxAskWanted) != 0)
		[self rgfxPostScheduledAsk];
	if (!posted) {
		// No block for it: the buffer fails as Metal's does when memory runs out, and its
		// handlers run off the committing thread.
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: command buffer 0x%llx: no region block for its %llu bytes: it fails "
			      @"with MTLCommandBufferErrorOutOfMemory",
			      (unsigned long long)[self rgfxObjId], (unsigned long long)messageBytes);
		RgfxCbComplete failed = {};
		failed.header.type = RGFX_EVT_CB_COMPLETE;
		failed.cb_obj_id = [self rgfxObjId];
		failed.status = RGFX_CB_INTERNAL;
		failed.error_code = MTLCommandBufferErrorOutOfMemory;
		RgfxCmdBuffer *strongSelf = self;
		[self rgfxHostDone:&failed];
		[queue rgfxDeliver:^{
			@autoreleasepool {
				[strongSelf rgfxDeliverCompletion:&failed];
			}
		}
		              here:NO];
	}
}

// The NSError Metal gives a command buffer that failed with `code`, an `MTLCommandBufferError`.
// Each description is the one Metal gave on the M3 Max (macOS 27.0.1), naming the IOGPU error
// behind the code: Apple's x86-64 guest stack gives the same for a hang and for revoked access.
// Metal puts that IOGPU error under it (`NSUnderlyingError`, `IOGPUCommandQueueErrorDomain`,
// its own description none): measured for InvalidResource, whose IOGPU code is 9 as its own
// is, so only it carries one. Internal's is a recovery's victim, the one failure Metal was seen
// to give that code for. A code with none measured has no description.
static NSError *rgfxCommandBufferError(NSInteger code)
{
	struct Known {
		NSInteger code;
		const char *description;
		bool underlying;
	};
	static const Known known[] = {
	    {MTLCommandBufferErrorInternal,
	     "Discarded (victim of GPU error/recovery) "
	     "(00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)",
	     false},
	    {MTLCommandBufferErrorTimeout,
	     "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)", false},
	    {MTLCommandBufferErrorPageFault,
	     "Caused GPU Address Fault Error (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)",
	     false},
	    {MTLCommandBufferErrorAccessRevoked,
	     "Ignored (for causing prior/excessive GPU errors) "
	     "(00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)",
	     false},
	    {MTLCommandBufferErrorOutOfMemory,
	     "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)", false},
	    {MTLCommandBufferErrorInvalidResource,
	     "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)", true},
	};
	for (const Known &k : known) {
		if (k.code != code)
			continue;
		NSMutableDictionary *info = [NSMutableDictionary
		    dictionaryWithObject:@(k.description)
		                  forKey:NSLocalizedDescriptionKey];
		if (k.underlying)
			info[NSUnderlyingErrorKey] = [NSError errorWithDomain:@"IOGPUCommandQueueErrorDomain"
			                                                 code:code
			                                             userInfo:nil];
		return [NSError errorWithDomain:MTLCommandBufferErrorDomain code:code userInfo:info];
	}
	return [NSError errorWithDomain:MTLCommandBufferErrorDomain code:code userInfo:nil];
}

// What the host is done with at the buffer's completion, on the thread that took it off the
// ring, before its record goes to the queue's completion queue: it is submitted where it was
// not yet (its CB_COMPLETE stands for both); its writes of surfaces all in place retire; the
// queue's slot goes back, so a handler that makes the next buffer never waits for handlers
// behind it; its message's block and its inline bytes' chunks go back, whether some found none
// noted for its status; and its times are taken.
// Whether every event it waits on is at its value as this process knows it: a shared event's
// value as the host posted it and the CPU set it, a plain event's as the host finished the
// buffers that signalled it (RgfxEvent's rgfxSignalledValue). A wait it cannot read is not met.
- (BOOL)rgfxWaitsMet
{
	for (const auto &wait : _eventWaits) {
		uint64_t now = 0;
		if ([wait.first isKindOfClass:[RgfxSharedEvent class]])
			now = [(RgfxSharedEvent *)wait.first signaledValue];
		else if ([wait.first isKindOfClass:[RgfxEvent class]])
			now = [(RgfxEvent *)wait.first rgfxSignalledValue];
		if (now < wait.second)
			return NO;
	}
	return YES;
}

// Names its surface accesses, once: its queue among the writers of each view it writes and the
// readers of each it only reads, so a present of the surface waits for it and a buffer of
// another queue posted after it that uses the surface runs after it where one of the two writes
// it (rgfxIOSurfaceNameOrder); and its writes to the kext, so a CPU lock of a surface it writes
// waits for it, as Metal's does once the buffer is submitted. At its post, before the commit
// returns, or, held, once it is posted and scheduled.
- (void)rgfxNameAccesses
{
	RgfxCommandQueue *queue = _queue;
	const uint64_t queueId = queue.rgfxObjId;
	std::lock_guard<std::mutex> g(_namedLock);
	if (!_namedWrites.empty())
		rgfxIOSurfaceNoteCommitted(_namedWrites, queue, queueId, _seqno, _namedOrder);
	if (!_namedReads.empty())
		rgfxIOSurfaceNoteRead(_namedReads, queue, queueId, _seqno, _namedOrder);
	if (_namedRegion != nullptr) {
		rgfxIOSurfaceWritesPosted(_namedRegion, _namedWrites, queueId, _seqno);
		_writesNamed.store(true, std::memory_order_release);
	}
	const uint32_t was = _surfaceSlots.fetch_or(kRgfxSlotsNamed, std::memory_order_acq_rel);
	if ((was & kRgfxSlotsHeld) != 0)
		[queue rgfxNoteHeldGone];
	// Where the host has finished it already, the slots just named go now.
	if ((was & kRgfxSlotsFinished) != 0)
		rgfxIOSurfaceNoteFinished(_namedWrites, _namedReads, queueId, _seqno);
}

// It is scheduled (its CB_SCHEDULED taken, or its scheduling reached): a buffer posted held has
// its accesses named now, where its post is done.
- (void)rgfxScheduledNamesAccesses
{
	const uint32_t was = _surfaceSlots.fetch_or(kRgfxSlotsScheduled, std::memory_order_acq_rel);
	if ((was & kRgfxSlotsScheduled) == 0 && (was & kRgfxSlotsHeld) != 0)
		[self rgfxNameAccesses];
}

- (void)rgfxHostDone:(const RgfxCbComplete *)msg
{
	[self rgfxNoteStage:kRgfxStageCompletionInterrupt];
	[self rgfxNoteStage:kRgfxStageKernelComplete];
	[self rgfxReachSubmitted];
	// The slots of its surfaces that still name it are free: a buffer that named it would wait
	// for nothing. Where its accesses are yet to be named, they are emptied once they are.
	if ((_surfaceSlots.fetch_or(kRgfxSlotsFinished, std::memory_order_acq_rel) &
	     kRgfxSlotsNamed) != 0) {
		std::lock_guard<std::mutex> g(_namedLock);
		rgfxIOSurfaceNoteFinished(_namedWrites, _namedReads, [_queue rgfxObjId], _seqno);
	}
	// The plain events it signalled are at their values, where it ran.
	if (msg->status == RGFX_CB_OK) {
		for (const auto &signal : _plainSignals)
			[signal.first rgfxNoteSignalled:signal.second];
	}
	_uploadsFailed = _uploads != nullptr && _uploads->failed();
	if (_uploads != nullptr) {
		_uploads->giveBack();
	}
	// The host read the message out of its block before it ran the buffer.
	if (_bodyPool != nullptr) {
		_bodyPool->giveBack(_body);
		_bodyPool = nullptr;
	}
	// A buffer that writes only surfaces in place has its results in their pages now that the
	// device has completed it: its writes retire here, so a CPU lock waiting for them goes on
	// at once, not behind the queue's earlier handlers. One with a copy to write back, behind
	// one, or not yet named (its post's naming still to come) writes its copies back and retires
	// on the queue's write-back queue, in the ring's order, and its delivery waits for that: not
	// behind the queue's earlier handlers either, so a handler that locks a surface a later
	// buffer of its queue writes waits for that buffer's write-back alone, as Metal's lock waits
	// for the GPU alone. The record's seqno is the buffer's: _seqno is written after the post,
	// which the host may complete first.
	if (_surfaceWritesRegion != nullptr) {
		const uint64_t seqno = msg->header.seqno;
		if ([_queue rgfxRetiresAtTake:_writesInPlace &&
		                              _writesNamed.load(std::memory_order_acquire)]) {
			rgfxIOSurfaceRetire(_surfaceWritesRegion, [_queue rgfxObjId], seqno);
			_surfaceWritesRegion = nullptr;
		} else {
			_writtenBack = dispatch_semaphore_create(0);
			RgfxCmdBuffer *strongSelf = self;
			dispatch_async([_queue rgfxWriteBackQueue], ^{
				[strongSelf rgfxWriteBackAndRetire:seqno];
			});
		}
	}
	// The copies it wrote that no retire waits for, of surfaces (no write named to the kext) and
	// of app memory under no-copy buffers, go back on the write-back queue too, in the ring's
	// order, as the completion is taken: not behind the queue's earlier handlers, so the CPU sees
	// the GPU's writes as the buffer is done, as on Metal's shared memory. Its delivery waits for
	// them.
	if (_writtenBack == nil &&
	    (!_shadowWrites.empty() ||
	     (!_surfaceWrites.empty() && !rgfxIOSurfaceWritesInPlace(_surfaceWrites)))) {
		dispatch_semaphore_t writtenBack = dispatch_semaphore_create(0);
		_writtenBack = writtenBack;
		RgfxCmdBuffer *strongSelf = self;
		dispatch_async([_queue rgfxWriteBackQueue], ^{
			[strongSelf rgfxWriteBackCopies];
			dispatch_semaphore_signal(writtenBack);
		});
	}
	// With no copy to write back, its results are where the CPU sees them: the resources it
	// kept are out of flight here, as Metal's are once the GPU is done, not behind the queue's
	// earlier handlers; else once the copies are written back (rgfxWriteBackCopies).
	if (_writtenBack == nil)
		_kept.land();
	[self rgfxGiveSlotBack];
	[self rgfxNoteGPUTimes:msg];
}

// On the queue's write-back queue, behind the buffers taken before it there: writes back the
// copies this buffer wrote, then retires the queue's writes up to its `seqno`, so the CPU locks
// waiting for them go on, and lets its delivery go on. Its accesses not yet named by then (a
// held buffer whose CB_SCHEDULED did not come, named at its delivery's rgfxReachScheduled; one
// whose post's naming is still to come) are named to no one: a naming already under way holds
// `_namedLock`, which this waits for.
- (void)rgfxWriteBackAndRetire:(uint64_t)seqno
{
	// Its accesses named from here would reach the kext after its retire: none are.
	{
		std::lock_guard<std::mutex> g(_namedLock);
		_namedWrites.clear();
		_namedReads.clear();
		_namedOrder.clear();
		_namedRegion = nullptr;
		_namedGone = true;
	}
	[self rgfxWriteBackCopies];
	rgfxIOSurfaceRetire(_surfaceWritesRegion, [_queue rgfxObjId], seqno);
	_surfaceWritesRegion = nullptr;
	[_queue rgfxRetiredLate];
	dispatch_semaphore_signal(_writtenBack);
}

// Writes back what this buffer wrote of its copies, on its queue's write-back queue: its
// surfaces' copies, then the app memory under its no-copy buffers; the resources it kept are
// then out of flight.
- (void)rgfxWriteBackCopies
{
	rgfxIOSurfaceWriteBack(_surfaceWrites);
	_surfaceWrites.clear();
	for (const std::shared_ptr<RgfxMemoryShadow> &shadow : _shadowWrites)
		shadow->writeBack();
	_shadowWrites.clear();
	_kept.land();
}

// Delivers the host's completion for this command buffer, on its queue's completion queue (or on
// the thread that took it off the ring, where no thread drives it), after rgfxHostDone: and
// behind the queue's earlier records: writes back what it wrote, flips the status (recording any
// error), fires the completed handlers in the order added with this buffer, and lets the waiters
// go. Its scheduled handlers run first where its CB_SCHEDULED did not come.
- (void)rgfxDeliverCompletion:(const RgfxCbComplete *)msg
{
	// A completed buffer was scheduled: where its CB_SCHEDULED did not come first (the host
	// could not carry it out), it is scheduled here, its scheduled handlers run before anything
	// of its completion.
	[self rgfxReachScheduled];
	// The surfaces and the app memory this buffer wrote hold its results before anyone sees it
	// complete: written back on the queue's write-back queue as its completion was taken, or
	// here where no take did.
	if (_writtenBack != nil)
		dispatch_semaphore_wait(_writtenBack, DISPATCH_TIME_FOREVER);
	rgfxIOSurfaceWriteBack(_surfaceWrites);
	_surfaceWrites.clear();
	_surfaceReads.clear();
	{
		std::lock_guard<std::mutex> g(_namedLock);
		_namedWrites.clear();
		_namedReads.clear();
		_namedOrder.clear();
		_namedRegion = nullptr;
		_namedGone = true;
	}
	// So does the app memory it may have written under no-copy buffers.
	for (const std::shared_ptr<RgfxMemoryShadow> &shadow : _shadowWrites)
		shadow->writeBack();
	_shadowWrites.clear();
	// The resources it kept are out of flight before it is seen complete, where its take did
	// not do it.
	_kept.land();
	// A buffer some of whose inline bytes found no chunk ran without those bindings, so it fails.
	if (msg->status == RGFX_CB_OK && _uploadsFailed) {
		// It ran without some of its bindings, so it fails as Metal's does when memory runs out.
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: a command buffer's inline bytes (set*Bytes:) found no room in the "
			      @"device's memory; it fails as out of memory");
		_cbError = rgfxCommandBufferError(MTLCommandBufferErrorOutOfMemory);
		_status.store(MTLCommandBufferStatusError);
	} else if (msg->status == RGFX_CB_OK) {
		_status.store(MTLCommandBufferStatusCompleted);
	} else {
		// The host preserved the raw MTLCommandBufferError code behind the status; it
		// travels back as the error Metal gives for it.
		_cbError = rgfxCommandBufferError(static_cast<NSInteger>(msg->error_code));
		_status.store(MTLCommandBufferStatusError);
	}
	[self rgfxNoteStage:kRgfxStageCompletionHandlerEnqueue];
	[self rgfxNoteStage:kRgfxStageCompletionHandlerExecution];
	for (MTLCommandBufferHandler block in _completedHandlers) {
		block(self);
	}
	[_completedHandlers removeAllObjects];
	// What it kept goes after its handlers, which may still name it, and before its waiters
	// wake, as Metal's buffer lets its retained references go; the blocks
	// __waitUntilCompletedAsync: was given run between the two, as Metal's do.
	_kept.clear();
	[self rgfxRunWaiters:_completedWaiters done:&_completedWaitersDone];
	if (_done != nil && !_left.exchange(true)) {
		dispatch_group_leave(_done);
		dispatch_group_leave(_queueCommitted);
	}
}

// The buffer is scheduled (its CB_SCHEDULED, or its completion where none came first), once:
// on its queue's completion queue, behind the queue's earlier records, it is submitted where it
// was not yet, becomes Scheduled where it was Committed, runs its scheduled handlers in the order
// added, then lets waitUntilScheduled return. The thread that took the record off the ring made
// it submitted already (commitAndWaitUntilSubmitted returns whatever handlers run). Metal's commitAndWaitUntilSubmitted returns before
// the scheduled handlers run, and Metal makes the buffer Scheduled only once every wait of it is
// satisfied, which the host's platform does too before it posts CB_SCHEDULED (measured on the
// M3 Max, macOS 27: on an idle queue the call returns with the buffer Committed and its
// scheduled handler about 10 us from running; one behind a long buffer returns in 0.1 ms,
// Scheduled; waitUntilScheduled behind a long buffer returns in 0.1 ms; a buffer that waits for
// an event nothing signals is neither submitted nor scheduled until the event is signalled).
- (void)rgfxReachScheduled
{
	if (_scheduledReached.exchange(true))
		return;
	[self rgfxReachSubmitted];
	[self rgfxScheduledNamesAccesses];
	MTLCommandBufferStatus committed = MTLCommandBufferStatusCommitted;
	_status.compare_exchange_strong(committed, MTLCommandBufferStatusScheduled);
	[self rgfxNoteStage:kRgfxStageSubmitToHW];
	for (MTLCommandBufferHandler block in _scheduledHandlers) {
		block(self);
	}
	[_scheduledHandlers removeAllObjects];
	[self rgfxRunWaiters:_scheduledWaiters done:&_scheduledWaitersDone];
	_scheduledRan.store(true);
	if (_scheduled != nil && !_scheduledLeft.exchange(true))
		dispatch_group_leave(_scheduled);
}

// The buffer is submitted (its CB_SUBMITTED, or its scheduling where none came first), once:
// commitAndWaitUntilSubmitted returns. Its status stays Committed until it is scheduled, as
// Metal's may at that return (measured on the M3 Max, an AMD 5300M and an Intel 630,
// docs/graphics-kext/dgpu-oracle.md, Q7). The resources it keeps are in flight from here
// (RgfxKept::begin), counted before anyone sees it submitted: a caller that finds it not yet
// submitted waits, in begin, for one counting them.
- (void)rgfxReachSubmitted
{
	if (!_submittedReached.load(std::memory_order_acquire)) {
		_kept.begin();
		_submittedReached.store(true, std::memory_order_release);
	}
	if (_submitted != nil && !_submittedLeft.exchange(true))
		dispatch_group_leave(_submitted);
}

// The GPU's times, in the guest's host time. The device's clock is the host's, not the
// guest's, so the guest's own time anchors them: the run ends when the completion is
// delivered (later than the GPU finished by the delivery's latency), and starts the run's
// length as the device measured it (gpu_end_ns - gpu_start_ns) before that, never before the
// CPU posted the buffer. With no length known (a failed buffer), the run starts as it ends.
- (void)rgfxNoteGPUTimes:(const RgfxCbComplete *)msg
{
	CFTimeInterval end = rgfxHostTime();
	CFTimeInterval unset = 0;
	_kernelEnd.compare_exchange_strong(unset, end);
	CFTimeInterval start = end;
	if (msg->gpu_start_ns != 0 && msg->gpu_end_ns >= msg->gpu_start_ns) {
		start = end - static_cast<CFTimeInterval>(msg->gpu_end_ns - msg->gpu_start_ns) / 1e9;
		start = std::max(start, _kernelEnd.load());
	}
	_gpuStart.store(start);
	_gpuEnd.store(end);
}

- (BOOL)rgfxFinished
{
	MTLCommandBufferStatus status = _status.load();
	return status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError;
}

// Invoked with this command buffer when its completion is delivered, in the order added
// (rgfxDeliverCompletion:). A handler added after the commit, before or after the buffer
// completes, ends the process with Metal's words, as Metal does on the M3 Max, an AMD 5300M and
// an Intel 630: so does presentDrawable: after the commit, which adds a scheduled handler.
- (void)addCompletedHandler:(MTLCommandBufferHandler)block
{
	if (block == nil)
		return;
	std::lock_guard<std::mutex> hold(_handlersLock);
	if (_committed)
		RgfxMetalAssertionFailed("-[_MTLCommandBuffer addCompletedHandler:]",
		                         @"Completed handler provided after commit call");
	[_completedHandlers addObject:[block copy]];
}

- (void)addScheduledHandler:(MTLCommandBufferHandler)block
{
	if (block == nil)
		return;
	std::lock_guard<std::mutex> hold(_handlersLock);
	if (_committed)
		RgfxMetalAssertionFailed("-[_MTLCommandBuffer addScheduledHandler:]",
		                         @"Scheduled handler provided after commit call");
	[_scheduledHandlers addObject:[block copy]];
}

// The command-buffer SPI SkyLight's compositor and VideoToolbox call: commits, then waits until
// the buffer is submitted (its CB_SUBMITTED where it asked for one, else its CB_SCHEDULED,
// rgfxReachSubmitted), not until it runs, as Metal's does; YES unless it failed by then. A CPU
// lock of a surface the buffer writes, in any process, waits for the GPU's writes from the
// commit on (the kext's surface locks, RgfxIOSurface.h).
- (BOOL)commitAndWaitUntilSubmitted
{
	if (!_committed)
		_waitsSubmitted = YES;
	[self commit];
	[self rgfxWaitFor:_submitted reached:&_submittedReached];
	return self.status != MTLCommandBufferStatusError;
}

- (void)waitUntilScheduled
{
	// Blocks until the buffer is scheduled and its scheduled handlers have run
	// (rgfxReachScheduled), as Metal's does, asking the host for its scheduling first.
	[self rgfxAskScheduled];
	[self rgfxWaitFor:_scheduled reached:&_scheduledRan];
}

// A thread waits for the committed buffer to be scheduled: where its queue's device answers
// RGFX_CMD_SCHEDULED_ASK (asksScheduled), and the buffer did not ask for its submission, whose
// records the host posts at its take where it may, and is not yet scheduled, the host is asked
// for its CB_SCHEDULED, once, after its message: here where the message is on its ring, else by
// its post (rgfxPost), whichever comes second. Metal makes a buffer that waits for nothing
// Scheduled within a fraction of a millisecond of its commit (measured on the M3 Max, an AMD
// 5300M and Apple's paravirtualized device: 0.05 to 0.2 ms, behind a 120 ms buffer too); the host
// posts the record at the ask where the buffer waits for nothing, not once its platform has
// submitted the buffer's work, which took 1.3 to 1.5 ms of each of the GPU process's waits during
// a window's resize on a 780M host (11-09).
- (void)rgfxAskScheduled
{
	if (_reader == nullptr || _asksSubmitted || !_queue.asksScheduled || _scheduledReached.load())
		return;
	const uint32_t prior = _scheduledAsk.fetch_or(kRgfxAskWanted, std::memory_order_acq_rel);
	if ((prior & (kRgfxAskWanted | kRgfxAskPosted)) == kRgfxAskPosted)
		[self rgfxPostScheduledAsk];
}

// Posts the buffer's RGFX_CMD_SCHEDULED_ASK on its queue's ring, after its message, waiting for
// room as a commit does.
- (void)rgfxPostScheduledAsk
{
	RgfxScheduledAsk ask = {};
	ask.cb_obj_id = [self rgfxObjId];
	ask.cb_seqno = _seqno;
	RgfxReplyReader *reader = _reader;
	std::function<void()> whileWaiting;
	if (!reader->driven())
		whileWaiting = [reader] { reader->pump(); };
	_ring->post(RGFX_CMD_SCHEDULED_ASK, &ask, sizeof(ask), nullptr, whileWaiting);
}

- (void)waitUntilCompleted
{
	// Blocks until this buffer's RGFX_EVT_CB_COMPLETE has been delivered: its status
	// flipped and its handlers run, as Metal's does.
	[self rgfxWaitFor:_done reached:nullptr];
}

// Blocks until `group` is left, which the buffer's record does on its queue's completion queue
// once `*reached` holds (the buffer's completion, for a null `reached`) and its handlers have run,
// as Metal's waits return only after the handlers (measured on the M3 Max, an AMD 5300M and an
// Intel 630). So a wait on the queue's completion queue for a buffer of that queue, a handler's
// for a later buffer of its queue or for its own buffer, waits for good, as Metal's does on all
// three. Where no thread drives the ring (a unit test, or its thread has ended), the ring is taken
// here meanwhile. Without a reader (a unit test without one, or before commit) there is nothing to
// wait on, so this returns at once.
- (void)rgfxWaitFor:(dispatch_group_t)group reached:(const std::atomic<bool> *)reached
{
	if (_reader == nullptr || group == nil) {
		return;
	}
	auto done = [self, reached] { return reached != nullptr ? reached->load() : [self rgfxFinished]; };
	while (!done()) {
		if (_reader->driven()) {
			// The completion thread takes the ring. A thread that ends (the process's
			// context gone) leaves the reader undriven, so look again each second.
			if (dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC)) == 0)
				return;
			continue;
		}
		_reader->pump();
	}
	// There, perhaps on the completion queue that is still running the handlers.
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
}

// The event of a command buffer's event op, which the buffer keeps while it runs: its id, or 0
// for an event no device of ours made (said in the log; the op is not carried).
- (uint64_t)rgfxEventOf:(id<MTLEvent>)event op:(const char *)op value:(uint64_t)value
{
	if ([(id)event isKindOfClass:[RgfxEvent class]] ||
	    [(id)event isKindOfClass:[RgfxSharedEvent class]])
		return rgfxNamed(_encoder, event);
	uint64_t n = 0;
	if (logEventOp(g_foreignEvents, &n)) {
		NSLog(@"RGFX: command buffer 0x%llx: %s an event %@ that no device of ours made, value "
		      @"%llu: not carried",
		      (unsigned long long)[self rgfxObjId], op, event, (unsigned long long)value);
	}
	return 0;
}

- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value
{
	// The work encoded after this waits, on the host's GPU, until the event reaches `value`:
	// signalled by another command buffer, or, for a shared event, set by the CPU.
	uint64_t obj = [self rgfxEventOf:event op:"encodeWaitForEvent" value:value];
	if (obj != 0 && _encoder != nullptr) {
		_encoder->waitEvent(obj, value);
		_eventWaits.emplace_back(event, value);
	}
}

- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value
{
	// The host signals it once the work encoded before this is done; a shared event's value
	// reaches this process as the host's event reaches it, where the buffer has work after this,
	// else once the buffer finishes, and always ahead of its completion (RGFX_EVT_EVENT_SIGNALLED,
	// which its queue's reader hands to the device's).
	uint64_t obj = [self rgfxEventOf:event op:"encodeSignalEvent" value:value];
	if (obj != 0 && _encoder != nullptr) {
		_encoder->signalEvent(obj, value);
		if ([(id)event isKindOfClass:[RgfxEvent class]])
			_plainSignals.emplace_back((RgfxEvent *)event, value);
	}
}

// A drawable's present, as MTLCommandBuffer.h describes it: `present` runs once this buffer is
// scheduled, from a scheduled handler added now, and only where the buffer has not failed by
// then, so the window server takes the frame as its work goes to the GPU. The drawable (a
// CAMetalLayer's) is then the layer's next frame: its presented handlers report the refresh that
// showed it, or 0 where a later frame took its place, and the layer gives it back to
// nextDrawable once the display is done with it. The block keeps the drawable until it runs.
- (void)rgfxPresent:(id<MTLDrawable>)drawable with:(void (^)(id<MTLDrawable>))present
{
	if (drawable == nil)
		return;
	[self addScheduledHandler:^(id<MTLCommandBuffer> cb) {
		if (cb.error == nil)
			present(drawable);
	}];
}

- (void)presentDrawable:(id<MTLDrawable>)drawable
{
	[self rgfxPresent:drawable
	             with:^(id<MTLDrawable> d) {
		             [d present];
	             }];
}

- (void)presentDrawable:(id<MTLDrawable>)drawable atTime:(CFTimeInterval)presentationTime
{
	[self rgfxPresent:drawable
	             with:^(id<MTLDrawable> d) {
		             [d presentAtTime:presentationTime];
	             }];
}

- (void)presentDrawable:(id<MTLDrawable>)drawable afterMinimumDuration:(CFTimeInterval)duration
{
	[self rgfxPresent:drawable
	             with:^(id<MTLDrawable> d) {
		             [d presentAfterMinimumDuration:duration];
	             }];
}

// --- MTLCommandBuffer state --------------------------------------------------

- (id<MTLCommandQueue>)commandQueue
{
	return _queue;
}
- (BOOL)retainedReferences
{
	return _retained;
}
- (MTLCommandBufferStatus)status
{
	return _status.load();
}
- (NSError *)error
{
	return _cbError;
}
- (MTLCommandBufferErrorOption)errorOptions
{
	return _errorOptions;
}
- (void)rgfxSetErrorOptions:(MTLCommandBufferErrorOption)options
{
	_errorOptions = options;
}
// What setLogs: gave, else an empty container.
- (id<MTLLogContainer>)logs
{
	return _logs != nil ? _logs : [[RgfxLogContainer alloc] init];
}
// The times (rgfxNoteGPUTimes:), 0 until the buffer has them.
- (CFTimeInterval)kernelStartTime
{
	return _kernelStart.load();
}
- (CFTimeInterval)kernelEndTime
{
	return _kernelEnd.load();
}
- (CFTimeInterval)GPUStartTime
{
	return _gpuStart.load();
}
- (CFTimeInterval)GPUEndTime
{
	return _gpuEnd.load();
}
// Its trace id (RgfxInterop.h), taken as it was made.
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

- (NSString *)label
{
	return _cbLabel;
}
- (void)setLabel:(NSString *)label
{
	_cbLabel = [label copy];
}

// What the client asked of the buffer's content protection, as set.
- (uint64_t)protectionOptions
{
	return _protectionOptions;
}

// The command-buffer SPI WindowServer's CoreDisplay calls before it fills a display surface.
// The device protects no content, so the options are kept and change nothing; options asking
// for protection are logged once.
- (void)setProtectionOptions:(uint64_t)options
{
	_protectionOptions = options;
	static std::atomic<bool> logged{false};
	if (options != 0 && !logged.exchange(true)) {
		NSLog(@"RGFX: a command buffer asks for protection options 0x%llx; the device protects "
		      @"no content",
		      static_cast<unsigned long long>(options));
	}
}

- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}

// --- unsupported: the device answers NO for these features -----------------------
// No encoder of these kinds can be made: nil, as Metal answers a device without them.

- (nullable id<MTLAccelerationStructureCommandEncoder>)accelerationStructureCommandEncoder
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}
- (id<MTLAccelerationStructureCommandEncoder>)accelerationStructureCommandEncoderWithDescriptor:
    (MTLAccelerationStructurePassDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}
- (nullable id<MTLResourceStateCommandEncoder>)resourceStateCommandEncoder
{
	RGFX_UNSUPPORTED("sparse textures");
	return nil;
}
- (nullable id<MTLResourceStateCommandEncoder>)resourceStateCommandEncoderWithDescriptor:
    (MTLResourceStatePassDescriptor *)resourceStatePassDescriptor
{
	RGFX_UNSUPPORTED("sparse textures");
	return nil;
}
// The device makes no residency set (newResidencySetWithDescriptor:error:).
- (void)useResidencySet:(id<MTLResidencySet>)residencySet
{
	if (residencySet != nil)
		RGFX_UNSUPPORTED("residency sets");
}
- (void)useResidencySets:(const id<MTLResidencySet> _Nonnull[_Nonnull])residencySets
                   count:(NSUInteger)count
{
	if (count != 0)
		RGFX_UNSUPPORTED("residency sets");
}

// --- what Core Image and MPS send a command buffer beyond MTLCommandBuffer ------------
// Each as Metal's command buffer answers it, measured on the M3 Max (macOS 27).

// A mutable dictionary of the caller's, the same one each time.
- (NSMutableDictionary *)userDictionary
{
	@synchronized(self) {
		if (_userDictionary == nil)
			_userDictionary = [NSMutableDictionary dictionary];
		return _userDictionary;
	}
}

// Metal signals the event for every agent mask tried (0, 1, 2, all ones): as
// encodeSignalEvent:value:.
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value agentMask:(uint64_t)agentMask
{
	(void)agentMask;
	[self encodeSignalEvent:event value:value];
}

// Statistics, as the queue's.
- (void)setStatEnabled:(BOOL)enabled
{
	if (enabled)
		RGFX_UNSUPPORTED("command statistics");
}
- (void)setStatLocations:(NSUInteger)locations
{
	(void)locations;
}
- (void)setStatOptions:(NSUInteger)options
{
	(void)options;
}

// Returns with nothing changed, before commit and after completion.
- (void)validate
{
}

// --- MTLCommandBufferSPI ---------------------------------------------------------------------
// The members of Metal's command-buffer SPI that the AMD 5300M's and the Intel 630's buffers both
// answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m),
// answered as Metal's base command buffer answers them (read from code: _MTLCommandBuffer and
// MTLIOAccelCommandBuffer, macOS 27.0.1); the getters of a fresh buffer answer what those two
// GPUs' buffers answer (measured: oracle_spi_members answers).

// When the buffer reached `stage`: profilingResults' times, kept while profiling is enabled.
- (void)rgfxNoteStage:(RgfxStage)stage
{
	if (_profilingEnabled.load(std::memory_order_relaxed))
		_stageTimes[stage].store(mach_absolute_time(), std::memory_order_relaxed);
}

// Runs the blocks `waiters` holds with the buffer, after its handlers of the stage, as Metal
// runs its Swift waiters after its callbacks and before its waits return, and marks them run, so
// a block given later runs at once (__waitUntilScheduledAsync:, __waitUntilCompletedAsync:). A record runs on the
// completion thread only where it runs nothing of the app's (rgfxDeliver:here:), so there the
// blocks go to the queue's completion queue instead.
- (void)rgfxRunWaiters:(NSMutableArray *)waiters done:(bool *)done
{
	NSArray *blocks = nil;
	{
		std::lock_guard<std::mutex> hold(_handlersLock);
		*done = true;
		if (waiters.count == 0)
			return;
		blocks = [waiters copy];
		[waiters removeAllObjects];
	}
	RgfxCmdBuffer *buffer = self;
	dispatch_block_t run = ^{
		for (void (^block)(id<MTLCommandBuffer>) in blocks)
			block(buffer);
	};
	if (RgfxCompletionThread::onCurrentThread())
		[_queue rgfxDeliver:run here:NO];
	else
		run();
}

// The wait Metal's Swift overlay awaits (scheduled()): `block` runs with the buffer once it is
// scheduled and its scheduled handlers have run, at once where they have.
- (void)__waitUntilScheduledAsync:(void (^)(id<MTLCommandBuffer>))block
{
	if (block == nil)
		return;
	{
		std::lock_guard<std::mutex> hold(_handlersLock);
		if (!_scheduledWaitersDone) {
			[_scheduledWaiters addObject:[block copy]];
			return;
		}
	}
	block(self);
}

// The same, once it has completed and its completed handlers have run (completed()).
- (void)__waitUntilCompletedAsync:(void (^)(id<MTLCommandBuffer>))block
{
	if (block == nil)
		return;
	{
		std::lock_guard<std::mutex> hold(_handlersLock);
		if (!_completedWaitersDone) {
			[_completedWaiters addObject:[block copy]];
			return;
		}
	}
	block(self);
}

// Metal's IOAccel buffer hands the resources and heaps named here to its kernel to purge as the
// buffer runs; the device has no such command.
- (void)addPurgedResource:(id)resource
{
	(void)resource;
	RGFX_UNSUPPORTED("purging a resource as a command buffer runs");
}

- (void)addPurgedHeap:(id)heap
{
	(void)heap;
	RGFX_UNSUPPORTED("purging a heap as a command buffer runs");
}

// Metal's base buffer raises an unrecognized selector for a commit with a deadline (it sends
// itself doesNotRecognizeSelector:), and so does its wait for the submission, which commits so.
- (void)commitWithDeadline:(uint64_t)deadline
{
	(void)deadline;
	[self doesNotRecognizeSelector:_cmd];
}

- (BOOL)commitAndWaitUntilSubmittedWithDeadline:(uint64_t)deadline
{
	[self commitWithDeadline:deadline];
	return NO;
}

// Metal takes only the descriptor's log state here, and a descriptor of the device's has none:
// the device makes no log state (newLogStateWithDescriptor:error:).
- (void)configWithCommandBufferDescriptor:(MTLCommandBufferDescriptor *)descriptor
{
	(void)descriptor;
}

- (id)deadlineProfile
{
	return nil;
}

// No debug buffer: Metal's base answers none, of length 0.
- (void *)debugBufferContentsWithLength:(NSUInteger *)length
{
	if (length != nullptr)
		*length = 0;
	return nullptr;
}

// A tool's hook that corrupts the buffer's kernel commands on purpose.
- (void)doCorruptCBSPI:(NSUInteger)how
{
	(void)how;
	RGFX_UNSUPPORTED("corrupting a command buffer on purpose");
}

// Metal's IOAccel buffer encodes a kernel command that aborts it where the event is signalled.
- (void)encodeConditionalAbortEvent:(id<MTLEvent>)event
{
	(void)event;
	RGFX_UNSUPPORTED("aborting a command buffer on an event");
}

// Resource-group dashboards: Metal's base buffer fails each as unsupported on the device, with
// its words.
- (void)encodeDashboardTagForResourceGroup:(id)group
{
	(void)group;
	RgfxMetalAssertionFailed("-[_MTLCommandBuffer encodeDashboardTagForResourceGroup:]",
	                         @"-[_MTLCommandBuffer encodeDashboardTagForResourceGroup:] not "
	                         @"supported on this device");
}

- (void)encodeDashboardFinalizeForResourceGroup:(id)group
                                      dashboard:(NSUInteger)dashboard
                                          value:(uint64_t)value
                                       forIndex:(NSUInteger)index
{
	(void)group;
	(void)dashboard;
	(void)value;
	(void)index;
	RgfxMetalAssertionFailed(
	    "-[_MTLCommandBuffer encodeDashboardFinalizeForResourceGroup:dashboard:value:forIndex:]",
	    @"-[_MTLCommandBuffer encodeDashboardFinalizeForResourceGroup:dashboard:value:forIndex:] "
	    @"not supported on this device");
}

- (void)encodeDashboardFinalizeForResourceGroup:(id)group
                                      dashboard:(NSUInteger)dashboard
                                         values:(const uint64_t *)values
                                        indices:(const NSUInteger *)indices
                                          count:(NSUInteger)count
{
	(void)group;
	(void)dashboard;
	(void)values;
	(void)indices;
	(void)count;
	RgfxMetalAssertionFailed("-[_MTLCommandBuffer "
	                         "encodeDashboardFinalizeForResourceGroup:dashboard:values:indices:count:]",
	                         @"-[_MTLCommandBuffer "
	                         @"encodeDashboardFinalizeForResourceGroup:dashboard:values:indices:count:] "
	                         @"not supported on this device");
}

- (NSUInteger)getListIndex
{
	return 0;
}

- (BOOL)isProfilingEnabled
{
	return _profilingEnabled.load();
}

- (void)setProfilingEnabled:(BOOL)enabled
{
	_profilingEnabled.store(enabled);
}

// With profiling enabled, when the buffer reached each stage, in mach_absolute_time's units under
// Metal's keys, 0 for one it has not reached; nil without. The device's stages: its creation,
// enqueue and commit; its post to the device (submitted to the kernel, and done scheduling
// there); its scheduling (submitted to the hardware); the completion's record taken off the ring
// (the interrupt, and the kernel's completion); its delivery (its handlers' enqueue and run).
- (NSDictionary *)profilingResults
{
	if (!_profilingEnabled.load())
		return nil;
	static NSString *const keys[kRgfxStageCount] = {
	    @"MTLCommandBufferCreationTime",
	    @"MTLCommandBufferEnqueueTime",
	    @"MTLCommandBufferCommitTime",
	    @"MTLCommandBufferSubmitToKernelTime",
	    @"MTLCommandBufferKernelScheduledTime",
	    @"MTLCommandBufferSubmitToHWTime",
	    @"MTLCommandBufferCompletionInterruptTime",
	    @"MTLCommandBufferKernelCompleteTime",
	    @"MTLCommandBufferCompletionHandlerEnqueueTime",
	    @"MTLCommandBufferCompletionHandlerExecutionTime",
	};
	NSMutableDictionary *results = [NSMutableDictionary dictionaryWithCapacity:kRgfxStageCount];
	for (unsigned i = 0; i < kRgfxStageCount; i++) {
		uint64_t at = _stageTimes[i].load(std::memory_order_relaxed);
		if (i == kRgfxStageKernelScheduled)
			at = _stageTimes[kRgfxStageSubmitToKernel].load(std::memory_order_relaxed);
		results[keys[i]] = @(at);
	}
	return [results copy];
}

- (id)privateData
{
	return _privateData;
}

- (void)setPrivateData:(id)privateData
{
	_privateData = privateData;
}

- (NSUInteger)privateDataOffset
{
	return _privateDataOffset;
}

- (void)setPrivateDataOffset:(NSUInteger)offset
{
	_privateDataOffset = offset;
}

// Not retained, as Metal's base keeps it.
- (id)privateLoggingBuffer
{
	return _privateLoggingBuffer;
}

- (void)setPrivateLoggingBuffer:(id)buffer
{
	_privateLoggingBuffer = buffer;
}

- (void)setLogs:(id)logs
{
	_logs = logs;
}

// Which tasks the buffer's work is for, which Metal's IOAccel buffer takes and does nothing with.
- (void)setResponsibleTaskIDs:(const int *)taskIDs count:(NSUInteger)count
{
	(void)taskIDs;
	(void)count;
}

- (BOOL)sharedIndirectionTable
{
	return _sharedIndirectionTable;
}

- (void)setSharedIndirectionTable:(BOOL)shared
{
	_sharedIndirectionTable = shared;
}

// Encoders that sample counters into a program-info buffer: Metal's base buffer answers nil for
// each, and the device samples no counters.
- (id)sampledBlitCommandEncoder:(uintptr_t)programInfoBuffer capacity:(NSUInteger)capacity
{
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledBlitCommandEncoderWithDescriptor:(uintptr_t)descriptor
                            programInfoBuffer:(uintptr_t)programInfoBuffer
                                     capacity:(NSUInteger)capacity
{
	(void)descriptor;
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledComputeCommandEncoderWithProgramInfoBuffer:(uintptr_t)programInfoBuffer
                                               capacity:(NSUInteger)capacity
{
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledComputeCommandEncoderWithDescriptor:(uintptr_t)descriptor
                               programInfoBuffer:(uintptr_t)programInfoBuffer
                                        capacity:(NSUInteger)capacity
{
	(void)descriptor;
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledComputeCommandEncoderWithDispatchType:(NSUInteger)dispatchType
                                 programInfoBuffer:(uintptr_t)programInfoBuffer
                                          capacity:(NSUInteger)capacity
{
	(void)dispatchType;
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledRenderCommandEncoderWithDescriptor:(uintptr_t)descriptor
                              programInfoBuffer:(uintptr_t)programInfoBuffer
                                       capacity:(NSUInteger)capacity
{
	(void)descriptor;
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledAccelerationStructureCommandEncoder:(uintptr_t)programInfoBuffer
                                        capacity:(NSUInteger)capacity
{
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

- (id)sampledAccelerationStructureCommandEncoderWithDescriptor:(uintptr_t)descriptor
                                             programInfoBuffer:(uintptr_t)programInfoBuffer
                                                      capacity:(NSUInteger)capacity
{
	(void)descriptor;
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

// A residency set of Metal's own: Metal's base buffer uses it as one of the caller's
// (useResidencySet:, useResidencySets:count:; read from code, _MTLCommandBuffer, macOS 27.0.1),
// which both x86 Macs' buffers answer (measured with respondsToSelector:, macOS 26.7.1,
// tools/oracle/oracle_spi_members.m ask), and so does this.
- (void)useInternalResidencySet:(id<MTLResidencySet>)residencySet
{
	[self useResidencySet:residencySet];
}

- (void)useInternalResidencySets:(const id<MTLResidencySet> _Nonnull[_Nonnull])residencySets
                           count:(NSUInteger)count
{
	[self useResidencySets:residencySets count:count];
}

@end

// =============================================================================
// RgfxRenderCommandEncoder
// =============================================================================

// The encoders' methods take the objects they are given unretained, as Metal's do: the caller
// holds each for the call, which only names it (the command buffer keeps what it names), so no
// retain and release of each argument is spent on every call.
#pragma clang attribute push(__attribute__((objc_externally_retained)), apply_to = objc_method)

@implementation RgfxRenderCommandEncoder {
	RgfxEncoder *_enc; // borrowed: the command buffer's encoder, or the segment below
	// A parallel pass's sub-encoder's own stream, which the pass joins into its command buffer's
	// at its end (RgfxEncoder::SegmentOf); null for an encoder of its own pass.
	std::unique_ptr<RgfxEncoder> _segment;
	BOOL _emitEnd;
	id _pipeline; // the last pipeline set, whose functions' arguments are below
	__weak id<MTLDevice> _device;
	// Its command buffer, held from the encoder's making until its endEncoding, as Metal's encoders
	// hold theirs (measured on the M3 Max, an AMD 5300M and an Intel 630): `_enc` is the buffer's.
	// `_commandBuffer` answers commandBuffer, nil once an ended encoder's buffer is gone.
	__weak id<MTLCommandBuffer> _commandBuffer;
	id<MTLCommandBuffer> _commandBufferHold;
	uint64_t _traceId;
	NSString *_encLabel;
	std::shared_ptr<RgfxUploadArena> _uploads; // the command buffer's, for inline bytes
	NSUInteger _tileWidth, _tileHeight;          // the pass descriptor's tile size
	RgfxBarrierSlot _barrier;                    // its queue barrier fence (QUEUE BARRIERS)
	BOOL _barrierWaited;                         // a barrier has waited for the earlier encoders
	RgfxSurfaceBindings _vertexSurfaces;         // the textures over IOSurfaces bound, by stage
	RgfxSurfaceBindings _fragmentSurfaces;
	RgfxSurfaceBindings _objectSurfaces;
	RgfxSurfaceBindings _meshSurfaces;
	// The pipeline's functions' arguments, by stage.
	std::shared_ptr<const RgfxAirFunction> _vertexAir;
	std::shared_ptr<const RgfxAirFunction> _fragmentAir;
	std::shared_ptr<const RgfxAirFunction> _objectAir;
	std::shared_ptr<const RgfxAirFunction> _meshAir;
	// Whether the host makes mesh pipelines, so the object and mesh stages' commands are sent.
	BOOL _meshes;
	// The index the encoder's indexed strips restart at since setPrimitiveRestartEnabled:YES
	// index:, where `_restartSet` (else their index type's all-ones index, as before any call).
	BOOL _restartSet;
	NSUInteger _restartIndex;
}

- (void)rgfxNoteDrawWrites
{
	_vertexSurfaces.noteWrites(_enc, _vertexAir.get());
	_fragmentSurfaces.noteWrites(_enc, _fragmentAir.get());
	_objectSurfaces.noteWrites(_enc, _objectAir.get());
	_meshSurfaces.noteWrites(_enc, _meshAir.get());
}

- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads
{
	_uploads = std::move(uploads);
}

- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height
{
	_tileWidth = width;
	_tileHeight = height;
}

- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot begin:(BOOL)begin
{
	_barrier = slot;
	if (begin && slot.waitOwn && slot.fences[slot.slot] != 0)
		_enc->renderWaitForFence(slot.fences[slot.slot], kRgfxBarrierRenderStages);
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc
                        emitEnd:(BOOL)emitEnd
                         device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_emitEnd = emitEnd;
		_device = device;
		_traceId = RgfxNextTraceObjectId();
		_meshes = [(id)device respondsToSelector:@selector(rgfxMakesMeshPipelines)] &&
		          [(id<RgfxMeshPipelineDevice>)device rgfxMakesMeshPipelines];
	}
	return self;
}

- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
	_commandBuffer = commandBuffer;
	_commandBufferHold = commandBuffer;
}

- (void)rgfxAdoptSegment:(std::unique_ptr<RgfxEncoder>)segment
{
	_segment = std::move(segment);
	_enc = _segment.get();
}

- (RgfxEncoder *)rgfxSegment
{
	return _segment.get();
}

// The command buffer that made the encoder, as Metal's encoders answer it (SPI, sent by MPS and
// SkyLight). A parallel pass's sub-encoder answers its pass's buffer: Metal's answers a buffer of
// its own on the same queue (measured on the M3 Max), which the bundle does not make.
- (id<MTLCommandBuffer>)commandBuffer
{
	return _commandBuffer;
}

// The encoder's trace id (RgfxInterop.h).
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

RGFX_LOG_UNRECOGNIZED

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}

- (void)endEncoding
{
	if (_emitEnd) {
		if (_barrier.fences[_barrier.slot] != 0)
			_enc->renderUpdateFence(_barrier.fences[_barrier.slot], kRgfxBarrierRenderStages);
		_enc->endEncoder();
	}
	_vertexSurfaces.clear();
	_fragmentSurfaces.clear();
	_objectSurfaces.clear();
	_meshSurfaces.clear();
	// Last: the buffer may go with it, and its RgfxEncoder with it.
	_commandBufferHold = nil;
}

// --- state -------------------------------------------------------------------

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipelineState
{
	if (pipelineState != _pipeline) {
		_pipeline = pipelineState;
		if ([(id)pipelineState isKindOfClass:[RgfxRenderPipelineState class]]) {
			RgfxRenderPipelineState *state = (RgfxRenderPipelineState *)pipelineState;
			_vertexAir = [state rgfxVertexAirFunction];
			_fragmentAir = [state rgfxFragmentAirFunction];
			_objectAir = [state rgfxObjectAirFunction];
			_meshAir = [state rgfxMeshAirFunction];
		} else {
			_vertexAir = nullptr;
			_fragmentAir = nullptr;
			_objectAir = nullptr;
			_meshAir = nullptr;
		}
		_vertexSurfaces.pipelineChanged();
		_fragmentSurfaces.pipelineChanged();
		_objectSurfaces.pipelineChanged();
		_meshSurfaces.pipelineChanged();
	}
	_enc->renderSetRenderPipelineState(rgfxNamed(_enc, pipelineState));
}
- (void)setDepthStencilState:(id<MTLDepthStencilState>)depthStencilState
{
	_enc->renderSetDepthStencilState(rgfxNamed(_enc, depthStencilState));
}
- (void)setViewport:(MTLViewport)viewport
{
	_enc->renderSetViewport(viewport.originX, viewport.originY, viewport.width, viewport.height,
	                        viewport.znear, viewport.zfar);
}
- (void)setViewports:(const MTLViewport *)viewports count:(NSUInteger)count
{
	std::vector<RgfxViewportElem> elems(count);
	for (NSUInteger i = 0; i < count; i++) {
		elems[i].origin_x_bits = f64bits(viewports[i].originX);
		elems[i].origin_y_bits = f64bits(viewports[i].originY);
		elems[i].width_bits = f64bits(viewports[i].width);
		elems[i].height_bits = f64bits(viewports[i].height);
		elems[i].znear_bits = f64bits(viewports[i].znear);
		elems[i].zfar_bits = f64bits(viewports[i].zfar);
	}
	_enc->renderSetViewports(elems.empty() ? nullptr : elems.data(), static_cast<uint32_t>(count));
}
- (void)setScissorRect:(MTLScissorRect)rect
{
	_enc->renderSetScissorRect(rect.x, rect.y, rect.width, rect.height);
}
- (void)setScissorRects:(const MTLScissorRect *)scissorRects count:(NSUInteger)count
{
	std::vector<RgfxScissorRectElem> elems(count);
	for (NSUInteger i = 0; i < count; i++) {
		elems[i].x = scissorRects[i].x;
		elems[i].y = scissorRects[i].y;
		elems[i].width = scissorRects[i].width;
		elems[i].height = scissorRects[i].height;
	}
	_enc->renderSetScissorRects(elems.empty() ? nullptr : elems.data(),
	                            static_cast<uint32_t>(count));
}
- (void)setCullMode:(MTLCullMode)cullMode
{
	_enc->renderSetCullMode(static_cast<uint64_t>(cullMode));
}
- (void)setFrontFacingWinding:(MTLWinding)frontFacingWinding
{
	_enc->renderSetFrontFacingWinding(static_cast<uint64_t>(frontFacingWinding));
}
- (void)setDepthCleared
{
}
- (void)setStencilCleared
{
}
- (void)setTriangleFillMode:(MTLTriangleFillMode)fillMode
{
	_enc->renderSetTriangleFillMode(static_cast<uint64_t>(fillMode));
}
- (void)setDepthClipMode:(MTLDepthClipMode)depthClipMode
{
	_enc->renderSetDepthClipModeSpi(static_cast<uint64_t>(depthClipMode));
}
- (void)setDepthBias:(float)depthBias slopeScale:(float)slopeScale clamp:(float)clamp
{
	_enc->renderSetDepthBias(depthBias, slopeScale, clamp);
}
- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha
{
	_enc->renderSetBlendColor(red, green, blue, alpha);
}
- (void)setStencilReferenceValue:(uint32_t)referenceValue
{
	_enc->renderSetStencilReferenceValues(referenceValue, referenceValue);
}
- (void)setStencilFrontReferenceValue:(uint32_t)frontReferenceValue
                   backReferenceValue:(uint32_t)backReferenceValue
{
	_enc->renderSetStencilReferenceValues(frontReferenceValue, backReferenceValue);
}
- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset
{
	_enc->renderSetVisibilityResultMode(static_cast<uint64_t>(mode), offset);
}
- (void)setVertexAmplificationCount:(NSUInteger)count
                       viewMappings:(const MTLVertexAmplificationViewMapping *)viewMappings
{
	uint32_t n = viewMappings != nullptr ? static_cast<uint32_t>(count) : 0;
	std::vector<RgfxViewMapping> elems(n);
	for (uint32_t i = 0; i < n; i++) {
		elems[i].viewport_array_index_offset = viewMappings[i].viewportArrayIndexOffset;
		elems[i].render_target_array_index_offset = viewMappings[i].renderTargetArrayIndexOffset;
	}
	_enc->renderSetVertexAmplificationCount(elems.empty() ? nullptr : elems.data(), n);
}

// --- store actions -----------------------------------------------------------

- (void)setColorStoreAction:(MTLStoreAction)storeAction atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreAction(static_cast<uint64_t>(storeAction), colorAttachmentIndex);
}
- (void)setDepthStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetDepthStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setStencilStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetStencilStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setColorStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
                           atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreActionOptions(static_cast<uint64_t>(storeActionOptions),
	                                       colorAttachmentIndex);
}
- (void)setStencilStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	_enc->renderSetStencilStoreActionOptions(static_cast<uint64_t>(storeActionOptions));
}
- (void)setDepthStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	_enc->renderSetDepthStoreActionOptions(static_cast<uint64_t>(storeActionOptions));
}

// --- vertex bindings ---------------------------------------------------------

- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetVertexBuffer(rgfxBinds(_enc, buffer), offset, index);
}
- (void)setVertexBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetVertexBufferOffset(offset, index);
}
- (void)setVertexBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
                 offsets:(const NSUInteger *)offsets
               withRange:(NSRange)range
{
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetVertexBuffers(range.location, b.empty() ? nullptr : b.data(),
	                             static_cast<uint32_t>(range.length));
}
- (void)setVertexBuffer:(id<MTLBuffer>)buffer
                 offset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	RgfxBufferStrideBinding e = {};
	e.buffer_obj_id = rgfxBinds(_enc, buffer);
	e.offset = offset;
	e.stride = stride;
	_enc->renderSetVertexBuffersAttributeStrides(index, &e, 1);
}
- (void)setVertexBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
                 offsets:(const NSUInteger *)offsets
        attributeStrides:(const NSUInteger *)strides
               withRange:(NSRange)range
{
	std::vector<RgfxBufferStrideBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
		b[i].stride = strides[i];
	}
	_enc->renderSetVertexBuffersAttributeStrides(range.location, b.empty() ? nullptr : b.data(),
	                                             static_cast<uint32_t>(range.length));
}
- (void)setVertexBufferOffset:(NSUInteger)offset
              attributeStride:(NSUInteger)stride
                      atIndex:(NSUInteger)index
{
	_enc->renderSetVertexBufferOffsetAttributeStride(offset, stride, index);
}
- (void)setVertexTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	bool overSurface = false;
	RgfxTextureBinding e = {};
	e.texture_obj_id = rgfxBinds(_enc, texture, &overSurface);
	_vertexSurfaces.bind(index, texture, overSurface);
	_enc->renderSetVertexTextures(index, &e, 1);
}
- (void)setVertexTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		bool overSurface = false;
		t[i].texture_obj_id = rgfxBinds(_enc, textures[i], &overSurface);
		_vertexSurfaces.bind(range.location + i, textures[i], overSurface);
	}
	_enc->renderSetVertexTextures(range.location, t.empty() ? nullptr : t.data(),
	                              static_cast<uint32_t>(range.length));
}
- (void)setVertexSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	_enc->renderSetVertexSamplerStates(index, &e, 1);
}
- (void)setVertexSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                     withRange:(NSRange)range
{
	std::vector<RgfxSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
	}
	_enc->renderSetVertexSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                                   static_cast<uint32_t>(range.length));
}
- (void)setVertexSamplerState:(id<MTLSamplerState>)sampler
                  lodMinClamp:(float)lodMinClamp
                  lodMaxClamp:(float)lodMaxClamp
                      atIndex:(NSUInteger)index
{
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetVertexSamplerStatesLodClamps(index, &e, 1);
}
- (void)setVertexSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                  lodMinClamps:(const float *)lodMinClamps
                  lodMaxClamps:(const float *)lodMaxClamps
                     withRange:(NSRange)range
{
	std::vector<RgfxSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetVertexSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                            static_cast<uint32_t>(range.length));
}

// --- fragment bindings -------------------------------------------------------

- (void)setFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetFragmentBuffer(rgfxBinds(_enc, buffer), offset, index);
}
- (void)setFragmentBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetFragmentBufferOffset(offset, index);
}
- (void)setFragmentBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
                   offsets:(const NSUInteger *)offsets
                 withRange:(NSRange)range
{
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetFragmentBuffers(range.location, b.empty() ? nullptr : b.data(),
	                               static_cast<uint32_t>(range.length));
}
- (void)setFragmentTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	bool overSurface = false;
	const uint64_t obj = rgfxBinds(_enc, texture, &overSurface);
	_fragmentSurfaces.bind(index, texture, overSurface);
	_enc->renderSetFragmentTexture(obj, index);
}
- (void)setFragmentTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		bool overSurface = false;
		t[i].texture_obj_id = rgfxBinds(_enc, textures[i], &overSurface);
		_fragmentSurfaces.bind(range.location + i, textures[i], overSurface);
	}
	_enc->renderSetFragmentTextures(range.location, t.empty() ? nullptr : t.data(),
	                                static_cast<uint32_t>(range.length));
}
- (void)setFragmentSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	_enc->renderSetFragmentSamplerStates(index, &e, 1);
}
- (void)setFragmentSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                       withRange:(NSRange)range
{
	std::vector<RgfxSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
	}
	_enc->renderSetFragmentSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                                     static_cast<uint32_t>(range.length));
}
- (void)setFragmentSamplerState:(id<MTLSamplerState>)sampler
                    lodMinClamp:(float)lodMinClamp
                    lodMaxClamp:(float)lodMaxClamp
                        atIndex:(NSUInteger)index
{
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetFragmentSamplerStatesLodClamps(index, &e, 1);
}
- (void)setFragmentSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                    lodMinClamps:(const float *)lodMinClamps
                    lodMaxClamps:(const float *)lodMaxClamps
                       withRange:(NSRange)range
{
	std::vector<RgfxSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetFragmentSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                             static_cast<uint32_t>(range.length));
}

// --- tile bindings -----------------------------------------------------------

- (void)setTileBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RgfxBufferBinding e = {};
	e.buffer_obj_id = rgfxBinds(_enc, buffer);
	e.offset = offset;
	_enc->renderSetTileBuffers(index, &e, 1);
}
- (void)setTileBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->renderSetTileBufferOffset(offset, index);
}
- (void)setTileBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
               offsets:(const NSUInteger *)offsets
             withRange:(NSRange)range
{
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetTileBuffers(range.location, b.empty() ? nullptr : b.data(),
	                           static_cast<uint32_t>(range.length));
}
- (void)setTileTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	RgfxTextureBinding e = {};
	e.texture_obj_id = rgfxBinds(_enc, texture);
	_enc->renderSetTileTextures(index, &e, 1);
}
- (void)setTileTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		t[i].texture_obj_id = rgfxBinds(_enc, textures[i]);
	}
	_enc->renderSetTileTextures(range.location, t.empty() ? nullptr : t.data(),
	                            static_cast<uint32_t>(range.length));
}
- (void)setTileSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	_enc->renderSetTileSamplerStates(index, &e, 1);
}
- (void)setTileSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers withRange:(NSRange)range
{
	std::vector<RgfxSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
	}
	_enc->renderSetTileSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                                 static_cast<uint32_t>(range.length));
}
- (void)setTileSamplerState:(id<MTLSamplerState>)sampler
                lodMinClamp:(float)lodMinClamp
                lodMaxClamp:(float)lodMaxClamp
                    atIndex:(NSUInteger)index
{
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetTileSamplerStatesLodClamps(index, &e, 1);
}
- (void)setTileSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
                lodMinClamps:(const float *)lodMinClamps
                lodMaxClamps:(const float *)lodMaxClamps
                   withRange:(NSRange)range
{
	std::vector<RgfxSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetTileSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                         static_cast<uint32_t>(range.length));
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length
                            offset:(NSUInteger)offset
                           atIndex:(NSUInteger)index
{
	_enc->renderSetThreadgroupMemoryLength(length, offset, index);
}
- (void)dispatchThreadsPerTile:(MTLSize)threadsPerTile
{
	_enc->renderDispatchThreadsPerTile(threadsPerTile.width, threadsPerTile.height,
	                                   threadsPerTile.depth);
}

// --- tessellation ------------------------------------------------------------

- (void)setTessellationFactorBuffer:(id<MTLBuffer>)buffer
                             offset:(NSUInteger)offset
                     instanceStride:(NSUInteger)instanceStride
{
	_enc->renderSetTessellationFactorBuffer(rgfxReads(_enc, buffer), offset, instanceStride);
}
- (void)setTessellationFactorScale:(float)scale
{
	_enc->renderSetTessellationFactorScale(scale);
}

// --- draws -------------------------------------------------------------------

- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawPrimitives(static_cast<uint64_t>(primitiveType), vertexStart, vertexCount);
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
         instanceCount:(NSUInteger)instanceCount
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawPrimitivesInstances(static_cast<uint64_t>(primitiveType), vertexStart,
	                                    vertexCount, instanceCount);
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
           vertexStart:(NSUInteger)vertexStart
           vertexCount:(NSUInteger)vertexCount
         instanceCount:(NSUInteger)instanceCount
          baseInstance:(NSUInteger)baseInstance
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawPrimitivesBase(static_cast<uint64_t>(primitiveType), vertexStart, vertexCount,
	                               instanceCount, baseInstance);
}
- (void)drawPrimitives:(MTLPrimitiveType)primitiveType
        indirectBuffer:(id<MTLBuffer>)indirectBuffer
  indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawPrimitivesIndirect(static_cast<uint64_t>(primitiveType),
	                                   rgfxReads(_enc, indirectBuffer), indirectBufferOffset);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawIndexedPrimitives(static_cast<uint64_t>(primitiveType), indexCount,
	                                  static_cast<uint64_t>(indexType), rgfxReads(_enc, indexBuffer),
	                                  indexBufferOffset);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
                instanceCount:(NSUInteger)instanceCount
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawIndexedPrimitivesInstances(static_cast<uint64_t>(primitiveType), indexCount,
	                                           static_cast<uint64_t>(indexType), rgfxReads(_enc, indexBuffer),
	                                           indexBufferOffset, instanceCount);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                   indexCount:(NSUInteger)indexCount
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
                instanceCount:(NSUInteger)instanceCount
                   baseVertex:(NSInteger)baseVertex
                 baseInstance:(NSUInteger)baseInstance
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawIndexedPrimitivesBase(static_cast<uint64_t>(primitiveType), indexCount,
	                                      static_cast<uint64_t>(indexType), rgfxReads(_enc, indexBuffer),
	                                      indexBufferOffset, instanceCount,
	                                      static_cast<int64_t>(baseVertex), baseInstance);
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)primitiveType
                    indexType:(MTLIndexType)indexType
                  indexBuffer:(id<MTLBuffer>)indexBuffer
            indexBufferOffset:(NSUInteger)indexBufferOffset
               indirectBuffer:(id<MTLBuffer>)indirectBuffer
         indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawIndexedPrimitivesIndirect(static_cast<uint64_t>(primitiveType),
	                                          static_cast<uint64_t>(indexType), rgfxReads(_enc, indexBuffer),
	                                          indexBufferOffset, rgfxReads(_enc, indirectBuffer),
	                                          indirectBufferOffset);
}

// --- patches -----------------------------------------------------------------

- (void)drawPatches:(NSUInteger)numberOfPatchControlPoints
                patchStart:(NSUInteger)patchStart
                patchCount:(NSUInteger)patchCount
          patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
    patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
             instanceCount:(NSUInteger)instanceCount
              baseInstance:(NSUInteger)baseInstance
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawPatches(numberOfPatchControlPoints, patchStart, patchCount,
	                        rgfxReads(_enc, patchIndexBuffer), patchIndexBufferOffset, instanceCount,
	                        baseInstance);
}
- (void)drawPatches:(NSUInteger)numberOfPatchControlPoints
          patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
    patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
            indirectBuffer:(id<MTLBuffer>)indirectBuffer
      indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawPatchesIndirect(numberOfPatchControlPoints, rgfxReads(_enc, patchIndexBuffer),
	                                patchIndexBufferOffset, rgfxReads(_enc, indirectBuffer),
	                                indirectBufferOffset);
}
- (void)drawIndexedPatches:(NSUInteger)numberOfPatchControlPoints
                     patchStart:(NSUInteger)patchStart
                     patchCount:(NSUInteger)patchCount
               patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
         patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
        controlPointIndexBuffer:(id<MTLBuffer>)controlPointIndexBuffer
  controlPointIndexBufferOffset:(NSUInteger)controlPointIndexBufferOffset
                  instanceCount:(NSUInteger)instanceCount
                   baseInstance:(NSUInteger)baseInstance
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawIndexedPatches(numberOfPatchControlPoints, patchStart, patchCount,
	                               rgfxReads(_enc, patchIndexBuffer), patchIndexBufferOffset,
	                               rgfxReads(_enc, controlPointIndexBuffer),
	                               controlPointIndexBufferOffset, instanceCount, baseInstance);
}
- (void)drawIndexedPatches:(NSUInteger)numberOfPatchControlPoints
               patchIndexBuffer:(id<MTLBuffer>)patchIndexBuffer
         patchIndexBufferOffset:(NSUInteger)patchIndexBufferOffset
        controlPointIndexBuffer:(id<MTLBuffer>)controlPointIndexBuffer
  controlPointIndexBufferOffset:(NSUInteger)controlPointIndexBufferOffset
                 indirectBuffer:(id<MTLBuffer>)indirectBuffer
           indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	[self rgfxNoteDrawWrites];
	_enc->renderDrawIndexedPatchesIndirect(numberOfPatchControlPoints, rgfxReads(_enc, patchIndexBuffer),
	                                       patchIndexBufferOffset, rgfxReads(_enc, controlPointIndexBuffer),
	                                       controlPointIndexBufferOffset, rgfxReads(_enc, indirectBuffer),
	                                       indirectBufferOffset);
}

// --- fences / residency / ICB ------------------------------------------------

- (void)updateFence:(id<MTLFence>)fence afterStages:(MTLRenderStages)stages
{
	_enc->renderUpdateFence(rgfxNamed(_enc, fence), static_cast<uint64_t>(stages));
}
- (void)waitForFence:(id<MTLFence>)fence beforeStages:(MTLRenderStages)stages
{
	_enc->renderWaitForFence(rgfxNamed(_enc, fence), static_cast<uint64_t>(stages));
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage
{
	RgfxResourceBinding e = {};
	e.obj_id = rgfxUses(_enc, resource, usage);
	e.kind = rgfxResourceKind(resource);
	rgfxNoteResident(_vertexSurfaces, resource, usage);
	rgfxNoteResident(_fragmentSurfaces, resource, usage);
	rgfxNoteResident(_objectSurfaces, resource, usage);
	rgfxNoteResident(_meshSurfaces, resource, usage);
	_enc->renderUseResources(&e, 1, static_cast<uint64_t>(usage));
}
- (void)useResources:(const id<MTLResource> __nonnull[__nonnull])resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
{
	std::vector<RgfxResourceBinding> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxUses(_enc, resources[i], usage);
		r[i].kind = rgfxResourceKind(resources[i]);
		rgfxNoteResident(_vertexSurfaces, resources[i], usage);
		rgfxNoteResident(_fragmentSurfaces, resources[i], usage);
		rgfxNoteResident(_objectSurfaces, resources[i], usage);
		rgfxNoteResident(_meshSurfaces, resources[i], usage);
	}
	_enc->renderUseResources(r.empty() ? nullptr : r.data(), static_cast<uint32_t>(count),
	                         static_cast<uint64_t>(usage));
}
- (void)useResource:(id<MTLResource>)resource
              usage:(MTLResourceUsage)usage
             stages:(MTLRenderStages)stages
{
	RgfxResourceBinding e = {};
	e.obj_id = rgfxUses(_enc, resource, usage);
	e.kind = rgfxResourceKind(resource);
	if (stages & MTLRenderStageVertex)
		rgfxNoteResident(_vertexSurfaces, resource, usage);
	if (stages & MTLRenderStageFragment)
		rgfxNoteResident(_fragmentSurfaces, resource, usage);
	if (stages & MTLRenderStageObject)
		rgfxNoteResident(_objectSurfaces, resource, usage);
	if (stages & MTLRenderStageMesh)
		rgfxNoteResident(_meshSurfaces, resource, usage);
	_enc->renderUseResourcesStages(&e, 1, static_cast<uint64_t>(usage),
	                               static_cast<uint64_t>(stages));
}
- (void)useResources:(const id<MTLResource> __nonnull[__nonnull])resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
              stages:(MTLRenderStages)stages
{
	std::vector<RgfxResourceBinding> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxUses(_enc, resources[i], usage);
		r[i].kind = rgfxResourceKind(resources[i]);
		if (stages & MTLRenderStageVertex)
			rgfxNoteResident(_vertexSurfaces, resources[i], usage);
		if (stages & MTLRenderStageFragment)
			rgfxNoteResident(_fragmentSurfaces, resources[i], usage);
		if (stages & MTLRenderStageObject)
			rgfxNoteResident(_objectSurfaces, resources[i], usage);
		if (stages & MTLRenderStageMesh)
			rgfxNoteResident(_meshSurfaces, resources[i], usage);
	}
	_enc->renderUseResourcesStages(r.empty() ? nullptr : r.data(), static_cast<uint32_t>(count),
	                               static_cast<uint64_t>(usage), static_cast<uint64_t>(stages));
}
- (void)useHeap:(id<MTLHeap>)heap
{
	RgfxHeapBinding e = {};
	e.heap_obj_id = rgfxNamed(_enc, heap);
	_enc->renderUseHeaps(&e, 1);
}
- (void)useHeaps:(const id<MTLHeap> __nonnull[__nonnull])heaps count:(NSUInteger)count
{
	std::vector<RgfxHeapBinding> h(count);
	for (NSUInteger i = 0; i < count; i++) {
		h[i].heap_obj_id = rgfxNamed(_enc, heaps[i]);
	}
	_enc->renderUseHeaps(h.empty() ? nullptr : h.data(), static_cast<uint32_t>(count));
}
- (void)useHeap:(id<MTLHeap>)heap stages:(MTLRenderStages)stages
{
	RgfxHeapBinding e = {};
	e.heap_obj_id = rgfxNamed(_enc, heap);
	_enc->renderUseHeapsStages(&e, 1, static_cast<uint64_t>(stages));
}
- (void)useHeaps:(const id<MTLHeap> __nonnull[__nonnull])heaps
           count:(NSUInteger)count
          stages:(MTLRenderStages)stages
{
	std::vector<RgfxHeapBinding> h(count);
	for (NSUInteger i = 0; i < count; i++) {
		h[i].heap_obj_id = rgfxNamed(_enc, heaps[i]);
	}
	_enc->renderUseHeapsStages(h.empty() ? nullptr : h.data(), static_cast<uint32_t>(count),
	                           static_cast<uint64_t>(stages));
}
// The range within the buffer's commands (RgfxIcbClamp): past its size a Mac draws part of it,
// faults or draws nothing, and the lane draws the commands it has.
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                      withRange:(NSRange)executionRange
{
	executionRange = RgfxIcbClamp(indirectCommandBuffer, executionRange);
	_enc->renderExecuteCommandsInBuffer(rgfxNamed(_enc, indirectCommandBuffer), executionRange.location,
	                                    executionRange.length);
}
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandbuffer
                 indirectBuffer:(id<MTLBuffer>)indirectRangeBuffer
           indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->renderExecuteCommandsIndirect(rgfxNamed(_enc, indirectCommandbuffer), rgfxReads(_enc, indirectRangeBuffer),
	                                    indirectBufferOffset);
}

// --- inline bytes ------------------------------------------------------------
// The wire carries no inline constant data: the bytes go to an upload chunk of the command
// buffer's (RgfxUploads.h), which is bound as a buffer at their offset there.

- (void)setVertexBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		_enc->renderSetVertexBuffer(rgfxBinds(_enc, buffer), offset, index);
}
- (void)setVertexBytes:(const void *)bytes
                length:(NSUInteger)length
       attributeStride:(NSUInteger)stride
               atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setVertexBuffer:buffer offset:offset attributeStride:stride atIndex:index];
}
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		_enc->renderSetFragmentBuffer(rgfxBinds(_enc, buffer), offset, index);
}
- (void)setTileBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setTileBuffer:buffer offset:offset atIndex:index];
}

// --- barriers ------------------------------------------------------------------

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope
                   afterStages:(MTLRenderStages)after
                  beforeStages:(MTLRenderStages)before
{
	_enc->renderMemoryBarrierWithScope(static_cast<uint64_t>(scope), static_cast<uint64_t>(after),
	                                   static_cast<uint64_t>(before));
}

// The wire's render barrier names scopes, not resources: the barrier covers the scope of every
// kind among `resources` (buffers; textures, which also covers them as render targets), a
// superset of the resources, so everything that waited on them still waits.
- (void)memoryBarrierWithResources:(const id<MTLResource> __nonnull[__nonnull])resources
                             count:(NSUInteger)count
                       afterStages:(MTLRenderStages)after
                      beforeStages:(MTLRenderStages)before
{
	MTLBarrierScope scope = 0;
	for (NSUInteger i = 0; i < count; i++) {
		if ([resources[i] conformsToProtocol:@protocol(MTLTexture)])
			scope |= MTLBarrierScopeTextures | MTLBarrierScopeRenderTargets;
		else
			scope |= MTLBarrierScopeBuffers;
	}
	if (scope != 0)
		[self memoryBarrierWithScope:scope afterStages:after beforeStages:before];
}

// What the deprecated texture barrier is: the render targets' writes before later fragment reads.
- (void)textureBarrier
{
	[self memoryBarrierWithScope:MTLBarrierScopeRenderTargets
	                 afterStages:MTLRenderStageFragment
	                beforeStages:MTLRenderStageFragment];
}

// --- the pass's tile size --------------------------------------------------------
// The pass descriptor's, where it named one; where it left the choice to the device, which
// size the host GPU picks is not on the wire: answered 0 and logged.

- (NSUInteger)tileWidth
{
	if (_tileWidth == 0)
		RGFX_UNSUPPORTED("the device's own tile size (no host query on the wire)");
	return _tileWidth;
}
- (NSUInteger)tileHeight
{
	if (_tileHeight == 0)
		RGFX_UNSUPPORTED("the device's own tile size (no host query on the wire)");
	return _tileHeight;
}

// --- unsupported: the device answers NO for these features -----------------------
// A nil or default argument asks for nothing, so it is exact; any other is logged and dropped.

- (void)setDepthTestMinBound:(float)minBound maxBound:(float)maxBound
{
	if (minBound != 0.0f || maxBound != 1.0f)
		RGFX_UNSUPPORTED("depth bounds testing (no op on the wire)");
}
- (void)setColorAttachmentMap:(nullable MTLLogicalToPhysicalColorAttachmentMap *)mapping
{
	if (mapping != nil)
		RGFX_UNSUPPORTED("color attachment mapping (no op on the wire)");
}
// Waits for every encoder made before this one on the queue (QUEUE BARRIERS), whatever the
// stages: once is enough, as no earlier encoder is added after.
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	if (_barrierWaited)
		return;
	_barrierWaited = YES;
	rgfxWaitForEarlierEncoders(_barrier, [&](RgfxObjId fence) {
		_enc->renderWaitForFence(fence, kRgfxBarrierRenderStages);
	});
}
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
	RGFX_UNSUPPORTED("counter sampling");
}

// --- mesh shading ----------------------------------------------------------------------------
// The object and mesh stages' bindings and the mesh draws, where the host makes mesh pipelines
// (RGFX_FEATURE_MESH_PIPELINES; `_meshes`). Where it makes none, no mesh pipeline can be made
// (newRenderPipelineStateWithMeshDescriptor: refuses), so these draw nothing, and a host without
// the feature is sent none of them. A mesh draw notes the writes of the object, mesh and fragment
// stages' bound IOSurface textures, as a vertex draw notes its stages' (rgfxNoteDrawWrites).

- (void)drawMeshThreadgroups:(MTLSize)threadgroupsPerGrid
    threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
      threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	[self rgfxNoteDrawWrites];
	const uint64_t grid[3] = {threadgroupsPerGrid.width, threadgroupsPerGrid.height,
	                          threadgroupsPerGrid.depth};
	const uint64_t object[3] = {threadsPerObjectThreadgroup.width,
	                            threadsPerObjectThreadgroup.height,
	                            threadsPerObjectThreadgroup.depth};
	const uint64_t mesh[3] = {threadsPerMeshThreadgroup.width, threadsPerMeshThreadgroup.height,
	                          threadsPerMeshThreadgroup.depth};
	_enc->renderDrawMeshThreadgroups(grid, object, mesh);
}
- (void)drawMeshThreads:(MTLSize)threadsPerGrid
    threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
      threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	[self rgfxNoteDrawWrites];
	const uint64_t grid[3] = {threadsPerGrid.width, threadsPerGrid.height, threadsPerGrid.depth};
	const uint64_t object[3] = {threadsPerObjectThreadgroup.width,
	                            threadsPerObjectThreadgroup.height,
	                            threadsPerObjectThreadgroup.depth};
	const uint64_t mesh[3] = {threadsPerMeshThreadgroup.width, threadsPerMeshThreadgroup.height,
	                          threadsPerMeshThreadgroup.depth};
	_enc->renderDrawMeshThreads(grid, object, mesh);
}
- (void)drawMeshThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)indirectBufferOffset
                   threadsPerObjectThreadgroup:(MTLSize)threadsPerObjectThreadgroup
                     threadsPerMeshThreadgroup:(MTLSize)threadsPerMeshThreadgroup
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	[self rgfxNoteDrawWrites];
	const uint64_t object[3] = {threadsPerObjectThreadgroup.width,
	                            threadsPerObjectThreadgroup.height,
	                            threadsPerObjectThreadgroup.depth};
	const uint64_t mesh[3] = {threadsPerMeshThreadgroup.width, threadsPerMeshThreadgroup.height,
	                          threadsPerMeshThreadgroup.depth};
	_enc->renderDrawMeshThreadgroupsIndirect(rgfxReads(_enc, indirectBuffer), indirectBufferOffset,
	                                         object, mesh);
}

- (void)setObjectBuffer:(nullable id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (buffer != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	RgfxBufferBinding e = {};
	e.buffer_obj_id = rgfxBinds(_enc, buffer);
	e.offset = offset;
	_enc->renderSetObjectBuffers(index, &e, 1);
}
- (void)setObjectBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	_enc->renderSetObjectBufferOffset(offset, index);
}
- (void)setObjectBuffers:(const id<MTLBuffer> __nullable[__nonnull])buffers
                offsets:(const NSUInteger[__nonnull])offsets
              withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetObjectBuffers(range.location, b.empty() ? nullptr : b.data(),
	                             static_cast<uint32_t>(range.length));
}
- (void)setObjectBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setObjectBuffer:buffer offset:offset atIndex:index];
}
- (void)setObjectSamplerState:(nullable id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (sampler != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	_enc->renderSetObjectSamplerStates(index, &e, 1);
}
- (void)setObjectSamplerState:(nullable id<MTLSamplerState>)sampler
                  lodMinClamp:(float)lodMinClamp
                  lodMaxClamp:(float)lodMaxClamp
                      atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (sampler != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetObjectSamplerStatesLodClamps(index, &e, 1);
}
- (void)setObjectSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                  lodMinClamps:(const float[__nonnull])lodMinClamps
                  lodMaxClamps:(const float[__nonnull])lodMaxClamps
                     withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxSamplerLodBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
		std::memcpy(&b[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&b[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetObjectSamplerStatesLodClamps(range.location, b.empty() ? nullptr : b.data(),
	                                            static_cast<uint32_t>(range.length));
}
- (void)setObjectSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                     withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxSamplerBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++)
		b[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
	_enc->renderSetObjectSamplerStates(range.location, b.empty() ? nullptr : b.data(),
	                                   static_cast<uint32_t>(range.length));
}
- (void)setObjectTexture:(nullable id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (texture != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	bool overSurface = false;
	RgfxTextureBinding e = {};
	e.texture_obj_id = rgfxBinds(_enc, texture, &overSurface);
	_objectSurfaces.bind(index, texture, overSurface);
	_enc->renderSetObjectTextures(index, &e, 1);
}
- (void)setObjectTextures:(const id<MTLTexture> __nullable[__nonnull])textures withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		bool overSurface = false;
		t[i].texture_obj_id = rgfxBinds(_enc, textures[i], &overSurface);
		_objectSurfaces.bind(range.location + i, textures[i], overSurface);
	}
	_enc->renderSetObjectTextures(range.location, t.empty() ? nullptr : t.data(),
	                              static_cast<uint32_t>(range.length));
}
- (void)setMeshBuffer:(nullable id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (buffer != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	RgfxBufferBinding e = {};
	e.buffer_obj_id = rgfxBinds(_enc, buffer);
	e.offset = offset;
	_enc->renderSetMeshBuffers(index, &e, 1);
}
- (void)setMeshBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	_enc->renderSetMeshBufferOffset(offset, index);
}
- (void)setMeshBuffers:(const id<MTLBuffer> __nullable[__nonnull])buffers
                offsets:(const NSUInteger[__nonnull])offsets
              withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->renderSetMeshBuffers(range.location, b.empty() ? nullptr : b.data(),
	                           static_cast<uint32_t>(range.length));
}
- (void)setMeshBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setMeshBuffer:buffer offset:offset atIndex:index];
}
- (void)setMeshSamplerState:(nullable id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (sampler != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	RgfxSamplerBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	_enc->renderSetMeshSamplerStates(index, &e, 1);
}
- (void)setMeshSamplerState:(nullable id<MTLSamplerState>)sampler
                  lodMinClamp:(float)lodMinClamp
                  lodMaxClamp:(float)lodMaxClamp
                      atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (sampler != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	RgfxSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->renderSetMeshSamplerStatesLodClamps(index, &e, 1);
}
- (void)setMeshSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                  lodMinClamps:(const float[__nonnull])lodMinClamps
                  lodMaxClamps:(const float[__nonnull])lodMaxClamps
                     withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxSamplerLodBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
		std::memcpy(&b[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&b[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->renderSetMeshSamplerStatesLodClamps(range.location, b.empty() ? nullptr : b.data(),
	                                          static_cast<uint32_t>(range.length));
}
- (void)setMeshSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
                     withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxSamplerBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++)
		b[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
	_enc->renderSetMeshSamplerStates(range.location, b.empty() ? nullptr : b.data(),
	                                 static_cast<uint32_t>(range.length));
}
- (void)setMeshTexture:(nullable id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	if (!_meshes) {
		if (texture != nil)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	bool overSurface = false;
	RgfxTextureBinding e = {};
	e.texture_obj_id = rgfxBinds(_enc, texture, &overSurface);
	_meshSurfaces.bind(index, texture, overSurface);
	_enc->renderSetMeshTextures(index, &e, 1);
}
- (void)setMeshTextures:(const id<MTLTexture> __nullable[__nonnull])textures withRange:(NSRange)range
{
	if (!_meshes) {
		if (range.length != 0)
			RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	std::vector<RgfxTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		bool overSurface = false;
		t[i].texture_obj_id = rgfxBinds(_enc, textures[i], &overSurface);
		_meshSurfaces.bind(range.location + i, textures[i], overSurface);
	}
	_enc->renderSetMeshTextures(range.location, t.empty() ? nullptr : t.data(),
	                            static_cast<uint32_t>(range.length));
}

- (void)setObjectThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	if (!_meshes) {
		RGFX_UNSUPPORTED(kNoMeshes);
		return;
	}
	_enc->renderSetObjectThreadgroupMemoryLength(length, index);
}

- (void)rgfxSetMeshPipelines:(BOOL)meshes
{
	_meshes = meshes;
}

// Ray tracing and function tables: the device makes no acceleration structure, intersection
// or visible function table, so only nil reaches these.
- (void)setVertexAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVertexIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVertexIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVertexVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setVertexVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setFragmentAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setFragmentIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setFragmentIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setFragmentVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setFragmentVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setTileAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setTileIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setTileIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setTileVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setTileVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}

// --- MTLRenderCommandEncoderSPI -----------------------------------------------------------
// The members of Metal's render-encoder SPI that the AMD 5300M's and the Intel 630's encoders both
// answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m).

// The object and mesh stages' acceleration structures and function tables, as the vertex,
// fragment and tile stages' (above): none bound is nothing to carry out.
- (void)setObjectAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                         atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setObjectIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                             atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setObjectIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                            withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setObjectVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                        atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setObjectVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                       withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setMeshAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setMeshIntersectionFunctionTable:
            (nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                           atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setMeshIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                          withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setMeshVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)functionTable
                      atBufferIndex:(NSUInteger)bufferIndex
{
	if (functionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setMeshVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])functionTables
                     withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}

// The depth clip mode, as setDepthClipMode: sets it: Metal's IOAccel encoder sends itself that
// (read from code, MTLIOAccelRenderCommandEncoder, macOS 27.0.1).
- (void)setDepthClipModeSPI:(MTLDepthClipMode)depthClipMode
{
	[self setDepthClipMode:depthClipMode];
}

// NO, as Metal's IOAccel encoder answers (read from code, MTLIOAccelRenderCommandEncoder, macOS
// 27.0.1).
- (BOOL)isMemorylessRender
{
	return NO;
}

// Ends the encoding and answers no program address table, as Metal's base encoder does (read
// from code, _MTLCommandEncoder, macOS 27.0.1).
- (nullable id)endEncodingAndRetrieveProgramAddressTable
{
	[self endEncoding];
	return nil;
}

// Metal's base encoder raises an unrecognized selector for these (it sends itself
// doesNotRecognizeSelector:; read from code, _MTLCommandEncoder and MTLIOAccelRenderCommandEncoder,
// macOS 27.0.1), and so does this.
- (void)setToolsDispatchBufferSPI:(NSUInteger)buffer
                          atIndex:(NSUInteger)index
                           stages:(NSUInteger)stages
{
	(void)buffer;
	(void)index;
	(void)stages;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)insertSplit
{
	[self doesNotRecognizeSelector:_cmd];
}

// A mode of vertex amplification beyond setVertexAmplificationCount:viewMappings:'s: none on the
// wire, so it is refused, and the draws amplify as that set them.
- (void)setVertexAmplificationMode:(NSUInteger)mode value:(NSUInteger)value
{
	(void)mode;
	(void)value;
	RGFX_UNSUPPORTED("a vertex amplification mode");
}

// A handler to call where the encoder's pass is split into several on the GPU: the device sends
// the pass to the host whole, so it reports no split and never calls one.
- (void)addSplitHandler:(void (^)(void))handler
{
	(void)handler;
}

// --- what QuartzCore, SkyLight and MPS send a render encoder beyond its protocol -----

// A fragment texture and its sampler, set together.
- (void)setFragmentTexture:(id<MTLTexture>)texture
            atTextureIndex:(NSUInteger)textureIndex
              samplerState:(id<MTLSamplerState>)sampler
            atSamplerIndex:(NSUInteger)samplerIndex
{
	[self setFragmentTexture:texture atIndex:textureIndex];
	[self setFragmentSamplerState:sampler atIndex:samplerIndex];
}

// The width of the lines drawn from here on, which the host's encoder sets. Metal draws a line
// whose width is not 1 as a band that many pixels wide along its minor axis (measured on the
// M3 Max: nothing at 0.5 along a pixel edge, 4 rows at 3.5).
- (void)setLineWidth:(float)width
{
	_enc->renderSetLineWidth(width);
}

// --- what the GL renderer (AppleMetalOpenGLRenderer) sends a render encoder beyond its protocol

// The index the encoder's indexed line and triangle strips restart at, as the Intel UHD 630's
// encoder answers it (measured, macOS 26.7.1; the AMD Radeon Pro 5300M's answers neither this nor
// setProvokingVertexMode:). Enabled, a strip drawn after it restarts at `index` alone, 16- and
// 32-bit alike, and draws its index type's all-ones index as a vertex; not enabled, at that
// all-ones index again, as with no call, whatever `index` is. The host draws the index the op
// carries in either case, so a call that is not enabled carries the all-ones index, and only where
// an enabled one came before it. The GL renderer sends this before its draws, not enabled on the
// 630 whatever GL's restart state, and restarts GL's strips itself; where an encoder answers
// neither, it writes "setPrimitiveRestartEnabled:index: unsupported!" to stderr once a process.
- (void)setPrimitiveRestartEnabled:(BOOL)enabled index:(NSUInteger)index
{
	if (enabled) {
		if (_restartSet && _restartIndex == index)
			return;
		_enc->renderSetPrimitiveRestartEnabled(index, true);
		_restartSet = YES;
		_restartIndex = index;
	} else if (_restartSet) {
		_enc->renderSetPrimitiveRestartEnabled(kRgfxRestartAtAllOnes, false);
		_restartSet = NO;
	}
}

// The vertex a primitive's flat inputs come from, which the host's encoder draws: mode 0 the
// first, as without a call, 1 the last (a triangle's third, strip triangle i's vertex i + 2, a
// line's second), as the Intel UHD 630's encoder answers it (measured, macOS 26.7.1).
- (void)setProvokingVertexMode:(NSUInteger)mode
{
	_enc->renderSetProvokingVertexMode(mode);
}

// A tile dispatch over part of the tiles: none on the wire (nor any tile pipeline to run).
- (void)dispatchThreadsPerTile:(MTLSize)threadsPerTile inRegion:(MTLRegion)region
{
	(void)threadsPerTile;
	(void)region;
	RGFX_UNSUPPORTED("tile dispatches over a region");
}

// Metal takes nil and its own sets; the device makes no residency set
// (newResidencySetWithDescriptor:error:), so nil is the only one this can be sent.
- (void)useResidencySet:(id<MTLResidencySet>)residencySet
{
	if (residencySet != nil)
		RGFX_UNSUPPORTED("residency sets");
}
- (void)useResidencySets:(const id<MTLResidencySet> _Nonnull[_Nonnull])residencySets
                   count:(NSUInteger)count
{
	if (count != 0)
		RGFX_UNSUPPORTED("residency sets");
}

@end

// =============================================================================
// RgfxComputeCommandEncoder
// =============================================================================

@implementation RgfxComputeCommandEncoder {
	RgfxEncoder *_enc;
	__weak id<MTLDevice> _device;
	// Its command buffer, held from the encoder's making until its endEncoding, as Metal's encoders
	// hold theirs (measured on the M3 Max, an AMD 5300M and an Intel 630): `_enc` is the buffer's.
	// `_commandBuffer` answers commandBuffer, nil once an ended encoder's buffer is gone.
	__weak id<MTLCommandBuffer> _commandBuffer;
	id<MTLCommandBuffer> _commandBufferHold;
	uint64_t _traceId;
	NSString *_encLabel;
	std::shared_ptr<RgfxUploadArena> _uploads; // the command buffer's, for inline bytes
	MTLDispatchType _dispatchType;              // the pass's; Serial, Metal's default, unless set
	RgfxBarrierSlot _barrier;                   // its queue barrier fence (QUEUE BARRIERS)
	BOOL _barrierWaited;
	RgfxSurfaceBindings _surfaces;              // the textures over IOSurfaces bound
	id _pipeline;                                // the last pipeline set, whose arguments are _air
	std::shared_ptr<const RgfxAirFunction> _air; // the pipeline's function's arguments
}

- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot
{
	_barrier = slot;
	if (slot.waitOwn && slot.fences[slot.slot] != 0)
		_enc->computeWaitForFence(slot.fences[slot.slot]);
}

- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads
{
	_uploads = std::move(uploads);
}

- (void)rgfxSetDispatchType:(MTLDispatchType)dispatchType
{
	_dispatchType = dispatchType;
}

- (MTLDispatchType)dispatchType
{
	return _dispatchType;
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_device = device;
		_traceId = RgfxNextTraceObjectId();
	}
	return self;
}

- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
	_commandBuffer = commandBuffer;
	_commandBufferHold = commandBuffer;
}

// The command buffer that made the encoder, as Metal's encoders answer it (SPI, sent by MPS and
// SkyLight). A parallel pass's sub-encoder answers its pass's buffer: Metal's answers a buffer of
// its own on the same queue (measured on the M3 Max), which the bundle does not make.
- (id<MTLCommandBuffer>)commandBuffer
{
	return _commandBuffer;
}

// The encoder's trace id (RgfxInterop.h).
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

RGFX_LOG_UNRECOGNIZED

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}
- (void)endEncoding
{
	if (_barrier.fences[_barrier.slot] != 0)
		_enc->computeUpdateFence(_barrier.fences[_barrier.slot]);
	_enc->endEncoder();
	_surfaces.clear();
	// Last: the buffer may go with it, and its RgfxEncoder with it.
	_commandBufferHold = nil;
}

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state
{
	if (state != _pipeline) {
		_pipeline = state;
		_air = [(id)state isKindOfClass:[RgfxComputePipelineState class]]
		           ? [(RgfxComputePipelineState *)state rgfxAirFunction]
		           : nullptr;
		_surfaces.pipelineChanged();
	}
	_enc->computeSetComputePipelineState(rgfxNamed(_enc, state));
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	RgfxComputeBufferBinding e = {};
	e.buffer_obj_id = rgfxBinds(_enc, buffer);
	e.offset = offset;
	_enc->computeSetBuffers(index, &e, 1);
}
- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	_enc->computeSetBufferOffset(offset, index);
}
- (void)setBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
           offsets:(const NSUInteger *)offsets
         withRange:(NSRange)range
{
	std::vector<RgfxComputeBufferBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
	}
	_enc->computeSetBuffers(range.location, b.empty() ? nullptr : b.data(),
	                        static_cast<uint32_t>(range.length));
}
- (void)setBuffer:(id<MTLBuffer>)buffer
          offset:(NSUInteger)offset
 attributeStride:(NSUInteger)stride
         atIndex:(NSUInteger)index
{
	RgfxComputeBufferStrideBinding e = {};
	e.buffer_obj_id = rgfxBinds(_enc, buffer);
	e.offset = offset;
	e.stride = stride;
	_enc->computeSetBuffersAttributeStrides(index, &e, 1);
}
- (void)setBuffers:(const id<MTLBuffer> __nullable [__nonnull])buffers
           offsets:(const NSUInteger *)offsets
  attributeStrides:(const NSUInteger *)strides
         withRange:(NSRange)range
{
	std::vector<RgfxComputeBufferStrideBinding> b(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		b[i].buffer_obj_id = rgfxBinds(_enc, buffers[i]);
		b[i].offset = offsets[i];
		b[i].stride = strides[i];
	}
	_enc->computeSetBuffersAttributeStrides(range.location, b.empty() ? nullptr : b.data(),
	                                        static_cast<uint32_t>(range.length));
}
- (void)setBufferOffset:(NSUInteger)offset
        attributeStride:(NSUInteger)stride
                atIndex:(NSUInteger)index
{
	_enc->computeSetBufferOffsetAttributeStride(offset, stride, index);
}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	bool overSurface = false;
	const uint64_t obj = rgfxBinds(_enc, texture, &overSurface);
	_surfaces.bind(index, texture, overSurface);
	_enc->computeSetTexture(obj, index);
}
- (void)setTextures:(const id<MTLTexture> __nullable [__nonnull])textures withRange:(NSRange)range
{
	std::vector<RgfxComputeTextureBinding> t(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		bool overSurface = false;
		t[i].texture_obj_id = rgfxBinds(_enc, textures[i], &overSurface);
		_surfaces.bind(range.location + i, textures[i], overSurface);
	}
	_enc->computeSetTextures(range.location, t.empty() ? nullptr : t.data(),
	                         static_cast<uint32_t>(range.length));
}
- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	RgfxComputeSamplerBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	_enc->computeSetSamplerStates(index, &e, 1);
}
- (void)setSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers withRange:(NSRange)range
{
	std::vector<RgfxComputeSamplerBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
	}
	_enc->computeSetSamplerStates(range.location, s.empty() ? nullptr : s.data(),
	                              static_cast<uint32_t>(range.length));
}
- (void)setSamplerState:(id<MTLSamplerState>)sampler
            lodMinClamp:(float)lodMinClamp
            lodMaxClamp:(float)lodMaxClamp
                atIndex:(NSUInteger)index
{
	RgfxComputeSamplerLodBinding e = {};
	e.sampler_obj_id = rgfxNamed(_enc, sampler);
	std::memcpy(&e.lod_min_clamp_bits, &lodMinClamp, 4);
	std::memcpy(&e.lod_max_clamp_bits, &lodMaxClamp, 4);
	_enc->computeSetSamplerStatesLodClamps(index, &e, 1);
}
- (void)setSamplerStates:(const id<MTLSamplerState> __nullable [__nonnull])samplers
            lodMinClamps:(const float *)lodMinClamps
            lodMaxClamps:(const float *)lodMaxClamps
               withRange:(NSRange)range
{
	std::vector<RgfxComputeSamplerLodBinding> s(range.length);
	for (NSUInteger i = 0; i < range.length; i++) {
		s[i].sampler_obj_id = rgfxNamed(_enc, samplers[i]);
		std::memcpy(&s[i].lod_min_clamp_bits, &lodMinClamps[i], 4);
		std::memcpy(&s[i].lod_max_clamp_bits, &lodMaxClamps[i], 4);
	}
	_enc->computeSetSamplerStatesLodClamps(range.location, s.empty() ? nullptr : s.data(),
	                                       static_cast<uint32_t>(range.length));
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index
{
	_enc->computeSetThreadgroupMemoryLength(length, index);
}
- (void)setImageblockWidth:(NSUInteger)width height:(NSUInteger)height
{
	_enc->computeSetImageblockWidth(width, height);
}
- (void)setStageInRegion:(MTLRegion)region
{
	_enc->computeSetStageInRegion(region.origin.x, region.origin.y, region.origin.z,
	                              region.size.width, region.size.height, region.size.depth);
}
- (void)setStageInRegionWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                      indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->computeSetStageInRegionIndirect(rgfxReads(_enc, indirectBuffer), indirectBufferOffset);
}
- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid
       threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	_surfaces.noteWrites(_enc, _air.get());
	_enc->computeDispatchThreadgroups(threadgroupsPerGrid.width, threadgroupsPerGrid.height,
	                                  threadgroupsPerGrid.depth, threadsPerThreadgroup.width,
	                                  threadsPerThreadgroup.height, threadsPerThreadgroup.depth);
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)indirectBufferOffset
                         threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	_surfaces.noteWrites(_enc, _air.get());
	_enc->computeDispatchThreadgroupsIndirect(rgfxReads(_enc, indirectBuffer), indirectBufferOffset,
	                                          threadsPerThreadgroup.width,
	                                          threadsPerThreadgroup.height,
	                                          threadsPerThreadgroup.depth);
}
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup
{
	_surfaces.noteWrites(_enc, _air.get());
	_enc->computeDispatchThreads(threadsPerGrid.width, threadsPerGrid.height, threadsPerGrid.depth,
	                             threadsPerThreadgroup.width, threadsPerThreadgroup.height,
	                             threadsPerThreadgroup.depth);
}
- (void)updateFence:(id<MTLFence>)fence
{
	_enc->computeUpdateFence(rgfxNamed(_enc, fence));
}
- (void)waitForFence:(id<MTLFence>)fence
{
	_enc->computeWaitForFence(rgfxNamed(_enc, fence));
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage
{
	RgfxComputeResourceRef e = {};
	e.obj_id = rgfxUses(_enc, resource, usage);
	e.kind = rgfxResourceKind(resource);
	rgfxNoteResident(_surfaces, resource, usage);
	_enc->computeUseResources(&e, 1, static_cast<uint64_t>(usage));
}
- (void)useResources:(const id<MTLResource> __nonnull[__nonnull])resources
               count:(NSUInteger)count
               usage:(MTLResourceUsage)usage
{
	std::vector<RgfxComputeResourceRef> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxUses(_enc, resources[i], usage);
		r[i].kind = rgfxResourceKind(resources[i]);
		rgfxNoteResident(_surfaces, resources[i], usage);
	}
	_enc->computeUseResources(r.empty() ? nullptr : r.data(), static_cast<uint32_t>(count),
	                          static_cast<uint64_t>(usage));
}
- (void)useHeap:(id<MTLHeap>)heap
{
	RgfxComputeHeapBinding e = {};
	e.heap_obj_id = rgfxNamed(_enc, heap);
	_enc->computeUseHeaps(&e, 1);
}
- (void)useHeaps:(const id<MTLHeap> __nonnull[__nonnull])heaps count:(NSUInteger)count
{
	std::vector<RgfxComputeHeapBinding> h(count);
	for (NSUInteger i = 0; i < count; i++) {
		h[i].heap_obj_id = rgfxNamed(_enc, heaps[i]);
	}
	_enc->computeUseHeaps(h.empty() ? nullptr : h.data(), static_cast<uint32_t>(count));
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope
{
	_enc->computeMemoryBarrierScope(static_cast<uint64_t>(scope));
}
- (void)memoryBarrierWithResources:(const id<MTLResource> __nonnull[__nonnull])resources
                             count:(NSUInteger)count
{
	std::vector<RgfxComputeResourceRef> r(count);
	for (NSUInteger i = 0; i < count; i++) {
		r[i].obj_id = rgfxNamed(_enc, resources[i]);
		r[i].kind = rgfxResourceKind(resources[i]);
	}
	_enc->computeMemoryBarrierResources(r.empty() ? nullptr : r.data(),
	                                    static_cast<uint32_t>(count));
}
// As the render encoder's.
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                      withRange:(NSRange)executionRange
{
	executionRange = RgfxIcbClamp(indirectCommandBuffer, executionRange);
	_enc->computeExecuteCommandsInBuffer(rgfxNamed(_enc, indirectCommandBuffer), executionRange.location,
	                                     executionRange.length);
}
- (void)executeCommandsInBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandbuffer
                 indirectBuffer:(id<MTLBuffer>)indirectRangeBuffer
           indirectBufferOffset:(NSUInteger)indirectBufferOffset
{
	_enc->computeExecuteCommandsIndirect(rgfxNamed(_enc, indirectCommandbuffer), rgfxReads(_enc, indirectRangeBuffer),
	                                     indirectBufferOffset);
}

// --- inline bytes ------------------------------------------------------------
// As the render encoder's: an upload chunk of the command buffer's, bound at its offset.

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setBuffer:buffer offset:offset atIndex:index];
}
- (void)setBytes:(const void *)bytes
             length:(NSUInteger)length
    attributeStride:(NSUInteger)stride
            atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint64_t offset = 0;
	if (rgfxPutBytes(_uploads.get(), bytes, length, &buffer, &offset))
		[self setBuffer:buffer offset:offset attributeStride:stride atIndex:index];
}

// --- unsupported: the device answers NO for these features -----------------------
// The device makes no acceleration structure or function table, so only nil reaches these.

- (void)setAccelerationStructure:(nullable id<MTLAccelerationStructure>)accelerationStructure
                   atBufferIndex:(NSUInteger)bufferIndex
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setIntersectionFunctionTable:(nullable id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                       atBufferIndex:(NSUInteger)bufferIndex
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                      withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}
- (void)setVisibleFunctionTable:(nullable id<MTLVisibleFunctionTable>)visibleFunctionTable
                  atBufferIndex:(NSUInteger)bufferIndex
{
	if (visibleFunctionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)setVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])visibleFunctionTables
                 withBufferRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
	RGFX_UNSUPPORTED("counter sampling");
}
// Waits for every encoder made before this one on the queue (QUEUE BARRIERS).
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	if (_barrierWaited)
		return;
	_barrierWaited = YES;
	rgfxWaitForEarlierEncoders(_barrier, [&](RgfxObjId fence) { _enc->computeWaitForFence(fence); });
}

// --- what MPS sends a compute encoder beyond its protocol ----------------------------

// Metal takes nil (and its own sets); the device makes no residency set, as the render
// encoder's.
- (void)useResidencySet:(id<MTLResidencySet>)residencySet
{
	if (residencySet != nil)
		RGFX_UNSUPPORTED("residency sets");
}
- (void)useResidencySets:(const id<MTLResidencySet> _Nonnull[_Nonnull])residencySets
                   count:(NSUInteger)count
{
	if (count != 0)
		RGFX_UNSUPPORTED("residency sets");
}

// A loop the GPU runs while a value in `buffer` compares true: none on the wire, so it is
// refused, and the encoder's work after it runs once.
- (void)encodeStartWhile:(id<MTLBuffer>)buffer
                  offset:(NSUInteger)offset
              comparison:(MTLCompareFunction)comparison
          referenceValue:(uint32_t)referenceValue
{
	(void)buffer;
	(void)offset;
	(void)comparison;
	(void)referenceValue;
	RGFX_UNSUPPORTED("GPU-side loops (none on the wire)");
}

// Metal answers YES closing a loop encodeStartWhile: opened, and NO with none open; no loop is
// ever open here.
- (BOOL)encodeEndWhile
{
	return NO;
}

// --- MTLComputeCommandEncoderSPI ----------------------------------------------------------
// The members of Metal's compute-encoder SPI that the AMD 5300M's and the Intel 630's encoders
// both answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m).

// A conditional the GPU evaluates on a value in `buffer`, and a loop that runs again while one
// compares true: none on the wire, so they are refused, as encodeStartWhile: is; the work of both
// branches runs, and a loop's once.
- (void)encodeStartIf:(id<MTLBuffer>)buffer
                offset:(NSUInteger)offset
            comparison:(MTLCompareFunction)comparison
        referenceValue:(uint32_t)referenceValue
{
	(void)buffer;
	(void)offset;
	(void)comparison;
	(void)referenceValue;
	RGFX_UNSUPPORTED("GPU-side conditionals (none on the wire)");
}
- (void)encodeStartElse
{
	RGFX_UNSUPPORTED("GPU-side conditionals (none on the wire)");
}
- (void)encodeStartDoWhile
{
	RGFX_UNSUPPORTED("GPU-side loops (none on the wire)");
}
- (BOOL)encodeEndDoWhile:(id<MTLBuffer>)buffer
                  offset:(NSUInteger)offset
              comparison:(MTLCompareFunction)comparison
          referenceValue:(uint32_t)referenceValue
{
	(void)buffer;
	(void)offset;
	(void)comparison;
	(void)referenceValue;
	RGFX_UNSUPPORTED("GPU-side loops (none on the wire)");
	return NO;
}
// NO, as for encodeEndWhile: no conditional is ever open here.
- (BOOL)encodeEndIf
{
	return NO;
}

// How the GPU spreads a dispatch's threadgroups over its cores: Metal's base encoder does
// nothing with either (read from code, _MTLCommandEncoder, macOS 27.0.1), and nor does this; the
// host's GPU spreads them.
- (void)setThreadgroupDistributionMode:(NSInteger)mode
{
	(void)mode;
}
- (void)setThreadgroupDistributionModeWithClusterGroupIndex:(uint32_t)index
{
	(void)index;
}

// Metal's base encoder raises an unrecognized selector for these (it sends itself
// doesNotRecognizeSelector:; read from code, _MTLCommandEncoder and MTLIOAccelComputeCommandEncoder,
// macOS 27.0.1), and so does this.
- (void)setThreadgroupPackingDisabled:(BOOL)disabled
{
	(void)disabled;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)setToolsDispatchBufferSPI:(NSUInteger)buffer atIndex:(NSUInteger)index
{
	(void)buffer;
	(void)index;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)insertSplit
{
	[self doesNotRecognizeSelector:_cmd];
}

// The debug info of the encoder's kernels: the device keeps none, so it answers nil.
- (nullable id)newKernelDebugInfo
{
	RGFX_UNSUPPORTED("a shader's debug info");
	return nil;
}

// Ends the encoding and answers no program address table, as Metal's base encoder does (read
// from code, _MTLCommandEncoder, macOS 27.0.1).
- (nullable id)endEncodingAndRetrieveProgramAddressTable
{
	[self endEncoding];
	return nil;
}

@end

// =============================================================================
// RgfxBlitCommandEncoder
// =============================================================================

// Metal's tensor slice (MTLBlitCommandEncoderSPI's copyFromTensor:sourceSlice:toTensor:
// destinationSlice:), which the device never reads: two objects, by its type encoding.
struct RgfxTensorSlice {
	__unsafe_unretained id field_00;
	__unsafe_unretained id field_08;
};

@implementation RgfxBlitCommandEncoder {
	RgfxEncoder *_enc;
	__weak id<MTLDevice> _device;
	// Its command buffer, held from the encoder's making until its endEncoding, as Metal's encoders
	// hold theirs (measured on the M3 Max, an AMD 5300M and an Intel 630): `_enc` is the buffer's.
	// `_commandBuffer` answers commandBuffer, nil once an ended encoder's buffer is gone.
	__weak id<MTLCommandBuffer> _commandBuffer;
	id<MTLCommandBuffer> _commandBufferHold;
	uint64_t _traceId;
	NSString *_encLabel;
	RgfxBarrierSlot _barrier; // its queue barrier fence (QUEUE BARRIERS)
	BOOL _barrierWaited;
}

- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot
{
	_barrier = slot;
	if (slot.waitOwn && slot.fences[slot.slot] != 0)
		_enc->blitWaitForFence(slot.fences[slot.slot]);
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_device = device;
		_traceId = RgfxNextTraceObjectId();
	}
	return self;
}

- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
	_commandBuffer = commandBuffer;
	_commandBufferHold = commandBuffer;
}

// The command buffer that made the encoder, as Metal's encoders answer it (SPI, sent by MPS and
// SkyLight). A parallel pass's sub-encoder answers its pass's buffer: Metal's answers a buffer of
// its own on the same queue (measured on the M3 Max), which the bundle does not make.
- (id<MTLCommandBuffer>)commandBuffer
{
	return _commandBuffer;
}

// The encoder's trace id (RgfxInterop.h).
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

RGFX_LOG_UNRECOGNIZED

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}
- (void)endEncoding
{
	if (_barrier.fences[_barrier.slot] != 0)
		_enc->blitUpdateFence(_barrier.fences[_barrier.slot]);
	_enc->endEncoder();
	// Last: the buffer may go with it, and its RgfxEncoder with it.
	_commandBufferHold = nil;
}

- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer
          sourceOffset:(NSUInteger)sourceOffset
              toBuffer:(id<MTLBuffer>)destinationBuffer
     destinationOffset:(NSUInteger)destinationOffset
                  size:(NSUInteger)size
{
	_enc->blitCopyBufferToBuffer(rgfxReads(_enc, sourceBuffer), rgfxWrites(_enc, destinationBuffer), sourceOffset,
	                             destinationOffset, size);
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer
          sourceOffset:(NSUInteger)sourceOffset
     sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
   sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
            sourceSize:(MTLSize)sourceSize
             toTexture:(id<MTLTexture>)destinationTexture
      destinationSlice:(NSUInteger)destinationSlice
      destinationLevel:(NSUInteger)destinationLevel
     destinationOrigin:(MTLOrigin)destinationOrigin
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyBufferToTexture(rgfxReads(_enc, sourceBuffer), rgfxWrites(_enc, destinationTexture), sourceOffset,
	                              sourceBytesPerRow, sourceBytesPerImage, sourceSize.width,
	                              sourceSize.height, sourceSize.depth, destinationSlice,
	                              destinationLevel, destinationOrigin.x, destinationOrigin.y,
	                              destinationOrigin.z);
}
- (void)copyFromBuffer:(id<MTLBuffer>)sourceBuffer
          sourceOffset:(NSUInteger)sourceOffset
     sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
   sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
            sourceSize:(MTLSize)sourceSize
             toTexture:(id<MTLTexture>)destinationTexture
      destinationSlice:(NSUInteger)destinationSlice
      destinationLevel:(NSUInteger)destinationLevel
     destinationOrigin:(MTLOrigin)destinationOrigin
               options:(MTLBlitOption)options
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyBufferToTextureOptions(rgfxReads(_enc, sourceBuffer), rgfxWrites(_enc, destinationTexture),
	                                     sourceOffset, sourceBytesPerRow, sourceBytesPerImage,
	                                     sourceSize.width, sourceSize.height, sourceSize.depth,
	                                     destinationSlice, destinationLevel, destinationOrigin.x,
	                                     destinationOrigin.y, destinationOrigin.z,
	                                     static_cast<uint64_t>(options));
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
           sourceOrigin:(MTLOrigin)sourceOrigin
             sourceSize:(MTLSize)sourceSize
               toBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
 destinationBytesPerRow:(NSUInteger)destinationBytesPerRow
destinationBytesPerImage:(NSUInteger)destinationBytesPerImage
{
	_enc->blitCopyTextureToBuffer(rgfxReads(_enc, sourceTexture), rgfxWrites(_enc, destinationBuffer), sourceSlice,
	                              sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
	                              sourceSize.width, sourceSize.height, sourceSize.depth,
	                              destinationOffset, destinationBytesPerRow,
	                              destinationBytesPerImage);
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
           sourceOrigin:(MTLOrigin)sourceOrigin
             sourceSize:(MTLSize)sourceSize
               toBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
 destinationBytesPerRow:(NSUInteger)destinationBytesPerRow
destinationBytesPerImage:(NSUInteger)destinationBytesPerImage
                options:(MTLBlitOption)options
{
	_enc->blitCopyTextureToBufferOptions(rgfxReads(_enc, sourceTexture), rgfxWrites(_enc, destinationBuffer),
	                                     sourceSlice, sourceLevel, sourceOrigin.x, sourceOrigin.y,
	                                     sourceOrigin.z, sourceSize.width, sourceSize.height,
	                                     sourceSize.depth, destinationOffset, destinationBytesPerRow,
	                                     destinationBytesPerImage, static_cast<uint64_t>(options));
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
           sourceOrigin:(MTLOrigin)sourceOrigin
             sourceSize:(MTLSize)sourceSize
              toTexture:(id<MTLTexture>)destinationTexture
       destinationSlice:(NSUInteger)destinationSlice
       destinationLevel:(NSUInteger)destinationLevel
      destinationOrigin:(MTLOrigin)destinationOrigin
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyTextureRegion(rgfxReads(_enc, sourceTexture), rgfxWrites(_enc, destinationTexture), sourceSlice,
	                            sourceLevel, sourceOrigin.x, sourceOrigin.y, sourceOrigin.z,
	                            sourceSize.width, sourceSize.height, sourceSize.depth,
	                            destinationSlice, destinationLevel, destinationOrigin.x,
	                            destinationOrigin.y, destinationOrigin.z);
}
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture
            sourceSlice:(NSUInteger)sourceSlice
            sourceLevel:(NSUInteger)sourceLevel
              toTexture:(id<MTLTexture>)destinationTexture
       destinationSlice:(NSUInteger)destinationSlice
       destinationLevel:(NSUInteger)destinationLevel
             sliceCount:(NSUInteger)sliceCount
             levelCount:(NSUInteger)levelCount
{
	rgfxIOSurfaceNoteWrite(_enc, destinationTexture);
	_enc->blitCopyTextureSlices(rgfxReads(_enc, sourceTexture), rgfxWrites(_enc, destinationTexture), sourceSlice,
	                            sourceLevel, destinationSlice, destinationLevel, sliceCount,
	                            levelCount);
}
// As many whole surfaces as both textures have, as Metal computes them: the level of one
// texture whose size is the other's first level's lines the two up; then the levels both have
// from there, and the slices both have (a cube's six faces each).
- (void)copyFromTexture:(id<MTLTexture>)sourceTexture toTexture:(id<MTLTexture>)destinationTexture
{
	NSUInteger sourceLevel = 0, destinationLevel = 0;
	if (!rgfxLevelMatching(sourceTexture, destinationTexture, &sourceLevel)) {
		if (!rgfxLevelMatching(destinationTexture, sourceTexture, &destinationLevel))
			return; // no level of one is the other's size: Metal's validation refuses the copy
	}
	NSUInteger levels = std::min(sourceTexture.mipmapLevelCount - sourceLevel,
	                             destinationTexture.mipmapLevelCount - destinationLevel);
	NSUInteger slices = std::min(rgfxSurfaceSlices(sourceTexture), rgfxSurfaceSlices(destinationTexture));
	[self copyFromTexture:sourceTexture
	          sourceSlice:0
	          sourceLevel:sourceLevel
	            toTexture:destinationTexture
	     destinationSlice:0
	     destinationLevel:destinationLevel
	           sliceCount:slices
	           levelCount:levels];
}
- (void)generateMipmapsForTexture:(id<MTLTexture>)texture
{
	rgfxIOSurfaceNoteWrite(_enc, texture);
	_enc->blitGenerateMipmaps(rgfxWrites(_enc, texture));
}
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value
{
	_enc->blitFillBuffer(rgfxWrites(_enc, buffer), range.location, range.length, value);
}
- (void)updateFence:(id<MTLFence>)fence
{
	_enc->blitUpdateFence(rgfxNamed(_enc, fence));
}
- (void)waitForFence:(id<MTLFence>)fence
{
	_enc->blitWaitForFence(rgfxNamed(_enc, fence));
}
- (void)optimizeContentsForGPUAccess:(id<MTLTexture>)texture
{
	_enc->blitOptimizeForGpu(rgfxNamed(_enc, texture));
}
- (void)optimizeContentsForGPUAccess:(id<MTLTexture>)texture
                               slice:(NSUInteger)slice
                               level:(NSUInteger)level
{
	_enc->blitOptimizeForGpuSlice(rgfxNamed(_enc, texture), slice, level);
}
- (void)optimizeContentsForCPUAccess:(id<MTLTexture>)texture
{
	_enc->blitOptimizeForCpu(rgfxNamed(_enc, texture));
}
- (void)optimizeContentsForCPUAccess:(id<MTLTexture>)texture
                               slice:(NSUInteger)slice
                               level:(NSUInteger)level
{
	_enc->blitOptimizeForCpuSlice(rgfxNamed(_enc, texture), slice, level);
}
// Each range within its buffer's commands (RgfxIcbClamp), as the encoders' executes; a copy
// takes no more commands than the destination has from its index.
- (void)resetCommandsInBuffer:(id<MTLIndirectCommandBuffer>)buffer withRange:(NSRange)range
{
	range = RgfxIcbClamp(buffer, range);
	_enc->blitResetCommandsInBuffer(rgfxNamed(_enc, buffer), range.location, range.length);
}
- (void)copyIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)source
                      sourceRange:(NSRange)sourceRange
                      destination:(id<MTLIndirectCommandBuffer>)destination
                 destinationIndex:(NSUInteger)destinationIndex
{
	sourceRange = RgfxIcbClamp(source, sourceRange);
	const NSRange into = RgfxIcbClamp(destination, NSMakeRange(destinationIndex, sourceRange.length));
	_enc->blitCopyIndirectCommandBuffer(rgfxNamed(_enc, source), rgfxNamed(_enc, destination), sourceRange.location,
	                                    into.length, into.location);
}
- (void)optimizeIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                            withRange:(NSRange)range
{
	range = RgfxIcbClamp(indirectCommandBuffer, range);
	_enc->blitOptimizeIndirectCommandBuffer(rgfxNamed(_enc, indirectCommandBuffer), range.location,
	                                        range.length);
}
- (void)synchronizeResource:(id<MTLResource>)resource
{
	_enc->blitSynchronizeResource(rgfxWrites(_enc, resource), rgfxResourceKind(resource));
}
- (void)synchronizeTexture:(id<MTLTexture>)texture slice:(NSUInteger)slice level:(NSUInteger)level
{
	_enc->blitSynchronizeTexture(rgfxWrites(_enc, texture), slice, level);
}

// --- unsupported: the device answers NO for these features -----------------------

- (void)resolveCounters:(id<MTLCounterSampleBuffer>)sampleBuffer
                inRange:(NSRange)range
      destinationBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)destinationOffset
{
	RGFX_UNSUPPORTED("counter sampling");
}
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier
{
	RGFX_UNSUPPORTED("counter sampling");
}
// The device makes no tensor (newTensorWithDescriptor:error:), so these copy nothing.
- (void)copyFromTensor:(id<MTLTensor>)sourceTensor
              sourceOrigin:(MTLTensorExtents *)sourceOrigin
          sourceDimensions:(MTLTensorExtents *)sourceDimensions
                  toTensor:(id<MTLTensor>)destinationTensor
         destinationOrigin:(MTLTensorExtents *)destinationOrigin
     destinationDimensions:(MTLTensorExtents *)destinationDimensions
{
	RGFX_UNSUPPORTED("tensors");
}
- (void)copyFromTensor:(id<MTLTensor>)sourceTensor
              sourceOrigin:(MTLTensorExtents *)sourceOrigin
          sourceDimensions:(MTLTensorExtents *)sourceDimensions
               sourcePlane:(MTLTensorPlaneType)sourcePlane
                  toTensor:(id<MTLTensor>)destinationTensor
         destinationOrigin:(MTLTensorExtents *)destinationOrigin
     destinationDimensions:(MTLTensorExtents *)destinationDimensions
          destinationPlane:(MTLTensorPlaneType)destinationPlane
{
	RGFX_UNSUPPORTED("tensors");
}
// Waits for every encoder made before this one on the queue (QUEUE BARRIERS).
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	if (_barrierWaited)
		return;
	_barrierWaited = YES;
	rgfxWaitForEarlierEncoders(_barrier, [&](RgfxObjId fence) { _enc->blitWaitForFence(fence); });
}

// --- MTLBlitCommandEncoderSPI -------------------------------------------------------------
// The members of Metal's blit-encoder SPI that the AMD 5300M's and the Intel 630's encoders both
// answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m).

// A copy between slices of tensors: the device makes no tensor, as for the copies above.
- (void)copyFromTensor:(id)sourceTensor
           sourceSlice:(RgfxTensorSlice)sourceSlice
              toTensor:(id)destinationTensor
      destinationSlice:(RgfxTensorSlice)destinationSlice
{
	(void)sourceTensor;
	(void)sourceSlice;
	(void)destinationTensor;
	(void)destinationSlice;
	RGFX_UNSUPPORTED("tensors");
}

// Metal's base encoder raises an unrecognized selector for these (it sends itself
// doesNotRecognizeSelector:; read from code, _MTLCommandEncoder, macOS 27.0.1), and so does this.
- (void)invalidateCompressedTexture:(id<MTLTexture>)texture
{
	(void)texture;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)invalidateCompressedTexture:(id<MTLTexture>)texture
                              slice:(NSUInteger)slice
                              level:(NSUInteger)level
{
	(void)texture;
	(void)slice;
	(void)level;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)optimizeContentsForTexture:(id<MTLTexture>)texture
                 readAccessPattern:(NSInteger)readAccessPattern
                      readAccessor:(NSInteger)readAccessor
{
	(void)texture;
	(void)readAccessPattern;
	(void)readAccessor;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)optimizeContentsForTexture:(id<MTLTexture>)texture
                 readAccessPattern:(NSInteger)readAccessPattern
                      readAccessor:(NSInteger)readAccessor
                             slice:(NSUInteger)slice
                             level:(NSUInteger)level
{
	(void)texture;
	(void)readAccessPattern;
	(void)readAccessor;
	(void)slice;
	(void)level;
	[self doesNotRecognizeSelector:_cmd];
}
// The fills of Metal's IOAccel blit encoder, which both x86 Macs' encoders are (read from code,
// MTLIOAccelBlitCommandEncoder, macOS 27.0.1): a buffer's and a texture's from bytes raise an
// unrecognized selector (it sends itself doesNotRecognizeSelector:), and a texture's from a color
// fills it from the color packed in the texture's own format, or in `pixelFormat`, which then
// raises. So do these; the bytes passed on are never read.
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range pattern4:(uint32_t)pattern
{
	(void)buffer;
	(void)range;
	(void)pattern;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)fillTexture:(id<MTLTexture>)texture
              level:(NSUInteger)level
              slice:(NSUInteger)slice
             region:(MTLRegion)region
              bytes:(const void *)bytes
             length:(NSUInteger)length
{
	(void)texture;
	(void)level;
	(void)slice;
	(void)region;
	(void)bytes;
	(void)length;
	[self doesNotRecognizeSelector:_cmd];
}
- (void)fillTexture:(id<MTLTexture>)texture
              level:(NSUInteger)level
              slice:(NSUInteger)slice
             region:(MTLRegion)region
              color:(MTLClearColor)color
{
	[self fillTexture:texture
	            level:level
	            slice:slice
	           region:region
	            color:color
	      pixelFormat:texture.pixelFormat];
}
- (void)fillTexture:(id<MTLTexture>)texture
              level:(NSUInteger)level
              slice:(NSUInteger)slice
             region:(MTLRegion)region
              color:(MTLClearColor)color
        pixelFormat:(MTLPixelFormat)pixelFormat
{
	(void)color;
	(void)pixelFormat;
	const uint8_t packed[16] = {};
	[self fillTexture:texture level:level slice:slice region:region bytes:packed length:sizeof packed];
}
- (void)insertSplit
{
	[self doesNotRecognizeSelector:_cmd];
}

// Ends the encoding and answers no program address table, as Metal's base encoder does (read
// from code, _MTLCommandEncoder, macOS 27.0.1).
- (nullable id)endEncodingAndRetrieveProgramAddressTable
{
	[self endEncoding];
	return nil;
}

@end

// =============================================================================
// RgfxParallelRenderCommandEncoder
// =============================================================================

@implementation RgfxParallelRenderCommandEncoder {
	RgfxEncoder *_enc;
	__weak id<MTLDevice> _device;
	// Its command buffer, held from the encoder's making until its endEncoding, as Metal's encoders
	// hold theirs (measured on the M3 Max, an AMD 5300M and an Intel 630): `_enc` is the buffer's.
	// `_commandBuffer` answers commandBuffer, nil once an ended encoder's buffer is gone.
	__weak id<MTLCommandBuffer> _commandBuffer;
	id<MTLCommandBuffer> _commandBufferHold;
	uint64_t _traceId;
	NSString *_encLabel;
	BOOL _ended;
	std::shared_ptr<RgfxUploadArena> _uploads; // the command buffer's, for its sub-encoders
	NSUInteger _tileWidth, _tileHeight;
	RgfxBarrierSlot _barrier; // the pass's queue barrier fence (QUEUE BARRIERS)
	BOOL _barrierWaited;
	MTLRenderPassDescriptor *_pass; // a copy of the pass's descriptor
	// Its sub-encoders, in the order they were made, which is the order they run in; any
	// thread may make one.
	os_unfair_lock _subsLock;
	NSMutableArray<RgfxRenderCommandEncoder *> *_subs;
}

- (void)rgfxSetPass:(MTLRenderPassDescriptor *)pass
{
	_pass = [pass copy];
}

- (void)rgfxSetBarrierSlot:(const RgfxBarrierSlot &)slot
{
	_barrier = slot;
	if (slot.waitOwn && slot.fences[slot.slot] != 0)
		_enc->renderWaitForFence(slot.fences[slot.slot], kRgfxBarrierRenderStages);
}

- (void)rgfxSetUploads:(std::shared_ptr<RgfxUploadArena>)uploads
{
	_uploads = std::move(uploads);
}

- (void)rgfxSetTileWidth:(NSUInteger)width height:(NSUInteger)height
{
	_tileWidth = width;
	_tileHeight = height;
}

- (instancetype)initWithEncoder:(RgfxEncoder *)enc device:(id<MTLDevice>)device
{
	self = [super init];
	if (self) {
		_enc = enc;
		_device = device;
		_traceId = RgfxNextTraceObjectId();
		_subsLock = OS_UNFAIR_LOCK_INIT;
		_subs = [NSMutableArray array];
	}
	return self;
}

- (void)rgfxSetCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
	_commandBuffer = commandBuffer;
	_commandBufferHold = commandBuffer;
}

// The command buffer that made the encoder, as Metal's encoders answer it (SPI, sent by MPS and
// SkyLight). A parallel pass's sub-encoder answers its pass's buffer: Metal's answers a buffer of
// its own on the same queue (measured on the M3 Max), which the bundle does not make.
- (id<MTLCommandBuffer>)commandBuffer
{
	return _commandBuffer;
}

// The encoder's trace id (RgfxInterop.h).
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

RGFX_LOG_UNRECOGNIZED

- (id<MTLDevice>)device
{
	return _device;
}
- (NSString *)label
{
	return _encLabel;
}
- (void)setLabel:(NSString *)label
{
	_encLabel = [label copy];
}
- (void)insertDebugSignpost:(NSString *)string
{
	(void)string;
}
- (void)pushDebugGroup:(NSString *)string
{
	(void)string;
}
- (void)popDebugGroup
{
}

// Sub-encoders of a parallel pass share the one open render pass, and write no framing of their
// own: the pass writes the one RGFX_OP_ENCODER_END. Each writes a stream of its own (a segment
// of the command buffer's), so threads encode them at once, as Metal allows; the pass joins them
// at its end in the order they were made, which is the order Metal runs them in, each from
// Metal's default state (measured on the M3 Max, an AMD 5300M and an Intel 630: the second made,
// encoded first, draws last).
- (nullable id<MTLRenderCommandEncoder>)renderCommandEncoder
{
	RgfxRenderCommandEncoder *sub = [[RgfxRenderCommandEncoder alloc] initWithEncoder:_enc
	                                                                          emitEnd:NO
	                                                                           device:_device];
	[sub rgfxAdoptSegment:std::make_unique<RgfxEncoder>(RgfxEncoder::SegmentOf{*_enc})];
	[sub rgfxSetCommandBuffer:_commandBuffer];
	[sub rgfxSetUploads:_uploads];
	[sub rgfxSetTileWidth:_tileWidth height:_tileHeight];
	[sub rgfxSetBarrierSlot:_barrier begin:NO];
	os_unfair_lock_lock(&_subsLock);
	[_subs addObject:sub];
	os_unfair_lock_unlock(&_subsLock);
	return sub;
}

- (void)endEncoding
{
	if (!_ended) {
		_ended = YES;
		os_unfair_lock_lock(&_subsLock);
		NSArray<RgfxRenderCommandEncoder *> *subs = [_subs copy];
		[_subs removeAllObjects];
		os_unfair_lock_unlock(&_subsLock);
		// Each sub-encoder's ops in turn, the render state the one before set back to Metal's
		// defaults first.
		uint32_t reset = 0;
		RgfxRenderTargetSize target = {};
		bool sized = false;
		for (RgfxRenderCommandEncoder *sub in subs) {
			RgfxEncoder *segment = [sub rgfxSegment];
			const uint32_t set = segment->renderStateSet();
			if (!sized && (reset & (kRgfxRenderStateViewport | kRgfxRenderStateScissor))) {
				target = rgfxTargetSize(_pass);
				sized = true;
			}
			_enc->appendSegment(*segment, reset, target);
			reset = set;
		}
		if (_barrier.fences[_barrier.slot] != 0)
			_enc->renderUpdateFence(_barrier.fences[_barrier.slot], kRgfxBarrierRenderStages);
		_enc->endEncoder();
	}
	// Last: the buffer may go with it, and its RgfxEncoder with it.
	_commandBufferHold = nil;
}

- (void)setColorStoreAction:(MTLStoreAction)storeAction atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreAction(static_cast<uint64_t>(storeAction), colorAttachmentIndex);
}
- (void)setDepthStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetDepthStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setStencilStoreAction:(MTLStoreAction)storeAction
{
	_enc->renderSetStencilStoreAction(static_cast<uint64_t>(storeAction));
}
- (void)setColorStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
                           atIndex:(NSUInteger)colorAttachmentIndex
{
	_enc->renderSetColorStoreActionOptions(static_cast<uint64_t>(storeActionOptions),
	                                       colorAttachmentIndex);
}
- (void)setStencilStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	_enc->renderSetStencilStoreActionOptions(static_cast<uint64_t>(storeActionOptions));
}
- (void)setDepthStoreActionOptions:(MTLStoreActionOptions)storeActionOptions
{
	_enc->renderSetDepthStoreActionOptions(static_cast<uint64_t>(storeActionOptions));
}
// Waits for every encoder made before this pass on the queue (QUEUE BARRIERS).
- (void)barrierAfterQueueStages:(MTLStages)afterQueueStages beforeStages:(MTLStages)beforeStages
{
	if (_barrierWaited)
		return;
	_barrierWaited = YES;
	rgfxWaitForEarlierEncoders(_barrier, [&](RgfxObjId fence) {
		_enc->renderWaitForFence(fence, kRgfxBarrierRenderStages);
	});
}

// --- MTLParallelRenderCommandEncoderSPI ---------------------------------------------------
// The members of Metal's parallel-encoder SPI that the AMD 5300M's and the Intel 630's encoders
// both answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m).

// A sub-encoder that samples the GPU's counters into `programInfoBuffer`: Metal's base parallel
// encoder asks its command buffer's sampledRenderCommandEncoderWithDescriptor:programInfoBuffer:
// capacity: for it (read from code, _MTLParallelRenderCommandEncoder, macOS 27.0.1), which
// answers nil on the device, which samples no counters; so does this.
- (nullable id)sampledRenderCommandEncoderWithProgramInfoBuffer:(void *)programInfoBuffer
                                                        capacity:(NSUInteger)capacity
{
	(void)programInfoBuffer;
	(void)capacity;
	return nil;
}

// Whether each sub-encoder's work is committed on its own: NO, and a set does nothing, as Metal's
// IOAccel parallel encoder answers (read from code, MTLIOAccelParallelRenderCommandEncoder, macOS
// 27.0.1). The sub-encoders' work goes to the host in the command buffer's.
- (BOOL)separateCommits
{
	return NO;
}
- (void)setSeparateCommits:(BOOL)separateCommits
{
	(void)separateCommits;
}

// Metal's base parallel encoder raises an unrecognized selector for this (it sends itself
// doesNotRecognizeSelector:; read from code, _MTLParallelRenderCommandEncoder, macOS 27.0.1), and
// so does this.
- (void)insertSplit
{
	[self doesNotRecognizeSelector:_cmd];
}

// Ends the encoding and answers no program address table, as Metal's base parallel encoder does
// (read from code, _MTLParallelRenderCommandEncoder, macOS 27.0.1).
- (nullable id)endEncodingAndRetrieveProgramAddressTable
{
	[self endEncoding];
	return nil;
}

@end

#pragma clang attribute pop
