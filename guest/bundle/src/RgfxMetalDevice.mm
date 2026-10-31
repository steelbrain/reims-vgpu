/*
 * RgfxMetalDevice: see RgfxMetalDevice.h for the class's role and the category
 * split. This file is slice 0: init, identity, and the limit/feature getters,
 * answered from RgfxCaps. The resource/pipeline/command-buffer/sync selectors live
 * in the category files the header names.
 *
 * Our own ObjC++; the only Apple-private names are the forward-declarations in
 * RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice.h"

#import "RgfxBuffer.h"
#import "RgfxTexture.h"

// RGFX DIAG (Wall-2 trace; remove before merge): the guest's logd is broken, so
// write diagnostics to a file any process (platform or not) can read.
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#include <IOKit/IOKitLib.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {
// RGFX DIAG (remove before merge): append one line to /tmp/rgfx-diag.log.
void rgfxDiag(const char *fmt, ...)
{
	char line[512];
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(line, sizeof(line), fmt, ap);
	va_end(ap);
	if (n <= 0)
		return;
	if (static_cast<size_t>(n) > sizeof(line))
		n = sizeof(line);
	int fd = open("/tmp/rgfx-diag.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd < 0)
		return;
	(void)write(fd, line, static_cast<size_t>(n));
	close(fd);
}
} // namespace

#include "RgfxCompletionThread.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"

// The accelerator's user-client ABI, mirroring guest/kext/src/RgfxAccelerator.hpp (that
// header is kernel-only and not included here, so the values are duplicated and kept in sync
// by hand): the only user-client type, and the selectors.
enum {
	kRgfxAccelUserClientType = 0x52474658, // 'RGFX', clear of the IOAccel surface type 0
	kRgfxAccelMethodDoorbell = 0,
	kRgfxAccelMethodOpenContext = 0x100,
	kRgfxAccelMethodAlloc = 0x101,
	kRgfxAccelMethodFree = 0x102,
	kRgfxAccelMethodQueueCreate = 0x103,
	kRgfxAccelMethodQueueDestroy = 0x104,
	kRgfxAccelMethodSubmit = 0x105,
	kRgfxAccelMethodResourceIds = 0x106,
	kRgfxAccelMethodWaitCompletions = 0x107,
	kRgfxAccelMethodSurfaceScanout = 0x108,
	kRgfxAccelMethodSurfaceBlock = 0x109,
	kRgfxAccelMethodSurfaceBlockRelease = 0x10a,
	kRgfxAccelMethodDeviceInfo = 0x10b,
};

namespace {
// The largest object-lifecycle message the kext takes (ReimsGraphics::kLifecycleMessageMax):
// half the control ring.
constexpr uint32_t kLifecycleMessageMax = RGFX_CONTROL_RING_DATA_SIZE / 2;

// What the lifecycle's forward reaches: the user client, and the reader its replies go to.
struct RgfxKextLink {
	io_connect_t conn;
	RgfxReplyReader *reader;
};

// The serializer's lifecycle route: the kext posts the message on the control ring, stamping
// its seqno, and hands back the answer the device owes it: a pipeline create's reflection, which
// goes to the reply reader, or a library create's or lookup's COMPLETE, whose status is
// `*out_status`.
bool rgfxKextSubmit(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
                    uint32_t *out_status)
{
	RgfxKextLink *link = static_cast<RgfxKextLink *>(context);
	uint64_t seqno = 0;
	uint32_t outCount = 1;
	uint8_t reply[128];
	size_t replySize = sizeof(reply);
	kern_return_t kr = IOConnectCallMethod(link->conn, kRgfxAccelMethodSubmit, nullptr, 0, message,
	                                       size, &seqno, &outCount, reply, &replySize);
	if (kr != KERN_SUCCESS)
		return false;
	RgfxMsgHeader header = {};
	if (replySize >= sizeof(header))
		std::memcpy(&header, reply, sizeof(header));
	if (header.type == RGFX_EVT_COMPLETE && replySize >= sizeof(RgfxComplete)) {
		RgfxComplete complete = {};
		std::memcpy(&complete, reply, sizeof(complete));
		if (out_status)
			*out_status = complete.status;
	} else if (replySize >= sizeof(RgfxMsgHeader) && link->reader) {
		link->reader->deliver(reply, static_cast<uint32_t>(replySize));
	}
	*out_seqno = seqno;
	return true;
}

// The kext's blocks, as RgfxRegion's block source; `context` is the user client's connect.
bool rgfxKextAlloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in[2] = {size, align};
	uint32_t outCount = 1;
	return IOConnectCallScalarMethod(conn, kRgfxAccelMethodAlloc, in, 2, offset, &outCount) ==
	       KERN_SUCCESS;
}

// The kext's chunks of texture resource ids, as RgfxIdSpace's chunk source; `context` is the
// user client's connect. False once every chunk of the guest's pool is held.
bool rgfxKextResourceIds(void *context, uint64_t *first_id)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint32_t outCount = 1;
	return IOConnectCallScalarMethod(conn, kRgfxAccelMethodResourceIds, nullptr, 0, first_id,
	                                 &outCount) == KERN_SUCCESS;
}

// The completion thread's wait: the kext returns once a completion ring of the process's holds
// records its last pass did not see (the wake count differs from `last`), or at once where it
// already does. `context` is the user client's connect.
RgfxCompletionThread::WaitResult rgfxKextWaitCompletions(void *context, uint64_t last,
                                                         uint64_t *now)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint32_t outCount = 1;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodWaitCompletions, &last, 1, now, &outCount);
	if (kr == KERN_SUCCESS)
		return RgfxCompletionThread::kWoke;
	if (kr == kIOReturnAborted)
		return RgfxCompletionThread::kRetry; // a signal broke the sleep
	rgfxDiag("RGFX: the completion wait ended kr=0x%x; waiters take the rings\n", kr);
	return RgfxCompletionThread::kStop;
}

void rgfxKextRelease(void *context, uint64_t offset)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodFree, &offset, 1, nullptr, nullptr);
}

// The kext's shared blocks of surface planes, as RgfxRegion's surface source: the block every
// process that holds the plane's surface gets, found by the pages it maps at `plane`.
bool rgfxKextSurfaceBlock(void *context, const void *plane, uint64_t length, uint64_t size,
                          uint64_t *offset, uint64_t *serial, bool *fresh)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in[3] = {reinterpret_cast<uintptr_t>(plane), length, size};
	uint64_t out[4] = {};
	uint32_t outCount = 4;
	if (IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlock, in, 3, out, &outCount) !=
	        KERN_SUCCESS ||
	    outCount != 4) {
		return false;
	}
	if (out[1] != size) {
		IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlockRelease, &out[0], 1, nullptr,
		                          nullptr);
		return false;
	}
	*offset = out[0];
	*serial = out[2];
	*fresh = out[3] != 0;
	return true;
}

void rgfxKextSurfaceRelease(void *context, uint64_t offset)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodSurfaceBlockRelease, &offset, 1, nullptr,
	                          nullptr);
}

// The serializer's queue closer: the kext stops watching the queue's completion ring and posts
// its destroy on the control ring, behind the process's messages before it.
void rgfxKextQueueCloser(void *context, RgfxObjId objId)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in = objId;
	IOConnectCallScalarMethod(conn, kRgfxAccelMethodQueueDestroy, &in, 1, nullptr, nullptr);
}

// The serializer's queue opener: the kext posts the queue's create on the control ring and
// returns once the device has its rings.
bool rgfxKextQueueOpener(void *context, const RgfxQueueCreate &q)
{
	io_connect_t conn = static_cast<io_connect_t>(reinterpret_cast<uintptr_t>(context));
	uint64_t in[6] = {q.obj_id, q.submit_offset, q.complete_offset, q.submit_data_size,
	                  q.complete_data_size, q.priority};
	uint64_t status = ~0ull;
	uint32_t outCount = 1;
	kern_return_t kr =
	    IOConnectCallScalarMethod(conn, kRgfxAccelMethodQueueCreate, in, 6, &status, &outCount);
	return kr == KERN_SUCCESS && status == RGFX_STATUS_OK;
}
} // namespace

// The device's doorbell trampoline the queue submission rings ring: it forwards the
// ring id to the device, which (once BAR 0's registers are mapped) writes it to the
// DOORBELL register. `context` is the device, valid for each ring writer's life (the
// device owns them).
@interface RgfxMetalDevice ()
- (void)rgfxRingDoorbell:(uint32_t)ringId;
@end

namespace {
void rgfxDeviceDoorbell(void *context, uint32_t ring_id)
{
	[(__bridge RgfxMetalDevice *)context rgfxRingDoorbell:ring_id];
}
} // namespace

// The device's `architecture`: Metal's class, whose name is the host's answer in the caps blob.
@interface RgfxArchitecture : MTLArchitecture
- (instancetype)initWithArchitectureName:(NSString *)name;
@end

@implementation RgfxArchitecture {
	NSString *_rgfxName;
}

- (instancetype)initWithArchitectureName:(NSString *)name
{
	self = [super init];
	if (self)
		_rgfxName = [name copy];
	return self;
}

- (NSString *)name
{
	return _rgfxName;
}

- (id)copyWithZone:(NSZone *)zone
{
	(void)zone;
	return self; // immutable
}
@end

@implementation RgfxMetalDevice {
	RgfxCaps _caps;              // host device caps; invalid until the region is read
	RgfxIdSpace _ids;            // this device's id/address/resource-id lines
	RgfxSerializer *_serializer; // the create route; null until the ring transport lands
	mach_port_t _acceleratorPort;
	RgfxRegion *_region;         // BAR-region allocator; null until the handshake maps it
	RgfxReplyReader *_replyReader; // the lifecycle's replies, from the kext; null until the handshake
	std::unique_ptr<RgfxCompletionThread> _completions; // the queues' completions; null until then
	RgfxKextLink _link;          // the lifecycle's route to the kext
	std::vector<std::unique_ptr<RgfxRingWriter>> _queueRings; // owned queue submit-ring writers
	std::mutex _queueRingsLock;  // queues are made and end on any thread
	io_connect_t _conn;          // our accelerator user client (BAR-2 region map + doorbell)
	std::unique_ptr<RgfxRingWriter> _controlRing; // the lifecycle, forwarded to the kext
	mach_vm_address_t _mapBase;  // the window's base in this process (0 until the context opens)
	mach_vm_size_t _mapSize;     // the window's size, the region's
}

// RGFX DIAG (Wall-2 trace; remove before merge): fires when the bundle is loaded
// and this class is realized in the ObjC runtime. If this logs but no init below
// does, Metal loaded the bundle but did not instantiate us by initWithAcceleratorPort
// (wrong entry point or a pre-instantiation gate); if this never logs, the class is
// not being realized at all.
+ (void)load
{
	rgfxDiag("RGFX: RgfxMetalDevice +load (class realized in runtime)\n");
}

// RGFX DIAG (remove before merge): catch a plain -init in case Metal instantiates us
// that way rather than via initWithAcceleratorPort:.
- (instancetype)init
{
	rgfxDiag("RGFX: -init called (not initWithAcceleratorPort:)\n");
	NSLog(@"RGFX: -init called");
	return [super init];
}

- (instancetype)initWithAcceleratorPort:(mach_port_t)port
{
	// RGFX DIAG (Wall-2 trace; remove before merge): confirm Metal enters our init
	// and report whether the IOGPUMetalDevice base accepts or declines. To a file
	// (the guest's logd is broken) plus the probe's stderr (NSLog).
	rgfxDiag("RGFX: initWithAcceleratorPort entered, port=0x%x\n", (unsigned)port);
	NSLog(@"RGFX: initWithAcceleratorPort entered, port=0x%x", (unsigned)port);
	self = [super initWithAcceleratorPort:port];
	rgfxDiag("RGFX: [super initWithAcceleratorPort] -> %s\n", self ? "non-nil" : "nil");
	NSLog(@"RGFX: [super initWithAcceleratorPort] -> %@", self ? @"non-nil" : @"nil");
	if (self == nil) {
		return nil; // Apple's base declined (or the port is not ours)
	}
	_acceleratorPort = port;
	_conn = IO_OBJECT_NULL;
	_serializer = nullptr;
	_region = nullptr;
	_replyReader = nullptr;
	_mapBase = 0;
	_mapSize = 0;

	// Wire transport handshake: open our own user client on the accelerator, read the caps blob
	// the host wrote, open the process's context, whose window is where its part of the device
	// region (BAR 2) lies in this task, and build the region allocator over the window and the
	// device submission ring (RgfxSerializer) + device completion ring (RgfxReplyReader). With
	// these non-null the resource / command-queue / command-buffer categories route over our
	// wire instead of falling back to the IOGPUMetalDevice base (Apple's IOGPU kernel ABI). Any
	// failure leaves the transport stubbed (the getters answer from defaults) rather than
	// crashing.
	kern_return_t kr = IOServiceOpen(port, mach_task_self(), kRgfxAccelUserClientType, &_conn);
	if (kr != KERN_SUCCESS) {
		rgfxDiag("RGFX: IOServiceOpen(type=%u) failed kr=0x%x; transport stubbed\n",
		         kRgfxAccelUserClientType, kr);
		return self;
	}

	// The caps blob the host wrote (read once at init), which the kext reads for us: its page
	// is no process's to map.
	RgfxDeviceInfo info = {};
	size_t infoSize = sizeof(info);
	kr = IOConnectCallStructMethod(_conn, kRgfxAccelMethodDeviceInfo, nullptr, 0, &info,
	                               &infoSize);
	if (kr != KERN_SUCCESS || infoSize != sizeof(info) || !_caps.load(info))
		rgfxDiag("RGFX: caps blob not read (kr=0x%x) or invalid; using defaults\n", kr);

	// The process's context: its id and address ranges, its window, its blocks and its chunks
	// of texture resource ids, from the kext, so no two processes' rings, blocks or numbers
	// meet, and this one maps nothing of the region but its own.
	uint64_t ctx[5] = {};
	uint32_t ctxCount = 5;
	kr = IOConnectCallScalarMethod(_conn, kRgfxAccelMethodOpenContext, nullptr, 0, ctx, &ctxCount);
	if (kr != KERN_SUCCESS || ctxCount != 5 || ctx[3] == 0 || ctx[4] < RGFX_ALLOCATIONS_OFFSET) {
		rgfxDiag("RGFX: OPEN_CONTEXT failed kr=0x%x (window 0x%llx, %llu bytes); transport "
		         "stubbed\n",
		         kr, (unsigned long long)ctx[3], (unsigned long long)ctx[4]);
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return self;
	}
	const mach_vm_address_t addr = ctx[3];
	const mach_vm_size_t mapSize = ctx[4];
	_mapBase = addr;
	_mapSize = mapSize;
	volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(addr);
	_ids.setBases(ctx[1], ctx[2]);
	void *connContext = reinterpret_cast<void *>(static_cast<uintptr_t>(_conn));
	_ids.setResourceIdSource(&rgfxKextResourceIds, connContext);
	_region = new RgfxRegion(base, mapSize);
	_region->setBlockSource(&rgfxKextAlloc, &rgfxKextRelease, connContext);
	_region->setSurfaceSource(&rgfxKextSurfaceBlock, &rgfxKextSurfaceRelease, connContext);

	// The object lifecycle goes to the kext, which posts it on the control ring and hands back
	// the replies (a pipeline's reflection).
	_replyReader = new RgfxReplyReader();
	_link.conn = _conn;
	_link.reader = _replyReader;
	_controlRing = std::make_unique<RgfxRingWriter>(&rgfxKextSubmit, &_link, kLifecycleMessageMax);
	_serializer = new RgfxSerializer(*_controlRing, _ids);
	_serializer->setQueueOpener(&rgfxKextQueueOpener, connContext);
	_serializer->setQueueCloser(&rgfxKextQueueCloser, connContext);
	// The queues' completions, taken as the kext wakes the thread for them.
	_completions = std::make_unique<RgfxCompletionThread>(&rgfxKextWaitCompletions, connContext);
	if (!_completions->start()) {
		rgfxDiag("RGFX: no completion thread; waiters take the rings\n");
		_completions.reset();
	}
	rgfxDiag("RGFX: context %llu: ids 0x%llx, va 0x%llx\n", (unsigned long long)ctx[0],
	         (unsigned long long)ctx[1], (unsigned long long)ctx[2]);
	rgfxDiag("RGFX: transport up: region=%llu bytes, caps=%s\n", (unsigned long long)mapSize,
	         _caps.valid() ? "valid" : "default");
	return self;
}

// RGFX DIAG (remove before merge): -[_MTLDevice initLimits] and other Metal bring-up code send
// MTLDeviceSPI selectors straight to us without -respondsToSelector:; any that neither our shell
// nor the IOGPUMetalDevice base implements hit -doesNotRecognizeSelector: and throw an
// unhandled ObjC exception (crash). Every unimplemented selector is forwarded here and answered
// with a zero scalar, so bring-up keeps going.
- (NSMethodSignature *)methodSignatureForSelector:(SEL)aSelector
{
	NSMethodSignature *sig = [super methodSignatureForSelector:aSelector];
	if (sig)
		return sig;
	return [NSMethodSignature signatureWithObjCTypes:"Q@:"]; // assume an 8-byte scalar return
}

- (void)forwardInvocation:(NSInvocation *)anInvocation
{
	unsigned long long zero = 0;
	[anInvocation setReturnValue:&zero];
}

// --- accessors the category slices route through -----------------------------

- (RgfxSerializer *)serializer
{
	return _serializer;
}

- (RgfxIdSpace *)ids
{
	return &_ids;
}

- (const RgfxCaps *)caps
{
	return &_caps;
}

- (RgfxRegion *)region
{
	return _region;
}

// A texture over a legacy IOAccel surface, which is how WindowServer's CoreDisplay reaches the
// surface a display is drawn into. The kext answers where the surface's memory lies, the
// framebuffer's scanout in the region, and lets this process name it; the texture lies over a
// Shared buffer over that span, so what the GPU draws there is what the display shows. Nil for
// a surface that is not the scanout, as the base answers for every surface.
- (id<MTLTexture>)newTextureWithSurface:(NSUInteger)surfaceId buffer:(NSUInteger)buffer
{
	if (_conn == IO_OBJECT_NULL || _region == nullptr || [self serializer] == nullptr) {
		return nil;
	}
	uint64_t in[2] = {surfaceId, buffer};
	uint64_t out[5] = {};
	uint32_t outCount = 5;
	kern_return_t kr =
	    IOConnectCallScalarMethod(_conn, kRgfxAccelMethodSurfaceScanout, in, 2, out, &outCount);
	static std::atomic<int> logged{0};
	if (logged.fetch_add(1) < 8) {
		NSLog(@"RGFX: newTextureWithSurface:0x%lx buffer:%lu -> 0x%x: offset 0x%llx, %llu bytes, "
		      @"%llux%llu, %llu bytes a row",
		      static_cast<unsigned long>(surfaceId), static_cast<unsigned long>(buffer), kr,
		      out[0], out[1], out[2], out[3], out[4]);
	}
	if (kr != KERN_SUCCESS || outCount != 5) {
		return nil;
	}
	uint64_t offset = out[0], length = out[1], width = out[2], height = out[3];
	uint64_t rowBytes = out[4];
	if (offset > _region->size() || length > _region->size() - offset ||
	    rowBytes * height > length) {
		return nil;
	}
	RgfxBuffer *span =
	    [RgfxBuffer bufferOverHostMemoryWithContents:const_cast<uint8_t *>(_region->base() + offset)
	                                    regionOffset:offset
	                                          length:static_cast<NSUInteger>(length)
	                                         options:MTLResourceStorageModeShared
	                                     deallocator:nil
	                                          device:self
	                                      serializer:[self serializer]];
	if (span == nil) {
		return nil;
	}
	// The scanout's pixels: BGRA8.
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:static_cast<NSUInteger>(width)
	                                                      height:static_cast<NSUInteger>(height)
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModeShared;
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
	return [RgfxTexture textureWithDescriptor:d
	                                   buffer:span
	                                   offset:0
	                              bytesPerRow:rowBytes
	                                   device:self
	                               serializer:[self serializer]];
}

- (RgfxReplyReader *)replyReader
{
	return _replyReader;
}

- (RgfxCompletionThread *)completionThread
{
	return _completions.get();
}

- (RgfxRingWriter *)addQueueSubmitRingAtOffset:(uint64_t)regionOffset dataSize:(uint32_t)dataSize
{
	if (_region == nullptr) {
		return nullptr;
	}
	// The host keys a per-queue ring's doorbell by its submission ring's region offset
	// (device.rs `ring_doorbell` -> `by_ring`, keyed on `submit_offset`), the control
	// ring's id being its offset 0. So the doorbell id is the region offset itself, not a
	// serial counter, or the host finds no ring for the wake and the queue never runs.
	uint32_t ringId = static_cast<uint32_t>(regionOffset);
	auto writer = std::make_unique<RgfxRingWriter>(_region->base() + regionOffset, dataSize, ringId,
	                                               &rgfxDeviceDoorbell, (__bridge void *)self);
	RgfxRingWriter *raw = writer.get();
	std::lock_guard<std::mutex> hold(_queueRingsLock);
	_queueRings.push_back(std::move(writer));
	return raw;
}

- (void)dropQueueSubmitRing:(RgfxRingWriter *)ring
{
	std::unique_ptr<RgfxRingWriter> dropped;
	{
		std::lock_guard<std::mutex> hold(_queueRingsLock);
		auto it = std::find_if(_queueRings.begin(), _queueRings.end(),
		                       [ring](const std::unique_ptr<RgfxRingWriter> &w) { return w.get() == ring; });
		if (it == _queueRings.end())
			return;
		dropped = std::move(*it);
		_queueRings.erase(it);
	}
}

- (void)rgfxRingDoorbell:(uint32_t)ringId
{
	// Ring the ring's doorbell through our accelerator user client: the kext writes `ringId`
	// to RGFX_REG_DOORBELL in BAR 0, waking the host's consumer of that ring.
	if (_conn == IO_OBJECT_NULL)
		return;
	uint64_t scalar = ringId;
	IOConnectCallScalarMethod(_conn, kRgfxAccelMethodDoorbell, &scalar, 1, nullptr, nullptr);
}

// --- identity ----------------------------------------------------------------

- (NSString *)name
{
	if (_caps.valid() && _caps.name()[0] != '\0') {
		return [NSString stringWithUTF8String:_caps.name()];
	}
	return @"reims-graphics";
}

- (uint64_t)registryID
{
	// The guest's own accelerator's registry entry ID: WindowServer finds the Metal device
	// that drives a display by matching this against the accelerator the display's
	// framebuffer names, so the host GPU's ID would match no display. The host's ID is
	// the answer only where the accelerator has none.
	uint64_t entryId = 0;
	if (_acceleratorPort != MACH_PORT_NULL &&
	    IORegistryEntryGetRegistryEntryID(_acceleratorPort, &entryId) == KERN_SUCCESS && entryId != 0)
		return entryId;
	return _caps.valid() ? _caps.registryId() : 0;
}

- (MTLSize)maxThreadsPerThreadgroup
{
	if (_caps.valid()) {
		return MTLSizeMake(_caps.maxThreadsPerThreadgroup(0), _caps.maxThreadsPerThreadgroup(1),
		                   _caps.maxThreadsPerThreadgroup(2));
	}
	return MTLSizeMake(1024, 1024, 1024);
}

// Every identity and capability getter below answers the caps blob's value, which the host
// writes from its platform's answers (and the lane's, where its device can do less); 0 or NO
// where the blob is invalid or the host has no answer.

- (MTLArchitecture *)architecture
{
	if (_caps.valid() && _caps.architectureName()[0] != '\0') {
		NSString *name = [NSString stringWithUTF8String:_caps.architectureName()];
		return [[RgfxArchitecture alloc] initWithArchitectureName:name];
	}
	// No answer from the host: the base's own, where it has one, else an empty name.
	Class base = class_getSuperclass([RgfxMetalDevice class]);
	if (class_getInstanceMethod(base, @selector(architecture)) != nullptr) {
		struct objc_super sup = {self, base};
		return ((MTLArchitecture * (*)(struct objc_super *, SEL)) objc_msgSendSuper)(
		    &sup, @selector(architecture));
	}
	return [[RgfxArchitecture alloc] initWithArchitectureName:@""];
}

- (BOOL)hasUnifiedMemory
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityHasUnifiedMemory);
}

- (BOOL)isLowPower
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityIsLowPower);
}

- (BOOL)isHeadless
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityIsHeadless);
}

- (BOOL)isRemovable
{
	return _caps.valid() && _caps.identityFlag(kRgfxIdentityIsRemovable);
}

- (MTLDeviceLocation)location
{
	return static_cast<MTLDeviceLocation>(_caps.size(kRgfxSizeLocation));
}

- (NSUInteger)locationNumber
{
	return _caps.size(kRgfxSizeLocationNumber);
}

- (uint64_t)maxTransferRate
{
	return _caps.size(kRgfxSizeMaxTransferRate);
}

- (uint64_t)peerGroupID
{
	return _caps.size(kRgfxSizePeerGroupId);
}

- (uint32_t)peerIndex
{
	return static_cast<uint32_t>(_caps.size(kRgfxSizePeerIndex));
}

- (uint32_t)peerCount
{
	return static_cast<uint32_t>(_caps.size(kRgfxSizePeerCount));
}

- (NSUInteger)currentAllocatedSize
{
	return 0; // no allocation accounting yet
}

- (uint64_t)recommendedMaxWorkingSetSize
{
	return _caps.size(kRgfxSizeRecommendedMaxWorkingSetSize);
}

- (NSUInteger)maximumConcurrentCompilationTaskCount
{
	return _caps.size(kRgfxSizeMaximumConcurrentCompilationTaskCount);
}

// --- limits (RgfxCaps) -------------------------------------------------------

- (NSUInteger)maxBufferLength
{
	return _caps.size(kRgfxSizeMaxBufferLength);
}

- (NSUInteger)maxThreadgroupMemoryLength
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxThreadgroupMemoryLength) : 32768;
}

- (NSUInteger)maxArgumentBufferSamplerCount
{
	return _caps.size(kRgfxSizeMaxArgumentBufferSamplerCount);
}

- (MTLArgumentBuffersTier)argumentBuffersSupport
{
	return static_cast<MTLArgumentBuffersTier>(_caps.size(kRgfxSizeArgumentBuffersSupport));
}

- (MTLReadWriteTextureTier)readWriteTextureSupport
{
	return static_cast<MTLReadWriteTextureTier>(_caps.size(kRgfxSizeReadWriteTextureSupport));
}

- (NSUInteger)sparseTileSizeInBytes
{
	return _caps.size(kRgfxSizeSparseTileSizeInBytes);
}

- (NSUInteger)minimumTextureBufferAlignmentForPixelFormat:(MTLPixelFormat)format
{
	return _caps.textureBufferAlignment(static_cast<uint64_t>(format));
}

- (MTLSize)sparseTileSizeWithTextureType:(MTLTextureType)textureType
                             pixelFormat:(MTLPixelFormat)pixelFormat
                             sampleCount:(NSUInteger)sampleCount
                          sparsePageSize:(MTLSparsePageSize)sparsePageSize
{
	// The host answers one tile for every texture type, as Metal does.
	(void)textureType;
	uint64_t tile[3];
	_caps.sparseTile(static_cast<uint64_t>(pixelFormat), sampleCount,
	                 static_cast<int64_t>(sparsePageSize), tile);
	return MTLSizeMake(tile[0], tile[1], tile[2]);
}

- (MTLSize)sparseTileSizeWithTextureType:(MTLTextureType)textureType
                             pixelFormat:(MTLPixelFormat)pixelFormat
                             sampleCount:(NSUInteger)sampleCount
{
	// Pages of 16 KiB, as Metal's query without a page size.
	return [self sparseTileSizeWithTextureType:textureType
	                               pixelFormat:pixelFormat
	                               sampleCount:sampleCount
	                            sparsePageSize:MTLSparsePageSize16];
}

// --- feature flags (RgfxCaps) ------------------------------------------------

- (BOOL)areProgrammableSamplePositionsSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagSupportsProgrammableSamplePositions);
}

- (BOOL)areRasterOrderGroupsSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagAreRasterOrderGroupsSupported);
}

- (BOOL)isDepth24Stencil8PixelFormatSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagIsDepth24Stencil8PixelFormatSupported);
}

- (BOOL)supports32BitFloatFiltering
{
	return _caps.identityFlag(kRgfxIdentitySupports32BitFloatFiltering);
}

- (BOOL)supports32BitMSAA
{
	return _caps.identityFlag(kRgfxIdentitySupports32BitMSAA);
}

- (BOOL)supportsQueryTextureLOD
{
	return _caps.identityFlag(kRgfxIdentitySupportsQueryTextureLOD);
}

- (BOOL)supportsBCTextureCompression
{
	return _caps.identityFlag(kRgfxIdentitySupportsBCTextureCompression);
}

- (BOOL)supportsPullModelInterpolation
{
	return _caps.identityFlag(kRgfxIdentitySupportsPullModelInterpolation);
}

- (BOOL)areBarycentricCoordsSupported
{
	return _caps.identityFlag(kRgfxIdentityAreBarycentricCoordsSupported);
}

- (BOOL)supportsShaderBarycentricCoordinates
{
	return _caps.identityFlag(kRgfxIdentitySupportsShaderBarycentricCoordinates);
}

- (BOOL)supportsDynamicLibraries
{
	return _caps.identityFlag(kRgfxIdentitySupportsDynamicLibraries);
}

- (BOOL)supportsRenderDynamicLibraries
{
	return _caps.identityFlag(kRgfxIdentitySupportsRenderDynamicLibraries);
}

- (BOOL)supportsRaytracing
{
	return _caps.identityFlag(kRgfxIdentitySupportsRaytracing);
}

- (BOOL)supportsFunctionPointers
{
	return _caps.identityFlag(kRgfxIdentitySupportsFunctionPointers);
}

- (BOOL)supportsFunctionPointersFromRender
{
	return _caps.identityFlag(kRgfxIdentitySupportsFunctionPointersFromRender);
}

- (BOOL)supportsRaytracingFromRender
{
	return _caps.identityFlag(kRgfxIdentitySupportsRaytracingFromRender);
}

- (BOOL)supportsPrimitiveMotionBlur
{
	return _caps.identityFlag(kRgfxIdentitySupportsPrimitiveMotionBlur);
}

- (BOOL)supportsPlacementSparse
{
	return _caps.identityFlag(kRgfxIdentitySupportsPlacementSparse);
}

- (BOOL)supportsFamily:(MTLGPUFamily)gpuFamily
{
	// By raw value: the blob lists every family of the host's SDK and -1009, the device-info
	// bit the host reports among them; any other value names no family, and is no.
	return _caps.supportsFamily(static_cast<int64_t>(gpuFamily));
}

- (BOOL)supportsFeatureSet:(MTLFeatureSet)featureSet
{
	return _caps.supportsFeatureSet(static_cast<uint64_t>(featureSet));
}

- (BOOL)supportsCounterSampling:(MTLCounterSamplingPoint)samplingPoint
{
	return _caps.supportsCounterSampling(static_cast<uint64_t>(samplingPoint));
}

- (BOOL)supportsVertexAmplificationCount:(NSUInteger)count
{
	// Metal takes every count from 0 up to its largest.
	uint64_t largest = _caps.valid() ? _caps.size(kRgfxSizeLargestVertexAmplificationCount) : 1;
	return count <= largest;
}

- (BOOL)supportsRasterizationRateMapWithLayerCount:(NSUInteger)layerCount
{
	return layerCount >= 1 && layerCount <= _caps.size(kRgfxSizeLargestRasterizationRateMapLayerCount);
}

- (BOOL)supportsTextureSampleCount:(NSUInteger)sampleCount
{
	return _caps.valid() && _caps.supportsSampleCount(static_cast<uint32_t>(sampleCount));
}

// --- MTLDeviceSPI (overridden; forward-declared in RgfxMetalPrivate.h) --------

- (NSUInteger)maxVertexSamplers
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxVertexSamplers) : 16;
}

- (NSUInteger)maxFragmentTextures
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxFragmentTextures) : 128;
}

- (NSUInteger)maxComputeTextures
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxComputeTextures) : 128;
}

- (NSUInteger)maxVertexAmplificationCount
{
	return _caps.valid() ? _caps.maxVertexAmplificationCount() : 1;
}

- (BOOL)supportsPrimitiveType:(NSUInteger)primitiveType
{
	return _caps.valid() && _caps.supportsPrimitiveType(static_cast<uint32_t>(primitiveType));
}

@end
