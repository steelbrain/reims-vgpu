/*
 * RgfxResource: see RgfxResource.h for the wrapper convention this base
 * establishes. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxResource.h"

#import <Metal/Metal.h>

#include <atomic>
#include <unistd.h>

#include "RgfxRetirement.hpp"
#include "RgfxSerializer.hpp"

@implementation RgfxResource {
	__weak id<MTLDevice> _device;
	id<MTLHeap> _heap; // the heap the resource lies in, kept alive
	// The process the resource is used for (setResponsibleProcess:): this one until told.
	int _responsibleProcess;
	// The purgeable state the client set (setPurgeableState:), NonVolatile until set.
	std::atomic<MTLPurgeableState> _purgeableState;
	// The resource's offset in its heap (rgfxSetHeapOffset:), and whether it was made
	// aliasable (makeAliasable).
	NSUInteger _heapOffset;
	std::atomic<bool> _aliasable;
}

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super init];
	if (self) {
		_objId = objId;
		_device = device;
		_serializer = serializer;
		_responsibleProcess = getpid();
		_purgeableState.store(MTLPurgeableStateNonVolatile);
	}
	return self;
}

- (uint64_t)rgfxObjId
{
	return _objId;
}

- (id<MTLDevice>)device
{
	return _device;
}

- (RgfxSerializer *)serializer
{
	return _serializer;
}

- (id<MTLHeap>)heap
{
	return _heap;
}

- (void)rgfxSetHeap:(id<MTLHeap>)heap
{
	_heap = heap;
}

- (void)rgfxRetireThen:(std::function<void()>)after
{
	RgfxSerializer *serializer = _serializer;
	if (serializer == nullptr) {
		if (after)
			after(); // no host object to wait for
		return;
	}
	uint64_t objId = _objId;
	RgfxRetirement::process().retire([serializer, objId, after = std::move(after)] {
		serializer->destroy(objId);
		if (after)
			after();
	});
}

// The resource SPI WindowServer calls as it prepares a window's surface for its compositor:
// the process the resource is used for. The device accounts memory to no process, so it is
// kept and answered.
- (void)setResponsibleProcess:(int)pid
{
	_responsibleProcess = pid;
}

- (int)responsibleProcess
{
	return _responsibleProcess;
}

- (void)rgfxSetHeapOffset:(NSUInteger)offset
{
	_heapOffset = offset;
}

// --- MTLResource's placement, aliasing and ownership --------------------------

- (NSUInteger)heapOffset
{
	return _heapOffset;
}

// Only a resource made in a heap can be made aliasable, as in Metal. The guest places every
// heap resource itself (a Placement heap), so aliasing changes nothing on the device: the flag
// is only answered.
- (void)makeAliasable
{
	if (_heap != nil) {
		_aliasable.store(true);
	}
}

- (BOOL)isAliasable
{
	return _aliasable.load();
}

// The task the resource's memory is accounted to. The device accounts memory to no task, so
// the change is taken and nothing else follows.
- (kern_return_t)setOwnerWithIdentity:(task_id_token_t)identity
{
	(void)identity;
	return KERN_SUCCESS;
}

// The state is kept as told and the previous one answered (KeepCurrent only answers it). The
// device never purges a resource, so a Volatile one keeps its contents and stays Volatile, and
// an Empty one keeps them too, which no client may read.
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
	if (state == MTLPurgeableStateKeepCurrent) {
		return _purgeableState.load();
	}
	return _purgeableState.exchange(state);
}

@end
