/*
 * RgfxResidencySet: the bundle's MTLResidencySet (RgfxResidencySet.h says what it answers).
 */

#import "RgfxResidencySet.h"

#include <os/lock.h>
#include <stdlib.h>

#import "RgfxResource.h"
#import "RgfxTexture.h"
#include "RgfxLog.hpp"

namespace {

// Ends the process at a nil allocation added or removed, as the M3 Max's set does (SIGSEGV;
// measured: macOS 27.0.1, tools/oracle/oracle_residency's add-nil, remove-nil, adds-nil and
// removes-nil).
[[noreturn]] void endsTheProcess(const char *call)
{
	rgfxLogLine(@"RGFX: -[RgfxResidencySet %s] of a nil allocation ends the process, as Metal's set does",
	            call);
	abort();
}

// The memory `allocation` lies in, which allocatedSize counts once however many of the set's
// allocations lie there: a view's texture, a texture's buffer, and the heap either lies in;
// nil for an allocation of no device of ours, which counts nothing, as the M3 Max's set counts
// an allocation of none (measured: tools/oracle/oracle_residency, add-foreign).
id memoryOf(id allocation)
{
	if (![allocation isKindOfClass:[RgfxResource class]])
		return nil;
	id memory = allocation;
	while ([memory isKindOfClass:[RgfxTexture class]]) {
		RgfxTexture *texture = memory;
		id under = texture.parentTexture;
		if (under == nil)
			under = texture.buffer;
		if (under == nil)
			break;
		memory = under;
	}
	id<MTLHeap> heap = ((RgfxResource *)memory).heap;
	return heap != nil ? heap : memory;
}

// Whether `allocation` is a texture over an IOSurface or a view of one (the root each texture
// records as it is made), which a command buffer using the set names at its commit (rgfxSurfaces).
bool overSurface(id allocation)
{
	return RgfxIsResource(allocation) && ((RgfxResource *)allocation)->_rgfxSurfaceRoot != nil;
}

} // namespace

@implementation RgfxResidencySet {
	__weak id<MTLDevice> _device;
	// Guards everything below.
	os_unfair_lock _lock;
	NSString *_label;
	// The committed allocations, in the order they were committed.
	NSMutableOrderedSet *_committed;
	// The adds since the last commit, none of them committed, in the order they were added.
	NSMutableOrderedSet *_added;
	// The committed allocations a remove since the last commit marks.
	NSMutableSet *_removed;
	// The memory the committed allocations lie in (memoryOf), each counted once an allocation.
	NSCountedSet *_memory;
	// The bytes of `_memory`'s objects: allocatedSize.
	uint64_t _allocatedSize;
	// The committed allocations over IOSurfaces (overSurface), in the order they were committed:
	// rgfxSurfaces.
	NSMutableOrderedSet *_surfaces;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device descriptor:(MTLResidencySetDescriptor *)descriptor
{
	if ((self = [super init])) {
		_device = device;
		_label = [descriptor.label copy];
		_lock = OS_UNFAIR_LOCK_INIT;
		_committed = [NSMutableOrderedSet orderedSet];
		_added = [NSMutableOrderedSet orderedSet];
		_removed = [NSMutableSet set];
		_memory = [NSCountedSet set];
		_surfaces = [NSMutableOrderedSet orderedSet];
	}
	return self;
}

- (id<MTLDevice>)device
{
	return _device;
}

- (NSString *)label
{
	os_unfair_lock_lock(&_lock);
	NSString *label = _label;
	os_unfair_lock_unlock(&_lock);
	return label;
}

- (void)setLabel:(NSString *)label
{
	NSString *copied = [label copy];
	os_unfair_lock_lock(&_lock);
	NS_VALID_UNTIL_END_OF_SCOPE NSString *was = _label; // let go after the lock
	_label = copied;
	os_unfair_lock_unlock(&_lock);
}

- (uint64_t)allocatedSize
{
	os_unfair_lock_lock(&_lock);
	const uint64_t size = _allocatedSize;
	os_unfair_lock_unlock(&_lock);
	return size;
}

// The device keeps every allocation resident, so neither a request nor its end has anything to
// do, and neither changes what the set answers, as on the M3 Max (measured).
- (void)requestResidency
{
}

- (void)endResidency
{
}

// Under the lock: `allocation` added, as the header gives it.
- (void)rgfxAdd:(id)allocation
{
	if ([_removed containsObject:allocation])
		[_removed removeObject:allocation];
	else if (![_committed containsObject:allocation])
		[_added addObject:allocation];
}

// Under the lock: `allocation` removed, as the header gives it.
- (void)rgfxRemove:(id)allocation
{
	if ([_added containsObject:allocation])
		[_added removeObject:allocation];
	else if ([_committed containsObject:allocation])
		[_removed addObject:allocation];
}

// An allocation's identity leaves through the set, so a no-copy buffer, or a texture over one,
// posts its escape note as it joins (RgfxEscapes), before the set's lock.
- (void)addAllocation:(id<MTLAllocation>)allocation
{
	if (allocation == nil)
		endsTheProcess("addAllocation:");
	if (RgfxIsResource(allocation))
		RgfxEscapes((RgfxResource *)allocation, RGFX_BUFFER_ESCAPED_RESIDENCY_SET);
	os_unfair_lock_lock(&_lock);
	[self rgfxAdd:allocation];
	os_unfair_lock_unlock(&_lock);
}

- (void)addAllocations:(const id<MTLAllocation> _Nonnull[_Nonnull])allocations count:(NSUInteger)count
{
	for (NSUInteger i = 0; i < count; i++) {
		if (allocations[i] == nil)
			endsTheProcess("addAllocations:count:");
	}
	for (NSUInteger i = 0; i < count; i++) {
		if (RgfxIsResource(allocations[i]))
			RgfxEscapes((RgfxResource *)allocations[i], RGFX_BUFFER_ESCAPED_RESIDENCY_SET);
	}
	os_unfair_lock_lock(&_lock);
	for (NSUInteger i = 0; i < count; i++)
		[self rgfxAdd:allocations[i]];
	os_unfair_lock_unlock(&_lock);
}

- (void)removeAllocation:(id<MTLAllocation>)allocation
{
	if (allocation == nil)
		endsTheProcess("removeAllocation:");
	// An add taken back lets the allocation go here, after the lock, where the set held it last.
	NS_VALID_UNTIL_END_OF_SCOPE id held = allocation;
	os_unfair_lock_lock(&_lock);
	[self rgfxRemove:held];
	os_unfair_lock_unlock(&_lock);
}

- (void)removeAllocations:(const id<MTLAllocation> _Nonnull[_Nonnull])allocations
                    count:(NSUInteger)count
{
	for (NSUInteger i = 0; i < count; i++) {
		if (allocations[i] == nil)
			endsTheProcess("removeAllocations:count:");
	}
	NS_VALID_UNTIL_END_OF_SCOPE NSArray *held = [[NSArray alloc] initWithObjects:allocations count:count];
	os_unfair_lock_lock(&_lock);
	for (id allocation in held)
		[self rgfxRemove:allocation];
	os_unfair_lock_unlock(&_lock);
}

- (void)removeAllAllocations
{
	os_unfair_lock_lock(&_lock);
	NS_VALID_UNTIL_END_OF_SCOPE NSOrderedSet *takenBack = [_added copy]; // let go after the lock
	[_added removeAllObjects];
	for (id allocation in _committed)
		[_removed addObject:allocation];
	os_unfair_lock_unlock(&_lock);
}

- (BOOL)containsAllocation:(id<MTLAllocation>)allocation
{
	if (allocation == nil)
		return NO;
	os_unfair_lock_lock(&_lock);
	const BOOL contains = [_added containsObject:allocation] ||
	                      ([_committed containsObject:allocation] && ![_removed containsObject:allocation]);
	os_unfair_lock_unlock(&_lock);
	return contains;
}

- (NSArray<id<MTLAllocation>> *)allAllocations
{
	os_unfair_lock_lock(&_lock);
	NSMutableArray *all = [[NSMutableArray alloc] initWithCapacity:_committed.count + _added.count];
	for (id allocation in _committed) {
		if (![_removed containsObject:allocation])
			[all addObject:allocation];
	}
	[all addObjectsFromArray:_added.array];
	os_unfair_lock_unlock(&_lock);
	return all.count != 0 ? [all copy] : @[];
}

- (NSUInteger)allocationCount
{
	os_unfair_lock_lock(&_lock);
	const NSUInteger count = _committed.count + _added.count;
	os_unfair_lock_unlock(&_lock);
	return count;
}

- (void)commit
{
	os_unfair_lock_lock(&_lock);
	// The allocations the commit drops, let go after the lock, and as it returns: an autoreleased
	// list (NSSet's allObjects) would keep them until the caller's pool drains, where the M3 Max's
	// set lets them go at its commit (oracle_residency's hold case).
	NS_VALID_UNTIL_END_OF_SCOPE NSMutableArray *dropped =
	    [[NSMutableArray alloc] initWithCapacity:_removed.count];
	for (id allocation in _removed)
		[dropped addObject:allocation];
	for (id allocation in dropped) {
		[_committed removeObject:allocation];
		if (overSurface(allocation))
			[_surfaces removeObject:allocation];
		id memory = memoryOf(allocation);
		if (memory == nil)
			continue;
		[_memory removeObject:memory];
		if ([_memory countForObject:memory] == 0)
			_allocatedSize -= [(id<MTLAllocation>)memory allocatedSize];
	}
	[_removed removeAllObjects];
	for (id allocation in _added) {
		[_committed addObject:allocation];
		if (overSurface(allocation))
			[_surfaces addObject:allocation];
		id memory = memoryOf(allocation);
		if (memory == nil)
			continue;
		if ([_memory countForObject:memory] == 0)
			_allocatedSize += [(id<MTLAllocation>)memory allocatedSize];
		[_memory addObject:memory];
	}
	[_added removeAllObjects];
	os_unfair_lock_unlock(&_lock);
}

- (NSArray *)rgfxSurfaces
{
	os_unfair_lock_lock(&_lock);
	NSArray *surfaces = _surfaces.count != 0 ? [_surfaces.array copy] : nil;
	os_unfair_lock_unlock(&_lock);
	return surfaces;
}

@end
