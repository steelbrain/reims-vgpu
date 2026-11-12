/*
 * RgfxFunctionTable: see RgfxFunctionTable.h. Our own ObjC++.
 */

#import "RgfxFunctionTable.h"

#include <mutex>
#include <vector>

#import "RgfxFunction.h"
#include "RgfxLogOnce.hpp"
#include "RgfxSerializer.hpp"

namespace {

void noteNoAnswer(const char *what, uint32_t status)
{
	static char site;
	if (rgfxLogFirst(&site, rgfxLogKindOf(what) ^ status))
		NSLog(@"RGFX: %s: the host made none (status 0x%x)", what, status);
}

} // namespace

@implementation RgfxFunctionHandle {
	NSString *_name;
	MTLFunctionType _functionType;
}

// The host drops the handle as the app lets it go; a table it was set in keeps it while the
// table holds it.
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)handleOfFunction:(id<MTLFunction>)function
                        pipeline:(uint64_t)pipeline
                          device:(id<MTLDevice>)device
                      serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || ![(id)function isKindOfClass:[RgfxFunction class]])
		return nil;
	RgfxFunction *rf = (RgfxFunction *)function;
	RgfxFunctionPointerState state = {};
	RgfxCreated created = serializer->createFunctionHandle(
	    pipeline, RgfxLinkedFunctionRefOf(rf, RGFX_LINKED_FUNCTIONS, nullptr), &state);
	// Not found is Metal's nil, a function the pipeline does not link, and no fault.
	if (created.status != RGFX_STATUS_OK) {
		if (created.status != RGFX_STATUS_NOT_FOUND)
			noteNoAnswer("functionHandleWithFunction:", created.status);
		return nil;
	}
	RgfxFunctionHandle *handle = [[RgfxFunctionHandle alloc] initWithObjId:created.obj_id
	                                                                 device:device
	                                                             serializer:serializer];
	if (handle) {
		handle->_name = [rf.name copy];
		handle->_functionType = rf.functionType;
	}
	return handle;
}

- (NSString *)name
{
	return _name;
}

- (MTLFunctionType)functionType
{
	return _functionType;
}

// The AMD Radeon Pro 5300M's handle raises an unrecognized selector here (measured: macOS
// 26.7.1, oracle_fnptr), and so does this: a handle reaches a shader through a table.
- (MTLResourceID)gpuResourceID
{
	[self doesNotRecognizeSelector:_cmd];
	return MTLResourceID{};
}

@end

@implementation RgfxVisibleFunctionTable {
	NSUInteger _count;
	uint64_t _resourceId;
	uint64_t _allocatedSize;
	MTLResourceOptions _options;
	// The handles set in it, by entry, under _lock: the table keeps each while it holds it.
	std::mutex _lock;
	std::vector<id<MTLFunctionHandle>> _entries;
}

// The host drops the table as the app lets it go, after any command buffer that names it.
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)tableOfPipeline:(uint64_t)pipeline
                          count:(NSUInteger)count
                         device:(id<MTLDevice>)device
                     serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr)
		return nil;
	RgfxFunctionPointerState state = {};
	RgfxCreated created = serializer->createFunctionTable(pipeline, count, &state);
	if (created.status != RGFX_STATUS_OK) {
		noteNoAnswer("newVisibleFunctionTableWithDescriptor:", created.status);
		return nil;
	}
	RgfxVisibleFunctionTable *table = [[RgfxVisibleFunctionTable alloc] initWithObjId:created.obj_id
	                                                                           device:device
	                                                                       serializer:serializer];
	if (table) {
		table->_count = count;
		table->_resourceId = state.gpu_resource_id;
		table->_allocatedSize = state.allocated_size;
		table->_options = static_cast<MTLResourceOptions>(state.resource_options);
		table->_entries.resize(count);
	}
	return table;
}

- (NSUInteger)rgfxFunctionCount
{
	return _count;
}

// --- MTLVisibleFunctionTable -----------------------------------------------------------

// The id the host's device numbered the table by, which an argument buffer holds.
- (MTLResourceID)gpuResourceID
{
	MTLResourceID rid;
	rid._impl = _resourceId;
	return rid;
}

- (void)setFunction:(id<MTLFunctionHandle>)function atIndex:(NSUInteger)index
{
	__unsafe_unretained id<MTLFunctionHandle> one[1] = {function};
	[self setFunctions:one withRange:NSMakeRange(index, 1)];
}

// The entries of `range` set to the handles, a nil handle a nil entry. A range past the table
// sets none, logged: the host's table takes none past its entries. A handle another device made
// names nothing on the host, and sets its entry nil.
- (void)setFunctions:(const id<MTLFunctionHandle> __nullable[__nonnull])functions
           withRange:(NSRange)range
{
	if (range.length == 0)
		return;
	if (range.location > _count || range.length > _count - range.location) {
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: -[MTLVisibleFunctionTable setFunctions:withRange:] range {%lu, %lu} "
			      @"past the table's %lu entries: none set",
			      (unsigned long)range.location, (unsigned long)range.length,
			      (unsigned long)_count);
		return;
	}
	std::vector<RgfxObjId> ids(range.length, 0);
	std::lock_guard<std::mutex> hold(_lock);
	for (NSUInteger i = 0; i < range.length; i++) {
		id<MTLFunctionHandle> h = functions[i];
		if (h != nil && ![(id)h isKindOfClass:[RgfxFunctionHandle class]]) {
			static char site;
			if (rgfxLogFirst(&site))
				NSLog(@"RGFX: -[MTLVisibleFunctionTable setFunctions:withRange:] a handle of "
				      @"another device (%@): its entry set nil",
				      NSStringFromClass([(id)h class]));
			h = nil;
		}
		ids[i] = h != nil ? ((RgfxFunctionHandle *)h).rgfxObjId : 0;
		_entries[range.location + i] = h;
	}
	if (_serializer != nullptr &&
	    !_serializer->setFunctionTable(_objId, range.location, ids.data(), ids.size())) {
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: -[MTLVisibleFunctionTable setFunctions:withRange:] did not reach the "
			      @"host");
	}
}

// --- MTLResource -----------------------------------------------------------------------

// The host table's options and size, as its create answered them.
- (MTLResourceOptions)resourceOptions
{
	return _options;
}

- (MTLStorageMode)storageMode
{
	return static_cast<MTLStorageMode>((_options & MTLResourceStorageModeMask) >>
	                                   MTLResourceStorageModeShift);
}

- (MTLCPUCacheMode)cpuCacheMode
{
	return static_cast<MTLCPUCacheMode>((_options & MTLResourceCPUCacheModeMask) >>
	                                    MTLResourceCPUCacheModeShift);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
	return static_cast<MTLHazardTrackingMode>((_options & MTLResourceHazardTrackingModeMask) >>
	                                          MTLResourceHazardTrackingModeShift);
}

- (NSUInteger)allocatedSize
{
	return static_cast<NSUInteger>(_allocatedSize);
}

// The resource SPI a buffer and an indirect command buffer answer (RgfxResource.h).
RGFX_RESOURCE_SPI

@end
