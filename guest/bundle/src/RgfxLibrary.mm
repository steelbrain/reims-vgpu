/*
 * RgfxLibrary: see RgfxLibrary.h. Our own ObjC++; no Apple private SPI except the
 * read-only accessors of MTLFunctionConstantValues' indexed constants, called only
 * when the object answers them.
 */

#import "RgfxLibrary.h"

#import <objc/message.h>

#import "RgfxFunction.h"
#import "RgfxMetalError.h"

#include <cstring>
#include <mutex>
#include <unistd.h>
#include <string>
#include <vector>

#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {

// One entry of a .metallib's function list: the entry point and its stage.
struct MetallibFunction {
	std::string name;
	MTLFunctionType type;
};

uint16_t rd16(const uint8_t *p)
{
	uint16_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

uint32_t rd32(const uint8_t *p)
{
	uint32_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

uint64_t rd64(const uint8_t *p)
{
	uint64_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

// A .metallib's TYPE tag byte, as MTLFunctionType. measured: macOS 27 Metal, on
// CoreDisplay's default.metallib (0 vertex, 1 fragment, 2 kernel).
MTLFunctionType metallibStage(uint8_t t)
{
	switch (t) {
	case 0:
		return MTLFunctionTypeVertex;
	case 1:
		return MTLFunctionTypeFragment;
	case 2:
		return MTLFunctionTypeKernel;
	default:
		return MTLFunctionTypeVisible;
	}
}

// Reads a .metallib's function list: the header's function-list offset (at 0x18),
// a u32 count, then per function a u32 group size (counting itself) and its tags,
// each a 4-byte name, a u16 size and the data, up to `ENDT`. Answers false, leaving
// `out` empty, when the bytes are not a .metallib or the list runs past them.
bool readMetallibFunctions(const uint8_t *p, size_t n, std::vector<MetallibFunction> &out)
{
	out.clear();
	if (p == nullptr || n < 0x28 || std::memcmp(p, "MTLB", 4) != 0)
		return false;
	uint64_t q = rd64(p + 0x18);
	if (q > n || n - q < 4)
		return false;
	uint32_t count = rd32(p + q);
	q += 4;
	for (uint32_t i = 0; i < count; i++) {
		if (n - q < 4)
			return false;
		uint32_t group = rd32(p + q);
		if (group < 4 || group > n - q)
			return false;
		const uint64_t end = q + group;
		uint64_t t = q + 4;
		MetallibFunction f{ std::string(), MTLFunctionTypeVisible };
		while (end - t >= 4) {
			if (std::memcmp(p + t, "ENDT", 4) == 0)
				break;
			if (end - t < 6)
				return false;
			uint16_t size = rd16(p + t + 4);
			const uint8_t *data = p + t + 6;
			if (size > end - t - 6)
				return false;
			if (std::memcmp(p + t, "NAME", 4) == 0)
				f.name.assign(reinterpret_cast<const char *>(data), strnlen(reinterpret_cast<const char *>(data), size));
			else if (std::memcmp(p + t, "TYPE", 4) == 0 && size >= 1)
				f.type = metallibStage(data[0]);
			t += 6 + static_cast<uint64_t>(size);
		}
		if (!f.name.empty())
			out.push_back(std::move(f));
		q = end;
	}
	return true;
}

// The byte size of a constant of `type` that fits the wire's inline value (at most
// eight bytes); 0 for a type the wire cannot carry inline.
uint32_t inlineConstantSize(MTLDataType type)
{
	switch (type) {
	case MTLDataTypeBool:
	case MTLDataTypeChar:
	case MTLDataTypeUChar:
		return 1;
	case MTLDataTypeBool2:
	case MTLDataTypeChar2:
	case MTLDataTypeUChar2:
	case MTLDataTypeShort:
	case MTLDataTypeUShort:
	case MTLDataTypeHalf:
		return 2;
	case MTLDataTypeBool3:
	case MTLDataTypeChar3:
	case MTLDataTypeUChar3:
		return 3;
	case MTLDataTypeBool4:
	case MTLDataTypeChar4:
	case MTLDataTypeUChar4:
	case MTLDataTypeShort2:
	case MTLDataTypeUShort2:
	case MTLDataTypeHalf2:
	case MTLDataTypeInt:
	case MTLDataTypeUInt:
	case MTLDataTypeFloat:
		return 4;
	case MTLDataTypeShort3:
	case MTLDataTypeUShort3:
	case MTLDataTypeHalf3:
		return 6;
	case MTLDataTypeShort4:
	case MTLDataTypeUShort4:
	case MTLDataTypeHalf4:
	case MTLDataTypeInt2:
	case MTLDataTypeUInt2:
	case MTLDataTypeFloat2:
	case MTLDataTypeLong:
	case MTLDataTypeULong:
		return 8;
	default:
		return 0;
	}
}

// The index-keyed constants set on `values`, read through the object's own
// `newIndexedConstantArray` (each element answering `index`, `dataType` and `data`).
// Constants set by name, and values wider than the wire's inline eight bytes, are
// not carried. measured: macOS 27 Metal
// (MTLFunctionConstantValuesInternal, MTLIndexedConstantValue).
std::vector<RgfxFunctionConstant> indexedConstants(MTLFunctionConstantValues *values)
{
	std::vector<RgfxFunctionConstant> out;
	if (values == nil)
		return out;
	SEL indexedSel = sel_registerName("newIndexedConstantArray");
	if (![values respondsToSelector:indexedSel])
		return out;
	NSArray *indexed = ((NSArray * (*)(id, SEL)) objc_msgSend)(values, indexedSel);
	SEL indexSel = sel_registerName("index");
	SEL typeSel = sel_registerName("dataType");
	SEL dataSel = sel_registerName("data");
	for (id c in indexed) {
		if (![c respondsToSelector:indexSel] || ![c respondsToSelector:typeSel] ||
		    ![c respondsToSelector:dataSel])
			continue;
		NSUInteger index = ((NSUInteger(*)(id, SEL))objc_msgSend)(c, indexSel);
		MTLDataType type = static_cast<MTLDataType>(((NSUInteger(*)(id, SEL))objc_msgSend)(c, typeSel));
		const void *data = ((const void *(*)(id, SEL))objc_msgSend)(c, dataSel);
		uint32_t size = inlineConstantSize(type);
		if (size == 0 || data == nullptr)
			continue;
		RgfxFunctionConstant k = {};
		k.index = static_cast<uint32_t>(index);
		k.data_type = static_cast<uint32_t>(type);
		std::memcpy(&k.value, data, size);
		out.push_back(k);
	}
	return out;
}

// The process's staging window for streamed libraries: two blocks of the device
// region used in turn, each with an ack slot the host writes once it has copied the
// block's chunk out. Made on the first large library and kept for the process's life
// (one device a process). The lock serialises streams, which share the window.
struct LibraryStaging {
	std::mutex lock;
	bool made = false;
	uint64_t chunk = 0;            // each block's size
	uint64_t blockOffset[2] = {};  // region offsets of the two blocks
	volatile uint8_t *block[2] = {};
	uint64_t ackOffset[2] = {};    // region offsets of the two ack slots
	volatile uint64_t *ack[2] = {};
	uint64_t expected[2] = {};     // the ack each block's last chunk asked for (0: none)
	uint64_t nextToken = 1;
};

LibraryStaging &libraryStaging()
{
	static LibraryStaging staging;
	return staging;
}

// Each staging block's size: 32 MiB, so QuartzCore's 160 MB library goes in five
// chunks per block; a smaller window when the region has no room for two of those.
constexpr uint64_t kStagingChunkSizes[] = { 32ull << 20, 8ull << 20, 1ull << 20 };

bool makeStaging(LibraryStaging &st, RgfxRegion &region)
{
	for (uint64_t chunk : kStagingChunkSizes) {
		// One block: a page of ack slots (one a cache line), then the two chunk blocks.
		RgfxRegion::Block b = region.allocate(0x1000 + 2 * chunk);
		if (!b.ok)
			continue;
		st.chunk = chunk;
		for (int i = 0; i < 2; i++) {
			st.ackOffset[i] = b.offset + 64 * static_cast<uint64_t>(i);
			st.ack[i] = reinterpret_cast<volatile uint64_t *>(b.ptr + 64 * i);
			__atomic_store_n(st.ack[i], 0, __ATOMIC_RELEASE);
			st.blockOffset[i] = b.offset + 0x1000 + chunk * static_cast<uint64_t>(i);
			st.block[i] = b.ptr + 0x1000 + chunk * i;
			st.expected[i] = 0;
		}
		st.made = true;
		return true;
	}
	return false;
}

// Waits, about ten seconds at most, for the host to have copied block `i`'s last chunk.
bool awaitBlock(LibraryStaging &st, int i)
{
	for (int t = 0; t < 50000; t++) {
		if (__atomic_load_n(st.ack[i], __ATOMIC_ACQUIRE) == st.expected[i])
			return true;
		usleep(200);
	}
	return false;
}

// Streams `length` bytes to the host as one library: each chunk copied into the next
// staging block once the host has copied out what it last held, then posted.
// The last chunk's answer is `*outStatus`.
bool streamLibrary(RgfxRegion &region, RgfxSerializer &ser, const uint8_t *bytes, uint64_t length,
                   const uint8_t *key, RgfxObjId *outObjId, uint32_t *outStatus)
{
	LibraryStaging &st = libraryStaging();
	std::lock_guard<std::mutex> guard(st.lock);
	if (!st.made && !makeStaging(st, region))
		return false;
	const RgfxObjId objId = ser.allocLibraryId();
	int i = 0;
	for (uint64_t at = 0; at < length; at += st.chunk, i ^= 1) {
		const uint64_t n = length - at < st.chunk ? length - at : st.chunk;
		if (!awaitBlock(st, i))
			return false;
		std::memcpy(const_cast<uint8_t *>(st.block[i]), bytes + at, n);
		const uint64_t token = st.nextToken++;
		st.expected[i] = token;
		bool posted = false;
		for (int r = 0; r < 2000 && !posted; r++) {
			posted = ser.postLibraryChunk(objId, st.blockOffset[i], n, length, at, st.ackOffset[i],
			                              token, key, outStatus);
			if (!posted)
				usleep(1000);
		}
		if (!posted)
			return false;
	}
	*outObjId = objId;
	return true;
}

} // namespace

@implementation RgfxLibrary {
	std::vector<MetallibFunction> _functions; // the .metallib's function list
	BOOL _haveFunctionTable;                  // the bytes were a readable .metallib
}

// The host drops the library's id once every command buffer in flight now has completed. Its
// functions keep it alive (RgfxFunction), and a pipeline made from them needs it no more: the
// host made it from the library before the destroy.
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

// Whether a create's answer is the device's library: it made it (RGFX_STATUS_OK), or the
// route hands no answer back (a ring the process writes itself).
static bool rgfxMade(uint32_t status)
{
	return status == RGFX_STATUS_OK || status == kRgfxNoAnswer;
}

// The library `objId` over `bytes`, its function table read from them.
+ (instancetype)rgfxLibraryWithObjId:(RgfxObjId)objId
                               bytes:(const void *)bytes
                              length:(NSUInteger)length
                              device:(id<MTLDevice>)device
                          serializer:(RgfxSerializer *)serializer
{
	RgfxLibrary *lib = [[RgfxLibrary alloc] initWithObjId:objId device:device serializer:serializer];
	if (lib != nil)
		lib->_haveFunctionTable =
		    readMetallibFunctions(static_cast<const uint8_t *>(bytes), length, lib->_functions);
	return lib;
}

+ (instancetype)libraryWithData:(const void *)bytes
                         length:(NSUInteger)length
                            key:(const uint8_t *)key
                         device:(id<MTLDevice>)device
                     serializer:(RgfxSerializer *)serializer
                         status:(uint32_t *)status
{
	*status = kRgfxNoAnswer;
	if (serializer == nullptr) {
		return nil; // a library must have a route to the host
	}
	RgfxCreated created = serializer->createLibrary(bytes, length, key);
	if (!created.posted) {
		return nil; // the bytes did not fit the control ring
	}
	*status = created.status;
	if (!rgfxMade(created.status))
		return nil;
	return [self rgfxLibraryWithObjId:created.obj_id
	                            bytes:bytes
	                           length:length
	                           device:device
	                       serializer:serializer];
}

+ (instancetype)libraryFoundWithKey:(const uint8_t *)key
                              bytes:(const void *)bytes
                             length:(NSUInteger)length
                             device:(id<MTLDevice>)device
                         serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr)
		return nil;
	RgfxCreated found = serializer->lookupLibrary(key, length);
	if (!found.posted || found.status != RGFX_STATUS_OK)
		return nil;
	return [self rgfxLibraryWithObjId:found.obj_id
	                            bytes:bytes
	                           length:length
	                           device:device
	                       serializer:serializer];
}

+ (instancetype)libraryStreamingBytes:(const void *)bytes
                              length:(NSUInteger)length
                                 key:(const uint8_t *)key
                              region:(RgfxRegion *)region
                              device:(id<MTLDevice>)device
                          serializer:(RgfxSerializer *)serializer
                              status:(uint32_t *)status
{
	*status = kRgfxNoAnswer;
	if (serializer == nullptr || region == nullptr || bytes == nullptr || length == 0) {
		return nil;
	}
	RgfxObjId objId = 0;
	if (!streamLibrary(*region, *serializer, static_cast<const uint8_t *>(bytes), length, key,
	                   &objId, status) ||
	    !rgfxMade(*status)) {
		return nil;
	}
	return [self rgfxLibraryWithObjId:objId
	                            bytes:bytes
	                           length:length
	                           device:device
	                       serializer:serializer];
}

// The table entry for `name`, or null when the table does not hold it.
- (const MetallibFunction *)rgfxEntryNamed:(NSString *)name
{
	const char *n = name.UTF8String;
	if (n == nullptr)
		return nullptr;
	for (const MetallibFunction &f : _functions) {
		if (f.name == n)
			return &f;
	}
	return nullptr;
}

// Vends `name` specialised by `constants`: nil when the library's function table
// does not hold it (as Metal answers a missing name).
- (nullable id<MTLFunction>)rgfxFunctionNamed:(NSString *)name
                                    constants:(const std::vector<RgfxFunctionConstant> &)constants
{
	if (name == nil) {
		return nil;
	}
	const MetallibFunction *entry = nullptr;
	if (_haveFunctionTable) {
		entry = [self rgfxEntryNamed:name];
		if (entry == nullptr)
			return nil;
	}
	RgfxFunction *fn = [RgfxFunction functionWithLibrary:self
	                                                name:name
	                                           constants:constants.empty() ? nullptr : constants.data()
	                                               count:static_cast<uint32_t>(constants.size())
	                                              device:self.device
	                                          serializer:[self serializer]];
	if (entry != nullptr)
		[fn rgfxSetFunctionType:entry->type];
	return fn;
}

// --- MTLLibrary --------------------------------------------------------------

- (id<MTLFunction>)newFunctionWithName:(NSString *)functionName
{
	return [self rgfxFunctionNamed:functionName constants:{}];
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                                 error:(__autoreleasing NSError **)error
{
	// Index-keyed constants are carried on the pipeline create's wire. A constant set
	// by name needs the function's constant table to find its index, which the shell
	// does not read; such constants are not carried (the host then compiles the
	// function unspecialised for them).
	std::vector<RgfxFunctionConstant> constants = indexedConstants(constantValues);
	id<MTLFunction> fn = [self rgfxFunctionNamed:name constants:constants];
	if (error)
		*error = fn ? nil
		            : RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
		                            [NSString stringWithFormat:@"no function named %@ in the library", name]);
	return fn;
}

- (void)newFunctionWithName:(NSString *)name
             constantValues:(MTLFunctionConstantValues *)constantValues
          completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	NSError *err = nil;
	id<MTLFunction> fn = [self newFunctionWithName:name constantValues:constantValues error:&err];
	if (completionHandler)
		completionHandler(fn, err);
}

- (id<MTLFunction>)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                                       error:(__autoreleasing NSError **)error
{
	// `specializedName` and `options` change only the host's compile, not which entry
	// point is named; the name and constants are what the wire carries.
	return [self newFunctionWithName:descriptor.name constantValues:descriptor.constantValues error:error];
}

- (void)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	NSError *err = nil;
	id<MTLFunction> fn = [self newFunctionWithDescriptor:descriptor error:&err];
	if (completionHandler)
		completionHandler(fn, err);
}

- (NSArray<NSString *> *)functionNames
{
	// The .metallib's function list; empty for raw AIR, whose functions callers
	// name directly by string.
	NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:_functions.size()];
	for (const MetallibFunction &f : _functions)
		[names addObject:[NSString stringWithUTF8String:f.name.c_str()]];
	return names;
}

- (MTLLibraryType)type
{
	return MTLLibraryTypeExecutable;
}

- (NSString *)installName
{
	return nil; // an executable library has no install name
}

@end
