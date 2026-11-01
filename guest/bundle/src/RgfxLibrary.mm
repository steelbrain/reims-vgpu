/*
 * RgfxLibrary: see RgfxLibrary.h. Our own ObjC++; no Apple private SPI except the
 * read-only accessors of MTLFunctionConstantValues' indexed and named constants,
 * called only when the object answers them, and the extern-function members of
 * Metal's own library (RgfxMetalPrivate.h), called on the shadow.
 */

#import "RgfxLibrary.h"
#import "RgfxUnsupported.h"

#import <objc/message.h>

#import "RgfxFunction.h"
#import "RgfxInterop.h"
#import "RgfxMetalError.h"
#import "RgfxMetalPrivate.h"

#include <cstring>
#include <map>
#include <mutex>
#include <unistd.h>
#include <string>
#include <vector>

#include "RgfxFunctionConstants.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {

// One entry of a .metallib's function list: the entry point, its stage, its HASH tag (the
// SHA-256 of its bitcode) when the entry has one, and the constants its public metadata declares
// (`haveConstants` where that table was read).
struct MetallibFunction {
	std::string name;
	MTLFunctionType type;
	bool hasHash = false;
	uint8_t hash[32] = {};
	uint64_t publicOffset = 0; // into the public metadata section (OFFT's first field)
	bool haveOffsets = false;
	std::vector<RgfxConstantDecl> constants;
	bool haveConstants = false;
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
// CoreDisplay's default.metallib (0 vertex, 1 fragment, 2 kernel), and on a Core Image
// kernel library on macOS 27 and 26.6.1 (5, its kernels: functionType 4, which
// `functionNames` leaves out).
MTLFunctionType metallibStage(uint8_t t)
{
	switch (t) {
	case 0:
		return MTLFunctionTypeVertex;
	case 1:
		return MTLFunctionTypeFragment;
	case 2:
		return MTLFunctionTypeKernel;
	case 5:
		return kRgfxFunctionTypeExtern;
	default:
		return MTLFunctionTypeVisible;
	}
}

// Reads a .metallib's function list: the header's function-list offset (at 0x18),
// a u32 count, then per function a u32 group size (counting itself) and its tags,
// each a 4-byte name, a u16 size and the data, up to `ENDT`; then each function's
// constants, from the public metadata its OFFT tag places. Answers false, leaving
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
		MetallibFunction f;
		f.type = MTLFunctionTypeVisible;
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
			else if (std::memcmp(p + t, "HASH", 4) == 0 && size == sizeof(f.hash)) {
				std::memcpy(f.hash, data, sizeof(f.hash));
				f.hasHash = true;
			} else if (std::memcmp(p + t, "OFFT", 4) == 0 && size >= 8) {
				f.publicOffset = rd64(data);
				f.haveOffsets = true;
			}
			t += 6 + static_cast<uint64_t>(size);
		}
		if (!f.name.empty()) {
			if (f.haveOffsets)
				f.haveConstants = rgfxReadConstantTable(p, n, f.publicOffset, f.constants);
			out.push_back(std::move(f));
		}
		q = end;
	}
	return true;
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

// An `NSArray` a `new`-family SPI hands back, retained.
using NewArrayFn = NSArray *__attribute__((ns_returns_retained)) (*)(id, SEL);

// The values set on `values`, by index (by index or by range, one per index, the last set) and by
// name, read through the object's own `newIndexedConstantArray` and `newNamedConstantArray` (each
// element answering `index` or `name`, `dataType` and `data`), each value's bytes as many as its
// type packs. measured: macOS 27 Metal (MTLFunctionConstantValuesInternal, MTLIndexedConstantValue,
// MTLNamedConstantValue; a range is one indexed value per index). nil sets none.
void readConstantValues(MTLFunctionConstantValues *values, std::map<uint32_t, RgfxConstantValue> &byIndex,
                    std::map<std::string, RgfxConstantValue> &byName)
{
	byIndex.clear();
	byName.clear();
	if (values == nil)
		return;
	SEL typeSel = sel_registerName("dataType");
	SEL dataSel = sel_registerName("data");
	SEL indexSel = sel_registerName("index");
	auto value = [&](id c, RgfxConstantValue *out) {
		if (![c respondsToSelector:typeSel] || ![c respondsToSelector:dataSel])
			return false;
		out->type = static_cast<uint32_t>(((NSUInteger(*)(id, SEL))objc_msgSend)(c, typeSel));
		const void *data = ((const void *(*)(id, SEL))objc_msgSend)(c, dataSel);
		const uint32_t size = rgfxConstantSize(out->type);
		if (data != nullptr && size != 0) {
			const uint8_t *b = static_cast<const uint8_t *>(data);
			out->bytes.assign(b, b + size);
		}
		return true;
	};
	SEL indexedSel = sel_registerName("newIndexedConstantArray");
	if ([values respondsToSelector:indexedSel]) {
		NSArray *indexed = ((NewArrayFn)objc_msgSend)(values, indexedSel);
		for (id c in indexed) {
			RgfxConstantValue v;
			if ([c respondsToSelector:indexSel] && value(c, &v)) {
				NSUInteger index = ((NSUInteger(*)(id, SEL))objc_msgSend)(c, indexSel);
				byIndex[static_cast<uint32_t>(index)] = std::move(v);
			}
		}
	}
	SEL namedSel = sel_registerName("newNamedConstantArray");
	if ([values respondsToSelector:namedSel]) {
		NSArray *named = ((NewArrayFn)objc_msgSend)(values, namedSel);
		for (id c in named) {
			RgfxConstantValue v;
			if ([c respondsToSelector:@selector(name)] && value(c, &v)) {
				NSString *name = [c name];
				if (name.UTF8String != nullptr)
					byName[name.UTF8String] = std::move(v);
			}
		}
	}
}

} // namespace

// One constant a function declares, as `functionConstantsDictionary` answers it.
@interface RgfxDeclaredConstant : MTLFunctionConstant
- (instancetype)initWithDecl:(const RgfxConstantDecl &)decl;
@end

@implementation RgfxDeclaredConstant {
	NSString *_name;
	MTLDataType _type;
	NSUInteger _index;
	BOOL _required;
}

RGFX_LOG_UNRECOGNIZED

- (instancetype)initWithDecl:(const RgfxConstantDecl &)decl
{
	self = [super init];
	if (self) {
		_name = [NSString stringWithUTF8String:decl.name.c_str()] ?: @"";
		_type = static_cast<MTLDataType>(decl.type);
		_index = decl.index;
		_required = decl.required ? YES : NO;
	}
	return self;
}

- (NSString *)name
{
	return _name;
}

- (MTLDataType)type
{
	return _type;
}

- (NSUInteger)index
{
	return _index;
}

- (BOOL)required
{
	return _required;
}

@end

@implementation RgfxLibrary {
	std::vector<MetallibFunction> _functions; // the .metallib's function list
	BOOL _haveFunctionTable;                  // the bytes were a readable .metallib
	std::mutex _shadowLock;                   // guards the three below
	dispatch_data_t _source;                  // the bytes, kept for the shadow
	RgfxShadowMaker _shadowMaker;
	id<MTLLibrary> _shadow;                   // Metal's own over _source, made on first use
	BOOL _shadowTried;
}

// The host drops the library's id as the app lets it go. Its functions keep it alive (RgfxFunction), and a pipeline made from them needs it no more: the
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
	if (entry != nullptr) {
		[fn rgfxSetFunctionType:entry->type];
		if (entry->hasHash)
			[fn rgfxSetBitCodeHash:entry->hash];
	}
	return fn;
}

// The NSError Metal answers for a name the library does not hold.
static NSError *rgfxFunctionNotFound(NSString *name)
{
	NSString *what = [NSString stringWithFormat:@"Function %@ was not found in the library", name];
	return [NSError errorWithDomain:MTLLibraryErrorDomain
	                           code:MTLLibraryErrorFunctionNotFound
	                       userInfo:@{NSLocalizedDescriptionKey : what}];
}

// --- MTLLibrary --------------------------------------------------------------

- (id<MTLFunction>)newFunctionWithName:(NSString *)functionName
{
	RgfxFunction *fn = (RgfxFunction *)[self rgfxFunctionNamed:functionName constants:{}];
	const MetallibFunction *entry = fn ? [self rgfxEntryNamed:functionName] : nullptr;
	if (entry != nullptr && entry->haveConstants && !entry->constants.empty()) {
		NSMutableDictionary<NSString *, MTLFunctionConstant *> *declared =
		    [NSMutableDictionary dictionaryWithCapacity:entry->constants.size()];
		for (const RgfxConstantDecl &d : entry->constants) {
			RgfxDeclaredConstant *c = [[RgfxDeclaredConstant alloc] initWithDecl:d];
			declared[c.name] = c;
		}
		[fn rgfxSetUnspecialisedConstants:declared];
	}
	return fn;
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                                 error:(__autoreleasing NSError **)error
{
	if (error)
		*error = nil;
	const MetallibFunction *entry = _haveFunctionTable ? [self rgfxEntryNamed:name] : nullptr;
	if (name == nil || (_haveFunctionTable && entry == nullptr)) {
		if (error)
			*error = rgfxFunctionNotFound(name);
		return nil;
	}
	std::map<uint32_t, RgfxConstantValue> byIndex;
	std::map<std::string, RgfxConstantValue> byName;
	readConstantValues(constantValues, byIndex, byName);
	std::vector<RgfxFunctionConstant> constants;
	if (entry != nullptr && entry->haveConstants) {
		// The values resolved against the function's own constants, as Metal resolves them.
		RgfxConstantMismatch mismatch;
		if (!rgfxResolveConstants(entry->constants, byIndex, byName, constants, &mismatch)) {
			const RgfxConstantDecl &d = entry->constants[mismatch.decl];
			NSString *what = [NSString
			    stringWithFormat:@"Function %@: constant %s (%u) is of type %s but value found has type %s",
			                     name, d.name.c_str(), d.index, rgfxDataTypeName(d.type).c_str(),
			                     rgfxDataTypeName(mismatch.valueType).c_str()];
			if (error)
				*error = [NSError errorWithDomain:MTLLibraryErrorDomain
				                             code:MTLLibraryErrorCompileFailure
				                         userInfo:@{NSLocalizedDescriptionKey : what}];
			return nil;
		}
	} else {
		// No table to resolve against (bytes that are no .metallib, or an entry whose
		// metadata did not read): the values set by index, as set; the host checks them
		// against the function. Values set by name need the table, and are not carried.
		for (const auto &[index, v] : byIndex) {
			if (v.bytes.empty())
				continue;
			RgfxFunctionConstant k = {};
			k.index = index;
			k.data_type = v.type;
			std::memcpy(k.value, v.bytes.data(),
			            v.bytes.size() < sizeof(k.value) ? v.bytes.size() : sizeof(k.value));
			constants.push_back(k);
		}
	}
	return [self rgfxFunctionNamed:name constants:constants];
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
	// The entry point and constants are what the wire names the function by; the options
	// and the specialized name travel with it to the pipelines that link it, and the
	// specialized name is the one it answers.
	id<MTLFunction> fn = [self newFunctionWithName:descriptor.name
	                                constantValues:descriptor.constantValues
	                                         error:error];
	[(RgfxFunction *)fn rgfxSetOptions:descriptor.options];
	if (descriptor.specializedName.length != 0)
		[(RgfxFunction *)fn rgfxSetSpecializedName:descriptor.specializedName];
	return fn;
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
	// The .metallib's function list but its extern functions, as Metal lists it; empty for raw
	// AIR, whose functions callers name directly by string.
	NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:_functions.size()];
	for (const MetallibFunction &f : _functions) {
		if (f.type != kRgfxFunctionTypeExtern)
			[names addObject:[NSString stringWithUTF8String:f.name.c_str()]];
	}
	return names;
}

// --- Core Image's extern functions, through the shadow -------------------------

- (void)rgfxSetSource:(dispatch_data_t)source shadowMaker:(RgfxShadowMaker)maker
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	_source = source;
	_shadowMaker = maker;
}

- (void)rgfxAdoptShadow:(id<MTLLibrary>)shadow
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	_shadow = shadow;
	_shadowTried = YES;
}

- (id<MTLLibrary>)rgfxShadow
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	if (!_shadowTried && _source != nil && _shadowMaker != nil) {
		_shadowTried = YES;
		_shadow = _shadowMaker(_source);
	}
	return _shadow;
}

- (NSArray<NSString *> *)externFunctionNames
{
	id<MTLLibrary> shadow = [self rgfxShadow];
	if (![shadow respondsToSelector:@selector(externFunctionNames)])
		return @[];
	return [(id<RgfxMetalLibraryInternals>)shadow externFunctionNames] ?: @[];
}

- (id<MTLFunction>)newExternFunctionWithName:(NSString *)name
{
	id<MTLLibrary> shadow = [self rgfxShadow];
	if (name == nil || ![shadow respondsToSelector:@selector(newExternFunctionWithName:)])
		return nil;
	id<MTLFunction> genuine = [(id<RgfxMetalLibraryInternals>)shadow newExternFunctionWithName:name];
	if (genuine == nil)
		return nil;
	RgfxFunction *fn = [RgfxFunction functionWithLibrary:self
	                                                name:name
	                                           constants:nullptr
	                                               count:0
	                                              device:self.device
	                                          serializer:[self serializer]];
	[fn rgfxSetFunctionType:genuine.functionType];
	const MetallibFunction *entry = [self rgfxEntryNamed:name];
	if (entry != nullptr && entry->hasHash)
		[fn rgfxSetBitCodeHash:entry->hash];
	[fn rgfxSetShadow:genuine];
	return fn;
}

- (MTLLibraryType)type
{
	return MTLLibraryTypeExecutable;
}

- (NSString *)installName
{
	return nil; // an executable library has no install name
}

// --- unsupported ----------------------------------------------------------------------

// Intersection functions are ray tracing's, which the device does not support: nil and
// Metal's own unsupported error.
- (id<MTLFunction>)newIntersectionFunctionWithDescriptor:(MTLIntersectionFunctionDescriptor *)descriptor
                                                   error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "ray tracing", MTLLibraryErrorDomain,
	                          MTLLibraryErrorUnsupported);
	return nil;
}

- (void)newIntersectionFunctionWithDescriptor:(MTLIntersectionFunctionDescriptor *)descriptor
                            completionHandler:(void (^)(id<MTLFunction> _Nullable,
                                                        NSError *_Nullable))completionHandler
{
	NSError *err = nil;
	id<MTLFunction> fn = [self newIntersectionFunctionWithDescriptor:descriptor error:&err];
	if (completionHandler)
		completionHandler(fn, err);
}

// A function's reflection: the shadow's, Metal's own of the same function, where the library
// keeps its source (Core Image reflects its kernels so; measured: macOS 26.6.1). With no shadow it
// would need the host's per-argument bindings, which no reply carries.
- (MTLFunctionReflection *)reflectionForFunctionWithName:(NSString *)functionName
{
	id<MTLLibrary> shadow = [self rgfxShadow];
	if ([shadow respondsToSelector:@selector(reflectionForFunctionWithName:)])
		return [shadow reflectionForFunctionWithName:functionName];
	RGFX_UNSUPPORTED("function reflection (the host sends no argument bindings)");
	return nil;
}

@end

NSArray<id<MTLFunction>> *RgfxShadowFunctions(NSArray<id<MTLFunction>> *functions, NSError **error)
{
	NSMutableArray<id<MTLFunction>> *shadows = [NSMutableArray arrayWithCapacity:functions.count];
	for (id<MTLFunction> fn in functions) {
		if (![(id)fn isKindOfClass:[RgfxFunction class]]) {
			[shadows addObject:fn];
			continue;
		}
		RgfxFunction *rf = (RgfxFunction *)fn;
		// Metal's newFunctionWithName: answers a Core Image kernel library's extern functions too,
		// of functionType 4 (measured: macOS 27).
		id<MTLFunction> shadow = [rf rgfxShadowOrMake];
		if (shadow == nil) {
			if (error)
				*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
				                       [NSString stringWithFormat:@"link: function %@ has no function of "
				                                                  @"Metal's own library",
				                                                  rf.name]);
			return nil;
		}
		[shadows addObject:shadow];
	}
	return shadows;
}

id<MTLLibrary> RgfxLinkOverShadows(NSArray<id<MTLFunction>> *functions, RgfxLinkBlock link,
                                   RgfxLibraryMaker makeLibrary, NSError **error)
{
	NSArray<id<MTLFunction>> *shadows = RgfxShadowFunctions(functions, error);
	if (shadows == nil)
		return nil;
	id<MTLLibrary> linked = link(shadows, error);
	if (linked == nil || [(id)linked isKindOfClass:[RgfxLibrary class]])
		return linked;
	NSData *bytes = [linked respondsToSelector:@selector(libraryDataContents)]
	                    ? [(id<RgfxMetalLibraryInternals>)linked libraryDataContents]
	                    : nil;
	if (bytes.length == 0) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRefused, @"link: Metal's linked library answered no bytes");
		return nil;
	}
	dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		(void)bytes;
	});
	id<MTLLibrary> lib = makeLibrary(data, error);
	if ([(id)lib isKindOfClass:[RgfxLibrary class]])
		[(RgfxLibrary *)lib rgfxAdoptShadow:linked];
	return lib;
}

id<MTLLibrary> RgfxStitchOverShadows(MTLStitchedLibraryDescriptor *descriptor, RgfxStitchBlock stitch,
                                     RgfxLibraryMaker makeLibrary, NSError **error)
{
	MTLStitchedLibraryDescriptor *copy = [descriptor copy];
	copy.binaryArchives = @[];
	return RgfxLinkOverShadows(
	    descriptor.functions,
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError *__autoreleasing *e) {
		    copy.functions = shadows;
		    return stitch(copy, e);
	    },
	    makeLibrary, error);
}
