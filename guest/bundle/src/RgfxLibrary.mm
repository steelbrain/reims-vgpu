/*
 * RgfxLibrary: see RgfxLibrary.h. Our own ObjC++; no Apple private SPI except the
 * read-only accessors of MTLFunctionConstantValues' indexed and named constants,
 * called only when the object answers them, and the extern-function members of
 * Metal's own library (RgfxMetalPrivate.h), called on the shadow.
 */

#import "RgfxLibrary.h"
#include "RgfxLogOnce.hpp"
#import "RgfxArgumentEncoder.h"
#import "RgfxBlockCall.h"
#import "RgfxUnsupported.h"

#import <objc/message.h>

#import "RgfxBinaryArchive.h"
#import "RgfxFunction.h"
#import "RgfxInterop.h"
#import "RgfxMetalError.h"
#import "RgfxMetalPrivate.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "RgfxAirArguments.hpp"
#include "RgfxFunctionConstants.hpp"
#include "RgfxLibraryStream.hpp"
#include "RgfxMetallibWriter.hpp"
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
	uint8_t tess = 0; // the TESS tag of a post-tessellation vertex function (rgfxPatchOf)
	bool hasTess = false;
	bool exported = false; // a dynamic library's exported function: TYPE 3 (metallibStage)
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
// `functionNames` leaves out); and on the AMD 5300M and Intel 630 under macOS 26.7.1, over
// libraries Apple's compiler built from our own sources (oracle_bundle, functypes): 4 for a
// [[visible]] and a [[stitchable]] function (Visible), 6 intersection, 7 mesh, 8 object. A
// dynamic library's exported function is 3, which both end the process at as a function is made
// of it (validateMTLFunctionType's assertion, oracle_bundle functypes.dyn), as the bundle does
// (rgfxFunctionNamed); it and any other byte answer Visible here.
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
	case 6:
		return MTLFunctionTypeIntersection;
	case 7:
		return MTLFunctionTypeMesh;
	case 8:
		return MTLFunctionTypeObject;
	default:
		return MTLFunctionTypeVisible;
	}
}

// A post-tessellation vertex function's patch type and control point count, from its TESS tag
// byte: the type (MTLPatchType) in the low two bits, the count above them, 63 where the
// shader's [[patch]] states none, which Metal answers as -1. measured: the tag Apple's metal
// compiler writes for [[patch(triangle|quad[, 3|4|16|32])]] (macOS 27 SDK, deployment targets
// 14.0 and 27.0), against what Metal's own functions answer for those bytes on the M3 Max
// (macOS 27) and on the AMD 5300M and Intel 630 (macOS 26.7.1). A function with no TESS tag
// answers MTLPatchTypeNone and -1 on all three.
void rgfxPatchOf(uint8_t tess, MTLPatchType *type, NSInteger *count)
{
	*type = static_cast<MTLPatchType>(tess & 3);
	*count = (tess >> 2) == 63 ? -1 : static_cast<NSInteger>(tess >> 2);
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
			else if (std::memcmp(p + t, "TYPE", 4) == 0 && size >= 1) {
				f.type = metallibStage(data[0]);
				f.exported = data[0] == 3;
			}
			else if (std::memcmp(p + t, "HASH", 4) == 0 && size == sizeof(f.hash)) {
				std::memcpy(f.hash, data, sizeof(f.hash));
				f.hasHash = true;
			} else if (std::memcmp(p + t, "OFFT", 4) == 0 && size >= 8) {
				f.publicOffset = rd64(data);
				f.haveOffsets = true;
			} else if (std::memcmp(p + t, "TESS", 4) == 0 && size >= 1) {
				f.tess = data[0];
				f.hasTess = true;
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

namespace {

// A .metallib's function list as readMetallibFunctions reads it: `readable` where the bytes were
// a readable .metallib; each function's place in `functions` by name (the first of a name); and
// each function's bitcode, as rgfxMetallibBitcode finds it in those bytes, by name.
struct FunctionTable {
	std::vector<MetallibFunction> functions;
	bool readable = false;
	std::unordered_map<std::string, size_t> byName;
	std::unique_ptr<RgfxMetallibBitcodeIndex> bitcode;
};

// The most function tables the process keeps by content key (functionTableOf).
constexpr size_t kKeptFunctionTables = 512;

// The function table of the `length` bytes at `bytes`, whose content key is `key` (32 bytes, or
// null for bytes with none): read from them, or, for a key the process read before, the table read
// then, shared. A key covers the function list and its metadata (RgfxLibraryKey.hpp), so bytes of
// one key have one table; a framework library a process loads again (MPSImage's, 1,837 functions:
// measured 4.4 to 5.1 ms a load, the cost following the function count) is not read again. The
// first kKeptFunctionTables keys are kept for the process's life.
std::shared_ptr<const FunctionTable> functionTableOf(const uint8_t *bytes, size_t length,
                                                     const uint8_t *key)
{
	auto read = [&] {
		auto table = std::make_shared<FunctionTable>();
		table->readable = readMetallibFunctions(bytes, length, table->functions);
		for (size_t i = 0; i < table->functions.size(); i++)
			table->byName.emplace(table->functions[i].name, i);
		table->bitcode = std::make_unique<RgfxMetallibBitcodeIndex>(bytes, length);
		return std::shared_ptr<const FunctionTable>(std::move(table));
	};
	if (key == nullptr)
		return read();
	typedef std::array<uint8_t, 32> Key;
	static std::mutex lock;
	// Never destroyed: a library may be made or freed on any thread while the process exits.
	static auto *kept = new std::map<Key, std::shared_ptr<const FunctionTable>>();
	Key k;
	std::memcpy(k.data(), key, k.size());
	{
		std::lock_guard<std::mutex> hold(lock);
		auto found = kept->find(k);
		if (found != kept->end())
			return found->second;
	}
	std::shared_ptr<const FunctionTable> table = read();
	std::lock_guard<std::mutex> hold(lock);
	if (kept->size() < kKeptFunctionTables)
		return kept->emplace(k, table).first->second;
	return table;
}

// What Metal's library keeps past its protocol for its tools and its compiler (MTLLibrarySPI): an
// override of the target triple, a copy of the one set (nil until set), and whether shader
// validation is on (NO until set), each answered as set (measured: M3 Max, macOS 27, a library
// made from bytes, from a URL and of a function array).
struct LibraryToolState {
	std::mutex lock; // guards overrideTriple
	NSString *overrideTriple = nil;
	std::atomic<bool> shaderValidation{false};

	NSString *triple()
	{
		std::lock_guard<std::mutex> hold(lock);
		return overrideTriple;
	}
	void setTriple(NSString *triple)
	{
		NSString *copy = [triple copy];
		std::lock_guard<std::mutex> hold(lock);
		overrideTriple = copy;
	}
};

} // namespace

@implementation RgfxLibrary {
	LibraryToolState _tools; // MTLLibrarySPI's triple and validation switch
	// The .metallib's function list, shared by every library of its content key (functionTableOf).
	std::shared_ptr<const FunctionTable> _table;
	std::mutex _shadowLock;                   // guards the three below
	dispatch_data_t _source;                  // the bytes, kept for the shadow
	RgfxShadowMaker _shadowMaker;
	id<MTLLibrary> _shadow;                   // Metal's own over _source, made on first use
	BOOL _shadowTried;
	BOOL _madeFromURL; // newLibraryWithURL:'s, which Metal does not serialize
	NSString *_fileName; // the last part of newLibraryWithURL:'s URL, nil for any other
	BOOL _dynamic;       // a dynamic library's bytes (rgfxMetallibIsDynamic)
	std::mutex _airLock; // guards the four below
	// The module `_source` holds for each function a read below asked for, by name, as the process
	// read it (rgfxAirModules), with why there is none (null where there is one).
	std::map<std::string, std::pair<std::shared_ptr<const RgfxAirModule>, const char *>> _modules;
	// Each entry point's arguments read from its bitcode in `_source`, by name, once each: null
	// where they did not read.
	std::map<std::string, std::shared_ptr<const RgfxAirFunction>> _air;
	// Each visible function's signature read likewise; null where it did not read.
	std::map<std::string, std::shared_ptr<const RgfxAirVisible>> _visible;
	// Each entry point's nodes whole, read likewise when a pipeline's reflection asks; null where
	// they did not read.
	std::map<std::string, std::shared_ptr<const RgfxAirEntries>> _entries;
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

// Whether `length` bytes at `bytes` are a dynamic library's metallib: its header's file type (the
// byte at 0x0a) is 2, where an executable's is 0 (measured on the AMD 5300M and the Intel 630
// under macOS 26.7.1: oracle_bundle libtype's and functypes' libraries, compiled from source and
// by Apple's compiler, of type 1 where it is 2).
static bool rgfxMetallibIsDynamic(const uint8_t *bytes, size_t length)
{
	return bytes != nullptr && length > 0x0a && std::memcmp(bytes, "MTLB", 4) == 0 &&
	       bytes[0x0a] == 2;
}

// The library `objId` over `bytes`, of content key `key` (32 bytes, or null), its function table
// read from them, or the one the process read for that key (functionTableOf).
+ (instancetype)rgfxLibraryWithObjId:(RgfxObjId)objId
                               bytes:(const void *)bytes
                              length:(NSUInteger)length
                                 key:(const uint8_t *)key
                              device:(id<MTLDevice>)device
                          serializer:(RgfxSerializer *)serializer
{
	RgfxLibrary *lib = [[RgfxLibrary alloc] initWithObjId:objId device:device serializer:serializer];
	if (lib != nil) {
		lib->_table = functionTableOf(static_cast<const uint8_t *>(bytes), length, key);
		lib->_dynamic = rgfxMetallibIsDynamic(static_cast<const uint8_t *>(bytes), length);
	}
	return lib;
}

- (const void *)rgfxFunctionTable
{
	return _table.get();
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
	                              key:key
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
	                              key:key
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
	if (!rgfxStreamLibrary(*region, *serializer, static_cast<const uint8_t *>(bytes), length, key,
	                       &objId, status) ||
	    !rgfxMade(*status)) {
		return nil;
	}
	return [self rgfxLibraryWithObjId:objId
	                            bytes:bytes
	                           length:length
	                              key:key
	                           device:device
	                       serializer:serializer];
}

// The table entry for `name` (the first of that name), or null when the table does not hold it.
- (const MetallibFunction *)rgfxEntryNamed:(NSString *)name
{
	const char *n = name.UTF8String;
	if (n == nullptr)
		return nullptr;
	auto found = _table->byName.find(n);
	return found != _table->byName.end() ? &_table->functions[found->second] : nullptr;
}

// Metal's own function `name` of the library's shadow, for a library whose function list did not
// read (bytes this reader does not follow, which the host and Metal took): the shadow, Metal's
// own library over the same bytes, answers which functions the library holds and what each is.
// nil where the list read, where there is no shadow, or where the shadow holds no such function;
// `*haveShadow` says whether there is one to ask.
- (nullable id<MTLFunction>)rgfxGenuineNamed:(NSString *)name haveShadow:(BOOL *)haveShadow
{
	*haveShadow = NO;
	if (_table->readable || name == nil)
		return nil;
	id<MTLLibrary> shadow = [self rgfxShadow];
	*haveShadow = shadow != nil;
	return [shadow newFunctionWithName:name];
}

// Vends `name` specialised by `constants`: nil when the library's function table, or for a
// library whose table did not read its shadow, does not hold it (as Metal answers a missing
// name).
- (nullable id<MTLFunction>)rgfxFunctionNamed:(NSString *)name
                                    constants:(const std::vector<RgfxFunctionConstant> &)constants
{
	if (name == nil) {
		return nil;
	}
	const MetallibFunction *entry = nullptr;
	BOOL haveShadow = NO;
	id<MTLFunction> genuine = nil;
	if (_table->readable) {
		entry = [self rgfxEntryNamed:name];
		if (entry == nullptr)
			return nil;
		// A dynamic library's exported function, of no MTLFunctionType: the x86 Macs' Metal ends
		// the process making a function of it (AMD 5300M and Intel 630, macOS 26.7.1).
		if (entry->exported)
			RgfxMetalAssertionFailed("validateMTLFunctionType",
			                         @"type is not a valid MTLFunctionType.");
	} else {
		genuine = [self rgfxGenuineNamed:name haveShadow:&haveShadow];
		if (haveShadow && genuine == nil)
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
		if (entry->hasTess) {
			MTLPatchType patchType;
			NSInteger controlPoints;
			rgfxPatchOf(entry->tess, &patchType, &controlPoints);
			[fn rgfxSetPatchType:patchType controlPointCount:controlPoints];
		}
		if (entry->type == MTLFunctionTypeVertex || entry->type == MTLFunctionTypeFragment ||
		    entry->type == MTLFunctionTypeKernel)
			[fn rgfxSetAirFunction:[self rgfxAirFunctionNamed:entry->name]];
		else if (entry->type == MTLFunctionTypeVisible)
			[fn rgfxSetAirVisible:[self rgfxAirVisibleNamed:entry->name]];
	} else if (genuine != nil) {
		[fn rgfxSetFunctionType:genuine.functionType];
		[fn rgfxSetPatchType:genuine.patchType controlPointCount:genuine.patchControlPointCount];
	}
	return fn;
}

bool rgfxAirUnreadFirstLog(const std::string &name)
{
	static std::mutex lock;
	static std::set<std::string> logged;
	std::lock_guard<std::mutex> hold(lock);
	return logged.insert(name).second;
}

// The module the library holds for the function `name`, in the bytes it was made from, as the
// process read it (rgfxAirModules: one read for every library over equal bytes), once for the
// library: null where the library kept no bytes (one Metal's link or stitching made), with `*why`
// null, or where it holds no bitcode for the function, with `*why` saying so. Called holding
// _airLock.
- (std::shared_ptr<const RgfxAirModule>)rgfxModuleOf:(const std::string &)name why:(const char **)why
{
	*why = nullptr;
	auto known = _modules.find(name);
	if (known != _modules.end()) {
		*why = known->second.second;
		return known->second.first;
	}
	dispatch_data_t source = nil;
	{
		std::lock_guard<std::mutex> shadowHold(_shadowLock);
		source = _source;
	}
	if (source == nil)
		return nullptr;
	const void *mapped = nullptr;
	size_t size = 0;
	dispatch_data_t map = dispatch_data_create_map(source, &mapped, &size);
	// The library was made from a universal .metallib's AIR slice; the bytes kept are whole. It
	// was made from the .metallib of their kernels where they are the bytes Metal's compiler
	// answers for OpenCL C source (RgfxMetallibOfClLibrary), which the process keeps as they are
	// for the shadow.
	const uint8_t *bytes = static_cast<const uint8_t *>(mapped);
	rgfxMetallibAirSlice(bytes, size);
	std::vector<uint8_t> clKernels;
	if (map != nil && RgfxMetallibOfClLibrary(bytes, size, clKernels)) {
		bytes = clKernels.data();
		size = clKernels.size();
	}
	const uint8_t *bitcode = nullptr;
	size_t bitcodeSize = 0;
	std::shared_ptr<const RgfxAirModule> module;
	if (map == nil || !_table->bitcode->find(bytes, size, name, &bitcode, &bitcodeSize))
		*why = "the library holds no bitcode for it";
	else
		module = rgfxAirModules().moduleOf(bitcode, bitcodeSize);
	_modules.emplace(name, std::make_pair(module, *why));
	return module;
}

// The arguments of the entry point `name`, read from its bitcode in the bytes the library was
// made from (rgfxModuleOf:why:), once: null where the library kept no bytes (one Metal's link or
// stitching made), or its module or its own arguments do not read, so which textures the function
// writes is unknown.
- (std::shared_ptr<const RgfxAirFunction>)rgfxAirFunctionNamed:(const std::string &)name
{
	std::lock_guard<std::mutex> hold(_airLock);
	auto known = _air.find(name);
	if (known != _air.end())
		return known->second;
	const char *why = nullptr;
	std::shared_ptr<const RgfxAirModule> module = [self rgfxModuleOf:name why:&why];
	std::shared_ptr<const RgfxAirFunction> air;
	if (module != nullptr && !module->functionsRead) {
		why = "its module did not read";
	} else if (module != nullptr) {
		const RgfxAirFunction *named = nullptr;
		for (const RgfxAirFunction &f : module->functions) {
			if (f.name == name) {
				named = &f;
				break;
			}
		}
		if (named == nullptr)
			why = "its module declares no such entry point";
		else if (!named->argumentsRead)
			why = "an argument of it is not one the reader follows";
		else
			air = std::shared_ptr<const RgfxAirFunction>(module, named);
	}
	if (air == nullptr && why != nullptr && rgfxAirUnreadFirstLog(name))
		NSLog(@"RGFX: %s: no arguments read (%s): the textures it binds are taken as written "
		      @"where their usage has ShaderWrite (logged once a process)",
		      name.c_str(), why);
	_air.emplace(name, air);
	return air;
}

- (std::shared_ptr<const RgfxAirEntries>)rgfxAirEntryNamed:(const std::string &)name
{
	std::lock_guard<std::mutex> hold(_airLock);
	auto known = _entries.find(name);
	if (known != _entries.end())
		return known->second;
	const char *why = nullptr;
	std::shared_ptr<const RgfxAirModule> module = [self rgfxModuleOf:name why:&why];
	std::shared_ptr<const RgfxAirEntries> entries;
	if (module != nullptr && !module->entriesRead) {
		why = "its module did not read";
	} else if (module != nullptr) {
		const RgfxAirEntry *named = nullptr;
		for (const RgfxAirEntry &e : module->entries.entries) {
			if (e.name == name) {
				named = &e;
				break;
			}
		}
		if (named == nullptr) {
			why = "its module declares no such entry point";
		} else if (!named->nodesRead) {
			why = "a node of it is not one the reader follows";
		} else {
			auto one = std::make_shared<RgfxAirEntries>();
			one->entries.push_back(*named);
			one->constants = module->entries.constants;
			one->code = module->entries.code;
			entries = std::move(one);
		}
	}
	static char site;
	if (entries == nullptr && why != nullptr && rgfxLogFirst(&site, rgfxLogKindOf(why)))
		NSLog(@"RGFX: %s: its nodes were not read (%s): pipelines of it answer no reflection",
		      name.c_str(), why);
	_entries.emplace(name, entries);
	return entries;
}

// The signature of the visible function `name`, read as rgfxAirFunctionNamed: reads arguments:
// null where it did not read, so Metal's own function answers its reflection.
- (std::shared_ptr<const RgfxAirVisible>)rgfxAirVisibleNamed:(const std::string &)name
{
	std::lock_guard<std::mutex> hold(_airLock);
	auto known = _visible.find(name);
	if (known != _visible.end())
		return known->second;
	const char *why = nullptr;
	std::shared_ptr<const RgfxAirModule> module = [self rgfxModuleOf:name why:&why];
	std::shared_ptr<const RgfxAirVisible> visible;
	if (module != nullptr && !module->visibleRead) {
		why = "its module did not read";
	} else if (module != nullptr) {
		for (const RgfxAirVisible &v : module->visible) {
			if (v.name == name) {
				visible = std::shared_ptr<const RgfxAirVisible>(module, &v);
				break;
			}
		}
		if (visible == nullptr)
			why = "its module lists no such visible function";
	}
	if (visible != nullptr && visible->signature == RgfxAirVisible::Signature::Unknown)
		why = "it has a type the reader does not know";
	if ((visible == nullptr || visible->signature == RgfxAirVisible::Signature::Unknown) && why != nullptr &&
	    rgfxAirUnreadFirstLog(name))
		NSLog(@"RGFX: %s: no signature read (%s): Metal's own function answers its reflection "
		      @"(logged once a process)",
		      name.c_str(), why);
	_visible.emplace(name, visible);
	return visible;
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

// The constants `decls` as `functionConstantsDictionary` answers them, by name.
static NSDictionary<NSString *, MTLFunctionConstant *> *
rgfxDeclaredConstants(const std::vector<RgfxConstantDecl> &decls)
{
	NSMutableDictionary<NSString *, MTLFunctionConstant *> *declared =
	    [NSMutableDictionary dictionaryWithCapacity:decls.size()];
	for (const RgfxConstantDecl &d : decls) {
		RgfxDeclaredConstant *c = [[RgfxDeclaredConstant alloc] initWithDecl:d];
		declared[c.name] = c;
	}
	return declared;
}

// The constants of `entry`, a function of a table that read whose metadata gives none (no CNST
// tag, or one that did not read), as its AIR module declares them (RgfxAirConstant), in index
// order: false where the module declares none or they did not read, or the function is of no
// stage the AIR reader follows. Apple's compiler writes a CNST tag for every function its module
// declares constants for (RgfxAirArguments.hpp), so this answers for metadata of another writer.
- (bool)rgfxAirConstantsOf:(const MetallibFunction &)entry into:(std::vector<RgfxConstantDecl> &)out
{
	out.clear();
	std::shared_ptr<const RgfxAirFunction> fn;
	std::shared_ptr<const RgfxAirVisible> visible;
	const std::vector<RgfxAirConstant> *air = nullptr;
	if (entry.type == MTLFunctionTypeVertex || entry.type == MTLFunctionTypeFragment ||
	    entry.type == MTLFunctionTypeKernel) {
		fn = [self rgfxAirFunctionNamed:entry.name];
		if (fn != nullptr && fn->constantsRead)
			air = &fn->constants;
	} else if (entry.type == MTLFunctionTypeVisible) {
		visible = [self rgfxAirVisibleNamed:entry.name];
		if (visible != nullptr && visible->constantsRead)
			air = &visible->constants;
	}
	if (air == nullptr || air->empty())
		return false;
	for (const RgfxAirConstant &c : *air)
		out.push_back(RgfxConstantDecl{c.name, c.type, c.index, c.required});
	std::sort(out.begin(), out.end(), [](const RgfxConstantDecl &a, const RgfxConstantDecl &b) {
		return a.index < b.index;
	});
	return true;
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)functionName
{
	RgfxFunction *fn = (RgfxFunction *)[self rgfxFunctionNamed:functionName constants:{}];
	const MetallibFunction *entry = fn ? [self rgfxEntryNamed:functionName] : nullptr;
	std::vector<RgfxConstantDecl> declaredByAir;
	if (entry != nullptr && entry->haveConstants && !entry->constants.empty()) {
		[fn rgfxSetUnspecialisedConstants:rgfxDeclaredConstants(entry->constants)];
	} else if (entry != nullptr && [self rgfxAirConstantsOf:*entry into:declaredByAir]) {
		// Metadata that names no constants: its module's.
		[fn rgfxSetUnspecialisedConstants:rgfxDeclaredConstants(declaredByAir)];
	} else if (fn != nil && !_table->readable) {
		// Where the table did not read, the shadow's function declares them.
		BOOL haveShadow = NO;
		id<MTLFunction> genuine = [self rgfxGenuineNamed:functionName haveShadow:&haveShadow];
		[fn rgfxSetUnspecialisedConstants:genuine.functionConstantsDictionary];
	}
	return fn;
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                                 error:(__autoreleasing NSError **)error
{
	if (error)
		*error = nil;
	const MetallibFunction *entry = _table->readable ? [self rgfxEntryNamed:name] : nullptr;
	if (name == nil || (_table->readable && entry == nullptr)) {
		if (error)
			*error = rgfxFunctionNotFound(name);
		return nil;
	}
	std::map<uint32_t, RgfxConstantValue> byIndex;
	std::map<std::string, RgfxConstantValue> byName;
	readConstantValues(constantValues, byIndex, byName);
	std::vector<RgfxFunctionConstant> constants;
	// The function's constants: its table entry's; where its entry names none, those its module
	// declares (rgfxAirConstantsOf:into:); where the table did not read, those its shadow's
	// function declares (Metal's own over the same bytes), in index order.
	const std::vector<RgfxConstantDecl> *decls =
	    entry != nullptr && entry->haveConstants ? &entry->constants : nullptr;
	std::vector<RgfxConstantDecl> declaredByAir;
	if (entry != nullptr && (decls == nullptr || decls->empty()) &&
	    [self rgfxAirConstantsOf:*entry into:declaredByAir])
		decls = &declaredByAir;
	std::vector<RgfxConstantDecl> declaredByShadow;
	if (!_table->readable) {
		BOOL haveShadow = NO;
		id<MTLFunction> genuine = [self rgfxGenuineNamed:name haveShadow:&haveShadow];
		if (haveShadow && genuine == nil) {
			if (error)
				*error = rgfxFunctionNotFound(name);
			return nil;
		}
		if (genuine != nil) {
			for (MTLFunctionConstant *c in genuine.functionConstantsDictionary.allValues) {
				declaredByShadow.push_back(RgfxConstantDecl{
				    c.name.UTF8String ?: "", static_cast<uint32_t>(c.type),
				    static_cast<uint32_t>(c.index), c.required ? true : false});
			}
			std::sort(declaredByShadow.begin(), declaredByShadow.end(),
			          [](const RgfxConstantDecl &a, const RgfxConstantDecl &b) {
				          return a.index < b.index;
			          });
			decls = &declaredByShadow;
		}
	}
	if (decls != nullptr) {
		// The values resolved against the function's own constants, as Metal resolves them.
		RgfxConstantMismatch mismatch;
		if (!rgfxResolveConstants(*decls, byIndex, byName, constants, &mismatch)) {
			const RgfxConstantDecl &d = (*decls)[mismatch.decl];
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
		// Nothing to resolve against (a table that did not read and no shadow, or an entry
		// whose metadata did not read): the values set by index, as set; the host checks them
		// against the function. Values set by name need the constants' names, and are not
		// carried, which is logged.
		if (!byName.empty())
			RGFX_UNSUPPORTED("function constants set by name, in a library whose table did not read");
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
	id<MTLFunction> fn = [self rgfxFunctionNamed:name constants:constants];
	// Its shadow is made with the same values, should a stitch or a link need it.
	if ([(id)fn isKindOfClass:[RgfxFunction class]])
		[(RgfxFunction *)fn rgfxSetShadowConstantValues:constantValues];
	return fn;
}

- (void)newFunctionWithName:(NSString *)name
             constantValues:(MTLFunctionConstantValues *)constantValues
          completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	name = [name copy];
	constantValues = [constantValues copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLFunction> fn = [self newFunctionWithName:name constantValues:constantValues error:&err];
		if (completionHandler)
			completionHandler(fn, err);
	});
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
	// binaryArchives reached MTLFunctionDescriptor in macOS 12; before it there are none to miss.
	NSArray<id<MTLBinaryArchive>> *archives = nil;
	if ([descriptor respondsToSelector:@selector(binaryArchives)])
		archives = descriptor.binaryArchives;
	NSError *miss = [RgfxBinaryArchive rgfxMissOfFunction:(RgfxFunction *)fn
	                                              options:descriptor.options
	                                             archives:archives];
	if (miss != nil) {
		if (error)
			*error = miss;
		return nil;
	}
	[(RgfxFunction *)fn rgfxSetOptions:descriptor.options];
	if (descriptor.specializedName.length != 0)
		[(RgfxFunction *)fn rgfxSetSpecializedName:descriptor.specializedName];
	return fn;
}

- (void)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLFunction> fn = [self newFunctionWithDescriptor:descriptor error:&err];
		if (completionHandler)
			completionHandler(fn, err);
	});
}

- (NSArray<NSString *> *)functionNames
{
	// The .metallib's function list but its extern functions, as Metal lists it; for bytes whose
	// list did not read, the shadow's, Metal's own over them (none where there is no shadow).
	if (!_table->readable)
		return [[self rgfxShadow] functionNames] ?: @[];
	NSMutableArray<NSString *> *names =
	    [NSMutableArray arrayWithCapacity:_table->functions.size()];
	for (const MetallibFunction &f : _table->functions) {
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

- (void)rgfxSetMadeFromURL:(NSURL *)url
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	_madeFromURL = YES;
	_fileName = [url.lastPathComponent copy];
}

// Metal answers nil for a library made from bytes, of data or of a URL (measured: macOS 27), as
// every library of the device is, and so does the shadow of one; a library Metal's link or
// stitching made answers its shadow's, Metal's own library.
- (id)bitcodeData
{
	id<MTLLibrary> shadow = nil;
	{
		std::lock_guard<std::mutex> hold(_shadowLock);
		shadow = _shadow;
	}
	return [shadow respondsToSelector:_cmd] ? ((id (*)(id, SEL))objc_msgSend)(shadow, _cmd) : nil;
}

// Metal writes a library made from data to `url` as the bytes it was made from, answering
// Foundation's error for a URL it cannot write, and refuses one made from a URL (measured: macOS
// 27): the shadow, Metal's own library over the same bytes, writes it, and a library made from a
// URL is refused with Metal's error.
- (BOOL)serializeToURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	BOOL fromURL = NO;
	{
		std::lock_guard<std::mutex> hold(_shadowLock);
		fromURL = _madeFromURL;
	}
	if (fromURL) {
		if (error)
			*error = [NSError
			    errorWithDomain:MTLLibraryErrorDomain
			               code:MTLLibraryErrorUnsupported
			           userInfo:@{NSLocalizedDescriptionKey : @"MTLLibrary is not formatted as a MetalLib file."}];
		return NO;
	}
	id<MTLLibrary> shadow = [self rgfxShadow];
	if (![shadow respondsToSelector:_cmd]) {
		RGFX_UNSUPPORTED_ERROR_IN(error, "serializing a library Metal's own does not hold",
		                          MTLLibraryErrorDomain, MTLLibraryErrorUnsupported);
		return NO;
	}
	return ((BOOL (*)(id, SEL, NSURL *, NSError *__autoreleasing *))objc_msgSend)(shadow, _cmd, url,
	                                                                              error);
}

// The library's type and install name as Metal answers them on the AMD 5300M and the Intel 630
// under macOS 26.7.1 (oracle_bundle libtype and functypes): a dynamic library's bytes answer
// Dynamic and the install name they hold (@executable_path/liboracle_bundle_dyn.metallib), made
// from a URL or from data, which the shadow, Metal's own library over them, answers; any other
// answers Executable, and as its install name the last part of the URL it was made from
// (oracle_bundle.metallib), or default.metallib for one made from data or source.
- (MTLLibraryType)type
{
	return _dynamic ? MTLLibraryTypeDynamic : MTLLibraryTypeExecutable;
}

- (NSString *)installName
{
	if (_dynamic) {
		id<MTLLibrary> shadow = [self rgfxShadow];
		return [shadow respondsToSelector:_cmd] ? shadow.installName : nil;
	}
	std::lock_guard<std::mutex> hold(_shadowLock);
	return _fileName ?: @"default.metallib";
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
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLFunction> fn = [self newIntersectionFunctionWithDescriptor:descriptor error:&err];
		if (completionHandler)
			completionHandler(fn, err);
	});
}

// A function's reflection: the shadow's, Metal's own of the same function, where the library
// keeps its source (Core Image reflects its kernels so; measured: macOS 26.6.1). With no shadow it
// would need the host's per-argument bindings, which no reply carries. Its buffer bindings are
// noted as the function's (RgfxNoteArgumentBinding), so an encoder of one is the function's.
- (MTLFunctionReflection *)reflectionForFunctionWithName:(NSString *)functionName
{
	id<MTLLibrary> shadow = [self rgfxShadow];
	if ([shadow respondsToSelector:@selector(reflectionForFunctionWithName:)]) {
		MTLFunctionReflection *reflection = [shadow reflectionForFunctionWithName:functionName];
		id<MTLFunction> function = nil;
		for (id<MTLBinding> binding in reflection.bindings) {
			if (binding.type != MTLBindingTypeBuffer)
				continue;
			if (function == nil)
				function = [self newFunctionWithName:functionName];
			if (![(id)function isKindOfClass:[RgfxFunction class]])
				break;
			RgfxNoteArgumentBinding(binding, (RgfxFunction *)function);
		}
		return reflection;
	}
	RGFX_UNSUPPORTED("function reflection (the host sends no argument bindings)");
	return nil;
}

// --- MTLLibrarySPI: what Metal's library keeps for its tools ---------------------------------
// As Metal's own library answers each (measured: M3 Max, macOS 27, libraries made from bytes,
// from a URL and of a function array; the AMD 5300M and the Intel 630, macOS 26.7.1, one made from
// source): the triple and the validation switch as set, nil and NO until set. The host compiles a
// library's functions for its own GPU, whatever the triple, and the device validates no shader.

// Metal answers nil for a library made from bytes or from a URL (measured: macOS 27), as every
// library of the device is, and a UUID for one it compiled from source (measured: macOS 26.7.1):
// the shadow of one Metal compiled, linked or stitched, Metal's own library, answers it, as for
// bitcodeData.
- (id)libraryIdentifier
{
	id<MTLLibrary> shadow = nil;
	{
		std::lock_guard<std::mutex> hold(_shadowLock);
		shadow = _shadow;
	}
	return [shadow respondsToSelector:_cmd] ? ((id (*)(id, SEL))objc_msgSend)(shadow, _cmd) : nil;
}

- (NSString *)overrideTriple
{
	return _tools.triple();
}

- (void)setOverrideTriple:(NSString *)overrideTriple
{
	_tools.setTriple(overrideTriple);
}

- (BOOL)shaderValidationEnabled
{
	return _tools.shaderValidation.load();
}

- (void)setShaderValidationEnabled:(BOOL)enabled
{
	_tools.shaderValidation.store(enabled);
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

// `metal`, Metal's own library, made ours by `makeLibrary` over the bytes it answers, keeping it as
// the shadow; `noBytes` is the error where it answers none.
static id<MTLLibrary> rgfxOursOver(id<MTLLibrary> metal, RgfxLibraryMaker makeLibrary, NSString *noBytes,
                                   NSError **error)
{
	NSData *bytes = [metal respondsToSelector:@selector(libraryDataContents)]
	                    ? [(id<RgfxMetalLibraryInternals>)metal libraryDataContents]
	                    : nil;
	if (bytes.length == 0) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRefused, noBytes);
		return nil;
	}
	dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		(void)bytes;
	});
	id<MTLLibrary> lib = makeLibrary(data, error);
	if ([(id)lib isKindOfClass:[RgfxLibrary class]])
		[(RgfxLibrary *)lib rgfxAdoptShadow:metal];
	return lib;
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
	return rgfxOursOver(linked, makeLibrary, @"link: Metal's linked library answered no bytes", error);
}

id<MTLLibrary> RgfxLibraryOverMetal(RgfxCompileBlock make, RgfxLibraryMaker makeLibrary, NSString *creator,
                                     NSError **error)
{
	NSError *metals = nil;
	id<MTLLibrary> made = make(&metals);
	if (made == nil || [(id)made isKindOfClass:[RgfxLibrary class]]) {
		if (error)
			*error = metals;
		return made;
	}
	id<MTLLibrary> lib = rgfxOursOver(
	    made, makeLibrary, [NSString stringWithFormat:@"%@ Metal's library answered no bytes", creator],
	    error);
	if (lib != nil && error)
		*error = metals;
	return lib;
}

id<MTLLibrary> RgfxCompileOverMetal(RgfxCompileBlock compile, RgfxLibraryMaker makeLibrary, NSError **error)
{
	return RgfxLibraryOverMetal(compile, makeLibrary, @"newLibraryWithSource:", error);
}

id<MTLLibrary> RgfxStitchOverShadows(MTLStitchedLibraryDescriptor *descriptor, id<MTLBinaryArchive> destination,
                                     RgfxStitchBlock stitch, RgfxLibraryMaker makeLibrary, NSError **error)
{
	if ((descriptor.options & MTLStitchedLibraryOptionFailOnBinaryArchiveMiss) != 0 &&
	    ![RgfxBinaryArchive rgfxStitched:descriptor heldBy:descriptor.binaryArchives]) {
		if (error)
			*error = [NSError errorWithDomain:MTLLibraryErrorDomain
			                             code:MTLLibraryErrorCompileFailure
			                         userInfo:@{
				                         NSLocalizedDescriptionKey : @"Failed to find library in binary archive."
			                         }];
		return nil;
	}
	MTLStitchedLibraryDescriptor *copy = [descriptor copy];
	// Metal would read our archives as its own, and finds nothing where it fails on a miss with none.
	copy.binaryArchives = @[];
	copy.options &= ~MTLStitchedLibraryOptionFailOnBinaryArchiveMiss;
	id<MTLLibrary> lib = RgfxLinkOverShadows(
	    descriptor.functions,
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError *__autoreleasing *e) {
		    copy.functions = shadows;
		    return stitch(copy, e);
	    },
	    makeLibrary, error);
	if (lib != nil && [(id)destination isKindOfClass:[RgfxBinaryArchive class]])
		[(RgfxBinaryArchive *)destination rgfxAddStitched:descriptor];
	return lib;
}

// --- libraries of a function array -----------------------------------------------------------

@implementation RgfxFunctionArrayLibrary {
	NSArray<id<MTLFunction>> *_functions; // in the order given
	LibraryToolState _tools;              // MTLLibrarySPI's triple and validation switch
}

+ (instancetype)libraryOfFunctions:(NSArray<id<MTLFunction>> *)functions
                            device:(id<MTLDevice>)device
                             error:(__autoreleasing NSError **)error
{
	if (functions.count == 0)
		RgfxMetalAssertionFailed("-[_MTLDevice(MTLDeviceInternal) newLibraryWithFunctionArray:error:]",
		                         @"Array must not be emty");
	NSMutableSet<NSString *> *names = [NSMutableSet setWithCapacity:functions.count];
	for (id<MTLFunction> fn in functions) {
		NSString *name = fn.name ?: @"";
		if ([names containsObject:name]) {
			if (error)
				*error = [NSError errorWithDomain:MTLLibraryErrorDomain
				                             code:MTLLibraryErrorUnsupported
				                         userInfo:@{NSLocalizedDescriptionKey : @"function names must be unique"}];
			return nil;
		}
		[names addObject:name];
	}
	RgfxFunctionArrayLibrary *lib = [[self alloc] initWithObjId:RGFX_OBJ_NONE
	                                                     device:device
	                                                 serializer:nullptr];
	lib->_functions = [functions copy];
	return lib;
}

- (id<MTLFunction>)rgfxGiven:(NSString *)name
{
	for (id<MTLFunction> fn in _functions)
		if ([fn.name isEqualToString:name])
			return fn;
	return nil;
}

- (id<MTLFunction>)rgfxGiven:(NSString *)name error:(__autoreleasing NSError **)error
{
	id<MTLFunction> fn = [self rgfxGiven:name];
	if (fn == nil && error)
		*error = rgfxFunctionNotFound(name);
	return fn;
}

- (NSArray<NSString *> *)functionNames
{
	NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:_functions.count];
	for (id<MTLFunction> fn in _functions.reverseObjectEnumerator)
		[names addObject:fn.name ?: @""];
	return names;
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)functionName
{
	return [self rgfxGiven:functionName];
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                                 error:(__autoreleasing NSError **)error
{
	(void)constantValues;
	return [self rgfxGiven:name error:error];
}

- (void)newFunctionWithName:(NSString *)name
             constantValues:(MTLFunctionConstantValues *)constantValues
          completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	name = [name copy];
	constantValues = [constantValues copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLFunction> fn = [self newFunctionWithName:name constantValues:constantValues error:&err];
		if (completionHandler)
			completionHandler(fn, err);
	});
}

- (id<MTLFunction>)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                                       error:(__autoreleasing NSError **)error
{
	return [self rgfxGiven:descriptor.name error:error];
}

- (void)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLFunction> fn = [self newFunctionWithDescriptor:descriptor error:&err];
		if (completionHandler)
			completionHandler(fn, err);
	});
}

// Intersection functions are ray tracing's, which the device does not support, as RgfxLibrary's.
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
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLFunction> fn = [self newIntersectionFunctionWithDescriptor:descriptor error:&err];
		if (completionHandler)
			completionHandler(fn, err);
	});
}

- (MTLFunctionReflection *)reflectionForFunctionWithName:(NSString *)functionName
{
	(void)functionName;
	return nil;
}

- (MTLLibraryType)type
{
	return MTLLibraryTypeExecutable;
}

- (NSString *)installName
{
	return nil;
}

// nil, as Metal's library of a function array answers (measured: M3 Max, macOS 27).
- (NSArray<NSString *> *)externFunctionNames
{
	return nil;
}

// The function given of that name, as Metal's library of a function array answers, and nil for a
// name none has (measured: M3 Max, macOS 27).
- (id<MTLFunction>)newExternFunctionWithName:(NSString *)name
{
	return [self rgfxGiven:name];
}

// No bitcode of its own, as Metal's library of a function array answers (measured: M3 Max, macOS
// 27): its functions keep theirs.
- (id)bitcodeData
{
	return nil;
}

- (BOOL)serializeToURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	(void)url;
	if (error)
		*error = [NSError
		    errorWithDomain:MTLLibraryErrorDomain
		               code:MTLLibraryErrorUnsupported
		           userInfo:@{NSLocalizedDescriptionKey : @"MTLLibrary is not formatted as a MetalLib file."}];
	return NO;
}

// --- MTLLibrarySPI: what Metal's library keeps for its tools ---------------------------------
// As Metal's own library of a function array answers each (measured: M3 Max, macOS 27): no
// identifier, and the triple and the validation switch as set, nil and NO until set. The host
// compiles a library's functions for its own GPU, whatever the triple, and the device validates
// no shader.

- (id)libraryIdentifier
{
	return nil;
}

- (NSString *)overrideTriple
{
	return _tools.triple();
}

- (void)setOverrideTriple:(NSString *)overrideTriple
{
	_tools.setTriple(overrideTriple);
}

- (BOOL)shaderValidationEnabled
{
	return _tools.shaderValidation.load();
}

- (void)setShaderValidationEnabled:(BOOL)enabled
{
	_tools.shaderValidation.store(enabled);
}

@end
