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
#include <string>
#include <vector>

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
// not carried: `*dropped` counts them. measured: macOS 27 Metal
// (MTLFunctionConstantValuesInternal, MTLIndexedConstantValue).
std::vector<RgfxFunctionConstant> indexedConstants(MTLFunctionConstantValues *values, uint32_t *dropped)
{
	std::vector<RgfxFunctionConstant> out;
	*dropped = 0;
	if (values == nil)
		return out;
	SEL indexedSel = sel_registerName("newIndexedConstantArray");
	SEL namedSel = sel_registerName("newNamedConstantArray");
	if ([values respondsToSelector:namedSel]) {
		NSArray *named = ((NSArray * (*)(id, SEL)) objc_msgSend)(values, namedSel);
		*dropped += static_cast<uint32_t>(named.count);
	}
	if (![values respondsToSelector:indexedSel])
		return out;
	NSArray *indexed = ((NSArray * (*)(id, SEL)) objc_msgSend)(values, indexedSel);
	SEL indexSel = sel_registerName("index");
	SEL typeSel = sel_registerName("dataType");
	SEL dataSel = sel_registerName("data");
	for (id c in indexed) {
		if (![c respondsToSelector:indexSel] || ![c respondsToSelector:typeSel] ||
		    ![c respondsToSelector:dataSel]) {
			(*dropped)++;
			continue;
		}
		NSUInteger index = ((NSUInteger(*)(id, SEL))objc_msgSend)(c, indexSel);
		MTLDataType type = static_cast<MTLDataType>(((NSUInteger(*)(id, SEL))objc_msgSend)(c, typeSel));
		const void *data = ((const void *(*)(id, SEL))objc_msgSend)(c, dataSel);
		uint32_t size = inlineConstantSize(type);
		if (size == 0 || data == nullptr) {
			(*dropped)++;
			continue;
		}
		RgfxFunctionConstant k = {};
		k.index = static_cast<uint32_t>(index);
		k.data_type = static_cast<uint32_t>(type);
		std::memcpy(&k.value, data, size);
		out.push_back(k);
	}
	return out;
}

} // namespace

@implementation RgfxLibrary {
	std::vector<MetallibFunction> _functions; // the .metallib's function list
	BOOL _haveFunctionTable;                  // the bytes were a readable .metallib
}

+ (instancetype)libraryWithData:(const void *)bytes
                         length:(NSUInteger)length
                         device:(id<MTLDevice>)device
                     serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a library must have a route to the host
	}
	RgfxCreated created = serializer->createLibrary(bytes, length);
	if (!created.posted) {
		return nil; // the bytes did not fit the control ring (a later transport
		            // piece places large AIR in a region block)
	}
	RgfxLibrary *lib = [[RgfxLibrary alloc] initWithObjId:created.obj_id
	                                               device:device
	                                           serializer:serializer];
	if (lib != nil)
		lib->_haveFunctionTable =
		    readMetallibFunctions(static_cast<const uint8_t *>(bytes), length, lib->_functions);
	return lib;
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
	RgfxFunction *fn = [RgfxFunction functionWithLibrary:self.rgfxObjId
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
	uint32_t dropped = 0;
	std::vector<RgfxFunctionConstant> constants = indexedConstants(constantValues, &dropped);
	if (dropped)
		NSLog(@"RGFX: %@: %u function constant(s) set by name or wider than 8 bytes are not carried",
		      name, dropped);
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
