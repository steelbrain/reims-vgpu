/*
 * RgfxFunction: see RgfxFunction.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxFunction.h"

#import "RgfxArgumentEncoder.h"
#import "RgfxUnsupported.h"

#import "RgfxMetalError.h"

#include <CommonCrypto/CommonDigest.h>

#include <cstring>
#include <string>
#include <vector>

@implementation RgfxFunction {
	RgfxResource *_library; // kept alive: a pipeline made from the function names it
	NSDictionary<NSString *, MTLFunctionConstant *> *_unspecialised; // declared, with no values
	uint64_t _libraryObjId;
	std::string _name;                          // the entry point, owned
	std::vector<RgfxFunctionConstant> _constants; // specialisation constants, owned
	MTLFunctionType _type;                        // the stage, from the library's table
	BOOL _haveType;
	RgfxBitCodeHash _hash;                        // the bitcode hash, from the library's table
	BOOL _haveHash;
	MTLFunctionOptions _options;                  // the descriptor's, where made from one
}

+ (instancetype)functionWithLibrary:(RgfxResource *)library
                               name:(NSString *)name
                          constants:(const RgfxFunctionConstant *)constants
                              count:(uint32_t)count
                             device:(id<MTLDevice>)device
                         serializer:(RgfxSerializer *)serializer
{
	return [[RgfxFunction alloc] initWithLibrary:library
	                                        name:name
	                                   constants:constants
	                                       count:count
	                                      device:device
	                                  serializer:serializer];
}

- (instancetype)initWithLibrary:(RgfxResource *)library
                           name:(NSString *)name
                      constants:(const RgfxFunctionConstant *)constants
                          count:(uint32_t)count
                         device:(id<MTLDevice>)device
                     serializer:(RgfxSerializer *)serializer
{
	if (name == nil) {
		return nil;
	}
	// A function has no host handle of its own: it is named by the library objId
	// and entry point. RGFX_OBJ_NONE (0) is the base objId ("0 names no object").
	self = [super initWithObjId:RGFX_OBJ_NONE device:device serializer:serializer];
	if (self) {
		_library = library;
		_libraryObjId = library != nil ? library.rgfxObjId : RGFX_OBJ_NONE;
		_name = name.UTF8String ? name.UTF8String : "";
		if (count && constants) {
			_constants.assign(constants, constants + count);
		}
	}
	return self;
}

- (uint64_t)rgfxLibraryObjId
{
	return _libraryObjId;
}

- (const char *)rgfxEntryPointUTF8
{
	return _name.c_str();
}

- (const RgfxFunctionConstant *)rgfxConstants
{
	return _constants.empty() ? nullptr : _constants.data();
}

- (uint32_t)rgfxConstantCount
{
	return static_cast<uint32_t>(_constants.size());
}

- (void)rgfxSetFunctionType:(MTLFunctionType)type
{
	_type = type;
	_haveType = YES;
}

- (void)rgfxSetBitCodeHash:(const uint8_t *)hash
{
	std::memcpy(_hash.bytes, hash, sizeof(_hash.bytes));
	_haveHash = YES;
}

// The entry's HASH tag. A function whose library gave none (one with no function table) has no
// bitcode of its own here, so it answers a stand-in no other function shares, as callers key
// their pipelines by it: the SHA-256 of "rgfx-function-v1", its library's id and its name.
- (const RgfxBitCodeHash *)bitCodeHash
{
	@synchronized(self) {
		if (!_haveHash) {
			static const char kPrefix[] = "rgfx-function-v1";
			CC_SHA256_CTX ctx;
			CC_SHA256_Init(&ctx);
			CC_SHA256_Update(&ctx, kPrefix, sizeof(kPrefix) - 1);
			CC_SHA256_Update(&ctx, &_libraryObjId, sizeof(_libraryObjId));
			CC_SHA256_Update(&ctx, _name.data(), static_cast<CC_LONG>(_name.size()));
			CC_SHA256_Final(_hash.bytes, &ctx);
			_haveHash = YES;
		}
	}
	return &_hash;
}

- (void)rgfxSetUnspecialisedConstants:(NSDictionary<NSString *, MTLFunctionConstant *> *)declared
{
	_unspecialised = declared.count ? [declared copy] : nil;
}

- (BOOL)rgfxUnspecialised
{
	return _unspecialised != nil;
}

+ (NSError *)rgfxUnspecialisedError:(id<MTLFunction>)function role:(NSString *)role
{
	if (![(id)function isKindOfClass:[RgfxFunction class]] ||
	    ![(RgfxFunction *)function rgfxUnspecialised])
		return nil;
	// Metal asserts here (measured: macOS 27 Metal, compute and render, with or without its
	// validation layer); the lane ends no process on its input, so the pipeline is refused.
	NSString *what = [NSString
	    stringWithFormat:@"%@ %@ cannot be used to build a pipeline state. Use "
	                     @"newFunctionWithName:constantValues:... to get the specialized function",
	                     role, function.name];
	return RgfxMakeError(RgfxMetalErrorInvalidDescriptor, what);
}

// --- MTLFunction -------------------------------------------------------------

- (NSString *)name
{
	return [NSString stringWithUTF8String:_name.c_str()];
}

- (MTLFunctionType)functionType
{
	if (_haveType)
		return _type; // the .metallib's TYPE tag for this entry point
	// DOCUMENTED PLACEHOLDER. A function's stage (vertex/fragment/kernel) is known
	// only to the AIR the host compiles, and the guest shell never parses the
	// metallib; the wire carries no per-function reflection (a pipeline create
	// names the function by stage through its position in the descriptor). Until a
	// function-reflection reply lands, this answers a fixed placeholder rather than
	// fabricating per-function the host's answer. Callers that need the real stage
	// read it from the pipeline, not the function. (Follow-up in the report.)
	return MTLFunctionTypeVertex;
}

- (NSDictionary<NSString *, MTLFunctionConstant *> *)functionConstantsDictionary
{
	// The constants the library's table declares, for a function made with no values; none
	// for one specialised by values, as Metal answers (measured: macOS 27 Metal).
	return _unspecialised ?: @{};
}

- (NSArray<MTLVertexAttribute *> *)vertexAttributes
{
	return nil; // reflection the shell does not read (see functionConstantsDictionary)
}

- (NSArray<MTLAttribute *> *)stageInputAttributes
{
	return nil;
}

- (MTLPatchType)patchType
{
	return MTLPatchTypeNone;
}

- (NSInteger)patchControlPointCount
{
	return 0;
}

- (MTLFunctionOptions)options
{
	return _options;
}

- (void)rgfxSetOptions:(MTLFunctionOptions)options
{
	_options = options;
}

// The host lays out the argument buffer the function takes at `bufferIndex` from the library's
// function (ARGUMENT_LAYOUT, RgfxArgumentEncoder.h); nil where it takes none there.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
{
	id<MTLDevice> device = self.device;
	RgfxRegion *region = [device conformsToProtocol:@protocol(RgfxRegionHolder)]
	                         ? [(id<RgfxRegionHolder>)device region]
	                         : nullptr;
	return [RgfxArgumentEncoder encoderOfFunction:self
	                                  bufferIndex:bufferIndex
	                                       region:region
	                                       device:device
	                                   serializer:[self serializer]];
}

// The argument's reflection is not on the wire: the encoder comes back without it.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
{
	if (reflection) {
		*reflection = nil;
		RGFX_UNSUPPORTED("an argument buffer's reflection (no argument reflection on the wire)");
	}
	return [self newArgumentEncoderWithBufferIndex:bufferIndex];
}

@end

id<MTLFunction> RgfxCollectLinkedFunctions(MTLLinkedFunctions *linked, RgfxLinkedFunctionList &out)
{
	if (linked == nil)
		return nil;
	auto add = [&](NSArray<id<MTLFunction>> *functions, uint32_t list,
	               const char *group) -> id<MTLFunction> {
		for (id<MTLFunction> fn in functions) {
			if (![(id)fn isKindOfClass:[RgfxFunction class]])
				return fn;
			RgfxFunction *rf = (RgfxFunction *)fn;
			RgfxLinkedFunctionRef r = {};
			r.function.library = rf.rgfxLibraryObjId;
			r.function.entry_point = [rf rgfxEntryPointUTF8];
			r.function.constants = [rf rgfxConstants];
			r.function.constant_count = [rf rgfxConstantCount];
			r.list = list;
			r.options = static_cast<uint32_t>(rf.options);
			r.group = group;
			out.refs.push_back(r);
		}
		return nil;
	};
	if (id<MTLFunction> foreign = add(linked.functions, RGFX_LINKED_FUNCTIONS, nullptr))
		return foreign;
	if (id<MTLFunction> foreign = add(linked.binaryFunctions, RGFX_LINKED_BINARY_FUNCTIONS, nullptr))
		return foreign;
	NSDictionary<NSString *, NSArray<id<MTLFunction>> *> *groups = linked.groups;
	for (NSString *name in [groups.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		out.groups.emplace_back(name.UTF8String != nullptr ? name.UTF8String : "");
		if (id<MTLFunction> foreign = add(groups[name], RGFX_LINKED_GROUP, out.groups.back().c_str()))
			return foreign;
	}
	return add(linked.privateFunctions, RGFX_LINKED_PRIVATE_FUNCTIONS, nullptr);
}
