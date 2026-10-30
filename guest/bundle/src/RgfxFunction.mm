/*
 * RgfxFunction: see RgfxFunction.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxFunction.h"

#include <string>
#include <vector>

@implementation RgfxFunction {
	uint64_t _libraryObjId;
	std::string _name;                          // the entry point, owned
	std::vector<RgfxFunctionConstant> _constants; // specialisation constants, owned
	MTLFunctionType _type;                        // the stage, from the library's table
	BOOL _haveType;
}

+ (instancetype)functionWithLibrary:(uint64_t)libraryObjId
                               name:(NSString *)name
                          constants:(const RgfxFunctionConstant *)constants
                              count:(uint32_t)count
                             device:(id<MTLDevice>)device
                         serializer:(RgfxSerializer *)serializer
{
	return [[RgfxFunction alloc] initWithLibrary:libraryObjId
	                                        name:name
	                                   constants:constants
	                                       count:count
	                                      device:device
	                                  serializer:serializer];
}

- (instancetype)initWithLibrary:(uint64_t)libraryObjId
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
		_libraryObjId = libraryObjId;
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
	// The function's constant table lives in the AIR the host compiles; the shell
	// does not read it, so it reports none.
	return @{};
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
	return MTLFunctionOptionNone;
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
{
	(void)bufferIndex;
	return nil; // needs the function's argument reflection, which the shell does not read
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
{
	if (reflection)
		*reflection = nil;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex];
}

@end
