/*
 * RgfxFunction: see RgfxFunction.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxFunction.h"

#import "RgfxArgumentEncoder.h"
#import "RgfxLibrary.h"
#import "RgfxUnsupported.h"

#import "RgfxMetalError.h"
#import "RgfxPipelineReflection.h"
#import "RgfxReflection.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <CommonCrypto/CommonDigest.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// Logs, the first time `cls` hands `cmd` to a shadow, that Metal's own function answered it.
void logAnsweredByShadow(Class cls, SEL cmd)
{
	static std::mutex lock;
	static std::set<std::pair<Class, SEL>> logged;
	{
		std::lock_guard<std::mutex> hold(lock);
		if (!logged.insert({cls, cmd}).second)
			return;
	}
	NSLog(@"RGFX: answered by Metal's own function: -[%s %s]", class_getName(cls), sel_getName(cmd));
}

} // namespace

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
	MTLPatchType _patchType;                      // the [[patch]] type and control point
	NSInteger _patchControlPointCount;            //   count, from the library's table
	MTLFunctionOptions _options;                  // the descriptor's, where made from one
	std::string _specializedName;                 // the descriptor's, where it gave one
	BOOL _haveSpecializedName;
	id<MTLFunction> _shadow;                      // Metal's own, for Core Image's link
	MTLFunctionConstantValues *_shadowValues;     // the values it was made with, for its shadow
	std::mutex _shadowLock;                       // guards the two above, set once in use
	std::shared_ptr<const RgfxAirFunction> _air;  // its arguments, from the library's bitcode
	std::shared_ptr<const RgfxAirVisible> _visible; // its signature, where it is a visible one
	std::once_flag _reflected;                      // the two below, made once from _visible
	id _returnType;                                 // what -returnType answers
	NSArray *_arguments;                            // what -arguments and -bindings answer
	BOOL _reflectedHere;                            // whether the two above are the answers
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
		_patchType = MTLPatchTypeNone;
		_patchControlPointCount = -1;
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

- (BOOL)rgfxFunctionTypeKnown
{
	return _haveType;
}

- (void)rgfxSetPatchType:(MTLPatchType)type controlPointCount:(NSInteger)count
{
	_patchType = type;
	_patchControlPointCount = count;
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
	if (_haveSpecializedName)
		return [NSString stringWithUTF8String:_specializedName.c_str()];
	return [NSString stringWithUTF8String:_name.c_str()];
}

- (MTLFunctionType)functionType
{
	if (_haveType)
		return _type; // the .metallib's TYPE tag, or its library's shadow's answer
	// A function of a library whose function list did not read and that has no shadow to ask
	// (RgfxLibrary): nothing on the guest tells its stage, and a pipeline create names the
	// function by its place in the descriptor, so this answers Vertex.
	return MTLFunctionTypeVertex;
}

- (NSDictionary<NSString *, MTLFunctionConstant *> *)functionConstantsDictionary
{
	// The constants the library's table declares, for a function made with no values; none
	// for one specialised by values, as Metal answers (measured: macOS 27 Metal).
	return _unspecialised ?: @{};
}

// A vertex function's attributes, and a kernel's stage-in attributes, which Metal reads from the
// function's reflection: the bundle reads none of it from the library, so it answers nil, logged.
- (NSArray<MTLVertexAttribute *> *)vertexAttributes
{
	RGFX_UNSUPPORTED("a function's vertex attributes (its reflection is not read)");
	return nil;
}

- (NSArray<MTLAttribute *> *)stageInputAttributes
{
	RGFX_UNSUPPORTED("a function's stage-in attributes (its reflection is not read)");
	return nil;
}

// The MTLFunction SPI reflectionWithOptions:, which the system's GL renderer sends each kernel of
// an OpenCL program it builds (gldBuildComputeProgram, with options 3, reading the arguments'
// access, type, dataTypeDescription's isConstantBuffer and bufferDataSize; read from its code):
// the function's arguments, as a pipeline's reflection of it lists them, in Metal's
// MTLFunctionReflectionInternal (RgfxFunctionReflectionOf). Metal's answers one for options 0, 1
// and 3 alike (measured: macOS 26.7.1, the AMD 5300M and the Intel 630), whose contents for 0 and
// 1 were not compared: the answer is the same for every value. nil, logged, where it is not made.
- (id)reflectionWithOptions:(NSUInteger)options
{
	(void)options;
	return RgfxFunctionReflectionOf(self);
}

- (MTLPatchType)patchType
{
	return _patchType;
}

- (NSInteger)patchControlPointCount
{
	return _patchControlPointCount;
}

- (MTLFunctionOptions)options
{
	return _options;
}

- (void)rgfxSetOptions:(MTLFunctionOptions)options
{
	_options = options;
}

// Metal's function keeps what it is given, and answers it from options (measured: macOS 27).
- (void)setOptions:(MTLFunctionOptions)options
{
	_options = options;
}

- (void)rgfxSetSpecializedName:(NSString *)specializedName
{
	_specializedName = specializedName.UTF8String ? specializedName.UTF8String : "";
	_haveSpecializedName = YES;
}

- (const char *)rgfxSpecializedNameUTF8
{
	return _haveSpecializedName ? _specializedName.c_str() : nullptr;
}

- (void)rgfxSetShadow:(id<MTLFunction>)shadow
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	_shadow = shadow;
}

- (id<MTLFunction>)rgfxShadow
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	return _shadow;
}

- (void)rgfxSetShadowConstantValues:(MTLFunctionConstantValues *)values
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	_shadowValues = [values copy];
}

// The shadow of a function made with constant values or a specialized name is Metal's own
// function of the same descriptor: its stitch finds a function by the name a graph's node gives,
// the specialized one where there is one, and links the specialised body. The descriptor carries
// no options and no archives, as the stitch over the shadows carries none
// (RgfxStitchOverShadows): Metal reads the shadows' AIR alone.
- (id<MTLFunction>)rgfxShadowOrMake
{
	std::lock_guard<std::mutex> hold(_shadowLock);
	if (_shadow != nil || ![_library isKindOfClass:[RgfxLibrary class]])
		return _shadow;
	id<MTLLibrary> metal = [(RgfxLibrary *)_library rgfxShadow];
	if (metal == nil)
		return nil;
	NSString *entry = [NSString stringWithUTF8String:_name.c_str()];
	if (_constants.empty() && _shadowValues == nil && !_haveSpecializedName) {
		_shadow = [metal newFunctionWithName:entry];
		return _shadow;
	}
	if (!_constants.empty() && _shadowValues == nil)
		return nil; // specialised by values this function does not hold: no shadow, not a wrong one
	MTLFunctionDescriptor *d = [MTLFunctionDescriptor functionDescriptor];
	d.name = entry;
	d.constantValues = _shadowValues;
	if (_haveSpecializedName)
		d.specializedName = [NSString stringWithUTF8String:_specializedName.c_str()];
	NSError *error = nil;
	_shadow = [metal newFunctionWithDescriptor:d error:&error];
	if (_shadow == nil) {
		static std::once_flag logged;
		std::call_once(logged, [&] {
			NSLog(@"RGFX: %s: Metal's own library made no function of its constant values: %@ "
			      @"(logged once a process)",
			      _name.c_str(), error.localizedDescription);
		});
	}
	return _shadow;
}

// The shadow's answer to `cmd`, Metal's own function's of the same entry point, logged once per
// class as forwarded selectors are; nil, logged, where there is none to ask.
- (id)rgfxShadowAnswer:(SEL)cmd
{
	id<MTLFunction> shadow = [self rgfxShadowOrMake];
	if (![shadow respondsToSelector:cmd]) {
		RgfxLogUnsupported([self class], cmd,
		                   "no function of Metal's own answers it (one of a library Metal's own "
		                   "does not hold, or that Metal's own library made none of)");
		return nil;
	}
	logAnsweredByShadow([self class], cmd);
	return ((id (*)(id, SEL))objc_msgSend)(shadow, cmd);
}

// --- MTLFunctionSPI, as Metal's own function answers it -----------------------------------
// Core Image and MPS read these. Metal's answers depend on the function (measured: macOS 27):
// nil arguments, imported libraries and return type for a plain library function, the bindings
// and the type of a stitchable one, the declared constants of one made with no values for them,
// and the function's AIR as its bitcode data. The shadow, Metal's own function of the same entry
// point, answers them here, but a visible function's reflection, which Metal makes with its
// device's compiler (RgfxReflection.h): Metal's own function on our device has none, and answers
// nil.

// A visible function's reflection, made once from its signature: false where Metal's own
// function answers it (no signature read, or none Metal's classes could be made of, logged once a
// process).
- (BOOL)rgfxReflected
{
	std::call_once(_reflected, [self] {
		if (_visible == nullptr || _visible->signature == RgfxAirVisible::Signature::Unknown)
			return;
		if (_visible->signature == RgfxAirVisible::Signature::None) {
			// A visible function that is not stitchable (measured: macOS 27, M3 Max; macOS 26.7.1,
			// AMD Radeon Pro 5300M).
			_arguments = @[];
			_reflectedHere = YES;
			return;
		}
		NSString *why = nil;
		id returnType = nil;
		NSArray *arguments = nil;
		if (!rgfxStitchableReflection(*_visible, &returnType, &arguments, &why)) {
			static std::once_flag logged;
			std::call_once(logged, [&] {
				NSLog(@"RGFX: %s: no reflection made (%@): Metal's own function answers it (logged "
				      @"once a process)",
				      _name.c_str(), why);
			});
			return;
		}
		_returnType = returnType;
		_arguments = arguments;
		_reflectedHere = YES;
	});
	return _reflectedHere;
}

- (NSArray *)arguments
{
	return [self rgfxReflected] ? _arguments : [self rgfxShadowAnswer:_cmd];
}

// The same object as -arguments (measured, as -arguments).
- (NSArray *)bindings
{
	return [self rgfxReflected] ? _arguments : [self rgfxShadowAnswer:_cmd];
}

- (NSArray *)importedLibraries
{
	return [self rgfxShadowAnswer:_cmd];
}

- (id)returnType
{
	return [self rgfxReflected] ? _returnType : [self rgfxShadowAnswer:_cmd];
}

- (id)bitcodeData
{
	return [self rgfxShadowAnswer:_cmd];
}

// Metal answers nil for a function specialised by values (measured).
- (NSArray *)functionConstants
{
	if (!_constants.empty())
		return nil;
	return [self rgfxShadowAnswer:_cmd];
}

// Metal answers nil for every function, one made of a descriptor with plugin data too (measured):
// its setPluginData: ends the process.
- (id)pluginData
{
	return nil;
}

// Metal's function aborts here, for any argument, nil too (measured: macOS 27), and so does ours.
- (void)setPluginData:(id)pluginData
{
	(void)pluginData;
	NSLog(@"RGFX: -[%s setPluginData:] ends the process, as Metal's function does",
	      class_getName([self class]));
	abort();
}

// What this class does not implement goes to the shadow, Metal's own function of the same entry
// point, where it answers it (rgfxSetShadow:); nothing else answers in its place.
- (id)forwardingTargetForSelector:(SEL)aSelector
{
	id<MTLFunction> shadow = [self rgfxShadow];
	if (shadow != nil && [shadow respondsToSelector:aSelector]) {
		logAnsweredByShadow([self class], aSelector);
		return shadow;
	}
	return [super forwardingTargetForSelector:aSelector];
}

- (BOOL)respondsToSelector:(SEL)aSelector
{
	if ([super respondsToSelector:aSelector])
		return YES;
	id<MTLFunction> shadow = [self rgfxShadow];
	return shadow != nil && [shadow respondsToSelector:aSelector];
}

- (RgfxResource *)rgfxLibrary
{
	return _library;
}

- (void)rgfxSetAirFunction:(std::shared_ptr<const RgfxAirFunction>)air
{
	_air = std::move(air);
}

- (std::shared_ptr<const RgfxAirFunction>)rgfxAirFunction
{
	return _air;
}

- (void)rgfxSetAirVisible:(std::shared_ptr<const RgfxAirVisible>)visible
{
	_visible = std::move(visible);
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

// The encoder with the buffer's reflection: the binding of that buffer index in the function's
// reflection, which its library answers from Metal's own over the same bytes, as both x86 Macs
// answer one there, an MTLBufferBindingInternal (oracle_bundle argbuf.reflection: the AMD 5300M
// and the Intel 630 under macOS 26.7.1). Their arrays' strides answer 0 there and the size of an
// element in the library's reflection, which this answers. nil, logged, where the library keeps
// no source to reflect.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
{
	id<MTLArgumentEncoder> encoder = [self newArgumentEncoderWithBufferIndex:bufferIndex];
	if (reflection)
		*reflection = encoder != nil ? [self rgfxArgumentOfBufferAtIndex:bufferIndex] : nil;
	return encoder;
}

- (MTLArgument *)rgfxArgumentOfBufferAtIndex:(NSUInteger)bufferIndex
{
	RgfxResource *library = [self rgfxLibrary];
	const char *entry = [self rgfxEntryPointUTF8];
	NSString *name = entry != nullptr ? @(entry) : nil;
	if (name != nil && [library respondsToSelector:@selector(reflectionForFunctionWithName:)]) {
		MTLFunctionReflection *reflection =
		    [(id<MTLLibrary>)library reflectionForFunctionWithName:name];
		for (id<MTLBinding> binding in reflection.bindings) {
			if (binding.type == MTLBindingTypeBuffer && binding.index == bufferIndex &&
			    [(id)binding isKindOfClass:[MTLArgument class]])
				return (MTLArgument *)binding;
		}
	}
	RGFX_UNSUPPORTED("an argument buffer's reflection, where the function's library has none");
	return nil;
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
			r.specialized_name = [rf rgfxSpecializedNameUTF8];
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
