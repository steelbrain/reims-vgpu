/*
 * RgfxGLFunction: see RgfxGLFunction.h. Our own ObjC++; no Apple private SPI.
 *
 * What a function of GL IR answers is Metal's own function's (measured: macOS 26.7.1, the Intel
 * UHD 630's device, over the IR and plugin data of a vertex shader the renderer gave it): the
 * name `__main`, no label, the stage it was made with, MTLPatchTypeNone and a control point
 * count of 0, no vertex or stage input attributes, no function constants and no options; and
 * newArgumentEncoderWithBufferIndex: asserts, for buffer 0 of a shader that binds none, "bufferIndex
 * 0 does not identify an argument buffer". Each newFunctionWithPluginData:bitcodeType: makes a
 * new variant.
 *
 * A variant's lowering, and a tessellated draw's kernel's under each stage input descriptor, are
 * kept across the app's runs (RgfxGLStore.hpp), as Metal's compiler service keeps what it
 * compiled for an app, in the app's own cache directory beside Metal's:
 * confstr(_CS_DARWIN_USER_CACHE_DIR), then the app's bundle identifier (its program's name where
 * it has none), then ai.aneesiqbal.reims-graphics/gl. Metal keeps an app's libraries at the same
 * place, in com.apple.metal (measured: macOS 26.7.2, a sandboxed agent's
 * <cache>/com.apple.iconservicesagent/com.apple.iconservicesagent/com.apple.metal/32023/
 * libraries.data, the sandbox's confstr already the app's own), so a sandboxed app writes there
 * as Metal does for it. A variant the process asks for again is answered from its memory
 * (RgfxGLMemo), with or without a store.
 */
#import "RgfxGLFunction.h"
#import "RgfxInterop.h"
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include "RgfxGLLowering.hpp"
#include "RgfxGLStore.hpp"
#include "RgfxLog.hpp"

#import <objc/runtime.h>

#include <CommonCrypto/CommonDigest.h>
#include <climits>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>

namespace {

char kReturnDataKey;
char kStageInKey;

// Logs `why` once, after `what`.
void logOnce(const char *what, const std::string &why)
{
	static std::mutex lock;
	static std::set<std::string> said;
	{
		std::lock_guard<std::mutex> hold(lock);
		if (!said.insert(std::string(what) + why).second)
			return;
	}
	rgfxLogLine(@"RGFX: %s: %s", what, why.c_str());
}

void logOnce(const std::string &why)
{
	logOnce("a GL shader variant was not made; answered nil", why);
}

// The bytes of entries an app's store keeps (RgfxGLStore.hpp): 64 MiB. Heaven, the heaviest GL
// app measured, asks for 298 variants and 34 stage-in kernels over all its scenes, which its
// store keeps in 332 entries of 22.3 MB (measured: macOS 26.7.2, an RTX 3090), so the cap holds
// Heaven's variants three times over, and an app's directory holds no more.
constexpr uint64_t kStoreCap = 64ull << 20;

// The bytes of keys and variants a process's memo holds (RgfxGLMemo): 32 MiB, above the 22.3 MB
// of Heaven's.
constexpr uint64_t kMemoCap = 32ull << 20;

// What decides a lowering besides its inputs (RgfxGLLoweringIdentity), or empty where the
// lowering cannot run.
const std::string &loweringIdentity()
{
	static std::string identity;
	static std::once_flag once;
	std::call_once(once, [] {
		std::string why;
		if (!RgfxGLLoweringIdentity(identity, why))
			identity.clear();
	});
	return identity;
}

// The app's store of lowered variants, or null where it has none: no libraries to lower with, or
// no cache directory (said once).
RgfxGLStore *store()
{
	static RgfxGLStore *kept = nullptr;
	static std::once_flag once;
	std::call_once(once, [] {
		const std::string &identity = loweringIdentity();
		if (identity.empty())
			return;
		char cache[PATH_MAX];
		const size_t n = confstr(_CS_DARWIN_USER_CACHE_DIR, cache, sizeof cache);
		NSString *app = NSBundle.mainBundle.bundleIdentifier;
		std::string name = app.length != 0 ? app.UTF8String : getprogname() != nullptr ? getprogname() : "";
		if (n == 0 || n > sizeof cache || name.empty() || name.find('/') != std::string::npos || name[0] == '.') {
			rgfxLogLine(@"RGFX: GL shader variants are lowered in each run: no cache directory for this app");
			return;
		}
		std::string directory = cache;
		if (!directory.empty() && directory.back() != '/')
			directory += '/';
		kept = new RgfxGLStore(directory + name + "/ai.aneesiqbal.reims-graphics/gl", kStoreCap, identity);
	});
	return kept;
}

// The process's memo of lowered variants (RgfxGLMemo).
RgfxGLMemo *memo()
{
	static RgfxGLMemo *kept = new RgfxGLMemo(kMemoCap);
	return kept;
}

// Logs the store's first failure, once a process.
void logStoreFailure(RgfxGLStore *kept)
{
	std::string why;
	if (kept != nullptr && kept->takeFirstFailure(why))
		rgfxLogLine(@"RGFX: the store of lowered GL shader variants failed, and its variants are lowered where it "
		            @"fails: %s",
		            why.c_str());
}

} // namespace

// A kernel variant's module text, which takes a patch control point input, and its kernels made
// so far under each stage input descriptor (by RgfxGLStageIn::key), guarded by _lock.
@interface RgfxGLStageInVariant : NSObject
@end

@implementation RgfxGLStageInVariant {
@public
	std::string _text;
	std::string _storeKey; // the variant's key in the store (RgfxGLVariantKey)
	std::mutex _lock;
	NSMutableDictionary<NSData *, id<MTLFunction>> *_kernels;
}
@end

NSData *RgfxGLPluginReturnData(id<MTLFunction> function)
{
	return function != nil ? objc_getAssociatedObject(function, &kReturnDataKey) : nil;
}

void RgfxGLSetPluginReturnData(id<MTLFunction> function, NSData *data)
{
	objc_setAssociatedObject(function, &kReturnDataKey, data, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

BOOL RgfxGLStageInKernel(id<MTLFunction> function, MTLStageInputOutputDescriptor *stageInput, id<MTLFunction> *kernel,
                         NSError **error)
{
	RgfxGLStageInVariant *variant =
	    function != nil && stageInput != nil ? objc_getAssociatedObject(function, &kStageInKey) : nil;
	if (variant == nil)
		return NO;
	RgfxGLStageIn in;
	for (NSUInteger i = 0; i < 31; i++) {
		MTLAttributeDescriptor *a = stageInput.attributes[i];
		in.attributes[i].format = static_cast<uint32_t>(a.format);
		in.attributes[i].offset = static_cast<uint32_t>(a.offset);
		in.attributes[i].bufferIndex = static_cast<uint32_t>(a.bufferIndex);
		MTLBufferLayoutDescriptor *l = stageInput.layouts[i];
		in.layouts[i].stride = static_cast<uint32_t>(l.stride);
		in.layouts[i].stepFunction = static_cast<uint32_t>(l.stepFunction);
		in.layouts[i].stepRate = static_cast<uint32_t>(l.stepRate);
	}
	in.indexType = static_cast<uint32_t>(stageInput.indexType);
	in.indexBufferIndex = static_cast<uint32_t>(stageInput.indexBufferIndex);
	const std::string k = in.key();
	NSData *key = [NSData dataWithBytes:k.data() length:k.size()];
	std::lock_guard<std::mutex> hold(variant->_lock);
	id<MTLFunction> made = variant->_kernels[key];
	if (made == nil) {
		RgfxGLLowered lowered;
		std::string why;
		RgfxGLStore *kept = store();
		const bool keyed = !variant->_storeKey.empty();
		const bool specialized = RgfxGLLowerKept(
		    keyed ? kept : nullptr, keyed ? RgfxGLStageInKey(variant->_storeKey, k) : std::string(),
		    [&](RgfxGLLowered &out, std::string &reason) {
			    return RgfxGLSpecializeStageIn(variant->_text, in, out, reason);
		    },
		    lowered, why, nullptr, keyed ? memo() : nullptr);
		logStoreFailure(kept);
		if (specialized) {
			dispatch_data_t data = dispatch_data_create(lowered.metallib.data(), lowered.metallib.size(), nullptr,
			                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
			NSError *refusal = nil;
			id<MTLLibrary> library = [function.device newLibraryWithData:data error:&refusal];
			if (library == nil)
				why = std::string("the device made no library of it: ") +
				      (refusal != nil ? refusal.localizedDescription.UTF8String : "no error");
			else if ((made = [library newFunctionWithName:@(lowered.entry.c_str())]) == nil)
				why = "its library has no function " + lowered.entry;
		}
		if (made == nil) {
			logOnce("a GL kernel was not made under its pipeline's stage input descriptor; the pipeline is refused",
			        why);
			*kernel = nil;
			if (error)
				*error = RgfxMakeError(RgfxMetalErrorUnsupported,
				                       [NSString stringWithFormat:@"compute pipeline: the GL kernel under its stage "
				                                                  @"input descriptor was not made: %s",
				                                                  why.c_str()]);
			return YES;
		}
		variant->_kernels[key] = made;
	}
	*kernel = made;
	return YES;
}

@implementation RgfxGLFunction {
	id<MTLDevice> _device;
	MTLFunctionType _functionType;
	NSString *_label;
	std::mutex _labelLock;    // guards _label and _relocations
	NSArray *_relocations;    // relocations
	uint8_t _irHash[32];      // bitCodeHash: SHA-256 of _rgfxIR
}
RGFX_LOG_UNRECOGNIZED

@synthesize rgfxIR = _rgfxIR;
@synthesize rgfxInputsDescription = _rgfxInputsDescription;

- (instancetype)initWithIR:(const void *)ir
              functionType:(MTLFunctionType)functionType
         inputsDescription:(id)inputsDescription
                    device:(id<MTLDevice>)device
{
	const uint8_t *payload = nullptr;
	size_t length = 0;
	if (!RgfxGLIRPayload(ir, &payload, &length))
		return nil;
	if ((self = [super init]) == nil)
		return nil;
	_rgfxIR = [NSData dataWithBytes:ir length:sizeof(uint32_t) + length];
	CC_SHA256(_rgfxIR.bytes, static_cast<CC_LONG>(_rgfxIR.length), _irHash);
	_functionType = functionType;
	_rgfxInputsDescription = inputsDescription;
	_device = device;
	return self;
}

- (id<MTLFunction>)newFunctionWithPluginData:(NSData *)pluginData bitcodeType:(uint8_t)bitcodeType
{
	const uint8_t *payload = nullptr;
	size_t length = 0;
	RgfxGLIRPayload(_rgfxIR.bytes, &payload, &length);
	const uint8_t *plugin = static_cast<const uint8_t *>(pluginData.bytes);
	const size_t pluginLength = pluginData.length;
	RgfxGLLowered lowered;
	std::string why;
	RgfxGLStore *kept = store();
	const std::string &identity = loweringIdentity();
	const std::string key = !identity.empty() ? RgfxGLVariantKey(identity, payload, length, plugin, pluginLength,
	                                                             bitcodeType, static_cast<uint64_t>(_functionType))
	                                          : std::string();
	const bool made = RgfxGLLowerKept(
	    !key.empty() ? kept : nullptr, key,
	    [&](RgfxGLLowered &out, std::string &reason) {
		    return RgfxLowerGLIR(payload, length, plugin, pluginLength, out, reason);
	    },
	    lowered, why, nullptr, !key.empty() ? memo() : nullptr);
	logStoreFailure(kept);
	if (!made) {
		logOnce(why);
		return nil;
	}
	dispatch_data_t data = dispatch_data_create(lowered.metallib.data(), lowered.metallib.size(), nullptr,
	                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	NSError *error = nil;
	id<MTLLibrary> library = [_device newLibraryWithData:data error:&error];
	if (library == nil) {
		logOnce(std::string("the device made no library of it: ") +
		        (error != nil ? error.localizedDescription.UTF8String : "no error"));
		return nil;
	}
	id<MTLFunction> variant = [library newFunctionWithName:@(lowered.entry.c_str())];
	if (variant == nil) {
		logOnce("its library has no function " + lowered.entry);
		return nil;
	}
	if (!lowered.returnData.empty())
		RgfxGLSetPluginReturnData(variant, [NSData dataWithBytes:lowered.returnData.data()
		                                                  length:lowered.returnData.size()]);
	if (!lowered.stageInText.empty()) {
		RgfxGLStageInVariant *stageIn = [RgfxGLStageInVariant new];
		stageIn->_text = std::move(lowered.stageInText);
		stageIn->_storeKey = key;
		stageIn->_kernels = [NSMutableDictionary dictionary];
		objc_setAssociatedObject(variant, &kStageInKey, stageIn, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	}
	return variant;
}

- (NSString *)label
{
	std::lock_guard<std::mutex> hold(_labelLock);
	return _label;
}

- (void)setLabel:(NSString *)label
{
	NSString *copy = [label copy];
	std::lock_guard<std::mutex> hold(_labelLock);
	_label = copy;
}

- (id<MTLDevice>)device
{
	return _device;
}

- (MTLFunctionType)functionType
{
	return _functionType;
}

- (MTLPatchType)patchType
{
	return MTLPatchTypeNone;
}

- (NSInteger)patchControlPointCount
{
	return 0;
}

- (NSArray<MTLVertexAttribute *> *)vertexAttributes
{
	return nil;
}

- (NSArray<MTLAttribute *> *)stageInputAttributes
{
	return nil;
}

- (NSString *)name
{
	return @"__main";
}

- (NSDictionary<NSString *, MTLFunctionConstant *> *)functionConstantsDictionary
{
	return @{};
}

- (MTLFunctionOptions)options
{
	return MTLFunctionOptionNone;
}

// Metal's function of GL IR has no arguments its encoders know: it asserts (measured, buffer 0),
// and so does ours, for any index, with its words.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
{
	RgfxMetalAssertionFailed("-[_MTLFunction newArgumentEncoderWithBufferIndex:reflection:functionReflection:]",
	                         [NSString stringWithFormat:@"bufferIndex %lu does not identify an argument buffer",
	                                                    static_cast<unsigned long>(bufferIndex)]);
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
{
	(void)reflection;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex];
}

// --- MTLFunctionSPI ----------------------------------------------------------------------
// What Metal's function of GL IR answers (measured: macOS 26.7.1, the Intel 630's device, the
// functions of a vertex and a fragment shader the renderer gave it; tools/gl_probe.m -s): it is a
// function of a library, _MTLFunctionInternal, with no arguments, bindings, return type, debug
// instrumentation, file, line, imports, relocations or render target array index type.

- (NSArray *)arguments
{
	return nil;
}

- (NSArray *)bindings
{
	return nil;
}

- (id)returnType
{
	return nil;
}

- (NSData *)debugInstrumentationData
{
	return nil;
}

- (NSString *)filePath
{
	return nil;
}

- (NSString *)unpackedFilePath
{
	return nil;
}

- (NSInteger)lineNumber
{
	return 0;
}

- (NSArray *)importedLibraries
{
	return nil;
}

- (NSArray *)importedSymbols
{
	return nil;
}

- (NSUInteger)renderTargetArrayIndexType
{
	return 0;
}

// Its relocations: none until set, then the same object, as Metal's function keeps them.
- (NSArray *)relocations
{
	std::lock_guard<std::mutex> hold(_labelLock);
	return _relocations;
}

- (void)setRelocations:(NSArray *)relocations
{
	std::lock_guard<std::mutex> hold(_labelLock);
	_relocations = relocations;
}

// The IR's bytes after its length word, as Metal's function answers them (measured: 810 and 628
// bytes, the two shaders' IR lengths).
- (dispatch_data_t)bitcodeData
{
	const uint8_t *payload = nullptr;
	size_t length = 0;
	RgfxGLIRPayload(_rgfxIR.bytes, &payload, &length);
	return dispatch_data_create(payload, length, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
}

// A hash of the function's IR, its length word with it: never null, as Metal's (measured), and the
// same for the same IR.
- (const void *)bitCodeHash
{
	return _irHash;
}

// Metal's base function ends the process here ("Driver needs to override this"; read from code,
// _MTLFunction, macOS 27.0.1), as its function of GL IR does, which answers it from the base
// (measured: the member's owner), and so does this.
- (id)functionHandle
{
	RgfxMetalAssertionFailed("-[_MTLFunction functionHandle]", @"Driver needs to override this");
}

// The reflection of a function of GL IR: the renderer takes a variant's from its pipeline's
// (newFunctionWithPluginData:bitcodeType:), so none is made of the function itself. Its forms that
// take an archive, a pipeline library or a handler are src/RgfxSpiForwarders.mm's.
- (id)reflectionWithOptions:(NSUInteger)options
{
	(void)options;
	RGFX_UNSUPPORTED("the reflection of a function of GL IR");
	return nil;
}

@end
