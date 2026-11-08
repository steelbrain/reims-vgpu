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
 */
#import "RgfxGLFunction.h"
#import "RgfxInterop.h"
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include "RgfxGLLowering.hpp"

#import <objc/runtime.h>

#include <CommonCrypto/CommonDigest.h>
#include <mutex>
#include <set>
#include <string>

namespace {

char kReturnDataKey;

void logOnce(const std::string &why)
{
	static std::mutex lock;
	static std::set<std::string> said;
	{
		std::lock_guard<std::mutex> hold(lock);
		if (!said.insert(why).second)
			return;
	}
	NSLog(@"RGFX: a GL shader variant was not made: %s; answered nil", why.c_str());
}

} // namespace

NSData *RgfxGLPluginReturnData(id<MTLFunction> function)
{
	return function != nil ? objc_getAssociatedObject(function, &kReturnDataKey) : nil;
}

void RgfxGLSetPluginReturnData(id<MTLFunction> function, NSData *data)
{
	objc_setAssociatedObject(function, &kReturnDataKey, data, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
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
	(void)bitcodeType;
	const uint8_t *payload = nullptr;
	size_t length = 0;
	RgfxGLIRPayload(_rgfxIR.bytes, &payload, &length);
	RgfxGLLowered lowered;
	std::string why;
	if (!RgfxLowerGLIR(payload, length, static_cast<const uint8_t *>(pluginData.bytes), pluginData.length, lowered,
	                   why)) {
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
