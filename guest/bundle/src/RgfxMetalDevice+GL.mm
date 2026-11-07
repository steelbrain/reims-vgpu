/*
 * RgfxMetalDevice (GL): see the header. Our own ObjC++.
 *
 * Each member's type encoding is Metal's base device's (`@32@0:8^v16Q24` without an inputs
 * description, `@40@0:8^v16@24Q32` with one), and each makes the same function: the IR is the
 * same GL IR for the three, which the GL code generator lowers alike (RgfxGLLowering.hpp).
 */

#import "RgfxMetalDevice+GL.h"

#import "RgfxGLFunction.h"

@implementation RgfxMetalDevice (GL)

- (id<MTLFunction>)rgfxFunctionWithGLIR:(void *)ir inputs:(id)inputs functionType:(NSUInteger)functionType
{
	return [[RgfxGLFunction alloc] initWithIR:ir
	                             functionType:static_cast<MTLFunctionType>(functionType)
	                        inputsDescription:inputs
	                                   device:self];
}

- (id<MTLFunction>)newFunctionWithGLIR:(void *)ir functionType:(NSUInteger)functionType
{
	return [self rgfxFunctionWithGLIR:ir inputs:nil functionType:functionType];
}

- (id<MTLFunction>)newFunctionWithGLIR:(void *)ir
                     inputsDescription:(id)inputsDescription
                          functionType:(NSUInteger)functionType
{
	return [self rgfxFunctionWithGLIR:ir inputs:inputsDescription functionType:functionType];
}

- (id<MTLFunction>)newFunctionWithGLCoreIR:(void *)ir functionType:(NSUInteger)functionType
{
	return [self rgfxFunctionWithGLIR:ir inputs:nil functionType:functionType];
}

- (id<MTLFunction>)newFunctionWithGLCoreIR:(void *)ir
                         inputsDescription:(id)inputsDescription
                              functionType:(NSUInteger)functionType
{
	return [self rgfxFunctionWithGLIR:ir inputs:inputsDescription functionType:functionType];
}

- (id<MTLFunction>)newFunctionWithGLESIR:(void *)ir functionType:(NSUInteger)functionType
{
	return [self rgfxFunctionWithGLIR:ir inputs:nil functionType:functionType];
}

- (id<MTLFunction>)newFunctionWithGLESIR:(void *)ir
                       inputsDescription:(id)inputsDescription
                            functionType:(NSUInteger)functionType
{
	return [self rgfxFunctionWithGLIR:ir inputs:inputsDescription functionType:functionType];
}

@end
