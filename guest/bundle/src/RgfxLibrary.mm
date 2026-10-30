/*
 * RgfxLibrary: see RgfxLibrary.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxLibrary.h"

#import "RgfxFunction.h"

#include "RgfxSerializer.hpp"

@implementation RgfxLibrary

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
	return [[RgfxLibrary alloc] initWithObjId:created.obj_id device:device serializer:serializer];
}

// --- MTLLibrary --------------------------------------------------------------

- (id<MTLFunction>)newFunctionWithName:(NSString *)functionName
{
	if (functionName == nil) {
		return nil;
	}
	return [RgfxFunction functionWithLibrary:self.rgfxObjId
	                                    name:functionName
	                               constants:nullptr
	                                   count:0
	                                  device:self.device
	                              serializer:[self serializer]];
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                                 error:(__autoreleasing NSError **)error
{
	// DOCUMENTED GAP. MTLFunctionConstantValues is write-only: it has no public
	// getter for the values set on it, so the shell cannot extract the constants to
	// carry on the pipeline create's wire. We vend the named function without
	// constants rather than fabricate values. (Follow-up in the report: a faithful
	// specialised-function path needs either a readable constant-values store or a
	// function-constant wire carried here.)
	(void)constantValues;
	if (error) {
		*error = nil;
	}
	return [self newFunctionWithName:name];
}

- (NSArray<NSString *> *)functionNames
{
	// DOCUMENTED PLACEHOLDER. The shell does not parse the metallib's symbol table;
	// the host holds the real function list. Returns an empty array until a library
	// reflection reply is carried. Callers name functions directly by string.
	return @[];
}

- (MTLLibraryType)type
{
	return MTLLibraryTypeExecutable;
}

@end
