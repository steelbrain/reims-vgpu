// RgfxReflection: the reflection Metal answers for a stitchable function (its -returnType, and its
// -arguments, which its -bindings is), built from the function's signature in its AIR module
// (RgfxAirArguments.hpp, rgfxReadAirVisible).
//
// Metal makes that reflection with its device's compiler, which our device does not have, so its
// functions answer it from here. It is made of Metal's own reflection objects, each through its
// class's initialiser, holding what Metal's own compiler gives the same function (measured:
// macOS 27 on an M3 Max and macOS 26.7.1 on an AMD Radeon Pro 5300M agree, over the stitchable
// functions of tests/air_arguments.metal; tools/reflection_check.mm compares ours with theirs
// getter by getter):
//   - the return type: a type object of the return's type; a void return's is of data type
//     kVoidDataType;
//   - one binding per argument, in order, named as the argument, of access 0, index 0 and array
//     length 0, inactive and unused: a scalar, vector, matrix or metal::array is of binding type
//     kValueBindingType and its type object; a struct of kStructBindingType and its struct type;
//     a pointer of kPointerBindingType and its pointer type; a texture is a texture binding of
//     its sample type's data type, its texture type, access and depth; a sampler is of binding
//     type 3 and no type object;
//   - a scalar's, vector's or matrix's type object is of its data type (a packed vector's is its
//     vector's), a metal::array's of none (MTLDataTypeNone);
//   - a struct type holds one member per field, each of its name, offset and data type, of
//     pixel format, ALU type, argument index and raster order group 0 and render target
//     0xFFFFFFFF, with the type object of a struct, array, pointer or texture field, and none for
//     another. It is named as the struct, but nameless where it is a field's type or a field's
//     pointee;
//   - an array type is of its length, its element type and a stride of the array's alignment,
//     with a type object of its element type alone (a texture type of that data type where its
//     elements are textures). Its element type is the element's data type, but
//     kArrayVectorElement for vectors and kArrayMatrixElement for matrices;
//   - a pointer type is of its pointee's data type, size and alignment, of access 0, neither a
//     constant buffer nor an argument buffer, with the pointee's type object: a struct's struct
//     type, an array's type object of its data type alone, and another's type object. A field's
//     pointee is described otherwise: an array's type object is of its element type, and one
//     that is no struct's or array's is of none;
//   - a texture type is of its sample type's data type, its texture type, its access (0 to sample
//     or read it, 2 to write it, 1 for both) and whether it is a depth texture.
#ifndef RGFX_REFLECTION_H_
#define RGFX_REFLECTION_H_

#import <Foundation/Foundation.h>

#include "RgfxAirArguments.hpp"

// Metal's answers for the stitchable function `fn` (its signature read: Signature::Read): its
// return type in `*returnType` and its arguments in `*arguments`. False where Metal has no class
// or initialiser the answer needs, with `*why` naming it; nothing is set then.
bool rgfxStitchableReflection(const RgfxAirVisible &fn, id __strong *returnType,
                              NSArray *__strong *arguments, NSString *__strong *why);

#endif // RGFX_REFLECTION_H_
