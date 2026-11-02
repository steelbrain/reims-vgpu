/*
 * RgfxBlockCall: calling a block whose type the caller chose, for the members of Metal's device
 * that no public header declares: the types of their blocks (a creation's completion handler, a
 * buffer's deallocator) are not ours to know, so the bundle reads each block's own signature
 * (clang's block ABI: a block made with a signature, BLOCK_HAS_SIGNATURE, carries its Objective-C
 * type encoding) and calls it with the arguments that signature states, rather than guessing them.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_BLOCK_CALL_H_
#define RGFX_BLOCK_CALL_H_

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/// The types of the arguments `block` takes by its own signature, the block itself left out:
/// each its encoding with any class name dropped (`@`, `@?`, `^v`, `Q`). Nil where the block is
/// nil or carries no signature.
NSArray<NSString *> *_Nullable RgfxBlockArgumentTypes(id _Nullable block);

/// Calls `handler`, a creation's completion handler, with `object` and `error`: (object, error)
/// for a handler of two object arguments, (object, nil, error) for one of three, whose second is
/// the creation's reflection, which the device does not give. A handler of any other shape, or
/// with no signature, is not called; that is logged, once per selector, with `cmd`, and answers
/// NO. A nil handler answers YES.
BOOL RgfxCallCreationHandler(id _Nullable handler, id _Nullable object, NSError *_Nullable error,
                             SEL cmd);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_BLOCK_CALL_H_
