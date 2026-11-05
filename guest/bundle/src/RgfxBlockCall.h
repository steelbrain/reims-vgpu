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
/// the creation's reflection. A handler of any other shape, or with no signature, is not called;
/// that is logged, once per selector, with `cmd`, and answers NO. A nil handler answers YES.
BOOL RgfxCallCreationHandler(id _Nullable handler, id _Nullable object, NSError *_Nullable error,
                             SEL cmd);

/// As RgfxCallCreationHandler, a handler of three object arguments given `reflection` second.
BOOL RgfxCallCreationHandlerWithReflection(id _Nullable handler, id _Nullable object,
                                           id _Nullable reflection, NSError *_Nullable error, SEL cmd);

/// Runs `creation`, an asynchronous creator's create and its call of its completion handler, after
/// the creator returns: on the global concurrent queue of default QoS, in an autorelease pool of
/// its own. That is where Metal runs them on an Apple M3 Max (macOS 27.0.1: the creator returns,
/// and its handler runs about 0.2 ms later on `com.apple.root.default-qos`); an AMD Radeon Pro
/// 5300M and an Intel UHD 630 (macOS 26.7.1) run theirs on a serial queue of the device's, about
/// 60 ms after the creator returned (docs/graphics-kext/dgpu-oracle.md, Q7). Never on the
/// caller's thread, so a handler that waits on the caller's own serial queue goes on. The
/// creator copies what it hands `creation` that its caller may change once it returns (the
/// descriptor), and the handler, which may be a block on the caller's stack.
void RgfxRunCreation(dispatch_block_t creation);

/// `object`'s copy where it can be copied (it answers `copyWithZone:`), else `object` itself:
/// a creator's descriptor of a type no header names, for RgfxRunCreation.
id _Nullable RgfxCopyOf(id _Nullable object);

/// Waits until every creation RgfxRunCreation started has ended and its block is released, or
/// `timeout` passes: for the tests, whose objects must outlive the creations that use them. YES
/// once none runs.
BOOL RgfxWaitForCreations(dispatch_time_t timeout);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_BLOCK_CALL_H_
