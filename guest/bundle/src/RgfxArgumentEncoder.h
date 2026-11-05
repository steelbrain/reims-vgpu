/*
 * RgfxArgumentEncoder: the reims-graphics bundle's MTLArgumentEncoder.
 *
 * An encoder is a layout and the argument buffer it writes into. The layout is the host's: the
 * device asks ARGUMENT_LAYOUT of the descriptors (newArgumentEncoderWithArguments:) or of a
 * function's buffer argument (newArgumentEncoderWithBufferIndex:), and the host writes the layout
 * its platform's encoder has into a block of the device's region (RgfxArgumentLayout.hpp). A
 * nested argument buffer's encoder (newArgumentEncoderForBufferAtIndex:) takes its layout from
 * the same answer. Nothing of an encoder lives on the host.
 *
 * The encoder writes each argument itself, at the place the layout gives its index, in the
 * argument buffer's contents: 8 bytes, little endian, a buffer's gpuAddress plus the offset, and
 * a texture's, sampler's or pipeline state's gpuResourceID, the words the host's GPU reads, since
 * the host takes the guest's addresses and resource ids as they are; a nil object writes 0. As
 * plain Metal does, an index no argument has writes at the arguments' offset 0, and a range
 * writes its words in turn. A setter with no argument buffer set writes nothing, and logs it
 * (plain Metal's process ends there). constantDataAtIndex: answers the place itself.
 *
 * A sampler's word is the resource id the host's device numbered it by (RgfxSampler.h). What the
 * device makes no object of (acceleration structures, function tables) only nil reaches, which
 * writes 0. A depth/stencil state has no resource id, as on the x86 Macs: setting one raises an
 * unrecognized selector at its gpuResourceID (RgfxDepthStencilState.h).
 *
 * An encoder of a buffer binding (newArgumentEncoderWithBufferBinding:) is laid out as Metal lays
 * one out, from the argument buffer the binding describes: the function's own encoder where the
 * binding is one of the device's reflections of a function's buffer (RgfxNoteArgumentBinding), as
 * both x86 Macs answer the same encoder for a function's buffer and for its library's reflection
 * of it; otherwise the encoder of descriptors of the binding's struct members.
 *
 * Our own ObjC++: it conforms to MTLArgumentEncoder but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_ARGUMENT_ENCODER_H_
#define RGFX_ARGUMENT_ENCODER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include <memory>

#include "RgfxArgumentLayout.hpp"

class RgfxRegion;
class RgfxSerializer;
#endif

@class RgfxFunction;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxArgumentEncoder : RgfxResource <MTLArgumentEncoder>

#ifdef __cplusplus
/// An encoder of `layout`. `device`/`serializer` are the owning device's (nil in a unit test).
- (instancetype)initWithLayout:(std::shared_ptr<const RgfxArgLayout>)layout
                        device:(nullable id<MTLDevice>)device
                    serializer:(nullable RgfxSerializer *)serializer;

/// The encoder of `arguments` (MTLArgumentDescriptor), laid out by the host into a block of
/// `region`; nil, logged, where the host answers none (descriptors Metal refuses) or no answer
/// comes back.
+ (nullable instancetype)encoderOfArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                                     region:(nullable RgfxRegion *)region
                                     device:(nullable id<MTLDevice>)device
                                 serializer:(nullable RgfxSerializer *)serializer;

/// The encoder of the argument buffer `function` takes at `bufferIndex`, laid out as
/// encoderOfArguments: lays out; nil where the function takes none there.
+ (nullable instancetype)encoderOfFunction:(RgfxFunction *)function
                               bufferIndex:(NSUInteger)bufferIndex
                                    region:(nullable RgfxRegion *)region
                                    device:(nullable id<MTLDevice>)device
                                serializer:(nullable RgfxSerializer *)serializer;

/// The encoder of the argument buffer `binding` describes: the encoder of the function's buffer
/// it was noted for (RgfxNoteArgumentBinding), or of descriptors of its struct's members; nil, as
/// the AMD 5300M answers, where the binding is not of an argument buffer, and nil, logged, where
/// a member is of a type no descriptor names (a struct).
+ (nullable instancetype)encoderOfBinding:(id<MTLBufferBinding>)binding
                                   region:(nullable RgfxRegion *)region
                                   device:(nullable id<MTLDevice>)device
                               serializer:(nullable RgfxSerializer *)serializer;

/// The layout the encoder writes by.
@property(nonatomic, readonly) std::shared_ptr<const RgfxArgLayout> rgfxLayout;
#endif

@end

/// Notes `binding`, a reflection the device answers of a buffer `function` takes, so an encoder of
/// it is the function's own encoder of that buffer (encoderOfBinding:). Held while the binding
/// lives.
void RgfxNoteArgumentBinding(id<MTLBinding> binding, RgfxFunction *function);

/// The descriptors of the members of an argument buffer's struct type, as a binding's encoder
/// takes them (encoderOfBinding:); nil where a member is of a type no descriptor names.
NSArray<MTLArgumentDescriptor *> *_Nullable RgfxDescriptorsOfArguments(MTLStructType *type);

/// What holds the device's region: the device answers it, so a function's encoder finds the
/// block its layout is written into.
@protocol RgfxRegionHolder <NSObject>
#ifdef __cplusplus
- (nullable RgfxRegion *)region;
#endif
@end

NS_ASSUME_NONNULL_END

#endif // RGFX_ARGUMENT_ENCODER_H_
