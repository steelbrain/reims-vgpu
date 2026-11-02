// sweep.h: what the sweep's cases share (sweep.m holds the tool's description and the helpers).
#ifndef SWEEP_H
#define SWEEP_H

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

// A case's tolerance: the largest difference allowed on a channel of a pixel compared, and the
// mean difference over them, in 8-bit levels (0..255) unless the case says otherwise.
typedef struct {
	double max, mean;
} SweepTol;

// What a case measured: its largest and mean difference, or why it did not run (skip), or why it
// failed whatever its numbers (fail). `note` follows the numbers on its line.
@interface SweepResult : NSObject
@property(nonatomic) double max, mean;
@property(nonatomic, copy, nullable) NSString *note, *skip, *fail;
+ (nonnull instancetype)max:(double)max mean:(double)mean;
+ (nonnull instancetype)skip:(nonnull NSString *)why;
+ (nonnull instancetype)fail:(nonnull NSString *)why;
@end

// RGBA8 pixels in sRGB, premultiplied alpha, rows top first, tightly packed.
@interface SweepPixels : NSObject
@property(nonatomic, readonly) size_t width, height;
@property(nonatomic, readonly, nonnull) NSMutableData *data;
+ (nonnull instancetype)width:(size_t)width height:(size_t)height;
// `image` drawn into new pixels of its size; nil for a null image.
+ (nullable instancetype)ofImage:(nullable CGImageRef)image;
- (nonnull uint8_t *)bytes;
- (nonnull uint8_t *)at:(size_t)x y:(size_t)y;
// A CGImage over a copy of the pixels; the caller releases it.
- (nonnull CGImageRef)newImage CF_RETURNS_RETAINED;
@end

// The patterns the cases start from, made on the CPU:
//   SweepSmooth      opaque, smooth gradients in each channel;
//   SweepStructured  SweepSmooth with sharp edges over it: a checkerboard band, diagonal lines,
//                    a disc and a frame of known colours;
//   SweepAlpha       SweepStructured with an alpha ramp from left (opaque) to right (clear).
typedef enum { SweepSmooth, SweepStructured, SweepAlpha } SweepPatternKind;
SweepPixels *_Nonnull sweepPattern(size_t width, size_t height, SweepPatternKind kind);

// The difference of `got` from `want` over `rect` (pixels, origin at the top left; CGRectNull for
// the whole image) on the first `channels` channels (3: colour, 4: with alpha): max and mean. A
// size mismatch fails.
SweepResult *_Nonnull sweepCompare(SweepPixels *_Nullable got, SweepPixels *_Nullable want,
                                   CGRect rect, int channels);
// As sweepCompare, but the max only over the pixels whose 3 x 3 neighbourhood in `want` is of one
// colour (away from edges, where anti-aliasing may differ); the mean over every pixel.
SweepResult *_Nonnull sweepCompareFlat(SweepPixels *_Nullable got, SweepPixels *_Nullable want,
                                       int channels);
// Whether `rect` of `p` (CGRectNull: all of it) holds one colour.
BOOL sweepOneColour(SweepPixels *_Nonnull p, CGRect rect);

// sRGB, which every case draws and reads in.
CGColorSpaceRef _Nonnull sweepSRGB(void);
// The device the cases that choose one use (-d NAME, else the system default device).
id<MTLDevice> _Nonnull sweepDevice(void);
// A scratch file path of this run, ending in `suffix`.
NSString *_Nonnull sweepTempPath(NSString *_Nonnull suffix);

// Runs the case `name` where it matches the command line's pattern: prints its line, counts it.
// A result over `tol` fails; an exception fails with its reason.
void sweepCase(NSString *_Nonnull name, SweepTol tol, SweepResult *_Nonnull (^_Nonnull run)(void));

void sweepImageIO(void);
void sweepCoreImage(void);
void sweepVideo(void);
void sweepMPS(void);
void sweepCoreAnimation(void);

#endif
