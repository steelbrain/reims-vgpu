// sweep_video: AVFoundation and VideoToolbox, whose decoders and pixel transfers may convert and
// scale on the Metal device.
//   video.<codec>.generator, video.<codec>.reader
//     a 1 s clip of 30 frames of 640 x 360, made on the CPU, encoded here with AVAssetWriter
//     (codec h264, hevc, hevcalpha; tagged BT.709 primaries and matrix and the sRGB transfer
//     function, so a read back as sRGB is not converted), then frames 0, 15 and 29 read back by
//     AVAssetImageGenerator (as CGImages) and every frame by AVAssetReader (as 32BGRA). Each frame
//     carries its number in a bar of eight blocks: a frame read back must carry the number it was
//     asked for (the reader: 0 to 29 in order), so a stale or repeated frame fails whatever its
//     pixels. The frames compared are lossy: a mean, no max. hevcalpha compares alpha too. A codec
//     AVAssetWriter has no encoder for here is SKIP.
//   vt.420v.planes      VTPixelTransferSession from 32BGRA to 420v (BT.601, video range) against
//                       vImage's conversion of the same pixels (the smooth pattern): the luma and
//                       chroma planes;
//   vt.420v.roundtrip   the same 420v back to 32BGRA by VTPixelTransferSession, against vImage's
//                       conversion of vImage's planes;
//   vt.scale            32BGRA scaled 640 x 360 to 320 x 180 by VTPixelTransferSession against
//                       vImageScale_ARGB8888.
// Tolerances (sweep.m): the starting values hold but for vt.420v.roundtrip's max, 4.5, set by
// stock's 3. The worst of the others: video.h264 155/0.98 and video.hevc 156/0.53 (one frame's
// moving box edge), video.hevcalpha 10/0.62 (AMD, Intel), vt.420v.planes 1/0.39 (stock),
// vt.scale 1/0.48 (AMD, Intel).
#import "sweep.h"

#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>

enum { kW = 640, kH = 360, kFrames = 30, kBits = 8 };

// Frame `i`: the smooth pattern, a box moving right, and the frame number in a bar of eight
// blocks (bit 7 first, light for 1) at the top. With `alpha`, alpha falls from 255 at the left
// to 64 at the right, premultiplied.
static SweepPixels *frame(int i, bool alpha)
{
	SweepPixels *p = sweepPattern(kW, kH, SweepSmooth);
	for (size_t y = 140; y < 220; y++)
		for (size_t x = 40 + (size_t)i * 16; x < 120 + (size_t)i * 16; x++) {
			uint8_t *q = [p at:x y:y];
			q[0] = 220, q[1] = 60, q[2] = 40;
		}
	for (int b = 0; b < kBits; b++) {
		const uint8_t v = (i >> (kBits - 1 - b)) & 1 ? 235 : 16;
		for (size_t y = 8; y < 32; y++)
			for (size_t x = 16 + (size_t)b * 72; x < 80 + (size_t)b * 72; x++) {
				uint8_t *q = [p at:x y:y];
				q[0] = q[1] = q[2] = v;
			}
	}
	if (alpha) {
		for (size_t y = 0; y < kH; y++)
			for (size_t x = 0; x < kW; x++) {
				uint8_t *q = [p at:x y:y];
				const int a = 255 - (int)(x * 191 / (kW - 1));
				for (int k = 0; k < 3; k++)
					q[k] = (uint8_t)((q[k] * a + 127) / 255);
				q[3] = (uint8_t)a;
			}
	}
	return p;
}

// The number a frame read back carries, from its bar's blocks (unpremultiplied by their alpha).
static int frameNumber(SweepPixels *p)
{
	int n = 0;
	for (int b = 0; b < kBits; b++) {
		const uint8_t *q = [p at:48 + (size_t)b * 72 y:20];
		const int a = q[3] ? q[3] : 1;
		n = n << 1 | (q[0] * 255 / a > 128 ? 1 : 0);
	}
	return n;
}

static CVPixelBufferRef newBuffer(size_t w, size_t h, OSType format) CF_RETURNS_RETAINED
{
	CVPixelBufferRef buf = NULL;
	CVPixelBufferCreate(NULL, w, h, format,
	                    (__bridge CFDictionaryRef) @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}}, &buf);
	return buf;
}

// RGBA pixels into a 32BGRA buffer, and back.
static void toBGRA(SweepPixels *p, CVPixelBufferRef buf)
{
	CVPixelBufferLockBaseAddress(buf, 0);
	uint8_t *base = CVPixelBufferGetBaseAddress(buf);
	const size_t rb = CVPixelBufferGetBytesPerRow(buf);
	for (size_t y = 0; y < p.height; y++)
		for (size_t x = 0; x < p.width; x++) {
			const uint8_t *s = [p at:x y:y];
			uint8_t *d = base + y * rb + x * 4;
			d[0] = s[2], d[1] = s[1], d[2] = s[0], d[3] = s[3];
		}
	CVPixelBufferUnlockBaseAddress(buf, 0);
}

static SweepPixels *fromBGRA(CVPixelBufferRef buf)
{
	CVPixelBufferLockBaseAddress(buf, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(buf);
	const size_t rb = CVPixelBufferGetBytesPerRow(buf);
	SweepPixels *p = [SweepPixels width:CVPixelBufferGetWidth(buf) height:CVPixelBufferGetHeight(buf)];
	for (size_t y = 0; y < p.height; y++)
		for (size_t x = 0; x < p.width; x++) {
			const uint8_t *s = base + y * rb + x * 4;
			uint8_t *d = [p at:x y:y];
			d[0] = s[2], d[1] = s[1], d[2] = s[0], d[3] = s[3];
		}
	CVPixelBufferUnlockBaseAddress(buf, kCVPixelBufferLock_ReadOnly);
	return p;
}

static bool waitFor(dispatch_semaphore_t s, int seconds)
{
	return dispatch_semaphore_wait(s, dispatch_time(DISPATCH_TIME_NOW, (int64_t)seconds * NSEC_PER_SEC)) == 0;
}

// The clip of `codec` made once: its URL, or nil with *skip (no encoder here) or *fail.
static NSURL *clip(NSString *codec, NSString **skip, NSString **fail)
{
	static NSMutableDictionary<NSString *, id> *made;
	if (made == nil)
		made = [NSMutableDictionary dictionary];
	if (made[codec] != nil) {
		id v = made[codec];
		if ([v isKindOfClass:[NSURL class]])
			return v;
		NSArray *why = v;
		if ([why[0] length])
			*skip = why[0];
		else
			*fail = why[1];
		return nil;
	}
	NSString *type = [codec isEqualToString:@"h264"]   ? AVVideoCodecTypeH264
	                 : [codec isEqualToString:@"hevc"] ? AVVideoCodecTypeHEVC
	                                                   : AVVideoCodecTypeHEVCWithAlpha;
	const bool alpha = [codec isEqualToString:@"hevcalpha"];
	NSURL *url = [NSURL fileURLWithPath:sweepTempPath(@".mov")];
	NSError *err = nil;
	AVAssetWriter *w = [AVAssetWriter assetWriterWithURL:url fileType:AVFileTypeQuickTimeMovie error:&err];
	NSDictionary *settings = @{
		AVVideoCodecKey : type,
		AVVideoWidthKey : @(kW),
		AVVideoHeightKey : @(kH),
		AVVideoColorPropertiesKey : @{
			AVVideoColorPrimariesKey : AVVideoColorPrimaries_ITU_R_709_2,
			AVVideoTransferFunctionKey : AVVideoTransferFunction_IEC_sRGB,
			AVVideoYCbCrMatrixKey : AVVideoYCbCrMatrix_ITU_R_709_2,
		},
		AVVideoCompressionPropertiesKey : @{AVVideoAverageBitRateKey : @20000000},
	};
	NSString *s = nil, *f = nil;
	if (w == nil) {
		f = [NSString stringWithFormat:@"no writer: %@", err.localizedDescription];
	} else if (![w canApplyOutputSettings:settings forMediaType:AVMediaTypeVideo]) {
		s = [NSString stringWithFormat:@"AVAssetWriter takes no %@ settings here", codec];
	} else {
		AVAssetWriterInput *in = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
		                                                            outputSettings:settings];
		in.expectsMediaDataInRealTime = NO;
		AVAssetWriterInputPixelBufferAdaptor *ad = [AVAssetWriterInputPixelBufferAdaptor
		    assetWriterInputPixelBufferAdaptorWithAssetWriterInput:in
		                               sourcePixelBufferAttributes:@{
			                               (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
			                               (id)kCVPixelBufferWidthKey : @(kW),
			                               (id)kCVPixelBufferHeightKey : @(kH),
		                               }];
		[w addInput:in];
		if (![w startWriting]) {
			NSError *e = w.error;
			if (e.code == AVErrorEncoderNotFound || e.code == AVErrorEncoderTemporarilyUnavailable)
				s = [NSString stringWithFormat:@"no %@ encoder here: %@", codec, e.localizedDescription];
			else
				f = [NSString stringWithFormat:@"startWriting: %@", e];
		} else {
			[w startSessionAtSourceTime:kCMTimeZero];
			for (int i = 0; i < kFrames && f == nil; i++) {
				for (int t = 0; t < 10000 && !in.readyForMoreMediaData; t++)
					usleep(1000);
				CVPixelBufferRef buf = newBuffer(kW, kH, kCVPixelFormatType_32BGRA);
				toBGRA(frame(i, alpha), buf);
				if (![ad appendPixelBuffer:buf withPresentationTime:CMTimeMake(i, 30)])
					f = [NSString stringWithFormat:@"append %d: %@", i, w.error];
				CVPixelBufferRelease(buf);
			}
			[in markAsFinished];
			dispatch_semaphore_t done = dispatch_semaphore_create(0);
			[w finishWritingWithCompletionHandler:^{
				dispatch_semaphore_signal(done);
			}];
			if (!waitFor(done, 60))
				f = f ?: @"the writer did not finish in 60 s";
			else if (w.status != AVAssetWriterStatusCompleted)
				f = f ?: [NSString stringWithFormat:@"the writer ended %ld: %@", (long)w.status, w.error];
		}
	}
	if (s == nil && f == nil) {
		made[codec] = url;
		return url;
	}
	made[codec] = @[ s ?: @"", f ?: @"" ];
	*skip = s, *fail = f;
	return nil;
}

// A read frame against its source: the number it carries first, then its pixels.
static SweepResult *against(SweepPixels *got, int i, bool alpha, double *worstMean, NSString **bad)
{
	if (got == nil) {
		*bad = [NSString stringWithFormat:@"frame %d: nothing read", i];
		return nil;
	}
	if (frameNumber(got) != i) {
		*bad = [NSString stringWithFormat:@"frame %d read back as frame %d", i, frameNumber(got)];
		return nil;
	}
	SweepResult *r = sweepCompare(got, frame(i, alpha), CGRectNull, alpha ? 4 : 3);
	if (r.fail != nil)
		*bad = [NSString stringWithFormat:@"frame %d: %@", i, r.fail];
	*worstMean = fmax(*worstMean, r.mean);
	return r;
}

static void codecCases(NSString *codec)
{
	const bool alpha = [codec isEqualToString:@"hevcalpha"];
	// Starting values: a 20 Mbit/s clip of smooth gradients and flat blocks is within a few
	// levels on average; the frame number is exact.
	sweepCase([NSString stringWithFormat:@"video.%@.generator", codec], (SweepTol){255, 6}, ^SweepResult * {
		NSString *skip = nil, *fail = nil;
		NSURL *url = clip(codec, &skip, &fail);
		if (url == nil)
			return skip ? [SweepResult skip:skip] : [SweepResult fail:fail];
		AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
		AVAssetImageGenerator *gen = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
		gen.requestedTimeToleranceBefore = kCMTimeZero;
		gen.requestedTimeToleranceAfter = kCMTimeZero;
		double worst = 0, mean = 0;
		for (int n = 0; n < 3; n++) {
			const int i = n == 0 ? 0 : n == 1 ? 15 : 29;
			__block CGImageRef image = NULL;
			__block NSError *error = nil;
			dispatch_semaphore_t done = dispatch_semaphore_create(0);
			[gen generateCGImageAsynchronouslyForTime:CMTimeMake(i, 30)
			                        completionHandler:^(CGImageRef im, CMTime t, NSError *e) {
				                        image = im ? CGImageRetain(im) : NULL;
				                        error = e;
				                        dispatch_semaphore_signal(done);
			                        }];
			if (!waitFor(done, 30))
				return [SweepResult fail:[NSString stringWithFormat:@"frame %d: no image in 30 s", i]];
			SweepPixels *got = [SweepPixels ofImage:image];
			if (image != NULL)
				CGImageRelease(image);
			if (got == nil)
				return [SweepResult fail:[NSString stringWithFormat:@"frame %d: %@", i, error]];
			NSString *bad = nil;
			SweepResult *r = against(got, i, alpha, &mean, &bad);
			if (bad != nil)
				return [SweepResult fail:bad];
			worst = fmax(worst, r.max);
		}
		return [SweepResult max:worst mean:mean];
	});
	sweepCase([NSString stringWithFormat:@"video.%@.reader", codec], (SweepTol){255, 6}, ^SweepResult * {
		NSString *skip = nil, *fail = nil;
		NSURL *url = clip(codec, &skip, &fail);
		if (url == nil)
			return skip ? [SweepResult skip:skip] : [SweepResult fail:fail];
		AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
		__block AVAssetTrack *track = nil;
		dispatch_semaphore_t loaded = dispatch_semaphore_create(0);
		[asset loadTracksWithMediaType:AVMediaTypeVideo
		             completionHandler:^(NSArray<AVAssetTrack *> *tracks, NSError *e) {
			             track = tracks.firstObject;
			             dispatch_semaphore_signal(loaded);
		             }];
		if (!waitFor(loaded, 30) || track == nil)
			return [SweepResult fail:@"no video track"];
		NSError *err = nil;
		AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&err];
		AVAssetReaderTrackOutput *out = [AVAssetReaderTrackOutput
		    assetReaderTrackOutputWithTrack:track
		                     outputSettings:@{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)}];
		[reader addOutput:out];
		if (![reader startReading])
			return [SweepResult fail:[NSString stringWithFormat:@"startReading: %@", reader.error]];
		double worst = 0, mean = 0;
		int i = 0;
		for (CMSampleBufferRef sb; (sb = [out copyNextSampleBuffer]) != NULL; i++) {
			CVPixelBufferRef px = CMSampleBufferGetImageBuffer(sb);
			SweepPixels *got = px ? fromBGRA(px) : nil;
			CFRelease(sb);
			if (got == nil)
				return [SweepResult fail:[NSString stringWithFormat:@"sample %d has no image", i]];
			if (frameNumber(got) != i)
				return [SweepResult fail:[NSString stringWithFormat:@"sample %d carries frame %d", i,
				                                                    frameNumber(got)]];
			if (i == 0 || i == 15 || i == 29) {
				NSString *bad = nil;
				SweepResult *r = against(got, i, alpha, &mean, &bad);
				if (bad != nil)
					return [SweepResult fail:bad];
				worst = fmax(worst, r.max);
			}
		}
		if (reader.status != AVAssetReaderStatusCompleted || i != kFrames)
			return [SweepResult fail:[NSString stringWithFormat:@"%d samples read, status %ld %@", i,
			                                                    (long)reader.status, reader.error]];
		return [SweepResult max:worst mean:mean];
	});
}

// ARGB from (and to) B G R A, as vImage's permute maps take it.
static const uint8_t kBGRAMap[4] = {3, 2, 1, 0};

// BT.601 video range as vImage takes it.
static const vImage_YpCbCrPixelRange kVideoRange = {16, 128, 235, 240, 235, 16, 240, 16};

static void tagged(CVPixelBufferRef buf)
{
	CVBufferSetAttachment(buf, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2,
	                      kCVAttachmentMode_ShouldPropagate);
	CVBufferSetAttachment(buf, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2,
	                      kCVAttachmentMode_ShouldPropagate);
	CVBufferSetAttachment(buf, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4,
	                      kCVAttachmentMode_ShouldPropagate);
}

static vImage_Buffer planeOf(CVPixelBufferRef buf, size_t plane)
{
	return (vImage_Buffer){CVPixelBufferGetBaseAddressOfPlane(buf, plane), CVPixelBufferGetHeightOfPlane(buf, plane),
	                       CVPixelBufferGetWidthOfPlane(buf, plane), CVPixelBufferGetBytesPerRowOfPlane(buf, plane)};
}

static vImage_Buffer whole(CVPixelBufferRef buf)
{
	return (vImage_Buffer){CVPixelBufferGetBaseAddress(buf), CVPixelBufferGetHeight(buf), CVPixelBufferGetWidth(buf),
	                       CVPixelBufferGetBytesPerRow(buf)};
}

static bool transfer(CVPixelBufferRef from, CVPixelBufferRef to, NSString **why)
{
	VTPixelTransferSessionRef s = NULL;
	OSStatus st = VTPixelTransferSessionCreate(NULL, &s);
	if (st == noErr)
		st = VTPixelTransferSessionTransferImage(s, from, to);
	if (s != NULL) {
		VTPixelTransferSessionInvalidate(s);
		CFRelease(s);
	}
	if (st != noErr)
		*why = [NSString stringWithFormat:@"VTPixelTransferSession: %d", (int)st];
	return st == noErr;
}

void sweepVideo(void)
{
	for (NSString *codec in @[ @"h264", @"hevc", @"hevcalpha" ])
		codecCases(codec);

	// The source of the transfers: the smooth pattern, where chroma and resampling filters of
	// either side agree.
	SweepPixels *src = sweepPattern(kW, kH, SweepSmooth);

	// Starting values: one matrix and range on both sides; rounding and chroma filters a level
	// or two apart.
	sweepCase(@"vt.420v.planes", (SweepTol){2, 1}, ^SweepResult * {
		CVPixelBufferRef bgra = newBuffer(kW, kH, kCVPixelFormatType_32BGRA);
		CVPixelBufferRef vt = newBuffer(kW, kH, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
		CVPixelBufferRef vi = newBuffer(kW, kH, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
		toBGRA(src, bgra);
		tagged(bgra), tagged(vt);
		NSString *why = nil;
		SweepResult *res = nil;
		if (!transfer(bgra, vt, &why)) {
			res = [SweepResult fail:why];
		} else {
			vImage_ARGBToYpCbCr info;
			vImageConvert_ARGBToYpCbCr_GenerateConversion(kvImage_ARGBToYpCbCrMatrix_ITU_R_601_4, &kVideoRange,
			                                              &info, kvImageARGB8888, kvImage420Yp8_CbCr8, 0);
			CVPixelBufferLockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
			CVPixelBufferLockBaseAddress(vi, 0);
			vImage_Buffer in = whole(bgra), y = planeOf(vi, 0), c = planeOf(vi, 1);
			vImageConvert_ARGB8888To420Yp8_CbCr8(&in, &y, &c, &info, kBGRAMap, kvImageNoFlags);
			CVPixelBufferUnlockBaseAddress(bgra, kCVPixelBufferLock_ReadOnly);
			CVPixelBufferLockBaseAddress(vt, kCVPixelBufferLock_ReadOnly);
			int worst = 0;
			double sum = 0, n = 0;
			for (size_t plane = 0; plane < 2; plane++) {
				vImage_Buffer a = planeOf(vt, plane), b = planeOf(vi, plane);
				const size_t bytes = a.width * (plane ? 2 : 1);
				for (size_t r = 0; r < a.height; r++)
					for (size_t k = 0; k < bytes; k++) {
						const int d = abs((int)((uint8_t *)a.data)[r * a.rowBytes + k] -
						                  (int)((uint8_t *)b.data)[r * b.rowBytes + k]);
						worst = d > worst ? d : worst, sum += d, n++;
					}
			}
			CVPixelBufferUnlockBaseAddress(vt, kCVPixelBufferLock_ReadOnly);
			CVPixelBufferUnlockBaseAddress(vi, 0);
			res = [SweepResult max:worst mean:sum / n];
			res.note = @"in code values";
		}
		CVPixelBufferRelease(bgra), CVPixelBufferRelease(vt), CVPixelBufferRelease(vi);
		return res;
	});
	// Starting values: VideoToolbox's chroma upsampling against vImage's (measured on the M3 Max:
	// max 3, mean 0.56). The max is stock's 3 times 1.5.
	sweepCase(@"vt.420v.roundtrip", (SweepTol){4.5, 1}, ^SweepResult * {
		CVPixelBufferRef bgra = newBuffer(kW, kH, kCVPixelFormatType_32BGRA);
		CVPixelBufferRef yuv = newBuffer(kW, kH, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
		CVPixelBufferRef back = newBuffer(kW, kH, kCVPixelFormatType_32BGRA);
		toBGRA(src, bgra);
		tagged(bgra), tagged(yuv), tagged(back);
		NSString *why = nil;
		SweepResult *res = nil;
		if (!transfer(bgra, yuv, &why) || !transfer(yuv, back, &why)) {
			res = [SweepResult fail:why];
		} else {
			// vImage's conversion back of VideoToolbox's own planes: the same input on both sides.
			vImage_YpCbCrToARGB info;
			vImageConvert_YpCbCrToARGB_GenerateConversion(kvImage_YpCbCrToARGBMatrix_ITU_R_601_4, &kVideoRange,
			                                              &info, kvImage420Yp8_CbCr8, kvImageARGB8888, 0);
			CVPixelBufferRef ref = newBuffer(kW, kH, kCVPixelFormatType_32BGRA);
			CVPixelBufferLockBaseAddress(yuv, kCVPixelBufferLock_ReadOnly);
			CVPixelBufferLockBaseAddress(ref, 0);
			vImage_Buffer y = planeOf(yuv, 0), c = planeOf(yuv, 1), out = whole(ref);
			vImageConvert_420Yp8_CbCr8ToARGB8888(&y, &c, &out, &info, kBGRAMap, 255, kvImageNoFlags);
			CVPixelBufferUnlockBaseAddress(ref, 0);
			CVPixelBufferUnlockBaseAddress(yuv, kCVPixelBufferLock_ReadOnly);
			res = sweepCompare(fromBGRA(back), fromBGRA(ref), CGRectNull, 3);
			CVPixelBufferRelease(ref);
		}
		CVPixelBufferRelease(bgra), CVPixelBufferRelease(yuv), CVPixelBufferRelease(back);
		return res;
	});
	// Starting values: two resamplers' filters differ, most at the pattern's sharp edges.
	sweepCase(@"vt.scale", (SweepTol){8, 2}, ^SweepResult * {
		CVPixelBufferRef big = newBuffer(kW, kH, kCVPixelFormatType_32BGRA);
		CVPixelBufferRef vt = newBuffer(kW / 2, kH / 2, kCVPixelFormatType_32BGRA);
		CVPixelBufferRef vi = newBuffer(kW / 2, kH / 2, kCVPixelFormatType_32BGRA);
		toBGRA(src, big);
		tagged(big), tagged(vt);
		NSString *why = nil;
		SweepResult *res = nil;
		if (!transfer(big, vt, &why)) {
			res = [SweepResult fail:why];
		} else {
			CVPixelBufferLockBaseAddress(big, kCVPixelBufferLock_ReadOnly);
			CVPixelBufferLockBaseAddress(vi, 0);
			vImage_Buffer in = whole(big), out = whole(vi);
			vImageScale_ARGB8888(&in, &out, NULL, kvImageHighQualityResampling);
			CVPixelBufferUnlockBaseAddress(vi, 0);
			CVPixelBufferUnlockBaseAddress(big, kCVPixelBufferLock_ReadOnly);
			res = sweepCompare(fromBGRA(vt), fromBGRA(vi), CGRectNull, 3);
		}
		CVPixelBufferRelease(big), CVPixelBufferRelease(vt), CVPixelBufferRelease(vi);
		return res;
	});
}
